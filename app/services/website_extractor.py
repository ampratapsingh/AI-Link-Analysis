import asyncio
import re
import zlib
from dataclasses import dataclass
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup, NavigableString, Tag

from app.config import Settings
from app.services.url_safety import UnsafeUrlError, parse_website_url, resolve_destination


class WebsiteFetchError(Exception):
    """A website response could not be fetched or processed."""

    def __init__(self, code: str, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


@dataclass(frozen=True)
class PageLink:
    text: str
    url: str


@dataclass(frozen=True)
class ExtractionResult:
    requested_url: str
    final_url: str
    title: str | None
    meta_description: str | None
    meta_keywords: list[str]
    h1_headings: list[str]
    h2_headings: list[str]
    h3_headings: list[str]
    main_content: str
    links: list[PageLink]
    image_alt_text: list[str]
    warnings: list[str]
    success: bool


_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_HTML_CONTENT_TYPES = {"text/html", "application/xhtml+xml"}
_SITE_ERRORS = {
    401: ("access_blocked", "The website requires authentication."),
    403: ("access_blocked", "The website denied access to the page."),
    404: ("page_not_found", "The website page was not found."),
    429: ("rate_limited", "The website is rate limiting requests. Try again later."),
}
_NOISE = (
    "script, style, nav, footer, noscript, template, iframe, svg, title, "
    "[hidden], [inert], [aria-hidden='true' i], [role='navigation'], [role='contentinfo'], "
    ".cookie-banner, #cookie-banner, .cookie-consent, #cookie-consent, #onetrust-banner-sdk"
)
_HIDDEN_STYLE = re.compile(r"(?:^|;)\s*(?:display\s*:\s*none|visibility\s*:\s*hidden)\b", re.I)
_BLOCK_TAGS = {
    "address", "article", "blockquote", "br", "dd", "div", "dl", "dt", "figcaption",
    "h1", "h2", "h3", "h4", "h5", "h6", "header", "li", "main", "p", "section", "td",
}
_CHALLENGE_TITLES = {
    "access denied", "attention required! | cloudflare", "just a moment...",
    "request blocked", "verify you are human",
}


class WebsiteExtractor:
    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._transport = transport

    async def extract(self, url: str) -> ExtractionResult:
        requested_url = parse_website_url(url)
        try:
            async with asyncio.timeout(self._settings.website_request_timeout):
                final_url, html, encoding = await self._fetch(requested_url)
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise WebsiteFetchError("timeout", "The website request timed out.") from exc
        except httpx.RequestError as exc:
            raise WebsiteFetchError("connection_failed", "Could not fetch the website.") from exc

        return await asyncio.to_thread(_extract_html, url, final_url, html, encoding)

    async def _fetch(self, url: httpx.URL) -> tuple[httpx.URL, bytes, str | None]:
        visited = {url}
        async with httpx.AsyncClient(
            headers={
                "Accept": "text/html, application/xhtml+xml",
                "Accept-Encoding": "gzip, identity",
                "User-Agent": self._settings.website_user_agent,
            },
            timeout=httpx.Timeout(self._settings.website_request_timeout),
            # Different hostnames can share a pinned IP but require different TLS identities.
            limits=httpx.Limits(max_keepalive_connections=0),
            follow_redirects=False,
            trust_env=False,
            transport=self._transport,
        ) as client:
            for hop in range(self._settings.max_website_redirects + 1):
                destination = await resolve_destination(url)
                client.cookies.clear()
                async with client.stream(
                    "GET",
                    destination.connection_url,
                    headers={"Host": url.netloc.decode("ascii")},
                    extensions={"sni_hostname": destination.tls_hostname},
                ) as response:
                    if response.status_code not in _REDIRECT_STATUSES:
                        _check_response(response)
                        return url, await _read_html(
                            response, self._settings.max_website_response_size
                        ), response.charset_encoding
                    if hop == self._settings.max_website_redirects:
                        break
                    location = response.headers.get("location")
                    if not location:
                        raise WebsiteFetchError(
                            "invalid_redirect", "The website returned an incomplete redirect."
                        )
                    # Check the raw location before urljoin can strip control characters.
                    if any(ord(char) <= 32 or ord(char) == 127 for char in location) or "\\" in location:
                        raise UnsafeUrlError("The website returned an invalid redirect URL.")
                    try:
                        redirected_url = urljoin(str(url), location)
                    except ValueError as exc:
                        raise UnsafeUrlError("The website returned an invalid redirect URL.") from exc
                    url = parse_website_url(redirected_url)
                    if url in visited:
                        raise WebsiteFetchError("redirect_loop", "The website redirected in a loop.")
                    visited.add(url)

        raise WebsiteFetchError("too_many_redirects", "The website exceeded the redirect limit.")


def _check_response(response: httpx.Response) -> None:
    if not response.is_success:
        code, message = _SITE_ERRORS.get(
            response.status_code,
            ("website_http_error", f"The website returned HTTP {response.status_code}."),
        )
        raise WebsiteFetchError(code, message, status_code=response.status_code)
    content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type not in _HTML_CONTENT_TYPES:
        raise WebsiteFetchError("unsupported_content_type", "The website did not return HTML.")


async def _read_html(response: httpx.Response, limit: int) -> bytes:
    too_large = WebsiteFetchError("response_too_large", "The HTML exceeded the configured size limit.")
    content_length = response.headers.get("content-length", "").strip().lstrip("0")
    if content_length.isascii() and content_length.isdecimal():
        if len(content_length) > len(str(limit)) or int(content_length) > limit:
            raise too_large

    encoding = response.headers.get("content-encoding", "identity").lower().strip()
    if encoding not in {"identity", "gzip"}:
        raise WebsiteFetchError("unsupported_encoding", "The website returned unsupported compression.")
    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS) if encoding == "gzip" else None

    wire_bytes = 0
    body = bytearray()
    try:
        async for chunk in response.aiter_raw():
            wire_bytes += len(chunk)
            if wire_bytes > limit:
                raise too_large
            remaining = limit - len(body)
            decoded = decoder.decompress(chunk, remaining + 1) if decoder else chunk
            if len(decoded) > remaining:
                raise too_large
            if decoder and decoder.unused_data:
                raise zlib.error("Data after the gzip member.")
            body.extend(decoded)
        if decoder and not decoder.eof:
            raise zlib.error("Incomplete gzip member.")
    except zlib.error as exc:
        raise WebsiteFetchError("invalid_compression", "The website returned unreadable compressed HTML.") from exc
    return bytes(body)


def _clean_text(value: str) -> str:
    return " ".join(value.split())


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _meta(soup: BeautifulSoup, name: str) -> str | None:
    tag = soup.select_one(f"meta[name='{name}' i]")
    value = tag.get("content") if tag else None
    return (_clean_text(value) or None) if isinstance(value, str) else None


def _strip_noise(body: Tag) -> None:
    for tag in body.select(_NOISE):
        if tag.parent is not None:
            tag.decompose()
    for tag in body.select("[style], header, [role='banner'], .site-header, #site-header"):
        if tag.parent is None:
            continue
        if _HIDDEN_STYLE.search(str(tag.get("style", ""))):
            tag.decompose()
        elif (
            tag.name == "header"
            or tag.get("role") == "banner"
            or tag.get("id") == "site-header"
            or "site-header" in tag.get("class", [])
        ) and not tag.select_one("h1, h2, h3, p"):
            tag.decompose()


def _text_blocks(body: Tag) -> list[str]:
    blocks: list[str] = []
    parts: list[str] = []
    stack = [(None, iter(body.children))]
    # Walk the tree iteratively and use block boundaries to keep inline text together.
    while stack:
        closing_tag, nodes = stack[-1]
        node = next(nodes, None)
        if node is None:
            stack.pop()
            boundary = closing_tag in _BLOCK_TAGS
        elif isinstance(node, Tag):
            stack.append((node.name, iter(node.children)))
            boundary = node.name in _BLOCK_TAGS
        else:
            if type(node) is NavigableString:
                parts.append(str(node))
            continue
        if boundary and parts:
            blocks.append(_clean_text("".join(parts)))
            parts.clear()
    if parts:
        blocks.append(_clean_text("".join(parts)))
    return _unique(blocks)


def _extract_links(body: Tag, final_url: httpx.URL) -> list[PageLink]:
    links: dict[PageLink, None] = {}
    for tag in body.find_all("a", href=True):
        text = _clean_text(tag.get_text(" ", strip=True))
        href = str(tag["href"]).strip()
        if not text or not href or href.startswith("#"):
            continue
        try:
            absolute_url = parse_website_url(urljoin(str(final_url), href))
        except ValueError:
            continue
        links[PageLink(text, str(absolute_url))] = None
    return list(links)


def _extract_html(
    requested_url: str, final_url: httpx.URL, html: bytes, encoding: str | None
) -> ExtractionResult:
    soup = BeautifulSoup(html, "html.parser", from_encoding=encoding)
    title = _clean_text(soup.title.get_text(" ", strip=True)) if soup.title else None
    description = _meta(soup, "description")
    keywords = _meta(soup, "keywords")
    javascript_hint = (
        soup.select_one("script") is not None
        and soup.select_one("#app, #root, #__next, #__nuxt") is not None
    ) or any(
        "javascript" in tag.get_text(" ", strip=True).lower()
        for tag in soup.find_all("noscript")
    )
    challenge_hint = soup.select_one("#challenge-form, .g-recaptcha, .h-captcha, .cf-turnstile")

    if soup.head is not None:
        soup.head.decompose()
    body = soup.body if soup.body is not None else soup
    _strip_noise(body)
    content = "\n".join(_text_blocks(body))

    warnings = []
    if not title:
        warnings.append("The page has no title.")
    if not description:
        warnings.append("The page has no meta description.")
    if (title or "").lower() in _CHALLENGE_TITLES or (challenge_hint is not None and len(content) < 200):
        failure = "The website returned a bot challenge or CAPTCHA."
    elif len(content) < 80 and javascript_hint:
        failure = "The page appears to require JavaScript and contains insufficient readable HTML."
    elif not content:
        failure = "The page contains no readable HTML content."
    else:
        failure = None
    if failure:
        warnings.append(failure)

    return ExtractionResult(
        requested_url=requested_url,
        final_url=str(final_url),
        title=title or None,
        meta_description=description,
        meta_keywords=_unique([_clean_text(word) for word in (keywords or "").split(",")]),
        h1_headings=_unique([_clean_text(tag.get_text(" ", strip=True)) for tag in body.find_all("h1")]),
        h2_headings=_unique([_clean_text(tag.get_text(" ", strip=True)) for tag in body.find_all("h2")]),
        h3_headings=_unique([_clean_text(tag.get_text(" ", strip=True)) for tag in body.find_all("h3")]),
        main_content=content,
        links=_extract_links(body, final_url),
        image_alt_text=_unique([_clean_text(str(tag["alt"])) for tag in body.find_all("img", alt=True)]),
        warnings=warnings,
        success=failure is None,
    )
