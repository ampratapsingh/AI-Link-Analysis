import asyncio
import gzip

import httpx
import pytest

from app.config import Settings
from app.services.url_safety import UnsafeUrlError
from app.services.website_extractor import PageLink, WebsiteExtractor, WebsiteFetchError

pytestmark = [pytest.mark.anyio, pytest.mark.parametrize("anyio_backend", ["asyncio"])]

URL = "https://example.com/"


class BodyStream(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk


def html_response(*chunks, status=200, headers=None):
    return httpx.Response(
        status,
        headers={"content-type": "text/html", **(headers or {})},
        stream=BodyStream(chunks),
    )


@pytest.fixture(autouse=True)
async def public_dns(monkeypatch):
    async def getaddrinfo(host, port, **kwargs):
        address = "10.0.0.5" if host == "internal.example.com." else "8.8.8.8"
        return [(0, 0, 0, "", (address, port))]

    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", getaddrinfo)


def extractor(handler, **settings):
    return WebsiteExtractor(Settings(**settings), transport=httpx.MockTransport(handler))


async def test_extracts_metadata_and_preserves_business_content():
    html = b"""
        <html><head>
            <title> Northwind   Services </title>
            <meta name="DESCRIPTION" content=" Managed IT for small businesses ">
            <meta name="keywords" content="IT support, cloud, IT support">
            <style>style noise</style>
        </head><body>
            <header class="site-header"><nav>Navigation noise</nav><a href="/">Brand boilerplate</a></header>
            <main>
              <header><h1>Northwind Services</h1></header>
              <section><h2>Managed IT</h2><h3>On-site support</h3>
                <p>We monitor <strong>business networks</strong> around the clock.</p>
                <p>We monitor business networks around the clock.</p>
                <a href="/services/managed-it">Managed IT plans</a>
                <a href="mailto:hello@example.com">Email us</a>
                <img alt="Network support team">
              </section>
              <div><h2>Cloud migration</h2><p>We manage business email migrations.</p></div>
              <div hidden>Hidden noise</div>
              <div style="color: black; DISPLAY: none !important">Styled noise</div>
              <div class="cookie-banner">Cookie noise</div>
            </main>
            <script>script noise</script><footer>Footer noise</footer>
        </body></html>
    """

    def handler(request):
        assert request.headers["accept-encoding"] == "gzip, identity"
        assert request.headers["user-agent"] == "AssessmentBot/1.0"
        return html_response(html)

    result = await extractor(handler, website_user_agent="AssessmentBot/1.0").extract(URL)

    assert result.success and result.warnings == []
    assert result.requested_url == result.final_url == URL
    assert result.title == "Northwind Services"
    assert result.meta_description == "Managed IT for small businesses"
    assert result.meta_keywords == ["IT support", "cloud"]
    assert result.h1_headings == ["Northwind Services"]
    assert result.h2_headings == ["Managed IT", "Cloud migration"]
    assert result.h3_headings == ["On-site support"]
    assert result.main_content.count("We monitor business networks around the clock.") == 1
    assert "We manage business email migrations." in result.main_content
    assert "noise" not in result.main_content.lower()
    assert "boilerplate" not in result.main_content.lower()
    assert result.links == [PageLink("Managed IT plans", "https://example.com/services/managed-it")]
    assert result.image_alt_text == ["Network support team"]


async def test_missing_title_and_description_are_warnings():
    service = extractor(lambda _: html_response(b"<body><p>Managed IT for businesses.</p>"))

    result = await service.extract(URL)

    assert result.success
    assert result.title is None
    assert result.meta_description is None
    assert result.warnings == ["The page has no title.", "The page has no meta description."]


@pytest.mark.parametrize(
    ("html", "warning"),
    [
        (b"", "no readable HTML"),
        (b"<head><title>Only metadata</title></head>", "no readable HTML"),
        (b'<div id="root"></div><script src="/app.js"></script>', "require JavaScript"),
        (b"<noscript>Please enable JavaScript to continue.</noscript>", "require JavaScript"),
        (b"<title>Just a moment...</title><p>Checking your browser.</p>", "bot challenge"),
        (b'<div class="h-captcha"></div><p>Verify to continue.</p>', "CAPTCHA"),
    ],
)
async def test_unusable_pages_return_clear_warnings(html, warning):
    result = await extractor(lambda _: html_response(html)).extract(URL)

    assert not result.success
    assert any(warning in message for message in result.warnings)


@pytest.mark.parametrize(
    ("status", "code"),
    [(403, "access_blocked"), (404, "page_not_found"), (429, "rate_limited"), (500, "website_http_error")],
)
async def test_website_http_errors(status, code):
    service = extractor(lambda _: html_response(b"upstream body", status=status))

    with pytest.raises(WebsiteFetchError) as exc:
        await service.extract(URL)

    assert exc.value.code == code
    assert exc.value.status_code == status


@pytest.mark.parametrize(
    ("error_type", "code"),
    [(httpx.ConnectTimeout, "timeout"), (httpx.ReadTimeout, "timeout"), (httpx.ConnectError, "connection_failed")],
)
async def test_network_errors(error_type, code):
    def handler(request):
        raise error_type("Internal networking detail", request=request)

    with pytest.raises(WebsiteFetchError) as exc:
        await extractor(handler).extract(URL)

    assert exc.value.code == code
    assert "Internal networking detail" not in str(exc.value)


async def test_deadline_bounds_dns_resolution(monkeypatch):
    async def getaddrinfo(*args, **kwargs):
        await asyncio.sleep(1)

    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", getaddrinfo)
    service = extractor(lambda _: pytest.fail("Unexpected HTTP request"), website_request_timeout=0.01)
    with pytest.raises(WebsiteFetchError, match="timed out"):
        await service.extract(URL)


async def test_non_html_response_is_rejected():
    service = extractor(lambda _: html_response(b"%PDF", headers={"content-type": "application/pdf"}))
    with pytest.raises(WebsiteFetchError) as exc:
        await service.extract(URL)
    assert exc.value.code == "unsupported_content_type"


@pytest.mark.parametrize(
    ("chunks", "headers"),
    [
        ([b"<p>1234", b"56789012345678901234567890</p>"], {}),
        ([b"<p>OK</p>"], {"content-length": "99999"}),
        ([gzip.compress(b"A" * 128)], {"content-encoding": "gzip"}),
    ],
)
async def test_size_limit_checks_declared_wire_and_decoded_sizes(chunks, headers):
    service = extractor(lambda _: html_response(*chunks, headers=headers), max_website_response_size=32)
    with pytest.raises(WebsiteFetchError) as exc:
        await service.extract(URL)
    assert exc.value.code == "response_too_large"


async def test_gzip_html_is_decoded_from_streamed_chunks():
    html = "<p>Café support</p>".encode("iso-8859-1")
    compressed = gzip.compress(html)
    headers = {"content-encoding": "gzip", "content-type": "text/html; charset=iso-8859-1"}
    service = extractor(lambda _: html_response(compressed[:10], compressed[10:], headers=headers))

    result = await service.extract(URL)

    assert result.success and result.main_content == "Café support"


@pytest.mark.parametrize(
    ("encoding", "body", "code"),
    [
        ("gzip", gzip.compress(b"<p>Valid HTML</p>")[:-4], "invalid_compression"),
        ("br", b"unsupported", "unsupported_encoding"),
    ],
)
async def test_unreadable_encodings_are_rejected(encoding, body, code):
    service = extractor(lambda _: html_response(body, headers={"content-encoding": encoding}))
    with pytest.raises(WebsiteFetchError) as exc:
        await service.extract(URL)
    assert exc.value.code == code


async def test_redirects_resolve_against_the_public_url_and_retain_page_evidence():
    requests = []

    def handler(request):
        requests.append((request.headers["host"], request.url.path))
        if request.url.path == "/":
            return httpx.Response(302, headers={"location": "/products/catalog"})
        return html_response(b'<p>Our products</p><a href="widget">Widget</a>')

    result = await extractor(handler).extract(URL)

    assert requests == [("example.com", "/"), ("example.com", "/products/catalog")]
    assert result.requested_url == URL
    assert result.final_url == "https://example.com/products/catalog"
    assert result.links == [PageLink("Widget", "https://example.com/products/widget")]


@pytest.mark.parametrize(
    "location",
    [
        "http://127.0.0.1/",
        "//internal.example.com/",
        "http://[fd00:ec2::254]/",
        "http://%31%32%37.0.0.1/",
        "http://0x7f.0.0.1/",
        "http://168.63.129.16/",
    ],
)
async def test_redirect_to_private_destination_is_never_fetched(location):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(302, headers={"location": location})

    with pytest.raises(UnsafeUrlError):
        await extractor(handler).extract(URL)

    assert len(requests) == 1


async def test_same_hostname_is_revalidated_when_dns_changes_after_redirect(monkeypatch):
    lookups = []
    requests = []

    async def getaddrinfo(host, port, **kwargs):
        lookups.append(host)
        address = "8.8.8.8" if len(lookups) == 1 else "10.0.0.5"
        return [(0, 0, 0, "", (address, port))]

    def handler(request):
        requests.append(request.url.host)
        return httpx.Response(302, headers={"location": "/next-page"})

    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", getaddrinfo)
    with pytest.raises(UnsafeUrlError):
        await extractor(handler).extract(URL)

    assert lookups == ["example.com.", "example.com."]
    assert requests == ["8.8.8.8"]


@pytest.mark.parametrize(
    ("max_redirects", "location", "code"),
    [(3, "/#top", "redirect_loop"), (0, "/other", "too_many_redirects"), (3, None, "invalid_redirect")],
)
async def test_invalid_redirect_chains(max_redirects, location, code):
    headers = {"location": location} if location else {}
    service = extractor(lambda _: httpx.Response(302, headers=headers), max_website_redirects=max_redirects)

    with pytest.raises(WebsiteFetchError) as exc:
        await service.extract(URL)

    assert exc.value.code == code
