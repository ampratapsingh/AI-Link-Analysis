import asyncio
import json
import logging
import re
from dataclasses import dataclass, replace

from pydantic import HttpUrl

from app.config import Settings
from app.providers.search import SearchError, SearchProvider, SearchResult, TavilySearchProvider
from app.schemas import Competitor, CompetitorResearchStatus, Evidence
from app.services.url_safety import parse_website_url
from app.services.website_extractor import ExtractionResult

logger = logging.getLogger(__name__)
_TITLE_SEPARATOR = re.compile(r"\s*(?:\||[–—])\s*|\s+-\s+|:\s+")
_WORD = re.compile(r"[^\W_]+")
_GENERIC_TERMS = set(
    "about and are best business businesses companies company contact customers for from home "
    "learn main more offering offerings our plans products providers services solutions that "
    "the their this website what with world your".split()
)
_LIMITATION = re.compile(
    r"\b(?:does not|do not|not available|unavailable|limited to|only available|only support)\b",
    re.I,
)
_FIRST_PERSON_COMPANY = re.compile(r"^(?:we|our)\b", re.I)
_OFFERING_LANGUAGE = re.compile(r"\b(?:we (?:offer|provide|specialize)|our (?:services|products))\b", re.I)


@dataclass(frozen=True)
class CompetitorSource:
    url: HttpUrl
    title: str
    excerpts: tuple[str, ...]
    company_name: str | None


@dataclass(frozen=True)
class CompetitorResearch:
    status: CompetitorResearchStatus
    sources: tuple[CompetitorSource, ...] = ()

    def fit_prompt(self, budget: int) -> tuple[str, "CompetitorResearch"]:
        """Return a JSON suffix and the exact sources sent within the prompt budget."""
        prefix = ',"competitor_research":['
        remaining = budget - len(prefix) - 1
        encoded = []
        sources = []
        for source in self.sources:
            entry = json.dumps(
                {
                    "source_url": str(source.url),
                    "title": source.title,
                    "excerpts": source.excerpts,
                    "company_name": source.company_name,
                },
                ensure_ascii=False, separators=(",", ":"),
            )
            cost = len(entry) + bool(encoded)
            if cost <= remaining:
                encoded.append(entry)
                sources.append(source)
                remaining -= cost
        if not sources:
            return "", replace(self, status="insufficient_budget", sources=())
        return prefix + ",".join(encoded) + "]", replace(self, sources=tuple(sources))


def _compact(text: str) -> str:
    return " ".join(text.split())


def _name_key(text: str) -> str:
    return "".join(_WORD.findall(text.casefold()))


def _terms(text: str) -> set[str]:
    return {
        word for word in _WORD.findall(text.casefold())
        if len(word) >= 3 and not word.isdecimal() and word not in _GENERIC_TERMS
    }


def _title_parts(title: str) -> list[str]:
    return [part for part in _TITLE_SEPARATOR.split(_compact(title)) if part]


def _company_name(title: str, host: str) -> str | None:
    host_labels = {
        _name_key(label) for label in host.removeprefix("www.").split(".")[:-1] if len(label) >= 3
    }
    return next(
        (part for part in _title_parts(title) if len(part) <= 100 and _name_key(part) in host_labels),
        None,
    )


def _bounded_phrase(value: str, words: int) -> str:
    # Search receives short text phrases, never page URLs, operators, or raw excerpts.
    return " ".join(word[:32] for word in _WORD.findall(_compact(value))[:words])


def _search_queries(page: ExtractionResult, host: str) -> tuple[list[str], set[str]]:
    title = page.title or ""
    title_parts = _title_parts(title)
    headings = list(page.h2_headings) + list(page.h3_headings)
    brand = _company_name(title, host) or next(
        (
            part for part in [*title_parts, *page.h1_headings[:1]]
            if 1 <= len(_WORD.findall(part)) <= 6 and _terms(part)
        ),
        host.removeprefix("www.").split(".")[0],
    )
    offering_lines = [
        re.sub(r"^we (?:offer|provide|specialize in)\s+", "", line, flags=re.I)
        for line in page.main_content.splitlines()
        if _OFFERING_LANGUAGE.search(line)
    ][:2]
    descriptors = [
        _bounded_phrase(text, 8)
        for text in (
            *headings[:10], *page.meta_keywords[:4], page.meta_description or "",
            *offering_lines, *title_parts,
        )
        if _name_key(text) != _name_key(brand)
    ]
    descriptors = list(dict.fromkeys(text for text in descriptors if _terms(text)))
    if not descriptors:
        return [], set()

    offering = max(descriptors, key=lambda text: min(len(_terms(text)), 3))
    category = next(
        (item for item in descriptors if item != offering and not _terms(item) <= _terms(offering)),
        "",
    )
    brand_label = _bounded_phrase(brand, 6)
    queries = [
        f'"{brand_label}" alternatives {offering}',
        f"{offering} providers {category}".strip(),
    ]
    terms = set().union(*(_terms(item) for item in descriptors)) - _terms(brand)
    return [query if len(query) <= 300 else query[:301].rsplit(" ", 1)[0] for query in queries], terms


def _same_site(first: str, second: str) -> bool:
    a = first.lower().rstrip(".").removeprefix("www.")
    b = second.lower().rstrip(".").removeprefix("www.")
    return a == b or a.endswith("." + b) or b.endswith("." + a)


def _sources(
    hits: list[SearchResult], brand_host: str, terms: set[str], limit: int
) -> tuple[CompetitorSource, ...]:
    ranked = []
    for hit in hits:
        try:
            url = HttpUrl(str(parse_website_url(hit.url)))
        except ValueError:
            continue
        host = url.host or ""
        if _same_site(host, brand_host):
            continue
        title = _compact(hit.title)[:180]
        excerpts = tuple(
            snippet for snippet in re.split(r"\s*\[\.\.\.\]\s*|(?<=[.!?])\s+", _compact(hit.snippet))
            if 12 <= len(snippet) <= 540
        )[:3]
        overlap = terms & _terms(title + " " + " ".join(excerpts))
        if not excerpts or len(overlap) < min(2, len(terms)):
            continue
        company = _company_name(title, host)
        ranked.append((bool(company), len(overlap), CompetitorSource(url, title, excerpts, company)))

    ranked.sort(key=lambda item: item[:2], reverse=True)
    sources = []
    hosts = []
    for _, _, source in ranked:
        host = source.url.host or ""
        if any(_same_site(host, previous) for previous in hosts):
            continue
        sources.append(source)
        hosts.append(host)
        if len(sources) == limit:
            break
    return tuple(sources)


async def discover_competitors(
    page: ExtractionResult, settings: Settings, *, provider: SearchProvider | None = None
) -> CompetitorResearch:
    if provider is None:
        key = settings.search_api_key
        selected = settings.search_provider.strip().casefold()
        if selected in {"", "none"} or not key or not key.get_secret_value().strip():
            return CompetitorResearch("not_configured")
        if selected != "tavily":
            logger.warning("Competitor search unavailable reason=unsupported_provider")
            return CompetitorResearch("unavailable")
        provider = TavilySearchProvider(key, settings.search_timeout_seconds)

    if settings.llm_max_content_chars // 3 < 300:
        return CompetitorResearch("insufficient_budget")
    host = HttpUrl(page.final_url).host or ""
    queries, terms = _search_queries(page, host)
    if not queries or not terms:
        return CompetitorResearch("no_relevant_results")

    hits = []
    failed = False
    try:
        async with asyncio.timeout(settings.search_timeout_seconds):
            for query in queries:
                hits.extend(await provider.search(query, max_results=settings.search_max_results))
    except SearchError as exc:
        failed = True
        logger.warning("Competitor search unavailable reason=%s", exc.code)
    except TimeoutError:
        failed = True
        logger.warning("Competitor search unavailable reason=deadline")

    evidence = _sources(hits, host, terms, settings.search_max_results)
    if evidence:
        return CompetitorResearch("search_snippets", evidence)
    return CompetitorResearch("unavailable" if failed else "no_relevant_results")


def _mentions(name: str, text: str) -> bool:
    return bool(re.search(r"(?<!\w)" + re.escape(_compact(name)) + r"(?!\w)", text, re.I))


def ground_researched_competitor(
    competitor: Competitor, sources: tuple[CompetitorSource, ...]
) -> Competitor | None:
    if not _compact(competitor.name):
        return None
    citations = []
    company_citations = []
    website = None
    for citation in competitor.sources:
        for source in sources:
            if citation.url != source.url or _compact(citation.excerpt) not in source.excerpts:
                continue
            name_matches = bool(source.company_name) and (
                _name_key(source.company_name or "") == _name_key(competitor.name)
            )
            if name_matches or _mentions(competitor.name, citation.excerpt):
                evidence = Evidence(url=source.url, excerpt=_compact(citation.excerpt))
                if evidence not in citations:
                    citations.append(evidence)
                if name_matches and (
                    _mentions(competitor.name, evidence.excerpt)
                    or _FIRST_PERSON_COMPANY.match(evidence.excerpt)
                ):
                    website = source.url
                    company_citations.append(evidence)
    if not citations:
        return None

    quotes = {item.excerpt for item in company_citations}

    def sourced(value: str | None) -> str | None:
        clean = _compact(value or "")
        return clean if clean in quotes else None

    facts = {
        "business_overview": sourced(competitor.business_overview),
        "category": sourced(competitor.category),
        "target_market": sourced(competitor.target_market),
        "main_offerings": list(dict.fromkeys(filter(None, map(sourced, competitor.main_offerings)))),
        "strengths": list(dict.fromkeys(filter(None, map(sourced, competitor.strengths)))),
        "gaps": [
            quote for quote in dict.fromkeys(map(sourced, competitor.gaps))
            if quote and _LIMITATION.search(quote)
        ],
    }
    has_facts = any(facts.values()) and competitor.evidence_status == "sourced"
    if not has_facts:
        facts = {
            field: [] if isinstance(value, list) else None for field, value in facts.items()
        }
    return competitor.model_copy(update={
        **facts,
        "website": website if has_facts else None,
        "evidence_status": "sourced" if has_facts else "potential",
        "sources": citations,
    })
