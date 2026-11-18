import asyncio
import json
import logging
import re
from dataclasses import dataclass, replace
from time import perf_counter

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
    r"\b(?:(?:does not|do not|doesn['’]t|don['’]t) (?:offer|support|provide|include|ship|deliver|serve)"
    r"|not available|unavailable|limited to|only available|available only|only supports?"
    r"|requires? (?:an? )?(?:paid|premium|enterprise) plan)\b",
    re.I,
)
_EXPLICIT_STRENGTH = re.compile(
    r"^(?:offers?|provides?|includes?) (?:(?:24/7|round-the-clock|around-the-clock) "
    r"(?:(?:customer|technical|live) )?support|free (?:trials?|shipping|migration|support)"
    r"|(?:customi[sz]able|flexible) (?:templates?|designs?)|design flexibility)\b",
    re.I,
)
_FIRST_PERSON_COMPANY = re.compile(r"^(?:we|our)\b", re.I)
_OFFERING_LANGUAGE = re.compile(r"\b(?:we (?:offer|provide|specialize)|our (?:services|products))\b", re.I)
_PRODUCT_DESCRIPTION = re.compile(
    r"^(?:an? )?(?:(?!(?:not|never|alternative|competitor|example)\b)[\w-]+ ){0,6}"
    r"(?:platform|plugin|software|store builder)\b",
    re.I,
)
_ALTERNATIVE_LIST = re.compile(
    r"\b(?:alternatives? (?:like|such as)|competitors? include|free tiers? where available\s*\()\s*"
    r"([^.!?():;]{1,180}?)(?=\s+(?:can|may|are|offer)\b|[.!?():;]|$)",
    re.I,
)
_COMPANY_LABEL = re.compile(r"[A-Z][\w-]*(?:\s+[A-Z0-9][\w-]*){0,3}")


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


def _company_from_excerpt(host: str, excerpts: tuple[str, ...]) -> str | None:
    # A hostname can disambiguate the company when a roundup-style title omits its name.
    labels = host.removeprefix("www.").split(".")
    if len(labels) != 2 or len(labels[0]) < 3:
        return None
    lead = re.compile(rf"^(?:why\s+)?({re.escape(labels[0])})(?=\s|['’])", re.I)
    for excerpt in excerpts:
        match = lead.match(excerpt)
        if match:
            return match.group(1)
    return None


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
        f"{offering} providers {category} features plan limits availability".strip(),
    ]
    terms = set().union(*(_terms(item) for item in descriptors)) - _terms(brand)
    return [query if len(query) <= 300 else query[:301].rsplit(" ", 1)[0] for query in queries], terms


def _same_site(first: str, second: str) -> bool:
    a = first.lower().rstrip(".").removeprefix("www.")
    b = second.lower().rstrip(".").removeprefix("www.")
    return a == b or a.endswith("." + b) or b.endswith("." + a)


def _named_leads(sources: tuple[CompetitorSource, ...]) -> list[str]:
    leads = {}
    for source in sources:
        for excerpt in source.excerpts:
            for group in _ALTERNATIVE_LIST.findall(excerpt):
                for candidate in re.split(r"\s*,\s*|\s+and\s+", group):
                    name = candidate.strip().removeprefix("and ")
                    if 2 <= len(name) <= 36 and _COMPANY_LABEL.fullmatch(name):
                        leads.setdefault(_name_key(name), name)
    return list(leads.values())[:8]


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
        company = _company_name(title, host) or _company_from_excerpt(host, excerpts)
        source = CompetitorSource(url, title, excerpts, company)
        comparisons = supported_competitors((source,))
        facets = frozenset(
            field for comparison in comparisons for field in ("main_offerings", "strengths", "gaps")
            if getattr(comparison, field)
        )
        ranked.append((len(facets), bool(company), len(overlap), source, facets))

    ranked.sort(key=lambda item: item[:3], reverse=True)
    sources = []
    primary = []
    complementary = []
    for _, _, _, source, facets in ranked:
        host = source.url.host or ""
        previous = next((entry for entry in primary if _same_site(host, entry[0])), None)
        if previous:
            _, first, first_facets = previous
            if (
                source.url != first.url and source.company_name and first.company_name
                and _name_key(source.company_name) == _name_key(first.company_name)
                and facets - first_facets
            ):
                complementary.append(source)
            continue
        sources.append(source)
        primary.append((host, source, facets))
        if len(sources) == limit:
            break
    return tuple((sources + complementary)[:limit])


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

    prompt_budget = settings.llm_max_content_chars // 3
    if prompt_budget < 300:
        return CompetitorResearch("insufficient_budget")
    host = HttpUrl(page.final_url).host or ""
    queries, terms = _search_queries(page, host)
    if not queries or not terms:
        return CompetitorResearch("no_relevant_results")

    hits = []
    evidence: tuple[CompetitorSource, ...] = ()
    failed = False
    completed = 0
    started = perf_counter()
    try:
        async with asyncio.timeout(settings.search_timeout_seconds):
            for query in queries:
                if completed:
                    leads = _named_leads(evidence)
                    if leads:
                        while sum(len(name) + 6 for name in leads) > 150:
                            leads.pop()
                        names = " or ".join(f'"{name}"' for name in leads)
                        query = f"{names} official product documentation {query}"
                        if len(query) > 300:
                            query = query[:301].rsplit(" ", 1)[0]
                hits.extend(await provider.search(query, max_results=settings.search_max_results))
                completed += 1
                evidence = _sources(hits, host, terms, settings.search_max_results)
                if completed < len(queries) and len(evidence) >= 3:
                    _, fitted = CompetitorResearch("search_snippets", evidence).fit_prompt(prompt_budget)
                    compared = supported_competitors(fitted.sources)
                    if len({_name_key(item.name) for item in compared if item.strengths and item.gaps}) >= 3:
                        break
    except SearchError as exc:
        failed = True
        logger.warning("Competitor search interrupted reason=%s completed_queries=%d", exc.code, completed)
    except TimeoutError:
        failed = True
        logger.warning("Competitor search interrupted reason=deadline completed_queries=%d", completed)

    logger.info(
        "Competitor search finished duration_ms=%d completed_queries=%d usable_sources=%d",
        (perf_counter() - started) * 1000,
        completed,
        len(evidence),
    )
    if evidence:
        return CompetitorResearch("search_snippets", evidence)
    return CompetitorResearch("unavailable" if failed else "no_relevant_results")


def _mentions(name: str, text: str) -> bool:
    return bool(re.search(r"(?<!\w)" + re.escape(_compact(name)) + r"(?!\w)", text, re.I))


def supported_competitors(sources: tuple[CompetitorSource, ...]) -> list[Competitor]:
    competitors = []
    for source in sources:
        name = source.company_name
        if not name:
            continue
        citations = [
            Evidence(url=source.url, excerpt=excerpt)
            for excerpt in source.excerpts
            if _mentions(name, excerpt) or _FIRST_PERSON_COMPANY.match(excerpt)
        ]
        if not citations:
            continue
        candidate = Competitor(
            name=name, main_offerings=[], strengths=[], gaps=[], evidence_status="potential",
            sources=citations,
        )
        grounded = ground_researched_competitor(candidate, (source,))
        if grounded and grounded.evidence_status == "sourced":
            existing = next((
                item for item in competitors
                if _name_key(item.name) == _name_key(grounded.name)
                and _same_site(item.website.host or "", source.url.host or "")
            ), None)
            if existing is None:
                competitors.append(grounded)
            else:
                existing.strengths = list(dict.fromkeys(existing.strengths + grounded.strengths))
                existing.gaps = list(dict.fromkeys(existing.gaps + grounded.gaps))
                existing.main_offerings = list(dict.fromkeys(existing.main_offerings + grounded.main_offerings))
                existing.sources.extend(item for item in grounded.sources if item not in existing.sources)
    return competitors


def ground_researched_competitor(
    competitor: Competitor, sources: tuple[CompetitorSource, ...]
) -> Competitor | None:
    if not _compact(competitor.name):
        return None
    company = re.escape(_compact(competitor.name))
    claim_pattern = re.compile(
        rf"^(?:(?:{company}|we)\s+|(?:{company}['’]s|our)\s+"
        r"(?:[\w-]+\s+){0,4}(?:plans?|services?|products?|platform|support|shipping|features?)\s+)"
        r"(?:(?:is|are)\s+)?(.+)",
        re.I,
    )
    heading_pattern = re.compile(rf"^why\s+{company}\b[^:]*:\s*", re.I)
    product_identity = re.compile(rf"^{company}\s+is\s+", re.I)
    product_prefix = re.compile(r"^(?:[A-Z][\w-]*\s+){1,3}(?=(?:offers?|provides?|includes?|requires?)\s)")

    def company_statement(excerpt: str) -> tuple[str, str] | None:
        statement = heading_pattern.sub("", excerpt)
        match = claim_pattern.match(statement)
        if match:
            return statement, product_prefix.sub("", match.group(1))
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
            direct = name_matches and company_statement(_compact(citation.excerpt))
            if direct or _mentions(competitor.name, citation.excerpt):
                evidence = Evidence(url=source.url, excerpt=_compact(citation.excerpt))
                if evidence not in citations:
                    citations.append(evidence)
                if direct:
                    website = source.url
                    company_citations.append(evidence)
    if not citations:
        return None

    quotes = {item.excerpt for item in company_citations}

    def sourced(value: str | None) -> str | None:
        clean = _compact(value or "")
        if clean.casefold().startswith("verified:"):
            clean = clean[len("verified:"):].strip()
        return clean if clean in quotes else None

    def is_gap(excerpt: str) -> bool:
        direct = company_statement(excerpt)
        return bool(direct and _LIMITATION.match(direct[1]))

    def strength_text(statement: str) -> str:
        return re.split(r"\s+that\s+[^.!?]+?\s+(?:can['’]t|cannot) match\b", statement, maxsplit=1)[0]

    facts = {
        "business_overview": sourced(competitor.business_overview),
        "category": sourced(competitor.category),
        "target_market": sourced(competitor.target_market),
        "main_offerings": list(dict.fromkeys(filter(None, map(sourced, competitor.main_offerings)))),
        "strengths": list(dict.fromkeys(
            strength_text(heading_pattern.sub("", quote)) for quote in map(sourced, competitor.strengths)
            if quote and not is_gap(quote)
        )),
        "gaps": [
            quote for quote in dict.fromkeys(map(sourced, competitor.gaps))
            if quote and is_gap(quote)
        ],
    }
    if competitor.evidence_status not in {"sourced", "verified"}:
        facts = {
            field: [] if isinstance(value, list) else None for field, value in facts.items()
        }

    # Reuse only company-attributed sources the model already anchored to this competitor.
    # Keep complete excerpts so qualifiers such as regions and plan names stay attached.
    company_urls = {item.url for item in company_citations}
    for source in sources:
        if source.url not in company_urls or _name_key(source.company_name or "") != _name_key(competitor.name):
            continue
        for excerpt in source.excerpts:
            direct = company_statement(excerpt)
            if direct is None:
                continue
            statement, claim = direct
            if _LIMITATION.match(claim):
                field = "gaps"
            elif _EXPLICIT_STRENGTH.match(claim):
                field = "strengths"
                # Keep the capability while leaving the publisher's rival comparison in the citation.
                statement = strength_text(statement)
            elif product_identity.match(statement) and _PRODUCT_DESCRIPTION.match(claim):
                field = "main_offerings"
            else:
                continue
            if statement not in facts[field]:
                facts[field].append(statement)
            evidence = Evidence(url=source.url, excerpt=excerpt)
            if evidence not in citations:
                citations.append(evidence)
    has_facts = any(facts.values())
    source_status = "verified" if competitor.evidence_status == "verified" else "sourced"
    return competitor.model_copy(update={
        **facts,
        "website": website if has_facts else None,
        "evidence_status": source_status if has_facts else "potential",
        "sources": citations,
    })


def combine_competitor_reasoning(
    candidate: Competitor, grounded: Competitor | None, page_evidence: str
) -> Competitor | None:
    named_on_page = _mentions(candidate.name, page_evidence)
    result = grounded or Competitor(
        name=_compact(candidate.name), main_offerings=[], strengths=[], gaps=[],
        evidence_status="ai_suggestion", sources=[],
    )
    inferred = result.evidence_status == "inferred"
    has_suggestions = False
    for field in ("main_offerings", "strengths", "gaps"):
        claims = getattr(result, field)
        for raw in getattr(candidate, field):
            text = _compact(raw)
            prefix, separator, body = text.partition(":")
            label = prefix.casefold() if separator else ""
            if label in {"verified", "sourced"}:
                continue
            if label in {"inferred", "ai_suggestion"}:
                text = body.strip()
            elif candidate.evidence_status in {"inferred", "ai_suggestion"}:
                label = candidate.evidence_status
            else:
                continue
            if not text or text in claims:
                continue
            if label == "inferred" and not (result.sources or named_on_page):
                label = "ai_suggestion"
            item = f"{label}: {text}"
            if item not in claims:
                claims.append(item)
            inferred |= label == "inferred"
            has_suggestions = True
    if result.evidence_status not in {"verified", "sourced"} and has_suggestions:
        result.evidence_status = "inferred" if inferred else "ai_suggestion"
    if grounded or has_suggestions or (named_on_page and candidate.evidence_status == "ai_suggestion"):
        return result
    return None
