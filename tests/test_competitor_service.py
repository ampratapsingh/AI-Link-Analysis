import asyncio
import json
from dataclasses import replace

import httpx
import pytest

from app.config import Settings
from app.providers.search import TavilySearchProvider
from app.schemas import Competitor
from app.services import competitor_service
from app.services.competitor_service import discover_competitors, ground_researched_competitor, supported_competitors
from app.services.website_extractor import ExtractionResult

pytestmark = [pytest.mark.anyio, pytest.mark.parametrize("anyio_backend", ["asyncio"])]

COMPANY_URL = "https://contoso.example/cloud"
OVERVIEW = "Contoso offers managed cloud hosting and Kubernetes migration."
LIMITATION = "Contoso does not offer on-premises hosting."


@pytest.fixture
def business_page():
    return ExtractionResult(
        requested_url="https://northwind.example/",
        final_url="https://northwind.example/",
        title="Managed Cloud Hosting | Northwind",
        meta_description="Managed cloud hosting for startups.",
        meta_keywords=[],
        h1_headings=["Northwind"],
        h2_headings=["Managed cloud hosting"],
        h3_headings=["Kubernetes migration"],
        main_content="We offer managed cloud hosting and Kubernetes migration.",
        links=[],
        image_alt_text=[],
        warnings=[],
        success=True,
    )


@pytest.fixture
def search_settings():
    return Settings(search_api_key="mock-tavily-key", search_max_results=3)


@pytest.fixture
def mock_search(monkeypatch):
    def install(handler):
        transport = httpx.MockTransport(handler)
        monkeypatch.setattr(
            competitor_service,
            "TavilySearchProvider",
            lambda key, timeout: TavilySearchProvider(key, timeout, transport=transport),
        )

    def unexpected(*args):
        pytest.fail("Unexpected search provider initialization")

    monkeypatch.setattr(competitor_service, "TavilySearchProvider", unexpected)
    return install


def hit(url=COMPANY_URL, title="Contoso | Cloud Hosting", snippet=f"{OVERVIEW} {LIMITATION}"):
    return {"title": title, "url": url, "content": snippet}


async def test_targeted_search_preserves_and_deduplicates_company_evidence(
    mock_search, business_page, search_settings
):
    queries = []

    def tavily(request):
        assert str(request.url) == "https://api.tavily.com/search"
        assert request.headers["Authorization"] == "Bearer mock-tavily-key"
        body = json.loads(request.content)
        assert body == {
            "query": body["query"],
            "max_results": 3,
            "search_depth": "basic",
            "chunks_per_source": 1,
            "include_answer": False,
            "include_raw_content": False,
            "auto_parameters": False,
        }
        queries.append(body["query"])
        hits = [
            hit("https://directory.example/cloud", "Cloud provider directory",
                "Contoso offers managed cloud hosting and Kubernetes migration."),
            hit(),
            hit("https://www.contoso.example/pricing"),
        ]
        return httpx.Response(200, json={"results": hits})

    mock_search(tavily)
    research = await discover_competitors(business_page, search_settings)

    assert len(queries) == 2
    assert '"Northwind" alternatives' in queries[0]
    assert all("managed cloud hosting" in query.lower() for query in queries)
    assert "kubernetes migration" in queries[1].lower()
    assert research.status == "search_snippets"
    company, directory = research.sources
    assert company.company_name == "Contoso"
    assert str(company.url) == COMPANY_URL
    assert company.title == "Contoso | Cloud Hosting"
    assert company.excerpts == (OVERVIEW, LIMITATION)
    assert directory.company_name is None
    suffix, sent = research.fit_prompt(4000)
    record = json.loads('{"page":true' + suffix + "}")["competitor_research"][0]
    assert record["source_url"] == COMPANY_URL
    assert record["excerpts"] == [OVERVIEW, LIMITATION]
    assert sent.sources == research.sources
    suffix, sent = research.fit_prompt(50)
    assert not suffix and not sent.sources and sent.status == "insufficient_budget"


async def test_roundup_title_preserves_identifiable_company_evidence(
    mock_search, business_page, search_settings
):
    strength = "Contoso provides 24/7 technical support for managed cloud hosting."
    roundup = f"{strength} Fabrikam also offers managed cloud hosting."
    mock_search(lambda _: httpx.Response(200, json={
        "results": [
            hit(title="Alternatives for managed cloud hosting", snippet=roundup),
            hit("https://directory.example/cloud", "Cloud Hosting Directory", roundup),
        ],
    }))

    research = await discover_competitors(business_page, search_settings)

    company, directory = research.sources
    assert str(company.url) == COMPANY_URL
    assert company.company_name == "Contoso"
    assert company.excerpts[0] == strength
    assert directory.company_name is None
    supported, = supported_competitors(research.sources)
    assert supported.name == "Contoso"
    assert supported.strengths == [strength]
    assert str(supported.sources[0].url) == COMPANY_URL


@pytest.mark.parametrize(
    "results",
    [
        [],
        [hit("https://baking.example/", "Cake Recipes", "We share gluten-free cake recipes.")],
        [hit("https://northwind.example/other"), hit("http://127.0.0.1/html")],
    ],
)
async def test_empty_irrelevant_and_unsafe_search_hits_are_filtered(
    mock_search, business_page, search_settings, results
):
    mock_search(lambda _: httpx.Response(200, json={"results": results}))
    research = await discover_competitors(business_page, search_settings)
    assert research.status == "no_relevant_results" and research.sources == ()


@pytest.mark.parametrize(
    "config", [Settings(), Settings(search_api_key=" "), Settings(search_provider="none", search_api_key="key")]
)
async def test_search_credentials_are_optional(mock_search, business_page, config):
    research = await discover_competitors(business_page, config)
    assert research.status == "not_configured" and research.sources == ()


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(429, json={"detail": "private provider response"}),
        httpx.Response(200, text="not JSON"),
        httpx.Response(200, json={"results": {"unexpected": "object"}}),
        httpx.Response(200, json={"results": [{"title": {}, "url": COMPANY_URL}]}),
    ],
)
async def test_rate_limit_and_malformed_response_fall_back_without_retry(
    mock_search, business_page, search_settings, response
):
    calls = []

    def handler(request):
        calls.append(request)
        return response

    mock_search(handler)
    research = await discover_competitors(business_page, search_settings)
    assert research.status == "unavailable" and research.sources == ()
    assert len(calls) == 1


async def test_total_search_deadline(mock_search, business_page, search_settings):
    async def timeout(_):
        await asyncio.sleep(10)

    mock_search(timeout)
    research = await discover_competitors(
        business_page, search_settings.model_copy(update={"search_timeout_seconds": 0.02})
    )
    assert research.status == "unavailable"


async def test_search_keeps_earlier_evidence_if_second_query_fails(
    mock_search, business_page, search_settings
):
    replies = iter([httpx.Response(200, json={"results": [hit()]}), httpx.Response(429)])
    mock_search(lambda _: next(replies))
    research = await discover_competitors(business_page, search_settings)
    assert research.status == "search_snippets"
    assert str(research.sources[0].url) == COMPANY_URL


async def test_second_query_timeout_keeps_earlier_evidence(mock_search, business_page, search_settings, caplog):
    calls = []

    async def tavily(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(200, json={"results": [hit()]})
        await asyncio.Event().wait()

    mock_search(tavily)
    research = await discover_competitors(
        business_page, search_settings.model_copy(update={"search_timeout_seconds": 0.1})
    )

    assert len(calls) == 2
    assert research.status == "search_snippets"
    assert str(research.sources[0].url) == COMPANY_URL
    assert "reason=deadline completed_queries=1" in caplog.text
    assert "mock-tavily-key" not in caplog.text and "Northwind" not in caplog.text


@pytest.mark.parametrize(("content_budget", "expected_calls"), [(12000, 1), (900, 2)])
async def test_search_stops_when_three_attributed_candidates_fit(
    mock_search, business_page, search_settings, content_budget, expected_calls
):
    calls = []
    results = [
        hit(
            f"https://{name.lower()}.example/cloud",
            f"{name} | Cloud Hosting",
            f"{name} offers managed cloud hosting and Kubernetes migration.",
        )
        for name in ("Contoso", "Fabrikam", "Adatum")
    ]

    def tavily(request):
        calls.append(request)
        assert request.extensions["timeout"]["read"] == 10.0
        return httpx.Response(200, json={"results": results})

    mock_search(tavily)
    settings = search_settings.model_copy(update={"llm_max_content_chars": content_budget})
    research = await discover_competitors(business_page, settings)

    assert research.status == "search_snippets"
    assert {source.company_name for source in research.sources} == {"Contoso", "Fabrikam", "Adatum"}
    assert len(calls) == expected_calls


async def test_claims_require_complete_company_excerpts_and_matching_urls(
    mock_search, business_page, search_settings
):
    mock_search(lambda _: httpx.Response(200, json={"results": [hit()]}))
    research = await discover_competitors(business_page, search_settings)
    candidate = Competitor.model_validate({
        "name": "Contoso",
        "website": "https://invented.example/",
        "business_overview": OVERVIEW,
        "category": "Managed hosting",
        "main_offerings": [OVERVIEW],
        "target_market": "Large enterprises",
        "strengths": ["Market leader"],
        "gaps": [OVERVIEW, LIMITATION, "Contoso offers on-premises hosting."],
        "evidence_status": "sourced",
        "sources": [
            {"url": COMPANY_URL, "excerpt": OVERVIEW},
            {"url": COMPANY_URL, "excerpt": LIMITATION},
            {"url": COMPANY_URL, "excerpt": "on-premises hosting."},
            {"url": "https://invented.example/", "excerpt": OVERVIEW},
        ],
    })
    grounded = ground_researched_competitor(candidate, research.sources)

    assert grounded is not None
    assert grounded.evidence_status == "sourced" and str(grounded.website) == COMPANY_URL
    assert grounded.business_overview == OVERVIEW and grounded.main_offerings == [OVERVIEW]
    assert grounded.category is None and grounded.target_market is None and grounded.strengths == []
    assert grounded.gaps == [LIMITATION]
    assert [source.excerpt for source in grounded.sources] == [OVERVIEW, LIMITATION]
    assert ground_researched_competitor(candidate.model_copy(update={"name": "FakeCo"}), research.sources) is None
    assert ground_researched_competitor(candidate, ()) is None

    directory = replace(research.sources[0], company_name=None)
    potential = ground_researched_competitor(candidate, (directory,))
    assert potential is not None and potential.evidence_status == "potential"
    assert potential.website is None and potential.business_overview is None and potential.gaps == []


@pytest.mark.parametrize("label", ["potential", "sourced"])
async def test_comparison_uses_explicit_strengths_and_gaps_from_existing_company_snippets(
    mock_search, business_page, search_settings, label
):
    strength = "Contoso provides 24/7 technical support on enterprise plans."
    gap = "Our managed hosting plans are only available in Canada."
    mock_search(lambda _: httpx.Response(200, json={
        "results": [hit(snippet=f"{OVERVIEW} {strength} {gap}")],
    }))
    research = await discover_competitors(business_page, search_settings)
    candidate = Competitor.model_validate({
        "name": "Contoso",
        "website": None,
        "main_offerings": [],
        "target_market": None,
        "strengths": ["Contoso is the market leader."],
        "gaps": ["Contoso has poor performance."],
        "evidence_status": label,
        "sources": [{"url": COMPANY_URL, "excerpt": OVERVIEW}],
    })

    grounded = ground_researched_competitor(candidate, research.sources)

    assert grounded is not None and grounded.evidence_status == "sourced"
    assert grounded.strengths == [strength]
    assert grounded.gaps == [gap]
    assert [(str(item.url), item.excerpt) for item in grounded.sources] == [
        (COMPANY_URL, OVERVIEW), (COMPANY_URL, strength), (COMPANY_URL, gap),
    ]


async def test_ambiguous_mentions_do_not_turn_a_potential_competitor_into_a_verified_comparison(
    mock_search, business_page, search_settings
):
    directory_url = "https://directory.example/cloud"
    directory_quote = "Contoso provides 24/7 technical support for cloud hosting."
    other_company = "Unlike Contoso, Fabrikam provides 24/7 technical support."
    positive_policy = "Contoso does not charge setup fees."
    mock_search(lambda _: httpx.Response(200, json={
        "results": [
            hit(snippet=f"{OVERVIEW} {other_company} {positive_policy}"),
            hit(directory_url, "Cloud Hosting Directory", directory_quote),
        ],
    }))
    research = await discover_competitors(business_page, search_settings)
    candidate = Competitor.model_validate({
        "name": "Contoso",
        "website": None,
        "main_offerings": [],
        "target_market": None,
        "strengths": [directory_quote, other_company],
        "gaps": [positive_policy],
        "evidence_status": "sourced",
        "sources": [
            {"url": directory_url, "excerpt": directory_quote},
            {"url": COMPANY_URL, "excerpt": other_company},
            {"url": COMPANY_URL, "excerpt": positive_policy},
        ],
    })

    grounded = ground_researched_competitor(candidate, research.sources)

    assert grounded is not None and grounded.evidence_status == "potential"
    assert grounded.strengths == [] and grounded.gaps == []
    assert grounded.sources == candidate.sources
