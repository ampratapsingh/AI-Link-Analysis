import httpx
import pytest
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from app.providers import anthropic_provider, openai_provider

URL = "https://example.com/services"
SERVICE_EXCERPT = "We monitor business networks around the clock."


@pytest.fixture
def analysis():
    return {
        "brand": {"name": "Northwind IT", "category": "Managed IT", "summary": "IT support for small businesses."},
        "analysis": {
            "competitor_analysis": {
                "brand_overview": "Small business IT support.",
                "main_products_and_services": ["Network monitoring"],
                "competitors": [],
                "strengths": ["Around-the-clock monitoring"],
                "opportunities": ["Explain response times on service pages"],
            },
            "target_search_queries": [
                {
                    "query": query,
                    "platforms": [platform],
                    "intent": "Find a provider",
                    "business_relevance": "Managed IT",
                    "target_audience": "Small businesses",
                }
                for query, platform in zip(
                    ["managed IT", "business network support", "IT support plans", "IT monitoring", "IT helpdesk"],
                    ["Google Search", "Facebook", "Instagram", "LinkedIn", "X"],
                )
            ],
            "seo_improvements": {
                "primary_keyword": "managed IT",
                "secondary_keywords": ["IT support"],
                "long_tail_keywords": ["managed IT for small businesses"],
                "related_business_keywords": ["network monitoring"],
                "recommended_title": "Managed IT for Small Businesses | Northwind IT",
                "title_structure": "Service and audience followed by brand",
                "meta_description": "Find managed IT for your small business.",
                "meta_description_keywords": ["managed IT"],
                "supporting_content": ["Network monitoring service page"],
            },
            "top_services": [
                {
                    "name": "Network monitoring",
                    "description": "Around-the-clock business network monitoring.",
                    "target_industry": None,
                    "target_customer": "Small businesses",
                    "main_benefit": "Ongoing monitoring",
                    "evidence_status": "website",
                    "sources": [{"url": URL, "excerpt": SERVICE_EXCERPT}],
                }
            ],
            "target_audience": {
                "primary_audience": ["Small businesses"],
                "secondary_audience": [],
                "industries": [],
                "business_types": ["Small businesses"],
                "customer_profiles": ["Owners without dedicated IT staff"],
                "geographical_focus": None,
                "customer_needs": ["Reliable network monitoring"],
                "customer_challenges": ["Limited in-house IT coverage"],
                "customer_interests": ["Managed IT"],
                "recommended_marketing_segments": ["Small business owners"],
            },
        },
    }


def _api_response(*, text=None, content=None, status="completed", reason=None):
    if content is None:
        content = [] if text is None else [{"type": "output_text", "text": text, "annotations": []}]
    return httpx.Response(
        200,
        json={
            "id": "resp_test",
            "object": "response",
            "created_at": 0,
            "model": "gpt-5.4-mini",
            "status": status,
            "incomplete_details": {"reason": reason} if reason else None,
            "output": [
                {"type": "message", "id": "msg_test", "role": "assistant", "status": status, "content": content}
            ],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "usage": {
                "input_tokens": 200,
                "output_tokens": 95,
                "total_tokens": 295,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        },
    )


@pytest.fixture(autouse=True)
def mock_openai(monkeypatch):
    def install(handler):
        requests = []

        def client(**options):
            requests.append(options)
            transport = httpx.MockTransport(handler)
            return AsyncOpenAI(**options, http_client=httpx.AsyncClient(transport=transport))

        monkeypatch.setattr(openai_provider, "AsyncOpenAI", client)
        return requests

    install(lambda _: pytest.fail("Unexpected OpenAI request"))
    return install


@pytest.fixture
def api_response():
    return _api_response


@pytest.fixture(autouse=True)
def mock_anthropic(monkeypatch):
    def install(handler):
        def client(**options):
            transport = httpx.MockTransport(handler)
            return AsyncAnthropic(**options, http_client=httpx.AsyncClient(transport=transport))

        monkeypatch.setattr(anthropic_provider, "AsyncAnthropic", client)

    def reject_client(**options):
        pytest.fail("Unexpected Anthropic client initialization")

    monkeypatch.setattr(anthropic_provider, "AsyncAnthropic", reject_client)
    return install


@pytest.fixture
def anthropic_response():
    def reply(text=None, *, stop_reason="end_turn"):
        return httpx.Response(200, json={
            "id": "msg_fixture",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-5-5",
            "content": [] if text is None else [{"type": "text", "text": text}],
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 200, "output_tokens": 95},
        })

    return reply
