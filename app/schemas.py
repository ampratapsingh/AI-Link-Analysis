from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


class LinkAnalysisRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"url": "https://example.com"}]},
    )

    url: str = Field(max_length=2083, json_schema_extra={"format": "uri"})

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        HttpUrl(value)
        # The fetcher's security checks need the original URL spelling.
        return value


class Brand(BaseModel):
    name: str | None
    category: str | None
    summary: str


class Evidence(BaseModel):
    url: HttpUrl
    excerpt: str


class Competitor(BaseModel):
    name: str
    website: HttpUrl | None
    main_offerings: list[str]
    target_market: str | None
    strengths: list[str]
    gaps: list[str]
    evidence_status: Literal["sourced", "potential", "ai_suggestion"]
    sources: list[Evidence]
    business_overview: str | None = None
    category: str | None = None


CompetitorResearchStatus = Literal[
    "search_snippets", "not_configured", "unavailable", "no_relevant_results", "insufficient_budget"
]


class CompetitorAnalysis(BaseModel):
    brand_overview: str
    main_products_and_services: list[str]
    competitors: list[Competitor]
    strengths: list[str]
    opportunities: list[str]
    research_status: CompetitorResearchStatus | None = None


class SearchPlatform(StrEnum):
    GOOGLE = "Google Search"
    FACEBOOK = "Facebook"
    INSTAGRAM = "Instagram"
    LINKEDIN = "LinkedIn"
    X = "X"


class TargetSearchQuery(BaseModel):
    query: str
    platforms: list[SearchPlatform] = Field(min_length=1)
    intent: str
    business_relevance: str
    target_audience: str


class SEOImprovements(BaseModel):
    primary_keyword: str
    secondary_keywords: list[str]
    long_tail_keywords: list[str]
    related_business_keywords: list[str]
    recommended_title: str
    title_structure: str
    meta_description: str
    meta_description_keywords: list[str]
    supporting_content: list[str]


class Service(BaseModel):
    name: str
    description: str
    target_industry: str | None
    target_customer: str | None
    main_benefit: str | None
    evidence_status: Literal["website", "inferred"]
    sources: list[Evidence]


class TargetAudience(BaseModel):
    primary_audience: list[str]
    secondary_audience: list[str]
    industries: list[str]
    business_types: list[str]
    customer_profiles: list[str]
    geographical_focus: str | None
    customer_needs: list[str]
    customer_challenges: list[str]
    customer_interests: list[str]
    recommended_marketing_segments: list[str]


class Analysis(BaseModel):
    competitor_analysis: CompetitorAnalysis
    target_search_queries: list[TargetSearchQuery] = Field(min_length=5, max_length=5)
    seo_improvements: SEOImprovements
    top_services: list[Service] = Field(max_length=5)
    target_audience: TargetAudience


class GeneratedAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    brand: Brand
    analysis: Analysis


class LinkAnalysisResponse(BaseModel):
    success: Literal[True] = True
    website: HttpUrl
    brand: Brand
    analysis: Analysis


class LinkAnalysisErrorResponse(BaseModel):
    success: Literal[False] = False
    code: str
    detail: str


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
