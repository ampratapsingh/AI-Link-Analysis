# AI-Powered Website Link Analysis API

Submit a public website URL to FastAPI to extract business evidence and generate a brand profile, competitor analysis, five search queries, SEO recommendations, services, and target audiences.

[Live demo](https://ai-link-analysis-k6ny.onrender.com/docs) · [API endpoint](https://ai-link-analysis-k6ny.onrender.com/link-analysis) · [Health check](https://ai-link-analysis-k6ny.onrender.com/health) · [Sanitized production logs](docs/DEMO_VERIFICATION.md)

## Local development

Use Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
test -f .env || cp .env.example .env
HOST=127.0.0.1 python -m app.main
```

The server defaults to port 4000. Swagger UI is available at [localhost:4000/docs](http://localhost:4000/docs). Configuration comes from environment variables and the optional local `.env` file. The process environment takes precedence. The API starts without provider credentials.

For Claude analysis, open `.env` in your editor and set `ANTHROPIC_API_KEY` privately. Set `LLM_PROVIDER=anthropic` and `ANTHROPIC_MODEL=claude-haiku-5-5` if your existing file contains an older provider choice. Restart the app after changing configuration.

## Verify startup and request validation

```bash
curl -i http://localhost:4000/health
curl -i http://localhost:4000/openapi.json
curl -i http://localhost:4000/link-analysis \
  -H 'Content-Type: application/json' \
  -d '{"url":"example.com"}'
python -m pytest
```

`GET /health` returns HTTP 200 with `{"status":"ok"}`. An invalid URL returns HTTP 422. Tests exercise the complete endpoint with mocked website, Anthropic, OpenAI, and Tavily connections.

## Manual end-to-end check

1. Set `ANTHROPIC_API_KEY` in your local `.env` or environment and restart the server. Use an Anthropic account with access to the configured model.
2. Substitute the URL of a public business page below and run the command. A successful model request uses API credits.
3. Inspect the HTTP status and JSON. A successful result has all five analysis sections and exactly five search queries. The `website` field is the final page URL after redirects.

```bash
curl -sS -i http://localhost:4000/link-analysis \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://your-business.example/services"}'
```

### Example HTTP 200 response

Illustrative output from mocked website and model responses.

```json
{
  "success": true,
  "website": "https://example.com/services",
  "brand": {
    "name": "Northwind IT",
    "category": "Managed IT",
    "summary": "IT support for small businesses."
  },
  "analysis": {
    "competitor_analysis": {
      "brand_overview": "Small business IT support.",
      "main_products_and_services": [
        "Network monitoring"
      ],
      "competitors": [],
      "research_status": "not_configured",
      "strengths": [
        "Around-the-clock monitoring"
      ],
      "opportunities": [
        "Explain response times on service pages"
      ]
    },
    "target_search_queries": [
      {
        "query": "managed IT",
        "platforms": [
          "Google Search"
        ],
        "intent": "Find a provider",
        "business_relevance": "Managed IT",
        "target_audience": "Small businesses"
      },
      {
        "query": "business network support",
        "platforms": [
          "Facebook"
        ],
        "intent": "Find a provider",
        "business_relevance": "Managed IT",
        "target_audience": "Small businesses"
      },
      {
        "query": "IT support plans",
        "platforms": [
          "Instagram"
        ],
        "intent": "Find a provider",
        "business_relevance": "Managed IT",
        "target_audience": "Small businesses"
      },
      {
        "query": "IT monitoring",
        "platforms": [
          "LinkedIn"
        ],
        "intent": "Find a provider",
        "business_relevance": "Managed IT",
        "target_audience": "Small businesses"
      },
      {
        "query": "IT helpdesk",
        "platforms": [
          "X"
        ],
        "intent": "Find a provider",
        "business_relevance": "Managed IT",
        "target_audience": "Small businesses"
      }
    ],
    "seo_improvements": {
      "primary_keyword": "managed IT",
      "secondary_keywords": [
        "IT support"
      ],
      "long_tail_keywords": [
        "managed IT for small businesses"
      ],
      "related_business_keywords": [
        "network monitoring"
      ],
      "recommended_title": "Managed IT for Small Businesses | Northwind IT",
      "title_structure": "Service and audience followed by brand",
      "meta_description": "Find managed IT for your small business.",
      "meta_description_keywords": [
        "managed IT"
      ],
      "supporting_content": [
        "Network monitoring service page"
      ]
    },
    "top_services": [
      {
        "name": "Network monitoring",
        "description": "Around-the-clock business network monitoring.",
        "target_industry": null,
        "target_customer": "Small businesses",
        "main_benefit": "Ongoing monitoring",
        "evidence_status": "website",
        "sources": [
          {
            "url": "https://example.com/services",
            "excerpt": "We monitor business networks around the clock."
          }
        ]
      }
    ],
    "target_audience": {
      "primary_audience": [
        "Small businesses"
      ],
      "secondary_audience": [],
      "industries": [],
      "business_types": [
        "Small businesses"
      ],
      "customer_profiles": [
        "Owners without dedicated IT staff"
      ],
      "geographical_focus": null,
      "customer_needs": [
        "Reliable network monitoring"
      ],
      "customer_challenges": [
        "Limited in-house IT coverage"
      ],
      "customer_interests": [
        "Managed IT"
      ],
      "recommended_marketing_segments": [
        "Small business owners"
      ]
    }
  }
}
```

### Error responses

Errors use `{"success":false,"code":"...","detail":"..."}`.

| Status | Meaning | Example code |
| --- | --- | --- |
| 408 | Production request body timed out | `request_body_timeout` |
| 413 | Production request body exceeded the size limit | `request_too_large` |
| 422 | Malformed request or unsafe website destination | `invalid_request`, `unsafe_url` |
| 429 | Production shared analysis limit reached | `analysis_rate_limited` |
| 502 | Website unavailable, HTML unusable, or model output rejected | `page_not_found`, `unusable_website`, `llm_invalid_output` |
| 503 | Analysis capacity, model configuration, or temporary provider failure | `analysis_busy`, `llm_missing_api_key`, `llm_rate_limited` |
| 504 | Website or model request timed out | `website_timeout`, `llm_timeout` |
| 500 | Unexpected internal failure | `internal_error` |

## Website extraction

`WebsiteExtractor(settings).extract(url)` fetches one page and returns its requested and final URLs, SEO metadata, headings, cleaned text, links, image alt text, and warnings. The final URL identifies the source of extracted content. Links are retained as evidence for later processing and are never fetched during extraction.

| Setting | Default |
| --- | --- |
| `WEBSITE_REQUEST_TIMEOUT` | 10 seconds for DNS and the entire fetch, including redirects |
| `MAX_WEBSITE_RESPONSE_SIZE` | 2,097,152 bytes each for received and decoded data |
| `MAX_WEBSITE_REDIRECTS` | 3 hops |
| `WEBSITE_USER_AGENT` | `AI-Link-Analysis/0.1` |

The API validates the URL without discarding its original spelling. The extractor rejects unsafe authorities and noncanonical numeric hosts. For a DNS name, it checks every IPv4 and IPv6 address returned by the resolver and rejects the entire destination if any answer fails. The address policy blocks non-public and IANA special-purpose space, including IPv4-mapped IPv6, translation and relay ranges, and the publicly numbered Azure WireServer endpoint. [IANA IPv4 registry](https://www.iana.org/assignments/iana-ipv4-special-registry) · [IANA IPv6 registry](https://www.iana.org/assignments/iana-ipv6-special-registry) · [Azure platform endpoint](https://learn.microsoft.com/en-us/azure/virtual-desktop/azurecommunicationips)

IPv6 destinations must also belong to an IANA allocation to a regional internet registry. This rejects reserved and unallocated space inside the global unicast block. Review the checked-in allocation list when IANA adds new prefixes. The check classifies addresses and does not determine how the deployment network routes them. [IANA IPv6 unicast allocations](https://www.iana.org/assignments/ipv6-unicast-address-assignments)

Each connection uses a validated numeric IP with environment proxies disabled. The HTTP Host header and TLS hostname come from the original URL for that hop. Redirects are followed manually, with DNS and address validation repeated even when the hostname stays the same. Connections close between hops. The final HTML is streamed with limits on both downloaded and decoded bytes. Tests cover destination pinning through simulated sockets and TLS, mixed DNS answers, a hostname changing to a private address between redirects, and blocked URL representations.

The extractor reads static HTML and supports identity and single-member gzip encoding. It checks for common bot challenges, interstitials, access-denied messages, and JavaScript shells. Detection is heuristic. Detected gates, empty pages, and explicitly partial HTML return an error with an explanation. DNS, blocked destinations, HTTP failures, timeouts, and unsupported or oversized responses raise typed errors.

For readable pages with very little business text, the API returns a limited report with the same five sections. The brand summary identifies the limitation and quotes the observed title if present. Products, services, competitors, and audiences remain unverified. Search queries and general SEO ideas are labeled as suggestions. The API skips both the LLM and external competitor research for these pages.

## LLM analysis

`generate_analysis(page, settings)` accepts a successful extraction result and returns the typed brand and five analysis sections. The model receives fixed instructions and a separate JSON user message containing prioritized website evidence and optional competitor snippets. The default Claude adapter uses the Messages API with a JSON Schema derived from Pydantic. If Anthropic rejects the structured-output schema with HTTP 400, the adapter makes one fallback request with the schema in the system prompt. Both paths check the stop reason and validate the JSON locally. The service checks query distinctness and verifies citations against the evidence sent to the model. [Claude structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)

| Setting | Default |
| --- | --- |
| `LLM_PROVIDER` | `anthropic` |
| `ANTHROPIC_API_KEY` | Unset. Required for Claude analysis. |
| `ANTHROPIC_MODEL` | `claude-haiku-5-5` |
| `OPENAI_API_KEY` | Unset. Required only with `LLM_PROVIDER=openai`. |
| `OPENAI_MODEL` | `gpt-5.4-mini` when OpenAI is selected. |
| `LLM_TIMEOUT_SECONDS` | 30 seconds for the provider call, including at most one SDK retry |
| `LLM_MAX_OUTPUT_TOKENS` | 5,000 for Claude, 3,000 for OpenAI. An explicit environment value overrides the default. |
| `LLM_MAX_CONTENT_CHARS` | 12,000 characters of prepared website JSON |

Haiku 5.5 runs with thinking disabled to leave the generation budget for the five-section JSON response. To use OpenAI, explicitly set `LLM_PROVIDER=openai` and provide `OPENAI_API_KEY`. The app uses the selected provider without automatic fallback. Logs include stage transitions, failure codes, duration, and available token counts. Keep SDK and HTTP client debug logging disabled when handling website data.

If Claude stops at its output token limit, the API returns `llm_output_truncated`. The adapter rejects the incomplete JSON and does not regenerate it, avoiding a second generation charge. The Claude log line shows the active token limit and whether the response came from native structured output or prompt-only JSON.

A service with a missing required citation or a citation that fails verification is omitted from `top_services`. The response still includes all five analysis sections, and `top_services` can be empty. Logs report omission counts.

### Model comparison and estimated cost

| Model | Input / output per 1M tokens | Context or input limit | Structured JSON | Latency and quality evidence |
| --- | ---: | ---: | --- | --- |
| Claude Haiku 5.5 | $0.10 / $0.50 in the standard short-prompt tier | 1M context | JSON Schema with prompt-only JSON fallback and local Pydantic validation | The recorded Shopify demo completed in 15.1 seconds end to end. Workload accuracy has not been benchmarked. |
| OpenAI GPT-5.4 mini | $0.75 / $4.50 | 400,000 context | Responses API Structured Outputs | Vendor positions it for efficient, high-volume work. Speed and accuracy for this task are unmeasured. |
| Gemini 3.5 Flash-Lite | $0.30 / $2.50 | 1,048,576 input | Schema-constrained JSON | Vendor positions it for low-latency extraction. Speed and accuracy for this task are unmeasured. |

**Recommendation:** Use Claude Haiku 5.5 for its estimated cost and the demonstrated five-section workflow. OpenAI GPT-5.4 mini is the configured alternative. At 6,000 input and 1,500 output tokens, estimated uncached generation costs are $0.00135 for Claude, $0.01125 for OpenAI, and $0.00555 for Gemini. Tavily credits are separate. The [full comparison](docs/LLM_COMPARISON.md) includes official sources, usage assumptions, and a repeatable benchmark procedure.

## Optional competitor research

Tavily receives up to two short searches based on the submitted site's title, headings, and service descriptions. The application favors results whose company name matches the source domain and filters self-matches, duplicates, and results without relevant business terms. It sends snippets as untrusted data within the existing LLM content budget. The app uses the search response as evidence and does not fetch the full competitor pages. [Tavily Search API](https://docs.tavily.com/documentation/api-reference/endpoint/search)

| Setting | Default |
| --- | --- |
| `SEARCH_PROVIDER` | `tavily`. Set to `none` to disable. |
| `SEARCH_API_KEY` | Unset. Set to your Tavily API key to enable search. |
| `SEARCH_MAX_RESULTS` | 5 per query and at most 5 retained sources with the default budget |
| `SEARCH_TIMEOUT_SECONDS` | 5 seconds for the search stage |

The API reports the search outcome in `analysis.competitor_analysis.research_status`. `search_snippets` means snippets were supplied to the model. `not_configured`, `unavailable`, `no_relevant_results`, and `insufficient_budget` indicate why the request used page-only evidence. A timeout, rate limit, or malformed search response falls back to page-only evidence unless earlier search results are usable.

A `sourced` competitor has claims matched verbatim to a cited snippet attributed to that business. An unsupported claim is removed. A `potential` competitor has a citation but too little evidence for a factual comparison. An `ai_suggestion` is an unverified candidate named on the submitted site. Domain and relevance checks are heuristic, and snippet citations do not verify the source's underlying claims.

To exercise search, set `SEARCH_PROVIDER=tavily` and add `SEARCH_API_KEY` and the selected model provider's key to your local `.env`. Restart and run the [manual end-to-end check](#manual-end-to-end-check) with a public business website. Inspect the research status, competitor evidence labels, and each source URL and excerpt. Repeat with `SEARCH_PROVIDER=none` to check page-only fallback. Enabled research can use two Tavily basic-search credits per analysis, up to $0.016 at the documented pay-as-you-go rate, plus model usage. [Tavily credits and pricing](https://docs.tavily.com/documentation/api-credits)

## Deploy on Render

Use a Render Free web service for the assessment demo. It can build directly from the GitHub repository, supplies an HTTPS URL, and supports environment variables and HTTP health checks. Use paid compute if the service needs to stay warm. [Render FastAPI guide](https://render.com/docs/deploy-fastapi) · [Render health checks](https://render.com/docs/health-checks)

| Platform | Setup | Free or entry-level limits |
| --- | --- | --- |
| Render | Native Python build, Blueprint configuration, dashboard secrets, automatic HTTPS URL, and continuous `/health` checks. | Free service sleeps after 15 minutes idle and typically takes about a minute to wake. The workspace shares 750 free instance hours per month. High outbound traffic can trigger suspension. Starter compute starts at $7/month and stays on. [Free limits](https://render.com/docs/free) · [Python hosting plans](https://render.com/hosting/python) |
| Railway | Python build with Railpack, dashboard variables, a generated HTTPS domain, and a deployment-time `/health` check. | Free plan includes $1 monthly usage credit. Hobby starts at $5/month. Optional serverless mode can sleep after at least five minutes without outbound traffic. [Pricing](https://docs.railway.com/pricing/plans) · [Health checks](https://docs.railway.com/deployments/healthchecks) · [Public networking](https://docs.railway.com/networking/public-networking) · [Serverless](https://docs.railway.com/deployments/serverless) |

### Prerequisites and repository setup

Create a GitHub account and a Render account if needed. Create a new empty public GitHub repository for the assessment. Use an Anthropic API key with access to `claude-haiku-5-5`. Tavily credentials are optional. Check the staged files for secrets and use the hosting dashboard to store keys. The application needs no database or persistent disk.

For a new local repository, after approving publication, run these commands from the project root. Replace the GitHub account placeholder.

```bash
git init -b main
git add .env.example .gitignore README.md requirements.txt render.yaml app docs tests
git diff --cached --name-only
git commit -m "Prepare website analysis API"
git remote add origin https://github.com/YOUR_ACCOUNT/AI-Link-Analysis.git
git push -u origin main
```

### Create the web service

1. In the Render dashboard, choose **New → Blueprint**. Connect the GitHub repository and choose the branch that contains the root `render.yaml`.
2. Review the single Free web service. Provide `ANTHROPIC_API_KEY` and `SEARCH_API_KEY` in the Blueprint secret prompts. For an existing service, enter missing keys directly under **Environment** because `sync: false` prompts only on initial Blueprint creation. Use **Save only** to stage environment changes for the next deploy. The service disables commit-triggered builds. Review Blueprint changes separately before syncing. [Render Blueprint guide](https://render.com/docs/infrastructure-as-code)
3. Wait for the build and `/health` check to pass. Use the service's assigned HTTPS URL.
4. For subsequent changes, choose **Manual Deploy** after reviewing the commit and rerunning tests.

| Render setting | Value |
| --- | --- |
| Runtime / Python | Native Python, pinned to `3.14.10` in the Blueprint. The application supports Python 3.11+. [Render Python versions](https://render.com/docs/python-version) |
| Build command | `python -m pip install -r requirements.txt` |
| Start command | `python -m app.main` |
| Listening address and port | `0.0.0.0` and Render's injected `PORT`, normally `10000`. [Render port binding](https://render.com/docs/web-services#port-binding) |
| Health check | `GET /health` |
| Swagger / OpenAPI | `GET /docs` and `GET /openapi.json` |

### Production environment variables

| Variable | Production value |
| --- | --- |
| `ANTHROPIC_API_KEY` | Required for the default analysis provider. Enter the key in Render's secret prompt or environment dashboard. |
| `ENVIRONMENT` | `production`, set by the Blueprint. This enables request limits and uses the process environment for configuration. |
| `HOST`, `PORT`, `PYTHON_VERSION` | The Blueprint sets `HOST` and `PYTHON_VERSION`. Leave `PORT` to Render. |
| `LLM_PROVIDER`, `ANTHROPIC_MODEL` | The Blueprint sets `anthropic` and `claude-haiku-5-5`. |
| `OPENAI_API_KEY`, `OPENAI_MODEL` | Optional. Select `LLM_PROVIDER=openai` to use OpenAI and add its key as a Render secret. |
| `SEARCH_PROVIDER`, `SEARCH_API_KEY` | The Blueprint selects `tavily`. Add the Tavily key as `SEARCH_API_KEY` in Render. Set `SEARCH_PROVIDER=none` to opt out. A later Blueprint sync reapplies the provider declared in `render.yaml`. |
| `ANALYSIS_REQUESTS_PER_HOUR`, `ANALYSIS_MAX_IN_FLIGHT` | Defaults are 20 admitted analysis requests per rolling hour and two in flight, shared per process. |
| `ANALYSIS_MAX_BODY_BYTES`, `ANALYSIS_BODY_TIMEOUT_SECONDS` | Defaults are 32 KiB and five seconds for reading the request body. |

Other fetch and model limits are listed in [`.env.example`](.env.example). Missing model-provider credentials leave liveness and documentation available. A request that reaches generation returns `503` until credentials are configured. Missing search credentials use the page-only analysis. Invalid numeric settings fail startup with the affected variable names.

### Verify the deployed service

Use the public demo URL below for the assessment service, or substitute your own service URL. These checks use no model or search credits.

```bash
base_url='https://ai-link-analysis-k6ny.onrender.com'
curl -fsS "$base_url/health"
curl -fsS -o /dev/null -w '%{http_code}\n' "$base_url/docs"
curl -fsS -o /dev/null -w '%{http_code}\n' "$base_url/openapi.json"
curl -sS -i "$base_url/link-analysis" -H 'Content-Type: application/json' -d '{"url":"example.com"}'
```

Expect `200` for the three GETs and `422` for invalid input. `/health` checks the running process. To verify the LLM and Tavily together, submit one public business website in [Swagger](https://ai-link-analysis-k6ny.onrender.com/docs). Check the five sections, `research_status`, and any returned competitor citations. With Tavily configured, that request can use up to two basic-search credits and one completed model generation in the normal path.

### Public service limits

Swagger runs on the API's own origin. CLI and server clients require no CORS policy. The production start command disables raw Uvicorn access logging and keeps stage, timing, failure codes, and available token counts in the application logs.

The shared in-memory quota resets on process restart. It has no per-client fairness, and additional workers or instances each get their own quota. Set a spend cap where available for the selected model provider, and keep Tavily disabled or capped for a public demo.

### Network isolation for the assessment demo

Render Free web services can initiate private-network connections to services in the same workspace and region. Its documented SMTP port restrictions and outbound usage limits do not provide a configurable destination-deny egress policy. Render documents cross-environment private network isolation for Pro workspaces and higher. Shared outbound IPs identify the source of traffic and are not an outbound firewall. [Render Free limits](https://render.com/docs/free#other-limitations) · [Render private network](https://render.com/docs/private-network) · [Render outbound IPs](https://render.com/docs/outbound-ip-addresses)

Use a dedicated, otherwise empty Render workspace for a time-limited assessment demo. Keep trusted workloads, private links, and destinations that trust the service's egress IPs outside that deployment. Use the existing quota and a dedicated provider project with a spend cap. The demo URL remains publicly reachable while it is running. Application IP checks cannot recognize arbitrary internal routes on globally numbered addresses or control a public server that proxies requests elsewhere. For broader exposure or any hosting network with access to sensitive systems, use an isolated fetcher or a hosting environment with enforceable destination-based egress controls. The live host's network behavior still needs verification after deployment approval.
