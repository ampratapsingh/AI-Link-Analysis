# LLM provider comparison for website link analysis

Provider documentation checked on **18 November 2026**.

The API turns extracted website text and optional competitor snippets into a brand profile and five analysis sections. The model needs to distinguish evidence from inference and produce specific SEO, service, query, and audience recommendations in one structured response. The default Anthropic adapter uses the Messages API and validates against Pydantic. The optional OpenAI adapter uses the Responses API. Application checks enforce query distinctness and verify competitor and service citations against supplied text.

## Research methodology

Model IDs, feature support, token limits, and prices come from the first-party OpenAI API, Gemini Developer API, and Claude API documentation. The baseline uses standard, uncached, online text generation in USD. Model listings establish advertised API availability. Access for a particular account or region requires a credentialed check. The suitability and integration recommendations below are project assessments.

## Provider comparison

| Provider and model ID | Standard input / output per 1M tokens | Context or input limit | Max output tokens | Structured JSON and Python compatibility |
| --- | ---: | --- | ---: | --- |
| OpenAI GPT-5.4 mini, `gpt-5.4-mini` | $0.75 / $4.50 | 400,000 context | 128,000 | Responses API Structured Outputs with strict JSON Schema and SDK-native Pydantic parsing. The app adapts the wire schema and enforces URL and list constraints locally. [Model and rates][openai-model] · [Structured Outputs][openai-schema] |
| Google Gemini 3.5 Flash-Lite, `gemini-3.5-flash-lite` | $0.30 / $2.50 | 1,048,576 input | 65,536 | Stable Gemini model with schema-constrained JSON. The Google Gen AI Python SDK can send a Pydantic-generated JSON Schema. The API supports a subset of JSON Schema, so validate the result locally. [Model][gemini-model] · [Rates][gemini-pricing] · [Structured output][gemini-schema] |
| Anthropic Claude Haiku 5.5, `claude-haiku-5-5` | $0.10 / $0.50 through 100,000 prompt tokens. $0.50 / $2.50 above that threshold. | 1 million context | 128,000 | Claude Messages API accepts a JSON Schema in `output_config.format`. The Python SDK's `messages.parse` accepts a Pydantic model, adapts its schema, and validates the result. [Model][claude-model] · [Rates][claude-pricing] · [Structured outputs][claude-schema] |

Google publishes separate input and output limits. The OpenAI and Anthropic rows use their published context windows, with the output ceiling listed separately. Every model has substantial headroom for the baseline workload.

| Model | Expected speed and basis | Website-analysis fit and trade-offs | Project recommendation |
| --- | --- | --- | --- |
| GPT-5.4 mini | OpenAI positions it for efficient, high-volume work and documents `none` as the default reasoning effort. A reasonable candidate for responsive, bounded structured generation. This workload's latency is unmeasured. [Model][openai-model] | The existing adapter parses the full contract and handles refusals, incomplete output, provider errors, and timeouts. It has the highest baseline token cost in this comparison. | **Optional alternative** and benchmark comparison point. |
| Gemini 3.5 Flash-Lite | Google positions it as low-latency and high-throughput for parsing and extraction. A candidate for low-latency generation. This workload's latency is unmeasured. [Model][gemini-model] | Lower cost and ample input capacity. A candidate for brand and service extraction and SEO suggestions. Test whether it preserves competitor attribution and stays specific across all five sections. A Gemini adapter and schema acceptance check are needed. | Evaluate as a second alternative. |
| Claude Haiku 5.5 | Anthropic describes it as the fastest option in its lineup. The app disables adaptive thinking to reserve the output budget for JSON. One production Shopify run took 11.7 seconds in the provider stage and 15.1 seconds end to end. [Model][claude-model] · [Demo logs](DEMO_VERIFICATION.md) | Lowest baseline cost. The adapter uses the existing URL and excerpt fields for citations and falls back to prompt-only JSON when Anthropic rejects the native schema. Both paths validate locally. | **Current default.** The demo exercised the fallback and returned all five sections. |

## Pricing and cost estimates

Assumptions are 6,000 billed input tokens, 1,500 billed output tokens, one generation call per analysis, and no cache hits. Input includes instructions, evidence, schema, and any provider overhead billed as input. Output is the total billable count, including reasoning or thinking tokens where the provider bills them as output. The same text can tokenize differently across providers, and `LLM_MAX_CONTENT_CHARS` controls characters in the evidence JSON.

```text
Estimated cost = (6,000 × input rate + 1,500 × output rate) / 1,000,000
```

| Model | Input subtotal | Output subtotal | 1 analysis | 100 analyses | 1,000 analyses | 10,000 analyses |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-5.4 mini | $0.00450 | $0.00675 | **$0.01125** | $1.125 | $11.25 | $112.50 |
| Gemini 3.5 Flash-Lite | $0.00180 | $0.00375 | **$0.00555** | $0.555 | $5.55 | $55.50 |
| Claude Haiku 5.5 | $0.00060 | $0.00075 | **$0.00135** | $0.135 | $1.35 | $13.50 |

These are estimates using unrounded per-analysis costs. They exclude retries, taxes, regional surcharges, free-tier credits, storage, and optional tools. If prompt input exceeds 100,000 tokens, Claude Haiku 5.5 applies higher rates to the entire request. Cache reads and writes count toward that threshold. Claude's Structured Outputs feature can add a billed system prompt. [Claude pricing][claude-pricing] · [Claude structured output token costs][claude-schema]

Cache discounts apply when eligible input is reused. OpenAI lists $0.075 per 1M cached input tokens for GPT-5.4 mini. Google lists $0.03 per 1M cached tokens plus $1.00 per 1M tokens per hour of context-cache storage. In Haiku's lower tier, cache writes are $0.125 per 1M tokens for five minutes or $0.20 for one hour, and reads are $0.01 per 1M. [OpenAI model rates][openai-model] · [OpenAI prompt caching][openai-cache] · [Gemini pricing][gemini-pricing] · [Claude pricing][claude-pricing]

**Tavily is billed separately.** Enabled research can make two basic searches at one credit each. At the published $0.008 pay-as-you-go price per credit, two searches add **$0.016** per analysis. [Tavily pricing][tavily-pricing]

## Response speed and accuracy

- **Provider-published information.** The model pages establish limits and feature availability. Their speed descriptions give a qualitative expectation within each provider's lineup.
- **Third-party measurements.** Artificial Analysis reports time to the first answer token and end-to-end output speed under its own workloads and network conditions. This API needs a complete, validated JSON response. Those measurements can inform hypotheses for a pilot but cannot establish website-analysis latency or accuracy. [Artificial Analysis performance methodology][aa-methodology]
- **Project measurements.** One successful production Shopify run took **15.1 seconds end to end**, including **11.7 seconds** in the provider stage. Claude reported **5,920 input** and **2,748 output** tokens on the successful generation, an estimated **$0.00197** at short-prompt list prices. This was a single warm-service observation with search disabled. It establishes neither a latency distribution nor comparative accuracy. Render Free cold starts and optional search can add latency. [Demo logs](DEMO_VERIFICATION.md) · [Claude pricing][claude-pricing]
- **Qualitative accuracy expectations.** Schema constraints help enforce fields, types, and enums. Website-specific accuracy requires checking whether services are actually offered, competitor claims are attributed, and SEO and audience recommendations are useful. Sparse or ambiguous evidence should lead to appropriate abstention. Compare these behaviors on identical frozen evidence.

## Reproducible benchmark procedure

1. **Freeze inputs.** Choose ten public business pages spanning local services, B2B, and e-commerce. Include a sparse page and one with ambiguous competitor evidence. Extract each page once. Save the prepared evidence JSON, matching `SYSTEM_PROMPT`, `GeneratedAnalysis.model_json_schema()`, app revision, capture date, and expected facts. Include fixed competitor snippets for some cases. Use the same evidence without repeating live website fetches or Tavily searches during timed model runs.
2. **Use equivalent requests.** Send identical instructions and evidence separately using each provider's schema output feature. Preserve fields, enums, and nullability. Adapt unsupported wire-schema keywords and enforce the original constraints locally. Set a common 5,000-token benchmark cap, omit provider tools, use standard pricing, and document each model's reasoning setting and exact version. Record fallback results separately from native structured output. [OpenAI Structured Outputs][openai-schema] · [Gemini structured output][gemini-schema] · [Claude structured outputs][claude-schema]
3. **Run comparable trials.** Make three non-streaming runs per case per model from the same host, rotating provider order. Disable retries for the measurement. Record cache hits and first-schema-call effects separately. Time from submission to complete validation using a monotonic clock. Save case ID, status, elapsed seconds, billed input and output tokens, and available reasoning and cache details.
4. **Validate and score.** Apply `GeneratedAnalysis.model_validate_json` and the application's query and evidence checks. Record raw schema pass rate and how many competitor claims grounding removes. Compare brand and category to the reference, assess service correctness and coverage, and review SEO, search queries, and audiences for relevance and specificity. Score each quality dimension 0 for wrong or unsupported, 1 for partially correct or generic, and 2 for grounded and specific. Score appropriate abstention as correct on sparse evidence. Review without provider labels.
5. **Report results.** Include each quality dimension, pass and failure rates, median and p95 completion times with sample counts, billed tokens, and estimated spend from actual usage. Record refusals and timeouts alongside successes. Measure a separate full API timing only after choosing a candidate and approving any adapter or configuration change.

## Final model and fallback recommendation

**Anthropic `claude-haiku-5-5` is the default.** It has the lowest estimated token cost in the comparison, **$0.00135 per uncached analysis** at 6,000 input and 1,500 output tokens. For initial planning, use **about 15 seconds** as a provisional reference for a warm, search-disabled demo request, based on the single observed Shopify run. Mocked tests cover schema validation and error handling. Comparative accuracy and performance across other sites remain unmeasured.

**OpenAI `gpt-5.4-mini` remains available as an optional alternative.** Select it explicitly with `LLM_PROVIDER=openai`. Gemini would require an adapter.

## Cost optimization

| Already implemented | Potential improvement |
| --- | --- |
| Extract cleaned text from HTML and remove boilerplate and duplicate passages. Prioritize titles, descriptions, headings, and offerings. Limit prepared evidence to 12,000 characters by default and reserve at most one third for competitor snippets. | Measure tokens for evidence, prompt, and schema together. Add a provider-specific token guard if the observed distribution warrants it. Keep the stable instructions concise and preserve grounding rules. |
| Request all five sections in one generation. Claude can make one prompt-only fallback after a native-schema rejection. Claude defaults to a 5,000-token output cap and OpenAI to 3,000. The adapter does not regenerate truncated output. | Benchmark smaller models on the complete task before switching or adding a separate extraction call. Tune the caps using observed truncation and quality. |
| Log request duration and available input and output usage. Bound optional search to two basic calls and filter their snippets. | Aggregate billed-token percentiles, cache-hit rates, and cost per successful response. Track Tavily credits and model tokens independently. |
| Deduplicate evidence within each analysis. | If repeat traffic warrants it, cache validated results by content fingerprint and model, prompt, and schema versions. Give competitor evidence a freshness window. Evaluate provider caching for common prefixes where eligible. |

## Limitations and official references

Rates and model availability can change. The Shopify run verifies Claude access on the demo deployment and provides one timing and usage sample. Live OpenAI and Gemini results and comparative accuracy remain unmeasured. The cost comparison assumes uniform token counts. Model-specific tokenization, reasoning settings, structured-output overhead, and rate limits can affect cost and latency. Static extraction and search snippets limit the available evidence.

- OpenAI [GPT-5.4 mini model and rates][openai-model], [API pricing][openai-pricing], [Structured Outputs][openai-schema], and [prompt caching][openai-cache].
- Google [Gemini 3.5 Flash-Lite model][gemini-model], [Gemini Developer API pricing][gemini-pricing], and [structured output][gemini-schema].
- Anthropic [Claude Haiku 5.5 model][claude-model], [Claude API pricing][claude-pricing], and [structured outputs][claude-schema].
- Tavily [search credit pricing][tavily-pricing].

[openai-model]: https://developers.openai.com/api/docs/models/gpt-5.4-mini
[openai-pricing]: https://developers.openai.com/api/docs/pricing
[openai-schema]: https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses
[openai-cache]: https://developers.openai.com/api/docs/guides/prompt-caching
[gemini-model]: https://aistudio.google.com/docs/models/gemini-3.5-flash-lite
[gemini-pricing]: https://aistudio.google.com/docs/pricing#gemini-3.5-flash-lite
[gemini-schema]: https://aistudio.google.com/docs/structured-output
[claude-model]: https://platform.claude.com/docs/en/models/haiku-5-5/overview
[claude-pricing]: https://platform.claude.com/docs/en/about-claude/pricing
[claude-schema]: https://platform.claude.com/docs/en/build-with-claude/structured-outputs
[tavily-pricing]: https://docs.tavily.com/documentation/api-credits
[aa-methodology]: https://artificialanalysis.ai/methodology/performance-benchmarking
