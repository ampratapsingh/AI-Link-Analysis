# Production demo verification

The Shopify request on 18 November 2026 used the Render deployment of commit `f3739a3`. The returned JSON reported `success: true` and the final URL `https://www.shopify.com/`. It contained all five analysis sections, five distinct search queries, and five services with cited excerpts. External competitor research was disabled for this recorded run, so it returned `research_status: "not_configured"` and an empty competitor list.

| Resource | URL |
| --- | --- |
| Public service | [Render demo](https://ai-link-analysis-k6ny.onrender.com) |
| Analysis endpoint | [POST /link-analysis](https://ai-link-analysis-k6ny.onrender.com/link-analysis) |
| Interactive documentation | [Swagger UI](https://ai-link-analysis-k6ny.onrender.com/docs) |
| Liveness | [GET /health](https://ai-link-analysis-k6ny.onrender.com/health) |
| API schema | [GET /openapi.json](https://ai-link-analysis-k6ny.onrender.com/openapi.json) |

## Sanitized Render log excerpt

Only application timestamps, stage transitions, and usage counters are included.

```text
2026-11-18T14:00:36.889724361Z Link analysis request received
2026-11-18T14:00:36.889751693Z Website extraction started
2026-11-18T14:00:40.195973859Z Website extraction completed
2026-11-18T14:00:40.195999051Z LLM analysis started
2026-11-18T14:00:51.912674334Z Claude analysis duration_ms=11714 input_tokens=5920 output_tokens=2748 output_mode=prompt_json max_output_tokens=5000
2026-11-18T14:00:51.995043062Z LLM analysis completed
2026-11-18T14:00:51.995270386Z Link analysis finished duration_ms=15105
```

This is a single production observation. See the [LLM comparison](LLM_COMPARISON.md) for cost assumptions and the proposed evaluation procedure.
