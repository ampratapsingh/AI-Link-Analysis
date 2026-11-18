from dataclasses import dataclass
from typing import Protocol

import httpx
from pydantic import SecretStr


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str


class SearchError(Exception):
    """An optional search request failed."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class SearchProvider(Protocol):
    async def search(self, query: str, *, max_results: int) -> list[SearchResult]: ...


class TavilySearchProvider:
    def __init__(
        self, key: SecretStr, timeout: float, *, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self._key = key
        self._timeout = timeout
        self._transport = transport

    async def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        try:
            async with httpx.AsyncClient(
                transport=self._transport, timeout=self._timeout,
                trust_env=False, follow_redirects=False,
            ) as client:
                response = await client.post(
                    "https://api.tavily.com/search",
                    headers={"Authorization": f"Bearer {self._key.get_secret_value().strip()}"},
                    json={
                        "query": query,
                        "max_results": max_results,
                        "search_depth": "basic",
                        "chunks_per_source": 1,
                        "include_answer": False,
                        "include_raw_content": False,
                        "auto_parameters": False,
                    },
                )
                response.raise_for_status()
                body = response.json()
        except httpx.TimeoutException:
            raise SearchError("timeout") from None
        except httpx.HTTPStatusError as exc:
            code = "rate_limited" if exc.response.status_code == 429 else "http_error"
            raise SearchError(code) from None
        except httpx.RequestError:
            raise SearchError("connection_error") from None
        except ValueError:
            raise SearchError("malformed_response") from None

        if not isinstance(body, dict) or not isinstance(body.get("results"), list):
            raise SearchError("malformed_response")
        results = []
        for item in body["results"][:max_results]:
            if isinstance(item, dict) and all(
                isinstance(item.get(field), str) and item[field].strip()
                for field in ("title", "url", "content")
            ):
                results.append(SearchResult(item["title"], item["url"], item["content"]))
        if body["results"] and not results:
            raise SearchError("malformed_response")
        return results
