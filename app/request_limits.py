import asyncio
import logging
from collections import deque
from math import ceil
from time import monotonic

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from app.config import Settings
from app.schemas import LinkAnalysisErrorResponse

logger = logging.getLogger(__name__)
_WINDOW_SECONDS = 3600


class AnalysisRequestLimits:
    """Limit memory, concurrency and paid work for one API process."""

    def __init__(self, app: ASGIApp, *, settings: Settings) -> None:
        self.app = app
        self.settings = settings
        self.in_flight = 0
        self.admitted: deque[float] = deque()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] != "/link-analysis":
            await self.app(scope, receive, send)
            return

        declared_size = next(
            (value.lstrip(b"0") for name, value in scope["headers"] if name == b"content-length"),
            b"",
        )
        max_size = self.settings.analysis_max_body_bytes
        if declared_size.isdigit() and (
            len(declared_size) > len(str(max_size)) or int(declared_size) > max_size
        ):
            await self._reject(
                scope, receive, send, 413, "request_too_large", "The JSON body is too large."
            )
            return

        if self.in_flight >= self.settings.analysis_max_in_flight:
            await self._reject(
                scope, receive, send, 503, "analysis_busy",
                "The service is handling other analyses. Try again shortly.", retry_after=5,
            )
            return

        self.in_flight += 1
        try:
            chunks = []
            size = 0
            try:
                async with asyncio.timeout(self.settings.analysis_body_timeout_seconds):
                    while True:
                        message = await receive()
                        if message["type"] == "http.disconnect":
                            return
                        chunk = message.get("body", b"")
                        size += len(chunk)
                        if size > max_size:
                            break
                        if chunk:
                            chunks.append(chunk)
                        if not message.get("more_body", False):
                            break
            except TimeoutError:
                await self._reject(
                    scope, receive, send, 408, "request_body_timeout", "Send the JSON body more quickly."
                )
                return

            if size > max_size:
                await self._reject(
                    scope, receive, send, 413, "request_too_large", "The JSON body is too large."
                )
                return

            now = monotonic()
            while self.admitted and now - self.admitted[0] >= _WINDOW_SECONDS:
                self.admitted.popleft()
            if len(self.admitted) >= self.settings.analysis_requests_per_hour:
                retry_after = max(1, ceil(_WINDOW_SECONDS - (now - self.admitted[0])))
                await self._reject(
                    scope, receive, send, 429, "analysis_rate_limited",
                    "The service has reached its shared analysis limit. Try again later.",
                    retry_after=retry_after,
                )
                return

            self.admitted.append(now)
            body_pending = True

            async def replay_body():
                nonlocal body_pending
                if not body_pending:
                    return await receive()
                body_pending = False
                return {"type": "http.request", "body": b"".join(chunks), "more_body": False}

            await self.app(scope, replay_body, send)
        finally:
            self.in_flight -= 1

    @staticmethod
    async def _reject(
        scope: Scope, receive: Receive, send: Send, status: int, code: str, detail: str,
        *, retry_after: int | None = None,
    ) -> None:
        logger.info("Link analysis rejected stage=ingress error_code=%s", code)
        body = LinkAnalysisErrorResponse(code=code, detail=detail)
        headers = {"Retry-After": str(retry_after)} if retry_after is not None else None
        await JSONResponse(status_code=status, content=body.model_dump(), headers=headers)(
            scope, receive, send
        )
