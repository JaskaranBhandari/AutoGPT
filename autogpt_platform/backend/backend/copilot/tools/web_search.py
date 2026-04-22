"""Web search tool — direct Exa client.

Skips the dispatch-model round-trip entirely: we call Exa directly, get
structured results back, and use Exa's ``cost_dollars.total`` for real
per-call billing.  No inference tax on top of the search fee —
``search_and_contents`` at 5 results is ~$0.012/call flat.
"""

import logging
from typing import Any

from exa_py import AsyncExa

from backend.copilot.model import ChatSession
from backend.copilot.token_tracking import persist_and_record_usage
from backend.util.settings import Settings

from .base import BaseTool
from .models import ErrorResponse, ToolResponseBase, WebSearchResponse, WebSearchResult

logger = logging.getLogger(__name__)

_DEFAULT_MAX_RESULTS = 5
_HARD_MAX_RESULTS = 20
_SNIPPET_MAX_CHARS = 500


class WebSearchTool(BaseTool):
    """Search the public web and return cited results."""

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Search the web for live info (news, recent docs). Returns "
            "{title, url, snippet}; use web_fetch to deep-dive a URL. "
            "Prefer one targeted query over many reformulations."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query.",
                },
                "max_results": {
                    "type": "integer",
                    "description": (
                        f"Max results (default {_DEFAULT_MAX_RESULTS}, "
                        f"cap {_HARD_MAX_RESULTS})."
                    ),
                    "default": _DEFAULT_MAX_RESULTS,
                },
            },
            "required": ["query"],
        }

    @property
    def requires_auth(self) -> bool:
        return False

    @property
    def is_available(self) -> bool:
        return bool(Settings().secrets.exa_api_key)

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        query: str = "",
        max_results: int = _DEFAULT_MAX_RESULTS,
        **kwargs: Any,
    ) -> ToolResponseBase:
        query = (query or "").strip()
        session_id = session.session_id if session else None
        if not query:
            return ErrorResponse(
                message="Please provide a non-empty search query.",
                error="missing_query",
                session_id=session_id,
            )

        try:
            max_results = int(max_results)
        except (TypeError, ValueError):
            max_results = _DEFAULT_MAX_RESULTS
        max_results = max(1, min(max_results, _HARD_MAX_RESULTS))

        api_key = Settings().secrets.exa_api_key
        if not api_key:
            return ErrorResponse(
                message=(
                    "Web search is unavailable — the deployment has no "
                    "Exa API key configured."
                ),
                error="web_search_not_configured",
                session_id=session_id,
            )

        client = AsyncExa(api_key=api_key)
        try:
            resp = await client.search_and_contents(
                query=query,
                num_results=max_results,
                text={"max_characters": _SNIPPET_MAX_CHARS},
            )
        except Exception as exc:
            logger.warning(
                "[web_search] Exa call failed for query=%r: %s", query, exc
            )
            return ErrorResponse(
                message=f"Web search failed: {exc}",
                error="web_search_failed",
                session_id=session_id,
            )

        results = _extract_results(resp, limit=max_results)
        cost_usd = _extract_cost_usd(resp)

        try:
            await persist_and_record_usage(
                session=session,
                user_id=user_id,
                prompt_tokens=0,
                completion_tokens=0,
                log_prefix="[web_search]",
                cost_usd=cost_usd,
                model="exa/search_and_contents",
                provider="exa",
            )
        except Exception as exc:
            logger.warning("[web_search] usage tracking failed: %s", exc)

        return WebSearchResponse(
            message=f"Found {len(results)} result(s) for {query!r}.",
            query=query,
            results=results,
            search_requests=1 if results else 0,
            session_id=session_id,
        )


def _extract_results(resp: Any, *, limit: int) -> list[WebSearchResult]:
    """Map Exa ``SearchResponse.results`` to our WebSearchResult shape."""
    out: list[WebSearchResult] = []
    for r in (getattr(resp, "results", None) or [])[:limit]:
        snippet = (_get(r, "text") or "")[:_SNIPPET_MAX_CHARS]
        out.append(
            WebSearchResult(
                title=_get(r, "title") or "",
                url=_get(r, "url") or "",
                snippet=snippet,
                page_age=_get(r, "published_date"),
            )
        )
    return out


def _extract_cost_usd(resp: Any) -> float | None:
    """Return ``cost_dollars.total`` from the Exa response, else None.

    Exa ships a structured ``cost_dollars`` object with a ``total`` field.
    Older SDK versions expose it as a string; guard both shapes so we
    never crash accounting on a schema variance.
    """
    cost = getattr(resp, "cost_dollars", None)
    if cost is None:
        return None
    total = getattr(cost, "total", None)
    if total is None and isinstance(cost, dict):
        total = cost.get("total")
    try:
        return float(total) if total is not None else None
    except (TypeError, ValueError):
        return None


def _get(obj: Any, key: str) -> Any:
    """Uniform attribute / dict key access — Exa's result shape varies
    between the pydantic SDK objects and raw dicts on older paths."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)
