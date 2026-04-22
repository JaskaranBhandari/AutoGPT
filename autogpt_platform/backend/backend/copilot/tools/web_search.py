"""Web search tool — wraps OpenRouter's ``openrouter:web_search`` server tool.

OpenRouter's server tool runs the search, injects results into the
assistant message as ``url_citation`` annotations, and bills the search
fee inside the same ``usage.cost`` line as the dispatch model.  Benefits:

* real per-call billing (no hard-coded pricing constants here); and
* the cost auto-flows through ``persist_and_record_usage`` into the
  daily / weekly microdollar rate-limit counter on the same rails as
  every other OpenRouter turn.

The older ``plugins: [{id: "web"}]`` + ``:online`` API are deprecated
upstream; the server tool is the supported path going forward.
"""

import logging
from typing import Any

from openai import AsyncOpenAI

from backend.copilot.config import ChatConfig
from backend.copilot.model import ChatSession
from backend.copilot.token_tracking import persist_and_record_usage

_chat_config = ChatConfig()

from .base import BaseTool
from .models import ErrorResponse, ToolResponseBase, WebSearchResponse, WebSearchResult

logger = logging.getLogger(__name__)

# A small, cheap model is fine — it only has to decide-and-call the
# server tool, not summarise anything.  Override via env if needed.
_WEB_SEARCH_DISPATCH_MODEL = "openai/gpt-4o-mini"
_MAX_DISPATCH_TOKENS = 64
_DEFAULT_MAX_RESULTS = 5
_HARD_MAX_RESULTS = 20


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
        return bool(_chat_config.api_key and _chat_config.base_url)

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

        if not _chat_config.api_key or not _chat_config.base_url:
            return ErrorResponse(
                message=(
                    "Web search is unavailable — the deployment has no "
                    "OpenRouter credentials configured."
                ),
                error="web_search_not_configured",
                session_id=session_id,
            )

        client = AsyncOpenAI(
            api_key=_chat_config.api_key, base_url=_chat_config.base_url
        )
        try:
            resp = await client.chat.completions.create(
                model=_WEB_SEARCH_DISPATCH_MODEL,
                max_tokens=_MAX_DISPATCH_TOKENS,
                messages=[{"role": "user", "content": query}],
                extra_body={
                    "tools": [
                        {
                            "type": "openrouter:web_search",
                            "openrouter:web_search": {
                                "max_results": max_results,
                            },
                        }
                    ],
                    "tool_choice": "required",
                    "usage": {"include": True},
                },
            )
        except Exception as exc:
            logger.warning(
                "[web_search] OpenRouter call failed for query=%r: %s", query, exc
            )
            return ErrorResponse(
                message=f"Web search failed: {exc}",
                error="web_search_failed",
                session_id=session_id,
            )

        results = _extract_results(resp, limit=max_results)
        cost_usd = _extract_cost_usd(resp)

        try:
            usage = getattr(resp, "usage", None)
            await persist_and_record_usage(
                session=session,
                user_id=user_id,
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                log_prefix="[web_search]",
                cost_usd=cost_usd,
                model=_WEB_SEARCH_DISPATCH_MODEL,
                provider="open_router",
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
    """Pull ``url_citation`` annotations from the OpenRouter response.

    OpenRouter's web plugin injects search results as annotations on the
    assistant message: ``{type: "url_citation", url_citation: {url, title,
    content, ...}}``.  Other annotation types (if any) are ignored.
    """
    results: list[WebSearchResult] = []
    choices = getattr(resp, "choices", []) or []
    if not choices:
        return results

    message = getattr(choices[0], "message", None)
    annotations = getattr(message, "annotations", None) or []
    for ann in annotations:
        if len(results) >= limit:
            break
        ann_type = _get(ann, "type")
        if ann_type != "url_citation":
            continue
        citation = _get(ann, "url_citation") or {}
        results.append(
            WebSearchResult(
                title=_get(citation, "title") or "",
                url=_get(citation, "url") or "",
                snippet=(_get(citation, "content") or "")[:500],
                page_age=None,
            )
        )
    return results


def _extract_cost_usd(resp: Any) -> float | None:
    """Return the real per-call cost from OpenRouter's ``usage.cost`` field."""
    usage = getattr(resp, "usage", None)
    if usage is None:
        return None
    val = getattr(usage, "cost", None)
    if val is None and hasattr(usage, "model_dump"):
        val = usage.model_dump().get("cost")
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _get(obj: Any, key: str) -> Any:
    """Uniform attribute / dict key access — OpenRouter's annotation shape
    varies across SDK versions (dict for raw JSON, pydantic for parsed)."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)
