"""Web search tool — routes through OpenRouter.

Two tiers, one handler, one billing path:

* ``deep=False`` (default) — OpenRouter's ``openrouter:web_search`` server
  tool with a cheap dispatch model.  Fast and shallow, ~$0.02/call at 5
  results.
* ``deep=True`` — Perplexity ``sonar-deep-research`` via OpenRouter.
  Multi-step reasoning grounded on the web; slower and more expensive
  (~$0.05–0.15/call depending on how many hops the model takes).

OpenRouter standardises ``url_citation`` annotations across engines and
models, so both paths share one extractor.  ``resp.usage.cost`` carries
the real billed value (search fee + tokens) and flows through
``persist_and_record_usage(provider='open_router')`` into the daily /
weekly microdollar rate-limit counter on the same rails as every other
OpenRouter turn — no separate provider ledger line, no estimation
drift.

The three helpers (``_build_web_search_extra_body`` /
``_extract_results`` / ``_extract_cost_usd``) take and return data with
zero coupling to the tool class — a future refactor can lift them into
``backend/util/openrouter_search.py`` so AI-generator blocks can reuse
them.
"""

import logging
from typing import Any

from openai import AsyncOpenAI

from backend.copilot.config import ChatConfig
from backend.copilot.model import ChatSession
from backend.copilot.token_tracking import persist_and_record_usage

from .base import BaseTool
from .models import ErrorResponse, ToolResponseBase, WebSearchResponse, WebSearchResult

logger = logging.getLogger(__name__)

_chat_config = ChatConfig()

# Quick path — cheap tokens, reliable tool-calling; the ~$0.02 Exa
# search fee dominates either way.
_QUICK_DISPATCH_MODEL = "google/gemini-2.5-flash"
_QUICK_MAX_TOKENS = 64

# Deep path — Perplexity sonar searches + reasons multi-step natively.
_DEEP_MODEL = "perplexity/sonar-deep-research"
_DEEP_MAX_TOKENS = 4096

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
            "Prefer one targeted query over many reformulations. "
            "Set deep=true for multi-step research — slower and pricier, "
            "use sparingly."
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
                        f"cap {_HARD_MAX_RESULTS}). Ignored when deep=true."
                    ),
                    "default": _DEFAULT_MAX_RESULTS,
                },
                "deep": {
                    "type": "boolean",
                    "description": (
                        "False = quick Exa-backed search (~$0.02). "
                        "True = Perplexity sonar-deep-research multi-step "
                        "(~$0.05-0.15)."
                    ),
                    "default": False,
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
        deep: bool = False,
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
        if deep:
            model_used = _DEEP_MODEL
            max_tokens = _DEEP_MAX_TOKENS
            extra_body: dict[str, Any] = {"usage": {"include": True}}
        else:
            model_used = _QUICK_DISPATCH_MODEL
            max_tokens = _QUICK_MAX_TOKENS
            extra_body = _build_web_search_extra_body(max_results)

        try:
            resp = await client.chat.completions.create(
                model=model_used,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": query}],
                extra_body=extra_body,
            )
        except Exception as exc:
            logger.warning(
                "[web_search] OpenRouter call failed (deep=%s) for query=%r: %s",
                deep,
                query,
                exc,
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
                model=model_used,
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


def _build_web_search_extra_body(max_results: int) -> dict[str, Any]:
    """``extra_body`` fragment enabling the OpenRouter web-search server
    tool and forcing it to fire.  Lifted out so blocks that want the
    same behaviour can reuse it as-is."""
    return {
        "tools": [
            {
                "type": "openrouter:web_search",
                "openrouter:web_search": {"max_results": max_results},
            }
        ],
        "tool_choice": "required",
        "usage": {"include": True},
    }


def _extract_results(resp: Any, *, limit: int) -> list[WebSearchResult]:
    """Pull ``url_citation`` annotations from an OpenRouter response.

    Shared between the quick (server tool) and deep (Perplexity sonar)
    paths — OpenRouter standardises the annotation schema across engines
    and models.
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
        if _get(ann, "type") != "url_citation":
            continue
        citation = _get(ann, "url_citation") or {}
        results.append(
            WebSearchResult(
                title=_get(citation, "title") or "",
                url=_get(citation, "url") or "",
                snippet=(_get(citation, "content") or "")[:_SNIPPET_MAX_CHARS],
                page_age=None,
            )
        )
    return results


def _extract_cost_usd(resp: Any) -> float | None:
    """Return ``usage.cost`` from the OpenRouter response, or None.

    Populated when the request includes ``extra_body.usage.include=True``.
    Falls back to ``usage.model_dump()`` for SDK versions that return
    dict-like usage objects without a typed ``cost`` attribute.
    """
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
    """Uniform attribute / dict key access — OpenRouter annotations ship
    as either dicts (raw JSON path) or pydantic objects (parsed path)."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)
