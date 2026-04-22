"""Tests for the ``web_search`` copilot tool.

Covers the annotation extractor + cost extractor as pure units (fed with
synthetic OpenRouter response objects), plus a light integration test
that mocks ``AsyncOpenAI.chat.completions.create`` and confirms the
handler plumbs through to ``persist_and_record_usage`` with
``provider='open_router'`` and the real ``usage.cost`` value.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.model import ChatSession

from .models import ErrorResponse, WebSearchResponse, WebSearchResult
from .web_search import (
    WebSearchTool,
    _extract_cost_usd,
    _extract_results,
)


def _fake_openrouter_response(
    *,
    citations: list[dict] | None = None,
    prompt_tokens: int = 120,
    completion_tokens: int = 40,
    cost: float | None = 0.02,
) -> SimpleNamespace:
    """Build a synthetic OpenRouter Chat Completions response.

    Matches the shape produced by ``AsyncOpenAI.chat.completions.create``
    when the ``openrouter:web_search`` server tool runs: search results
    arrive as ``url_citation`` annotations on ``choices[0].message``,
    and ``usage.cost`` carries the real billed cost (tokens + search fee).
    """
    annotations = []
    for c in citations or []:
        annotations.append(
            {
                "type": "url_citation",
                "url_citation": {
                    "url": c.get("url", ""),
                    "title": c.get("title", "untitled"),
                    "content": c.get("content", ""),
                },
            }
        )
    message = SimpleNamespace(
        role="assistant",
        content="ok",
        annotations=annotations,
    )
    choices = [SimpleNamespace(message=message, finish_reason="stop")]
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        cost=cost,
    )
    return SimpleNamespace(choices=choices, usage=usage)


class TestExtractResults:
    """Pin the annotation-shape contact point; an OpenRouter schema
    change surfaces here first."""

    def test_extracts_title_url_snippet(self):
        resp = _fake_openrouter_response(
            citations=[
                {
                    "title": "Kimi K2.6 launch",
                    "url": "https://example.com/kimi",
                    "content": "Moonshot released K2.6 on 2026-04-20.",
                },
                {
                    "title": "OpenRouter pricing",
                    "url": "https://openrouter.ai/moonshotai/kimi-k2.6",
                    "content": "",
                },
            ]
        )
        out = _extract_results(resp, limit=10)
        assert len(out) == 2
        assert out[0].title == "Kimi K2.6 launch"
        assert out[0].url == "https://example.com/kimi"
        assert out[0].snippet.startswith("Moonshot released")
        assert out[1].snippet == ""

    def test_limit_caps_returned_results(self):
        resp = _fake_openrouter_response(
            citations=[
                {"title": f"r{i}", "url": f"https://e/{i}"} for i in range(10)
            ]
        )
        out = _extract_results(resp, limit=3)
        assert len(out) == 3
        assert [r.title for r in out] == ["r0", "r1", "r2"]

    def test_missing_choices_returns_empty(self):
        resp = SimpleNamespace(choices=[], usage=None)
        out = _extract_results(resp, limit=10)
        assert out == []

    def test_non_url_citation_annotations_are_ignored(self):
        resp = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        role="assistant",
                        content="ok",
                        annotations=[
                            {"type": "file_citation", "file_citation": {}},
                            {
                                "type": "url_citation",
                                "url_citation": {
                                    "url": "https://real.example",
                                    "title": "real",
                                    "content": "body",
                                },
                            },
                        ],
                    )
                )
            ],
            usage=None,
        )
        out = _extract_results(resp, limit=10)
        assert len(out) == 1 and out[0].title == "real"


class TestExtractCostUsd:
    """Read real ``usage.cost`` from OpenRouter — no hard-coded rates,
    so a future provider price change is automatically reflected."""

    def test_returns_cost_value(self):
        resp = _fake_openrouter_response(cost=0.023456)
        assert _extract_cost_usd(resp) == pytest.approx(0.023456)

    def test_returns_none_when_usage_missing(self):
        resp = SimpleNamespace(choices=[], usage=None)
        assert _extract_cost_usd(resp) is None

    def test_returns_none_when_cost_missing(self):
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
        resp = SimpleNamespace(choices=[], usage=usage)
        assert _extract_cost_usd(resp) is None

    def test_survives_string_cost(self):
        # OpenRouter has shipped both float and string ``cost`` at
        # different points; coerce safely.
        usage = SimpleNamespace(cost="0.017")
        resp = SimpleNamespace(choices=[], usage=usage)
        assert _extract_cost_usd(resp) == pytest.approx(0.017)


class TestWebSearchToolDispatch:
    """Lightweight integration test: mock the OpenAI client, confirm
    the handler returns a ``WebSearchResponse`` and the usage tracker is
    called with ``provider='open_router'`` and the real ``usage.cost``."""

    def _session(self) -> ChatSession:
        s = ChatSession.new("test-user", dry_run=False)
        s.session_id = "sess-1"
        return s

    @pytest.mark.asyncio
    async def test_returns_response_with_results_and_tracks_cost(self, monkeypatch):
        fake_resp = _fake_openrouter_response(
            citations=[
                {
                    "title": "hello",
                    "url": "https://example.com",
                    "content": "greeting",
                }
            ],
            cost=0.0214,
        )
        mock_client = type(
            "MC",
            (),
            {
                "chat": type(
                    "C",
                    (),
                    {
                        "completions": type(
                            "CC",
                            (),
                            {"create": AsyncMock(return_value=fake_resp)},
                        )()
                    },
                )()
            },
        )()

        monkeypatch.setattr(
            "backend.copilot.tools.web_search._chat_config",
            SimpleNamespace(
                api_key="sk-test",
                base_url="https://openrouter.ai/api/v1",
            ),
        )

        with (
            patch(
                "backend.copilot.tools.web_search.AsyncOpenAI",
                return_value=mock_client,
            ),
            patch(
                "backend.copilot.tools.web_search.persist_and_record_usage",
                new=AsyncMock(return_value=160),
            ) as mock_track,
        ):
            tool = WebSearchTool()
            result = await tool._execute(
                user_id="u1",
                session=self._session(),
                query="kimi k2.6 launch",
                max_results=5,
            )

        assert isinstance(result, WebSearchResponse)
        assert result.query == "kimi k2.6 launch"
        assert len(result.results) == 1
        assert isinstance(result.results[0], WebSearchResult)
        assert result.search_requests == 1

        # Cost tracker called with provider="open_router" and the real
        # ``usage.cost`` value (NOT an estimate).
        assert mock_track.await_count == 1
        kwargs = mock_track.await_args.kwargs
        assert kwargs["provider"] == "open_router"
        assert kwargs["model"] == "openai/gpt-4o-mini"
        assert kwargs["user_id"] == "u1"
        assert kwargs["cost_usd"] == pytest.approx(0.0214)

    @pytest.mark.asyncio
    async def test_missing_credentials_returns_error(self, monkeypatch):
        monkeypatch.setattr(
            "backend.copilot.tools.web_search._chat_config",
            SimpleNamespace(api_key="", base_url=""),
        )
        openai_stub = AsyncMock()
        with (
            patch(
                "backend.copilot.tools.web_search.AsyncOpenAI",
                return_value=openai_stub,
            ),
            patch(
                "backend.copilot.tools.web_search.persist_and_record_usage",
                new=AsyncMock(),
            ) as mock_track,
        ):
            tool = WebSearchTool()
            assert tool.is_available is False
            result = await tool._execute(
                user_id="u1",
                session=self._session(),
                query="anything",
            )
        assert isinstance(result, ErrorResponse)
        assert result.error == "web_search_not_configured"
        openai_stub.chat.completions.create.assert_not_called()
        mock_track.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_query_rejected_without_api_call(self, monkeypatch):
        monkeypatch.setattr(
            "backend.copilot.tools.web_search._chat_config",
            SimpleNamespace(
                api_key="sk-test",
                base_url="https://openrouter.ai/api/v1",
            ),
        )
        openai_stub = AsyncMock()
        with patch(
            "backend.copilot.tools.web_search.AsyncOpenAI",
            return_value=openai_stub,
        ):
            tool = WebSearchTool()
            result = await tool._execute(
                user_id="u1", session=self._session(), query="   "
            )
        assert isinstance(result, ErrorResponse)
        assert result.error == "missing_query"
        openai_stub.chat.completions.create.assert_not_called()


class TestToolRegistryIntegration:
    """The tool must be registered under the ``web_search`` name so the
    MCP layer exposes it as ``mcp__copilot__web_search`` — which is
    what the SDK path now dispatches to (see
    ``sdk/tool_adapter.py::SDK_DISALLOWED_TOOLS`` which blocks the CLI's
    native ``WebSearch`` in favour of the MCP route)."""

    def test_web_search_is_in_tool_registry(self):
        from backend.copilot.tools import TOOL_REGISTRY

        assert "web_search" in TOOL_REGISTRY
        assert isinstance(TOOL_REGISTRY["web_search"], WebSearchTool)

    def test_sdk_native_websearch_is_disallowed(self):
        from backend.copilot.sdk.tool_adapter import SDK_DISALLOWED_TOOLS

        assert "WebSearch" in SDK_DISALLOWED_TOOLS
