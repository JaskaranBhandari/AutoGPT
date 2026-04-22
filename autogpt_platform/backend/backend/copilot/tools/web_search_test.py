"""Tests for the ``web_search`` copilot tool.

Covers the result + cost extractors as pure units (fed with synthetic
Exa response objects), plus a light integration test that mocks
``AsyncExa.search_and_contents`` and confirms the handler plumbs
through to ``persist_and_record_usage`` with ``provider='exa'`` and the
real ``cost_dollars.total`` value.
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


def _fake_exa_response(
    *,
    results: list[dict] | None = None,
    cost_total: float | None = 0.005,
) -> SimpleNamespace:
    """Build a synthetic Exa ``SearchResponse`` object.

    Matches the shape the ``exa-py`` SDK produces from
    ``client.search_and_contents``: a list of result objects with
    ``title`` / ``url`` / ``text`` / ``published_date`` plus a
    ``cost_dollars`` object with a ``total`` field.
    """
    result_items = []
    for r in results or []:
        result_items.append(
            SimpleNamespace(
                title=r.get("title", "untitled"),
                url=r.get("url", ""),
                text=r.get("text", ""),
                published_date=r.get("published_date"),
            )
        )
    cost = SimpleNamespace(total=cost_total) if cost_total is not None else None
    return SimpleNamespace(results=result_items, cost_dollars=cost)


class TestExtractResults:
    """Pin the Exa SDK shape; an SDK bump surfaces here first."""

    def test_extracts_title_url_text_and_published_date(self):
        resp = _fake_exa_response(
            results=[
                {
                    "title": "Kimi K2.6 launch",
                    "url": "https://example.com/kimi",
                    "text": "Moonshot released K2.6 on 2026-04-20.",
                    "published_date": "2026-04-20",
                },
                {
                    "title": "OpenRouter pricing",
                    "url": "https://openrouter.ai/moonshotai/kimi-k2.6",
                    "text": "",
                },
            ]
        )
        out = _extract_results(resp, limit=10)
        assert len(out) == 2
        assert out[0].title == "Kimi K2.6 launch"
        assert out[0].url == "https://example.com/kimi"
        assert out[0].snippet.startswith("Moonshot released")
        assert out[0].page_age == "2026-04-20"
        assert out[1].snippet == ""

    def test_limit_caps_returned_results(self):
        resp = _fake_exa_response(
            results=[{"title": f"r{i}", "url": f"https://e/{i}"} for i in range(10)]
        )
        out = _extract_results(resp, limit=3)
        assert len(out) == 3
        assert [r.title for r in out] == ["r0", "r1", "r2"]

    def test_missing_results_returns_empty(self):
        resp = SimpleNamespace(results=None, cost_dollars=None)
        out = _extract_results(resp, limit=10)
        assert out == []

    def test_snippet_clamped_to_max_chars(self):
        long_body = "x" * 5000
        resp = _fake_exa_response(
            results=[{"title": "t", "url": "https://e", "text": long_body}]
        )
        out = _extract_results(resp, limit=1)
        assert len(out) == 1
        assert len(out[0].snippet) == 500


class TestExtractCostUsd:
    """Read real ``cost_dollars.total`` from Exa — no hard-coded rates,
    so a future Exa price change is automatically reflected."""

    def test_returns_total_value(self):
        resp = _fake_exa_response(cost_total=0.00823)
        assert _extract_cost_usd(resp) == pytest.approx(0.00823)

    def test_returns_none_when_cost_dollars_missing(self):
        resp = SimpleNamespace(results=[], cost_dollars=None)
        assert _extract_cost_usd(resp) is None

    def test_supports_dict_shape(self):
        resp = SimpleNamespace(results=[], cost_dollars={"total": 0.012})
        assert _extract_cost_usd(resp) == pytest.approx(0.012)

    def test_survives_string_total(self):
        resp = SimpleNamespace(results=[], cost_dollars=SimpleNamespace(total="0.017"))
        assert _extract_cost_usd(resp) == pytest.approx(0.017)


class TestWebSearchToolDispatch:
    """Lightweight integration test: mock ``AsyncExa.search_and_contents``,
    confirm the handler returns a ``WebSearchResponse`` and the usage
    tracker is called with ``provider='exa'`` and the real
    ``cost_dollars.total`` value."""

    def _session(self) -> ChatSession:
        s = ChatSession.new("test-user", dry_run=False)
        s.session_id = "sess-1"
        return s

    @pytest.mark.asyncio
    async def test_returns_response_with_results_and_tracks_cost(self, monkeypatch):
        fake_resp = _fake_exa_response(
            results=[
                {
                    "title": "hello",
                    "url": "https://example.com",
                    "text": "greeting",
                }
            ],
            cost_total=0.005,
        )
        mock_client = SimpleNamespace(
            search_and_contents=AsyncMock(return_value=fake_resp)
        )

        monkeypatch.setattr(
            "backend.copilot.tools.web_search.Settings",
            lambda: SimpleNamespace(
                secrets=SimpleNamespace(exa_api_key="exa-test")
            ),
        )

        with (
            patch(
                "backend.copilot.tools.web_search.AsyncExa",
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

        assert mock_track.await_count == 1
        kwargs = mock_track.await_args.kwargs
        assert kwargs["provider"] == "exa"
        assert kwargs["model"] == "exa/search_and_contents"
        assert kwargs["user_id"] == "u1"
        assert kwargs["cost_usd"] == pytest.approx(0.005)

    @pytest.mark.asyncio
    async def test_missing_api_key_returns_error(self, monkeypatch):
        monkeypatch.setattr(
            "backend.copilot.tools.web_search.Settings",
            lambda: SimpleNamespace(secrets=SimpleNamespace(exa_api_key="")),
        )
        exa_stub = AsyncMock()
        with (
            patch(
                "backend.copilot.tools.web_search.AsyncExa",
                return_value=exa_stub,
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
        exa_stub.search_and_contents.assert_not_called()
        mock_track.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_query_rejected_without_api_call(self, monkeypatch):
        monkeypatch.setattr(
            "backend.copilot.tools.web_search.Settings",
            lambda: SimpleNamespace(
                secrets=SimpleNamespace(exa_api_key="exa-test")
            ),
        )
        exa_stub = AsyncMock()
        with patch(
            "backend.copilot.tools.web_search.AsyncExa",
            return_value=exa_stub,
        ):
            tool = WebSearchTool()
            result = await tool._execute(
                user_id="u1", session=self._session(), query="   "
            )
        assert isinstance(result, ErrorResponse)
        assert result.error == "missing_query"
        exa_stub.search_and_contents.assert_not_called()


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
