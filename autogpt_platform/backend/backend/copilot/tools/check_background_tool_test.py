"""Tests for CheckBackgroundToolTool."""

import asyncio
import contextlib
from unittest.mock import MagicMock

import pytest

from backend.copilot.response_model import StreamToolOutputAvailable
from backend.copilot.sdk.background_registry import (
    init_registry,
    register_background_task,
)

from .check_background_tool import CheckBackgroundToolTool
from .models import BackgroundToolList, BackgroundToolStatus


def _make_session() -> MagicMock:
    session = MagicMock()
    session.session_id = "s1"
    session.dry_run = False
    return session


def _completed_result(output: str = "ok") -> StreamToolOutputAvailable:
    return StreamToolOutputAvailable(
        toolCallId="tc-1",
        output=output,
        toolName="slow_tool",
        success=True,
    )


@pytest.fixture(autouse=True)
def _init_registry_for_each_test():
    init_registry()


class TestCheckBackgroundTool:
    @pytest.mark.asyncio
    async def test_missing_background_id_returns_error(self):
        tool = CheckBackgroundToolTool()
        response = await tool._execute(
            user_id="u",
            session=_make_session(),
            background_id="",
        )
        assert response.type.value == "error"

    @pytest.mark.asyncio
    async def test_unknown_background_id_returns_error(self):
        tool = CheckBackgroundToolTool()
        response = await tool._execute(
            user_id="u",
            session=_make_session(),
            background_id="bg-does-not-exist",
        )
        assert response.type.value == "error"
        assert "No background task" in response.message

    @pytest.mark.asyncio
    async def test_wait_zero_returns_still_running(self):
        async def slow():
            await asyncio.sleep(10)
            return _completed_result()

        task = asyncio.create_task(slow())
        bg_id = register_background_task(task, "slow_tool")

        tool = CheckBackgroundToolTool()
        response = await tool._execute(
            user_id="u",
            session=_make_session(),
            background_id=bg_id,
            wait_seconds=0,
        )
        assert isinstance(response, BackgroundToolStatus)
        assert response.status == "still_running"
        assert response.background_id == bg_id

        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_wait_returns_completed_when_task_finishes(self):
        async def fast():
            await asyncio.sleep(0.05)
            return _completed_result("final-output")

        task = asyncio.create_task(fast())
        bg_id = register_background_task(task, "slow_tool")

        tool = CheckBackgroundToolTool()
        response = await tool._execute(
            user_id="u",
            session=_make_session(),
            background_id=bg_id,
            wait_seconds=5,
        )
        assert isinstance(response, BackgroundToolStatus)
        assert response.status == "completed"
        assert response.output == "final-output"

    @pytest.mark.asyncio
    async def test_wait_times_out_and_returns_still_running(self):
        async def slow():
            await asyncio.sleep(10)
            return _completed_result()

        task = asyncio.create_task(slow())
        bg_id = register_background_task(task, "slow_tool")

        tool = CheckBackgroundToolTool()
        response = await tool._execute(
            user_id="u",
            session=_make_session(),
            background_id=bg_id,
            wait_seconds=1,
        )
        assert isinstance(response, BackgroundToolStatus)
        assert response.status == "still_running"
        assert response.waited_seconds == 1

        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_cancel_true_cancels_and_removes_from_registry(self):
        observed_cancel = asyncio.Event()

        async def stays_until_cancelled():
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                observed_cancel.set()
                raise
            return _completed_result()

        task = asyncio.create_task(stays_until_cancelled())
        # Let the task start before we cancel it.
        await asyncio.sleep(0)

        bg_id = register_background_task(task, "slow_tool")

        tool = CheckBackgroundToolTool()
        response = await tool._execute(
            user_id="u",
            session=_make_session(),
            background_id=bg_id,
            cancel=True,
        )
        assert isinstance(response, BackgroundToolStatus)
        assert response.status == "cancelled"

        with contextlib.suppress(asyncio.CancelledError):
            await task
        assert observed_cancel.is_set()

        from backend.copilot.sdk.background_registry import get_background_task

        assert get_background_task(bg_id) is None

    @pytest.mark.asyncio
    async def test_cancel_after_task_completed_returns_real_result(self):
        """If the task completes between registration and the agent's
        cancel=true call, surface the real result instead of reporting
        'cancelled' and losing the output (race guard)."""

        async def finish_quickly():
            return _completed_result("final-value")

        task = asyncio.create_task(finish_quickly())
        await task  # definitely done by the time we register
        bg_id = register_background_task(task, "slow_tool")

        tool = CheckBackgroundToolTool()
        response = await tool._execute(
            user_id="u",
            session=_make_session(),
            background_id=bg_id,
            cancel=True,
        )
        assert isinstance(response, BackgroundToolStatus)
        assert response.status == "completed"
        assert response.output == "final-value"

    @pytest.mark.asyncio
    async def test_errored_task_reports_error_status(self):
        async def raises():
            raise ValueError("boom")

        task = asyncio.create_task(raises())
        # Let the task complete before we query it.
        try:
            await task
        except ValueError:
            pass
        bg_id = register_background_task(task, "broken_tool")

        tool = CheckBackgroundToolTool()
        response = await tool._execute(
            user_id="u",
            session=_make_session(),
            background_id=bg_id,
        )
        assert isinstance(response, BackgroundToolStatus)
        assert response.status == "error"
        assert "boom" in response.message

    @pytest.mark.asyncio
    async def test_finished_task_with_success_false_reports_error(self):
        """A tool that completes with success=False (without raising) is
        reported as status='error', not 'completed', so the agent doesn't
        treat it as a win."""

        async def finish_with_failure():
            return StreamToolOutputAvailable(
                toolCallId="tc-1",
                output="partial",
                toolName="broken_tool",
                success=False,
            )

        task = asyncio.create_task(finish_with_failure())
        await task
        bg_id = register_background_task(task, "broken_tool")

        tool = CheckBackgroundToolTool()
        response = await tool._execute(
            user_id="u",
            session=_make_session(),
            background_id=bg_id,
        )
        assert isinstance(response, BackgroundToolStatus)
        assert response.status == "error"
        assert response.output == "partial"

    @pytest.mark.asyncio
    async def test_list_true_returns_active_background_tasks(self):
        """list=true enumerates registered tasks so the agent can recover
        forgotten background_ids."""

        async def hang():
            await asyncio.sleep(60)

        tasks = [asyncio.create_task(hang()) for _ in range(2)]
        await asyncio.sleep(0)
        bg_ids = [register_background_task(t, f"tool_{i}") for i, t in enumerate(tasks)]

        tool = CheckBackgroundToolTool()
        response = await tool._execute(
            user_id="u",
            session=_make_session(),
            list=True,
        )
        assert isinstance(response, BackgroundToolList)
        assert len(response.tasks) == 2
        returned_ids = {entry.background_id for entry in response.tasks}
        assert returned_ids == set(bg_ids)
        for entry in response.tasks:
            assert entry.tool.startswith("tool_")
            assert entry.age_seconds >= 0
            assert entry.done is False

        for t in tasks:
            t.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await t

    @pytest.mark.asyncio
    async def test_list_true_empty_when_no_tasks(self):
        tool = CheckBackgroundToolTool()
        response = await tool._execute(
            user_id="u",
            session=_make_session(),
            list=True,
        )
        assert isinstance(response, BackgroundToolList)
        assert response.tasks == []

    @pytest.mark.asyncio
    async def test_cancel_in_dry_run_does_not_actually_cancel_task(self):
        """Under session.dry_run, cancel=true must not kill the real task."""

        async def hang():
            await asyncio.sleep(60)

        task = asyncio.create_task(hang())
        await asyncio.sleep(0)
        bg_id = register_background_task(task, "slow_tool")

        session = _make_session()
        session.dry_run = True

        tool = CheckBackgroundToolTool()
        response = await tool._execute(
            user_id="u",
            session=session,
            background_id=bg_id,
            cancel=True,
        )
        assert isinstance(response, BackgroundToolStatus)
        assert response.status == "cancelled"
        assert "[dry-run]" in response.message
        # Real task is still running.
        assert not task.done()

        # Cleanup.
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    def test_requires_auth_is_true(self):
        tool = CheckBackgroundToolTool()
        assert tool.requires_auth is True
