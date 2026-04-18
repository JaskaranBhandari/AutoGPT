"""Per-session registry of backgrounded tool calls.

When a tool exceeds its per-call ``timeout_seconds`` budget the in-flight
``asyncio.Task`` is parked here rather than being cancelled. The agent can
then use the ``check_background_tool`` tool (keyed by ``background_id``) to
wait longer, poll status, or cancel — keeping the autopilot in control of
slow sub-agents and graph executions.

Lives in its own module so that both ``tool_adapter.py`` (which registers
tasks during tool dispatch) and ``tools/check_background_tool.py`` (which
inspects them) can import the registry without creating a cycle via the
tool-registry import chain.

Scoping: the registry is a :class:`ContextVar`, so each execution context
(parent AutoPilot, and any sub-AutoPilot invoked via ``run_block``) gets an
independent registry. A sub-AutoPilot cannot see or cancel a parent's
background tasks — this is intentional isolation.
"""

import asyncio
import logging
import time
import uuid
from contextvars import ContextVar
from typing import Any

logger = logging.getLogger(__name__)

# Max wait a single check_background_tool call may block for. Kept below the
# stream-level idle timeout so the outer safety net still triggers if the
# whole session genuinely stalls.
MAX_BACKGROUND_WAIT_SECONDS = 9 * 60  # 9 minutes

# Upper bound on concurrent background tasks per session. Prevents a
# pathological agent from leaking asyncio.Tasks by timing out hundreds of
# tools back-to-back. When full, the oldest entry is cancelled and evicted
# so the newest registration still succeeds.
MAX_BACKGROUND_TASKS_PER_SESSION = 32

_background_tasks: ContextVar[dict[str, dict[str, Any]]] = ContextVar(
    "_background_tasks",
    default=None,  # type: ignore[arg-type]
)


def init_registry() -> None:
    """Install a fresh per-session registry in the current context."""
    _background_tasks.set({})


def register_background_task(task: asyncio.Task, tool_name: str) -> str:
    """Register *task* in the per-session background registry, returning the id.

    If the registry is already at :data:`MAX_BACKGROUND_TASKS_PER_SESSION`,
    the oldest entry is cancelled and evicted to make room.
    """
    bg_id = f"bg-{uuid.uuid4().hex[:12]}"
    registry = _background_tasks.get(None)
    if registry is None:
        # Registry isn't initialized (e.g. unit tests that bypass
        # set_execution_context). Fall back to a fresh dict so we at least
        # don't drop the task silently.
        registry = {}
        _background_tasks.set(registry)

    if len(registry) >= MAX_BACKGROUND_TASKS_PER_SESSION:
        oldest_id, oldest_entry = min(
            registry.items(), key=lambda kv: kv[1]["started_at"]
        )
        oldest_task: asyncio.Task = oldest_entry["task"]
        if not oldest_task.done():
            oldest_task.cancel()
        registry.pop(oldest_id, None)
        logger.warning(
            "Background registry full — evicted oldest entry %s (tool=%s)",
            oldest_id,
            oldest_entry["tool_name"],
        )

    registry[bg_id] = {
        "task": task,
        "tool_name": tool_name,
        "started_at": time.monotonic(),
    }
    return bg_id


def get_background_task(background_id: str) -> dict[str, Any] | None:
    """Return the registered entry for *background_id*, or ``None``."""
    registry = _background_tasks.get(None)
    if registry is None:
        return None
    return registry.get(background_id)


def list_background_tasks() -> list[dict[str, Any]]:
    """Return a snapshot of every registered task in the current session.

    Each entry: ``{background_id, tool_name, started_at, done}``. Used by
    ``check_background_tool(list=true)`` so the agent can recover IDs after
    context compaction or a long pause.
    """
    registry = _background_tasks.get(None)
    if not registry:
        return []
    return [
        {
            "background_id": bg_id,
            "tool_name": entry["tool_name"],
            "started_at": entry["started_at"],
            "done": entry["task"].done(),
        }
        for bg_id, entry in registry.items()
    ]


def unregister_background_task(background_id: str) -> None:
    """Drop a finished/cancelled task from the registry."""
    registry = _background_tasks.get(None)
    if registry is None:
        return
    registry.pop(background_id, None)


def cancel_all_background_tasks(reason: str = "stream ended") -> int:
    """Cancel every task in the registry and empty it.

    Called from the stream's ``finally`` block so orphaned long-running
    tools don't keep executing after the user leaves or the stream errors.
    Returns the number of tasks that were cancelled.
    """
    registry = _background_tasks.get(None)
    if not registry:
        return 0
    cancelled = 0
    for bg_id, entry in list(registry.items()):
        task: asyncio.Task = entry["task"]
        if not task.done():
            task.cancel()
            cancelled += 1
        registry.pop(bg_id, None)
    if cancelled:
        logger.info("Cancelled %d orphaned background task(s) on %s", cancelled, reason)
    return cancelled
