"""Tests for OrchestratorBlock per-iteration cost charging.

The OrchestratorBlock in agent mode makes multiple LLM calls in a single
node execution. The executor uses ``Block.charge_per_llm_call`` to detect
this and charge ``base_cost * (llm_call_count - 1)`` extra credits after
the block completes.
"""

from unittest.mock import MagicMock

import pytest

from backend.blocks.orchestrator import OrchestratorBlock

# ── Class flag ──────────────────────────────────────────────────────


class TestChargePerLlmCallFlag:
    """OrchestratorBlock opts into per-LLM-call billing."""

    def test_orchestrator_opts_in(self):
        assert OrchestratorBlock.charge_per_llm_call is True

    def test_default_block_does_not_opt_in(self):
        from backend.blocks._base import Block

        assert Block.charge_per_llm_call is False


# ── charge_extra_iterations math ───────────────────────────────────


@pytest.fixture()
def fake_node_exec():
    node_exec = MagicMock()
    node_exec.user_id = "u"
    node_exec.graph_exec_id = "g"
    node_exec.graph_id = "g"
    node_exec.node_exec_id = "ne"
    node_exec.node_id = "n"
    node_exec.block_id = "b"
    node_exec.inputs = {}
    return node_exec


@pytest.fixture()
def patched_processor(monkeypatch):
    """ExecutionProcessor with stubbed db client / block lookup helpers.

    Returns the processor and a list of credit amounts spent so tests can
    assert on what was charged.
    """
    from backend.executor import manager

    spent: list[int] = []

    class FakeDb:
        def spend_credits(self, *, user_id, cost, metadata):
            spent.append(cost)
            return 1000  # remaining balance

    fake_block = MagicMock()
    fake_block.name = "FakeBlock"

    monkeypatch.setattr(manager, "get_db_client", lambda: FakeDb())
    monkeypatch.setattr(manager, "get_block", lambda block_id: fake_block)
    monkeypatch.setattr(
        manager,
        "block_usage_cost",
        lambda block, input_data, **_kw: (10, {"model": "claude-sonnet-4-6"}),
    )

    proc = manager.ExecutionProcessor.__new__(manager.ExecutionProcessor)
    return proc, spent


class TestChargeExtraIterations:
    def test_zero_extra_iterations_charges_nothing(
        self, patched_processor, fake_node_exec
    ):
        proc, spent = patched_processor
        cost, balance = proc.charge_extra_iterations(fake_node_exec, extra_iterations=0)
        assert cost == 0
        assert balance == 0
        assert spent == []

    def test_extra_iterations_multiplies_base_cost(
        self, patched_processor, fake_node_exec
    ):
        proc, spent = patched_processor
        cost, balance = proc.charge_extra_iterations(fake_node_exec, extra_iterations=4)
        assert cost == 40  # 4 × 10
        assert balance == 1000
        assert spent == [40]

    def test_negative_extra_iterations_charges_nothing(
        self, patched_processor, fake_node_exec
    ):
        proc, spent = patched_processor
        cost, balance = proc.charge_extra_iterations(
            fake_node_exec, extra_iterations=-1
        )
        assert cost == 0
        assert balance == 0
        assert spent == []

    def test_capped_at_max(self, monkeypatch, fake_node_exec):
        """Runaway llm_call_count is capped at _MAX_EXTRA_ITERATIONS."""
        from backend.executor import manager

        spent: list[int] = []

        class FakeDb:
            def spend_credits(self, *, user_id, cost, metadata):
                spent.append(cost)
                return 1000

        fake_block = MagicMock()
        fake_block.name = "FakeBlock"

        monkeypatch.setattr(manager, "get_db_client", lambda: FakeDb())
        monkeypatch.setattr(manager, "get_block", lambda block_id: fake_block)
        monkeypatch.setattr(
            manager,
            "block_usage_cost",
            lambda block, input_data, **_kw: (10, {}),
        )

        proc = manager.ExecutionProcessor.__new__(manager.ExecutionProcessor)
        cap = manager.ExecutionProcessor._MAX_EXTRA_ITERATIONS
        cost, _ = proc.charge_extra_iterations(
            fake_node_exec, extra_iterations=cap * 100
        )
        # Charged at most cap × 10
        assert cost == cap * 10
        assert spent == [cap * 10]

    def test_zero_base_cost_skips_charge(self, monkeypatch, fake_node_exec):
        from backend.executor import manager

        spent: list[int] = []

        class FakeDb:
            def spend_credits(self, *, user_id, cost, metadata):
                spent.append(cost)
                return 0

        fake_block = MagicMock()
        fake_block.name = "FakeBlock"

        monkeypatch.setattr(manager, "get_db_client", lambda: FakeDb())
        monkeypatch.setattr(manager, "get_block", lambda block_id: fake_block)
        monkeypatch.setattr(
            manager, "block_usage_cost", lambda block, input_data, **_kw: (0, {})
        )

        proc = manager.ExecutionProcessor.__new__(manager.ExecutionProcessor)
        cost, balance = proc.charge_extra_iterations(fake_node_exec, extra_iterations=4)
        assert cost == 0
        assert balance == 0
        assert spent == []

    def test_block_not_found_skips_charge(self, monkeypatch, fake_node_exec):
        from backend.executor import manager

        spent: list[int] = []

        class FakeDb:
            def spend_credits(self, *, user_id, cost, metadata):
                spent.append(cost)
                return 0

        monkeypatch.setattr(manager, "get_db_client", lambda: FakeDb())
        monkeypatch.setattr(manager, "get_block", lambda block_id: None)
        monkeypatch.setattr(
            manager, "block_usage_cost", lambda block, input_data, **_kw: (10, {})
        )

        proc = manager.ExecutionProcessor.__new__(manager.ExecutionProcessor)
        cost, balance = proc.charge_extra_iterations(fake_node_exec, extra_iterations=3)
        assert cost == 0
        assert balance == 0
        assert spent == []

    def test_propagates_insufficient_balance_error(self, monkeypatch, fake_node_exec):
        """Out-of-credits errors must propagate, not be silently swallowed."""
        from backend.executor import manager
        from backend.util.exceptions import InsufficientBalanceError

        class FakeDb:
            def spend_credits(self, *, user_id, cost, metadata):
                raise InsufficientBalanceError(
                    user_id=user_id,
                    message="Insufficient balance",
                    balance=0,
                    amount=cost,
                )

        fake_block = MagicMock()
        fake_block.name = "FakeBlock"

        monkeypatch.setattr(manager, "get_db_client", lambda: FakeDb())
        monkeypatch.setattr(manager, "get_block", lambda block_id: fake_block)
        monkeypatch.setattr(
            manager, "block_usage_cost", lambda block, input_data, **_kw: (10, {})
        )

        proc = manager.ExecutionProcessor.__new__(manager.ExecutionProcessor)
        with pytest.raises(InsufficientBalanceError):
            proc.charge_extra_iterations(fake_node_exec, extra_iterations=4)


# ── charge_node_usage ──────────────────────────────────────────────


class TestChargeNodeUsage:
    """charge_node_usage delegates to _charge_usage with execution_count=0."""

    def test_delegates_with_zero_execution_count(self, monkeypatch, fake_node_exec):
        """Nested tool charges should NOT inflate the per-execution counter."""
        from backend.executor import manager

        captured: dict = {}

        def fake_charge_usage(self, node_exec, execution_count):
            captured["execution_count"] = execution_count
            captured["node_exec"] = node_exec
            return (5, 100)

        monkeypatch.setattr(
            manager.ExecutionProcessor, "_charge_usage", fake_charge_usage
        )

        proc = manager.ExecutionProcessor.__new__(manager.ExecutionProcessor)
        cost, balance = proc.charge_node_usage(fake_node_exec)
        assert cost == 5
        assert balance == 100
        assert captured["execution_count"] == 0
