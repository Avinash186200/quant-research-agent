"""Agent loop tests with a fake client. No network, no API key."""
import json
from types import SimpleNamespace as NS

import pytest

from conftest import STRATS
from qra import agent as agent_mod
from qra.agent import MODEL, AgentError, run_agent
from qra.tools.registry import AuditContext

REPORT = {
    "strategy": "x",
    "verdict": "biased",
    "summary": "same-bar fill",
    "findings": [{"bias_type": "same_bar_fill", "severity": "high",
                  "evidence": "check_signal_timing reads_decision_bar", "fix": "shift(1)"}],
}


def msg(stop, content):
    return NS(stop_reason=stop, content=content, usage=NS(input_tokens=100, output_tokens=20))


def tool_use(i, name, inp=None):
    return NS(type="tool_use", id=f"tu_{i}", name=name, input=inp or {})


def text(t):
    return NS(type="text", text=t)


class FakeClient:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.messages = self

    def create(self, **kw):
        self.calls.append({**kw, "messages": list(kw["messages"])})  # snapshot: the loop keeps appending
        return self.script.pop(0)


@pytest.fixture
def ctx():
    return AuditContext.from_path(STRATS / "momentum_same_bar.py")


def test_loop_runs_tools_and_parses_report(ctx):
    c = FakeClient([
        msg("tool_use", [tool_use(1, "get_strategy_source"), tool_use(2, "check_signal_timing")]),
        msg("end_turn", [text(json.dumps(REPORT))]),
    ])
    res = run_agent(ctx, client=c)
    assert res.report.verdict == "biased"
    assert res.report.strategy == "momentum_same_bar"
    assert [t["name"] for t in res.tool_calls] == ["get_strategy_source", "check_signal_timing"]
    assert res.usage == {"input_tokens": 200, "output_tokens": 40}
    kw = c.calls[0]
    assert kw["model"] == MODEL == "claude-opus-5-5"
    assert kw["thinking"] == {"type": "adaptive"}
    assert kw["tool_choice"] == {"type": "auto"}
    assert kw["output_config"]["effort"] == "high"
    assert kw["output_config"]["format"]["type"] == "json_schema"
    results = c.calls[1]["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["tu_1", "tu_2"]
    assert "same_bar_fill" in json.loads(results[1]["content"])["issues"]


def test_tool_error_is_returned_to_model(ctx):
    c = FakeClient([
        msg("tool_use", [tool_use(1, "deflated_sharpe", {"n_trials": "many"})]),
        msg("end_turn", [text(json.dumps(REPORT))]),
    ])
    res = run_agent(ctx, client=c)
    tr = c.calls[1]["messages"][-1]["content"][0]
    assert tr["is_error"] is True and res.tool_calls[0]["ok"] is False


def test_refusal_is_inconclusive_and_runs_no_tools(ctx):
    c = FakeClient([msg("refusal", [tool_use(1, "run_backtest")])])
    res = run_agent(ctx, client=c)
    assert res.report.verdict == "inconclusive" and res.tool_calls == []


def test_max_tokens_never_executes_truncated_tools(ctx):
    c = FakeClient([msg("max_tokens", [tool_use(1, "noise_test")])])
    res = run_agent(ctx, client=c)
    assert res.report.verdict == "inconclusive" and res.tool_calls == []


def test_nudges_once_when_no_tool_called(ctx):
    c = FakeClient([
        msg("end_turn", [text(json.dumps(REPORT))]),
        msg("end_turn", [text(json.dumps(REPORT))]),
    ])
    res = run_agent(ctx, client=c)
    assert len(c.calls) == 2
    assert res.report.verdict == "inconclusive"  # a report with no tool evidence is not trusted


def test_invalid_json_is_inconclusive(ctx):
    c = FakeClient([
        msg("tool_use", [tool_use(1, "run_backtest")]),
        msg("end_turn", [text(json.dumps({"verdict": "great"}))]),
    ])
    assert run_agent(ctx, client=c).report.verdict == "inconclusive"


def test_tool_budget_enforced(ctx):
    many = [tool_use(i, "get_strategy_source") for i in range(5)]
    c = FakeClient([msg("tool_use", many), msg("end_turn", [text(json.dumps(REPORT))])])
    res = run_agent(ctx, client=c, max_tool_calls=3)
    assert len(res.tool_calls) == 3
    assert sum(1 for r in c.calls[1]["messages"][-1]["content"] if r.get("is_error")) == 2


def test_turn_limit(ctx):
    loop = [msg("tool_use", [tool_use(i, "get_strategy_source")]) for i in range(3)]
    res = run_agent(ctx, client=FakeClient(loop), max_turns=3)
    assert res.report.verdict == "inconclusive" and res.turns == 3


def test_api_errors_are_typed(ctx):
    import anthropic

    class Boom(FakeClient):
        def create(self, **kw):
            raise anthropic.APIConnectionError(request=None)

    with pytest.raises(AgentError, match="network"):
        run_agent(ctx, client=Boom([]))


def test_missing_key_raises(ctx, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(AgentError):
        run_agent(ctx)


def test_tool_definitions_are_strict_and_closed():
    for t in agent_mod.tool_definitions():
        assert t["strict"] is True and t["input_schema"]["additionalProperties"] is False
