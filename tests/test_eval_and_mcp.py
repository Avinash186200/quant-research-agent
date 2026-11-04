import asyncio
import json

from conftest import STRATS
from qra.baseline import audit_deterministic
from qra.scoring import score
from qra.tools.registry import TOOLS, AuditContext


def test_scoring_math():
    truth = {"a": {"same_bar_fill"}, "b": set(), "c": {"best_of_n"}}
    pred = {"a": {"same_bar_fill"}, "b": {"future_leak"}, "c": set()}
    s = score(pred, truth)
    assert (s["tp"], s["fp"], s["fn"]) == (1, 1, 1)
    assert s["precision"] == 0.5 and s["recall"] == 0.5


def test_baseline_has_no_false_alarms_on_clean():
    labels = json.loads((STRATS / "labels.json").read_text())["labels"]
    for name, truth in labels.items():
        if not truth:
            rep, _ = audit_deterministic(AuditContext.from_path(STRATS / f"{name}.py"))
            assert rep.verdict == "clean", name


def test_mcp_server_lists_every_tool(monkeypatch):
    from qra import mcp_server

    monkeypatch.setattr(mcp_server, "STRATEGY_ROOT", STRATS)
    mcp_server._ctx.cache_clear()
    names = {t.name for t in asyncio.run(mcp_server.server.list_tools())}
    assert names == {t.name for t in TOOLS}
    assert "same_bar_fill" in mcp_server.check_signal_timing("momentum_same_bar")["issues"]
