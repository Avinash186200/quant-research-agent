import numpy as np
import pytest

from conftest import STRATS
from qra.data import replace_returns_from, synthetic_prices
from qra.tools import StrategyError, load_strategy
from qra.tools.registry import AuditContext, call_tool
from qra.tools.stats import deflated_sharpe, expected_max_sharpe, sharpe


def ctx(name):
    return AuditContext.from_path(STRATS / f"{name}.py")


def test_synthetic_is_reproducible():
    a, b = synthetic_prices(seed=3), synthetic_prices(seed=3)
    assert a.equals(b) and len(a) == 1500


def test_replace_returns_keeps_past_identical():
    p = synthetic_prices()
    q = replace_returns_from(p, 700, np.random.default_rng(1), shock_sign=1)
    assert np.array_equal(p.values[:700], q.values[:700])
    assert not np.allclose(p.values[700:], q.values[700:])


@pytest.mark.parametrize("name", ["sma_crossover", "breakout_open", "vol_target_trend"])
def test_clean_timing(name):
    assert call_tool(ctx(name), "check_signal_timing")["issues"] == []


@pytest.mark.parametrize("name,issue", [
    ("momentum_same_bar", "same_bar_fill"),
    ("zscore_open", "current_bar_window"),
    ("centered_smoother", "future_leak"),
])
def test_timing_catches_planted_leaks(name, issue):
    assert issue in call_tool(ctx(name), "check_signal_timing")["issues"]


def test_fit_window():
    assert call_tool(ctx("threshold_fit_train"), "check_fit_window")["issues"] == []
    assert call_tool(ctx("threshold_fit_full"), "check_fit_window")["issues"] == ["test_fitted_threshold"]
    assert call_tool(ctx("sma_crossover"), "check_fit_window")["applicable"] is False


def test_purge_embargo():
    assert call_tool(ctx("purged_cv_classifier"), "check_purge_embargo")["overlapping_train_labels_total"] == 0
    assert call_tool(ctx("overlapping_labels"), "check_purge_embargo")["overlapping_train_labels_total"] > 0


def test_expected_max_sharpe_grows_with_trials():
    assert expected_max_sharpe(1, 0.01) == 0.0
    assert expected_max_sharpe(100, 0.01) > expected_max_sharpe(10, 0.01) > 0


def test_deflation_lowers_dsr():
    r = np.random.default_rng(0).normal(0.0005, 0.01, 600)
    one, many = deflated_sharpe(r, 1), deflated_sharpe(r, 200)
    assert one["dsr"] == one["psr"] and many["dsr"] < one["dsr"]


def test_deflated_sharpe_tool_flags_sweep_and_accepts_override():
    assert call_tool(ctx("lookback_sweep"), "deflated_sharpe")["issues"] == ["best_of_n"]
    c = ctx("tuned_mean_reversion")
    assert call_tool(c, "deflated_sharpe")["issues"] == []
    assert call_tool(c, "deflated_sharpe", {"n_trials": 116})["issues"] == ["best_of_n"]


def test_noise_test_flags_leak_not_clean():
    assert call_tool(ctx("momentum_same_bar"), "noise_test", {"n_paths": 8})["flag"] is True
    assert call_tool(ctx("sma_crossover"), "noise_test", {"n_paths": 8})["flag"] is False


def test_compare_fills_gap_on_same_bar():
    assert call_tool(ctx("momentum_same_bar"), "compare_fills")["gap"] > 2


def test_sharpe_zero_variance():
    assert sharpe([0.0, 0.0, 0.0]) == 0.0


def test_call_tool_validates_args():
    c = ctx("sma_crossover")
    with pytest.raises(ValueError):
        call_tool(c, "deflated_sharpe", {"n_trials": "lots"})
    with pytest.raises(ValueError):
        call_tool(c, "run_backtest", {"path": "elsewhere.py"})
    with pytest.raises(KeyError):
        call_tool(c, "delete_everything")


def test_root_guard_blocks_traversal():
    with pytest.raises(StrategyError):
        load_strategy("../src/qra/agent", root=STRATS)
    assert load_strategy("sma_crossover", root=STRATS).name == "sma_crossover"
