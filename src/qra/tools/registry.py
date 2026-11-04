"""One registry of audit tools, shared by the MCP server, the agent and the baseline.

Every tool takes an ``AuditContext`` (strategy + data, fixed by the caller)
plus a few small, validated arguments, and returns a JSON-safe dict. The
model never chooses which file or dataset a tool runs on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from ..data import synthetic_prices
from . import backtest, checks
from .loader import Strategy, load_strategy
from .stats import deflated_sharpe

MAX_SOURCE_CHARS = 12_000


@dataclass
class AuditContext:
    strategy: Strategy
    prices: pd.Series

    @classmethod
    def from_path(cls, path, root=None, seed: int = 0, n_bars: int = 1500) -> "AuditContext":
        return cls(load_strategy(path, root), synthetic_prices(n=n_bars, seed=seed))


def t_get_strategy_source(ctx: AuditContext) -> dict[str, Any]:
    src = ctx.strategy.source()
    return {
        "name": ctx.strategy.name,
        "spec": ctx.strategy.spec,
        "has_param_grid": bool(ctx.strategy.param_grid),
        "param_grid_size": len(ctx.strategy.param_grid),
        "has_fit": ctx.strategy.has_fit,
        "has_cv_splits": ctx.strategy.has_cv,
        "source": src[:MAX_SOURCE_CHARS],
        "truncated": len(src) > MAX_SOURCE_CHARS,
    }


def t_run_backtest(ctx: AuditContext) -> dict[str, Any]:
    return backtest.run_backtest(ctx.strategy, ctx.prices)


def t_compare_fills(ctx: AuditContext) -> dict[str, Any]:
    return backtest.compare_fills(ctx.strategy, ctx.prices)


def t_check_signal_timing(ctx: AuditContext) -> dict[str, Any]:
    return checks.check_signal_timing(ctx.strategy, ctx.prices)


def t_check_fit_window(ctx: AuditContext) -> dict[str, Any]:
    return checks.check_fit_window(ctx.strategy, ctx.prices)


def t_check_purge_embargo(ctx: AuditContext) -> dict[str, Any]:
    return checks.check_purge_embargo(ctx.strategy, n=len(ctx.prices))


def t_deflated_sharpe(ctx: AuditContext, n_trials: int | None = None) -> dict[str, Any]:
    ev = backtest.evaluate(ctx.strategy, ctx.prices)
    declared = ctx.strategy.declared_trials
    n = declared if n_trials is None else int(n_trials)
    if n < 1 or n > 100_000:
        raise ValueError("n_trials must be between 1 and 100000")
    out = deflated_sharpe(ev["test_returns"], n, ev["trial_sharpes"] if n == declared else None)
    out["declared_trials"] = declared
    out["issues"] = ["best_of_n"] if (n > 1 and out["dsr"] < 0.95) else []
    return out


def t_noise_test(ctx: AuditContext, n_paths: int = 20) -> dict[str, Any]:
    return backtest.noise_test(ctx.strategy, n_paths=n_paths, n_bars=len(ctx.prices))


NO_ARGS = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    fn: Callable[..., dict[str, Any]]


TOOLS: list[ToolSpec] = [
    ToolSpec(
        "get_strategy_source",
        "Return the strategy's source code, SPEC and which optional hooks it defines. "
        "Call this first, before any other tool, so you know what to test.",
        NO_ARGS,
        t_get_strategy_source,
    ),
    ToolSpec(
        "run_backtest",
        "Backtest the strategy as its author reports it (best grid point if it has a PARAM_GRID) "
        "and return test-period Sharpe, return, drawdown and turnover. Call this to see the claimed result.",
        NO_ARGS,
        t_run_backtest,
    ),
    ToolSpec(
        "check_signal_timing",
        "Redraw every bar from a cut point on and report whether earlier signals change (future leak) "
        "or the signal at the cut bar changes (reads its own decision bar). Call this for every strategy.",
        NO_ARGS,
        t_check_signal_timing,
    ),
    ToolSpec(
        "compare_fills",
        "Test-period Sharpe with same-bar vs next-bar fills for the same signal. "
        "Call this when the strategy fills on the same bar or when timing looks suspicious.",
        NO_ARGS,
        t_compare_fills,
    ),
    ToolSpec(
        "check_fit_window",
        "Redraw the test period and refit. Reports which fitted parameters move, which means fit() "
        "saw test data. Call this whenever the strategy defines fit().",
        NO_ARGS,
        t_check_fit_window,
    ),
    ToolSpec(
        "check_purge_embargo",
        "Count training labels whose forward horizon overlaps a test fold. "
        "Call this whenever the strategy defines cv_splits() or a label_horizon above 1.",
        NO_ARGS,
        t_check_purge_embargo,
    ),
    ToolSpec(
        "deflated_sharpe",
        "Deflated Sharpe ratio of the reported result given the number of trials. Defaults to the trials "
        "the strategy declares; pass n_trials if the source or comments suggest more variants were tried.",
        {
            "type": "object",
            "properties": {
                "n_trials": {
                    "type": "integer",
                    "description": "Number of variants tried before this one was picked (1 to 100000).",
                }
            },
            "required": [],
            "additionalProperties": False,
        },
        t_deflated_sharpe,
    ),
    ToolSpec(
        "noise_test",
        "Run the full reporting procedure on zero-drift random walks. A real, honestly reported strategy "
        "shows no edge on noise. Call this to corroborate any suspected leak.",
        {
            "type": "object",
            "properties": {
                "n_paths": {"type": "integer", "description": "Random walks to run, 5 to 100. Default 20."}
            },
            "required": [],
            "additionalProperties": False,
        },
        t_noise_test,
    ),
]

TOOLS_BY_NAME = {t.name: t for t in TOOLS}


def call_tool(ctx: AuditContext, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    spec = TOOLS_BY_NAME.get(name)
    if spec is None:
        raise KeyError(f"unknown tool: {name}")
    args = dict(args or {})
    allowed = set(spec.input_schema["properties"])
    extra = set(args) - allowed
    if extra:
        raise ValueError(f"unexpected arguments for {name}: {sorted(extra)}")
    for k, v in args.items():
        if not isinstance(v, int) or isinstance(v, bool):
            raise ValueError(f"{k} must be an integer")
    return spec.fn(ctx, **args)
