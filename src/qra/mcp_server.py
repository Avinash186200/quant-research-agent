"""MCP server exposing the audit tools, so any MCP client (Claude Code, Claude
Desktop, another agent) can audit a strategy.

Run:  python -m qra.mcp_server          (stdio transport)

Guardrail: strategies are Python files and loading one executes it, so the
server only loads files under QRA_STRATEGY_ROOT (default: ./strategies).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

try:  # mcp >= 2: FastMCP was renamed MCPServer
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

from .tools.registry import AuditContext, call_tool

STRATEGY_ROOT = Path(os.environ.get("QRA_STRATEGY_ROOT", Path.cwd() / "strategies"))

server = _Server(
    "quant-research-auditor",
    instructions="Deterministic look-ahead and selection-bias checks for trading strategies. "
    "Pass a strategy file name from the strategy root, e.g. 'sma_crossover'.",
)


@lru_cache(maxsize=32)
def _ctx(strategy: str, seed: int) -> AuditContext:
    return AuditContext.from_path(strategy, root=STRATEGY_ROOT, seed=seed)


def _call(name: str, strategy: str, seed: int, **kw: Any) -> dict[str, Any]:
    if not 0 <= seed <= 1_000_000:
        raise ValueError("seed must be between 0 and 1000000")
    return call_tool(_ctx(strategy, seed), name, kw)


@server.tool()
def get_strategy_source(strategy: str, seed: int = 0) -> dict[str, Any]:
    """Source code, SPEC and optional hooks of a strategy. Call this first."""
    return _call("get_strategy_source", strategy, seed)


@server.tool()
def run_backtest(strategy: str, seed: int = 0) -> dict[str, Any]:
    """Backtest as the author reports it; test-period Sharpe, return, drawdown, turnover."""
    return _call("run_backtest", strategy, seed)


@server.tool()
def check_signal_timing(strategy: str, seed: int = 0) -> dict[str, Any]:
    """Perturb bars from a cut point on; report future leaks and decision-bar reads."""
    return _call("check_signal_timing", strategy, seed)


@server.tool()
def compare_fills(strategy: str, seed: int = 0) -> dict[str, Any]:
    """Same-bar vs next-bar fill Sharpe for the same signal."""
    return _call("compare_fills", strategy, seed)


@server.tool()
def check_fit_window(strategy: str, seed: int = 0) -> dict[str, Any]:
    """Redraw the test period and refit; moving params mean fit() saw test data."""
    return _call("check_fit_window", strategy, seed)


@server.tool()
def check_purge_embargo(strategy: str, seed: int = 0) -> dict[str, Any]:
    """Count training labels overlapping test folds."""
    return _call("check_purge_embargo", strategy, seed)


@server.tool()
def deflated_sharpe(strategy: str, n_trials: int | None = None, seed: int = 0) -> dict[str, Any]:
    """Deflated Sharpe of the reported result for a given trial count."""
    kw = {} if n_trials is None else {"n_trials": n_trials}
    return _call("deflated_sharpe", strategy, seed, **kw)


@server.tool()
def noise_test(strategy: str, n_paths: int = 20, seed: int = 0) -> dict[str, Any]:
    """Run the reporting procedure on zero-drift random walks."""
    return _call("noise_test", strategy, seed, n_paths=n_paths)


def main() -> None:
    server.run()


if __name__ == "__main__":
    main()
