"""Load a strategy module and normalise its interface.

Strategy contract (see ``strategies/README.md``):

* ``SPEC``: dict with ``name``, ``decides_at`` ("close" or "open"),
  ``fill`` ("next_bar" or "same_bar"), optional ``label_horizon``,
  ``embargo`` and ``n_trials``.
* ``signal(prices, **params) -> pd.Series`` of positions in [-1, 1].
* optional ``PARAM_GRID``: list of param dicts the author searched.
* optional ``fit(prices, split) -> dict``: fit params; must use only
  ``prices.iloc[:split]``.
* optional ``cv_splits(n) -> list[(train_idx, test_idx)]``.

Loading a module executes it, so only load strategies you trust. The MCP
server restricts loading to one root directory for that reason.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd

DEFAULT_SPEC: dict[str, Any] = {
    "decides_at": "close",
    "fill": "next_bar",
    "label_horizon": 1,
    "embargo": 0,
}


class StrategyError(ValueError):
    """The strategy file is missing or breaks the contract."""


@dataclass
class Strategy:
    name: str
    path: Path
    module: ModuleType
    spec: dict[str, Any] = field(default_factory=dict)

    @property
    def param_grid(self) -> list[dict[str, Any]]:
        return list(getattr(self.module, "PARAM_GRID", []) or [])

    @property
    def has_fit(self) -> bool:
        return callable(getattr(self.module, "fit", None))

    @property
    def has_cv(self) -> bool:
        return callable(getattr(self.module, "cv_splits", None))

    @property
    def declared_trials(self) -> int:
        if "n_trials" in self.spec:
            return int(self.spec["n_trials"])
        return max(1, len(self.param_grid))

    def fit(self, prices: pd.Series, split: int) -> dict[str, Any]:
        return dict(self.module.fit(prices, split))

    def cv_splits(self, n: int):
        return self.module.cv_splits(n)

    def signal(self, prices: pd.Series, params: dict[str, Any] | None = None) -> pd.Series:
        raw = self.module.signal(prices, **(params or {}))
        s = pd.Series(raw, index=prices.index) if not isinstance(raw, pd.Series) else raw
        s = s.reindex(prices.index).astype(float).fillna(0.0).clip(-1.0, 1.0)
        return s

    def source(self) -> str:
        return self.path.read_text(encoding="utf-8")


def resolve_path(strategy: str | Path, root: Path | None = None) -> Path:
    """Resolve a strategy path; if ``root`` is given, refuse anything outside it."""
    p = Path(strategy)
    if root is not None:
        root = root.resolve()
        if not p.is_absolute():
            p = root / p
        if p.suffix != ".py":
            p = p.with_suffix(".py")
        p = p.resolve()
        if root not in p.parents:
            raise StrategyError(f"strategy must live under the strategy root: {strategy!r}")
    if not p.is_file():
        raise StrategyError(f"strategy file not found: {strategy!r}")
    return p


def load_strategy(strategy: str | Path, root: Path | None = None) -> Strategy:
    path = resolve_path(strategy, root)
    spec_obj = importlib.util.spec_from_file_location(f"qra_strategy_{path.stem}", path)
    if spec_obj is None or spec_obj.loader is None:
        raise StrategyError(f"cannot import {path.name}")
    module = importlib.util.module_from_spec(spec_obj)
    spec_obj.loader.exec_module(module)
    if not callable(getattr(module, "signal", None)):
        raise StrategyError(f"{path.name} has no signal(prices, **params) function")
    spec = {**DEFAULT_SPEC, **dict(getattr(module, "SPEC", {}) or {})}
    if spec["decides_at"] not in ("close", "open"):
        raise StrategyError("SPEC['decides_at'] must be 'close' or 'open'")
    if spec["fill"] not in ("next_bar", "same_bar"):
        raise StrategyError("SPEC['fill'] must be 'next_bar' or 'same_bar'")
    name = str(spec.get("name", path.stem))
    return Strategy(name=name, path=path, module=module, spec=spec)
