# Walkthrough

This follows the deterministic auditor on three strategies from the corpus, tool by tool. Every number here comes
from running the code in this repo (seed 0). After `pip install -e .`, you can repeat it in a Python shell started
at the repo root:

```python
from qra.tools.registry import AuditContext, call_tool
from qra.baseline import audit_deterministic
ctx = AuditContext.from_path("strategies/momentum_same_bar.py", seed=0)
```

## 1. A clean strategy: `sma_crossover`

Long when the 20-bar mean is above the 100-bar mean, decided at the close, filled on the next bar.

- `compare_fills`: Sharpe 1.13 with a same-bar fill and 1.05 with a next-bar fill. The gap is small.
- `check_signal_timing`: 12 bars tested. No earlier signal changes, and the decision-bar signal does not change
  either.
- `deflated_sharpe`: one trial, so DSR equals PSR (0.95). Below the 0.95 threshold, but with one trial there is no
  selection to deflate, so nothing is flagged.

The verdict is `clean`. Note the 1.05 Sharpe on a random walk: one 600-bar test period is short, and a decent
Sharpe can come from luck alone. That is why the noise test runs many paths.

## 2. A same-bar fill: `momentum_same_bar`

5-bar momentum, decided at the close and credited with that same bar's return.

- `compare_fills`: Sharpe 5.83 with the declared same-bar fill and 0.18 with a next-bar fill. Almost all of the
  edge is the bar the signal was computed on.
- `check_signal_timing`: the decision-bar signal changes at all 12 tested bars, and the strategy decides at the
  close and fills on the same bar, so the tool returns `same_bar_fill`.

The report is `biased`, with one finding and the fix "Fill on the next bar: hold signal[t] over bar t+1".

## 3. A parameter sweep: `lookback_sweep`

The module declares a `PARAM_GRID` of 40 settings, and the reported result is the best one.

- `deflated_sharpe`: observed Sharpe 1.41. The best of 40 zero-skill trials would be expected to reach about 1.01.
  PSR is 0.99 but DSR is 0.73, below 0.95, so the tool returns `best_of_n`.

This is the case the baseline can only catch when the search is declared. In `tuned_mean_reversion` the
parameters were picked by a search that the code does not show, so the baseline misses it. The agent can read the
source and call `deflated_sharpe` with the trial count it infers.

## 4. The eval

`python evals/run.py` does the same for all 13 strategies, scores the findings against `strategies/labels.json`,
and writes `evals/results/baseline.json` and `evals/results/results.md`. See the README for the table.

## Things to try

- Write a new strategy with a `rolling(..., center=True)` and check that `check_signal_timing` reports
  `future_leak`.
- Call `call_tool(ctx, "deflated_sharpe", {"n_trials": 200})` on `sma_crossover` and see how DSR falls.
- Run `call_tool(ctx, "noise_test")` on a few strategies and compare the mean Sharpe on noise.
