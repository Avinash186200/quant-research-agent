# Methodology

This page describes what each check does, exactly as the code in `src/qra/` does it, and how the eval is scored.
For the system design (why MCP, guardrails, cost control, failure modes) see [architecture.md](architecture.md).

## The strategy contract

A strategy is a Python module in `strategies/` (see `src/qra/tools/loader.py`):

| Name | Required | Meaning |
|---|---|---|
| `SPEC` | yes | `name`, `decides_at` (`"open"` or `"close"`), `fill` (`"same_bar"` or `"next_bar"`); optional `n_trials`, `label_horizon`, `embargo` |
| `signal(prices, **params)` | yes | Position for each bar, from a pandas Series of closes |
| `PARAM_GRID` | no | List of parameter dicts the author searched |
| `fit(prices, split)` | no | Fitted parameters; must use `prices.iloc[:split]` only |
| `cv_splits(n)` | no | List of `(train_idx, test_idx)` for a cross-validated model |

`signal[t]` is the position decided at bar t. With `fill="next_bar"` it is held over bar t+1. With
`fill="same_bar"` it earns bar t's own return, which is only honest if the signal was decided at the open from data
up to t-1.

## Data

All checks run on a seeded geometric random walk (`data.synthetic_prices`: 1,500 business days, zero drift, 1%
daily vol). On a random walk no strategy has a real edge, so a large edge found there points to a bug. The first
60% of bars is the training period and the last 40% is the test period (`backtest.SPLIT_FRAC`).

## The checks

| Tool | What it does | Flags |
|---|---|---|
| `run_backtest` | Runs the strategy the way its author would report it. With a `PARAM_GRID`, every grid point is run and the best test Sharpe is kept | nothing on its own |
| `compare_fills` | Test-period Sharpe with same-bar and with next-bar fills, same parameters | evidence only; a big positive gap means the edge lives in the decision bar |
| `check_signal_timing` | At 12 random bars k, redraws every return from k on (with an 8-sigma shock at k) and compares signals | `future_leak` if any signal before k changes; `current_bar_window` if signal k changes and the strategy decides at the open; `same_bar_fill` if signal k changes, it decides at the close and fills on the same bar |
| `check_fit_window` | Redraws the test period 10 times and refits | `test_fitted_threshold` if any fitted parameter moves |
| `check_purge_embargo` | For each CV fold, counts training rows whose label (bars i+1 .. i+h) reaches into the test fold, or that sit within h bars after it | `label_overlap` if the count is above zero |
| `deflated_sharpe` | Probabilistic Sharpe (PSR) and deflated Sharpe (DSR) of the reported test returns, given the number of trials | `best_of_n` if trials > 1 and DSR < 0.95 |
| `noise_test` | Runs the full reporting procedure on 20 fresh random walks | flags if the mean Sharpe is above 0.3 with a t-stat above 3 |

### PSR and DSR

`tools/stats.py` follows Bailey and López de Prado. PSR is the probability that the true per-period Sharpe is above
a benchmark, given the sample length, skew and kurtosis. DSR is PSR with the benchmark set to the Sharpe you would
expect from the best of N zero-skill trials. When the trial Sharpes are known (a declared `PARAM_GRID`) their
variance is used; otherwise the null variance of about 1/T. The number of trials comes from `SPEC["n_trials"]`, or
the size of `PARAM_GRID`, or 1. The agent can pass a larger `n_trials` when the source suggests an undeclared search.

## The two auditors

- **Baseline** (`baseline.py`): runs `check_signal_timing`, `check_fit_window`, `check_purge_embargo` and
  `deflated_sharpe` (plus `compare_fills` as evidence), and turns every issue into a finding with a fixed fix
  text. Verdict is `biased` if any check fired, otherwise `clean`. No LLM.
- **Agent** (`agent.py`): Claude gets the same tools, bound to one strategy, chooses which to call, reads the
  source, and returns an `AuditReport` that is schema-checked. It runs only with `--agent` and an API key.

## Scoring

`evals/run.py` audits every strategy in `strategies/` and compares the predicted bias types with
`strategies/labels.json` (`scoring.py`). Per bias type, a predicted type that is in the truth set is a true
positive, a predicted type that is not is a false positive, and a true type that was not predicted is a false
negative. Precision is TP / (TP + FP), recall is TP / (TP + FN), and verdict accuracy is the share of strategies
where "biased or clean" is right.

## Limits

- Everything runs on synthetic prices. Biases that only appear on real data (survivorship, corporate actions,
  stale quotes) are out of scope.
- The corpus is small and I wrote both the bugs and the checks, so the baseline result is a sanity check, not a
  benchmark.
- The timing check samples 12 bars. A leak that only fires on rare bars could be missed.
