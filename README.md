# Quant Research Agent

An agent that audits trading strategies for **look-ahead bias** and **selection bias**.

You hand it a strategy file. It runs deterministic checks (perturb future bars, compare same-bar and next-bar fills, perturb only the test period and see if fitted parameters move, deflate the Sharpe for the number of trials, run the strategy on a pure random walk) and returns a structured audit report: a verdict, and each finding with its bias type, the evidence behind it, and a fix.

I built this because the bugs that make a backtest look great are almost always the same handful, and they are cheap to test for mechanically. What is not cheap is reading the code and deciding which test matters. So the tools do the maths, and Claude does the reading and the judgement.

[![CI](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)

## At a glance

| | |
|---|---|
| **Input** | A Python strategy module exposing `signals(prices, **params)` (plus optional parameter grid, fit and label hooks) |
| **Method** | Deterministic, tested bias checks on seeded synthetic prices, exposed over MCP and to a Claude agent that chooses which to run and interprets the evidence |
| **Output** | JSON `AuditReport`: verdict, findings (bias type, evidence, fix), plus the tool calls and token usage behind it |
| **Guardrails** | Tools bound to one strategy, schema-enforced output, tool-call budget, refusal and max_tokens handled as inconclusive, no API calls in CI |
| **Status** | Working. Deterministic baseline measured below; agent numbers are yours to produce with your own key |

## How it works

```
strategy.py ──► loader ──► tools (backtest, fills, timing, fit window, purge, DSR, noise)
                               │                          │
                         baseline.py              agent.py / mcp_server.py
                               └──────────► AuditReport ◄─┘ ──► evals/run.py
```

| Bias type | How the tools catch it |
|---|---|
| `same_bar_fill` | Return collapses when the fill moves from bar t to bar t+1 |
| `current_bar_window` | Signals meant to be decided at the open change when the current bar is perturbed |
| `future_leak` | Past signals change when only future bars are perturbed |
| `test_fitted_threshold` | Fitted parameters move when only the test period is perturbed |
| `best_of_n` | Deflated Sharpe given the number of trials |
| `label_overlap` | Label horizon longer than the purge / embargo gap |

More detail, including why MCP, guardrails, cost control and failure modes, is in [docs/architecture.md](docs/architecture.md).

## Results: deterministic baseline

`python evals/run.py` on the 13-strategy corpus in `strategies/` (5 clean, 8 with planted bugs, ground truth in `strategies/labels.json`), seed 0:

**Precision 1.00 | Recall 0.78 | Verdict accuracy 0.92**

| Bias type | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| same_bar_fill | 2 | 0 | 0 | 1.00 | 1.00 |
| current_bar_window | 1 | 0 | 0 | 1.00 | 1.00 |
| future_leak | 1 | 0 | 0 | 1.00 | 1.00 |
| test_fitted_threshold | 1 | 0 | 0 | 1.00 | 1.00 |
| best_of_n | 1 | 0 | 2 | 1.00 | 0.33 |
| label_overlap | 1 | 0 | 0 | 1.00 | 1.00 |

The baseline misses `best_of_n` twice. `tuned_mean_reversion` hard-codes parameters that were picked from a search, and in `sweep_same_bar` the sweep sits behind a bigger same-bar bug. A fixed rule cannot see either. That gap is what the agent is for: it reads the source and can call `deflated_sharpe` with the trial count it infers.

### Agent numbers

I do not publish agent numbers here, because I only report what I have measured. To produce them:

```bash
export ANTHROPIC_API_KEY=<your key>
python evals/run.py --agent --effort high
```

This writes `evals/results/agent.json` (every report, tool call and token count) and adds an agent table to `evals/results/results.md`. Without a key the flag is skipped.

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q      # fast and offline; the agent loop runs against a mocked client
python evals/run.py      # deterministic baseline
```

Serve the tools to any MCP client over stdio:

```bash
pip install -e .
qra-mcp
```

## Repo layout

```
src/qra/tools/         deterministic checks, strategy loader, PSR / DSR, tool registry
src/qra/agent.py       Claude agent: manual tool loop, adaptive thinking, structured output
src/qra/mcp_server.py  FastMCP server exposing the same tools
src/qra/baseline.py    rule-based auditor (the floor the agent must beat)
strategies/            13 example strategies + labels.json
evals/run.py           precision / recall per bias type
tests/                 tools, eval, MCP server and agent loop (mocked client)
```

## Limits

- Everything runs on synthetic prices. Biases that only show up on real data (survivorship, corporate actions, stale quotes) are out of scope.
- The corpus is small and I wrote both the bugs and the checks, so the baseline is a sanity check, not a benchmark.

## License

MIT
