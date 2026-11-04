# Architecture

```
                 +-------------------------------+
 strategy.py --> | loader (import from corpus,   |
                 | signals(prices, **params))    |
                 +---------------+---------------+
                                 |
        +------------------------+-------------------------+
        |        deterministic tools (src/qra/tools)       |
        |  backtest | compare_fills | signal_timing |      |
        |  fit_window | purge_embargo | deflated_sharpe |  |
        |  noise_test | synthetic prices (data.py)         |
        +-----------+--------------------+-----------------+
                    |                    |
          +---------v-------+   +--------v-----------+
          | baseline.py     |   | agent.py (Claude)  |   <-- same tools also served
          | fixed rules     |   | chooses tools,     |       over MCP (mcp_server.py)
          | -> AuditReport  |   | reads evidence     |       to any MCP client
          +---------+-------+   | -> AuditReport     |
                    |           +--------+-----------+
                    +---------+----------+
                              v
                  evals/run.py: precision / recall per bias type
                  against strategies/labels.json
```

## Why deterministic tools plus LLM reasoning

Every number comes from a tested Python function on seeded synthetic data. The
model never computes a Sharpe ratio itself. What the model adds is the part
rules are bad at: reading the strategy source, noticing that a "parameter" was
picked from a sweep the code does not declare, deciding which check to run
next, and writing a fix a person can apply. If the model is wrong, the evidence
it cites is still reproducible, so a reviewer can check it.

I keep the deterministic baseline as a first-class auditor, not just a test
fixture. It is the floor the agent has to beat, and it runs in CI for free.

## Why MCP

The same tools are exposed through a FastMCP server. Any MCP client (Claude
Code, Claude Desktop, another agent) can audit a strategy without importing this
package, and there is one definition of each tool. The in-process agent calls
the same registry directly, which keeps the tests fast and offline.

## Guardrails

- Tools are bound to one strategy per audit; the model cannot point them at
  arbitrary files.
- Strategy names resolve inside the corpus directory only.
- The final output is a JSON schema (`AuditReport`) enforced by structured
  outputs and re-validated with pydantic; invalid output becomes an
  `inconclusive` report.
- `tool_choice` stays `auto`. If the model answers without running a tool it is
  asked once to gather evidence; a report with no tool calls is inconclusive.
- `refusal` and `max_tokens` stop reasons end the audit as inconclusive; tools
  requested in a refused or truncated turn are never executed.
- Tool exceptions go back to the model as `is_error` results instead of crashing.
- Typed API errors (auth, permission, rate limit, connection, server) map to one
  `AgentError` with a clear message.

## Cost control

- Hard tool-call budget per audit (default 16).
- Tool results are truncated before they go back to the model.
- `output_config.effort` is explicit and set from the CLI (`--effort`).
- Token usage is summed per audit and written to `evals/results/agent.json`.
- CI never calls the API; the agent path runs only with `--agent` and a key.

## Failure modes

| Failure | What happens |
|---|---|
| Bias the tools cannot see (an undeclared parameter sweep) | The baseline misses it; the agent has to infer it from the source. This is where the baseline loses recall. |
| Model over-flags a clean strategy | Shows up as a false alarm in the eval; precision drops. |
| Strategy raises or returns bad shapes | The tool error goes back to the model and the report can still be written. |
| Synthetic data does not trigger the bias | Checks use perturbations and several paths, but a bias that only appears on real data can be missed. |
