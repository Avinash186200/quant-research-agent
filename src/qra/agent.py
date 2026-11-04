"""Claude agent that audits one strategy with the deterministic tools.

The model decides which tools to run and how to read the evidence; the tools
do every calculation. The final answer is an ``AuditReport`` enforced by
structured outputs (``output_config.format``).

I use a manual tool loop rather than the SDK's beta tool runner because the
loop needs a hard tool-call budget, binds every tool to one fixed strategy,
and must never execute tools from a refused or truncated turn. It also keeps
the agent free of beta APIs and easy to test with a fake client.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import anthropic
from pydantic import ValidationError

from .report import BIAS_TYPES, REPORT_JSON_SCHEMA, AuditReport
from .tools.registry import TOOLS, AuditContext, call_tool

MODEL = "claude-opus-5-5"
MAX_TOKENS = 16000
MAX_TOOL_RESULT_CHARS = 8000

SYSTEM_PROMPT = f"""You audit trading strategies for look-ahead and selection bias.

You have deterministic tools bound to ONE strategy and ONE synthetic dataset. You cannot
change either. Do every calculation with the tools; never estimate a number yourself.

Procedure:
1. Call get_strategy_source first. Read the SPEC, the code and the comments.
2. Run check_signal_timing on every strategy. Run check_fit_window if it defines fit(),
   check_purge_embargo if it defines cv_splits() or label_horizon > 1, compare_fills if it
   fills on the same bar, and deflated_sharpe on the reported result.
3. If the source or comments say more variants were tried than the strategy declares,
   call deflated_sharpe again with your best estimate of n_trials.
4. Use noise_test to corroborate a suspected leak.

Bias types (use only these labels): {", ".join(BIAS_TYPES)}.
- same_bar_fill: decided on bar t's close but credited with bar t's return.
- current_bar_window: decided at bar t's open but its inputs include bar t.
- future_leak: signal[t] depends on bars after t.
- test_fitted_threshold: fit() uses data after the train/test split.
- best_of_n: the reported result was selected from many variants and does not survive deflation.
- label_overlap: training labels overlap test folds (no purge or embargo).

Report only biases the tool evidence supports. Every finding needs the tool name and the
numbers in `evidence`, and a concrete `fix`. Verdict: "biased" if any finding, "clean" if the
checks you ran are clean, "inconclusive" if a check you needed failed.
Your final message must be the JSON report only."""


class AgentError(RuntimeError):
    """The API call failed in a way the caller should see (auth, rate limit, bad request)."""


@dataclass
class AgentResult:
    report: AuditReport
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})
    stop_reason: str = ""
    turns: int = 0


def tool_definitions() -> list[dict[str, Any]]:
    return [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema, "strict": True}
        for t in TOOLS
    ]


def _inconclusive(name: str, why: str) -> AuditReport:
    return AuditReport(strategy=name, verdict="inconclusive", findings=[], summary=why)


def _create(client: Any, messages: list, effort: str) -> Any:
    try:
        return client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            thinking={"type": "adaptive"},
            output_config={
                "effort": effort,
                "format": {"type": "json_schema", "schema": REPORT_JSON_SCHEMA},
            },
            tools=tool_definitions(),
            tool_choice={"type": "auto"},
            messages=messages,
        )
    except anthropic.AuthenticationError as e:
        raise AgentError("authentication failed: check ANTHROPIC_API_KEY") from e
    except anthropic.PermissionDeniedError as e:
        raise AgentError("API key lacks permission for this model") from e
    except anthropic.RateLimitError as e:
        raise AgentError("rate limited after SDK retries; try again later") from e
    except anthropic.BadRequestError as e:
        raise AgentError(f"bad request: {e.message}") from e
    except anthropic.APIStatusError as e:
        raise AgentError(f"API error {e.status_code}: {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise AgentError("network error reaching the API") from e


def _add_usage(total: dict[str, int], resp: Any) -> None:
    u = getattr(resp, "usage", None)
    if u is None:
        return
    for k in ("input_tokens", "output_tokens"):
        total[k] = total.get(k, 0) + int(getattr(u, k, 0) or 0)


def _run_tool(ctx: AuditContext, block: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    record = {"name": block.name, "input": dict(block.input or {})}
    try:
        out = call_tool(ctx, block.name, block.input)
        text = json.dumps(out, default=str)
        if len(text) > MAX_TOOL_RESULT_CHARS:
            text = text[:MAX_TOOL_RESULT_CHARS] + "...[truncated]"
        record["ok"] = True
        return {"type": "tool_result", "tool_use_id": block.id, "content": text}, record
    except Exception as e:  # tool failures go back to the model, not up the stack
        record["ok"] = False
        record["error"] = f"{type(e).__name__}: {e}"
        return (
            {"type": "tool_result", "tool_use_id": block.id, "content": record["error"], "is_error": True},
            record,
        )


def run_agent(
    ctx: AuditContext,
    client: Any | None = None,
    effort: str = "high",
    max_turns: int = 12,
    max_tool_calls: int = 16,
) -> AgentResult:
    if client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise AgentError("ANTHROPIC_API_KEY is not set")
        client = anthropic.Anthropic()
    name = ctx.strategy.name
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": f"Audit the strategy '{name}'. Use the tools, then return the JSON report."}
    ]
    result = AgentResult(report=_inconclusive(name, "agent did not finish"))
    nudged = False

    for turn in range(1, max_turns + 1):
        result.turns = turn
        resp = _create(client, messages, effort)
        _add_usage(result.usage, resp)
        result.stop_reason = resp.stop_reason

        if resp.stop_reason == "refusal":
            details = getattr(resp, "stop_details", None)
            why = getattr(details, "explanation", None) or "no details"
            result.report = _inconclusive(name, f"model refused: {why}")
            return result
        if resp.stop_reason == "max_tokens":
            # A truncated turn may hold a half-written tool call or report: never act on it.
            result.report = _inconclusive(name, "response hit max_tokens before finishing")
            return result

        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason == "pause_turn":
            continue

        tool_uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
        if resp.stop_reason == "tool_use" and tool_uses:
            results = []
            for block in tool_uses:
                if len(result.tool_calls) >= max_tool_calls:
                    results.append({
                        "type": "tool_result", "tool_use_id": block.id, "is_error": True,
                        "content": "Tool budget exhausted. Write the JSON report now from the evidence you have.",
                    })
                    continue
                tr, rec = _run_tool(ctx, block)
                results.append(tr)
                result.tool_calls.append(rec)
            messages.append({"role": "user", "content": results})
            continue

        # end_turn (or stop_sequence): expect the JSON report.
        if not result.tool_calls and not nudged:
            # tool_choice must stay "auto" on this model, so a call is not guaranteed; ask once.
            nudged = True
            messages.append({"role": "user", "content": "Run the audit tools before reporting. Start with get_strategy_source."})
            continue
        text = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), "")
        try:
            report = AuditReport.model_validate_json(text)
        except ValidationError as e:
            result.report = _inconclusive(name, f"report failed schema validation: {e.error_count()} error(s)")
            return result
        report.strategy = name
        if not result.tool_calls:
            report = _inconclusive(name, "model reported without running any tool")
        result.report = report
        return result

    result.report = _inconclusive(name, f"stopped after {max_turns} turns without a report")
    return result
