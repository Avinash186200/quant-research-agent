"""Deterministic auditor: run every tool, map tool issues to findings. No LLM."""

from __future__ import annotations

from .report import AuditReport, Finding
from .tools.registry import AuditContext, call_tool

FIXES = {
    "same_bar_fill": "Fill on the next bar: hold signal[t] over bar t+1 (pos.shift(1)).",
    "current_bar_window": "Shift inputs by one bar (prices.shift(1)) so an open-decided signal sees data up to t-1 only.",
    "future_leak": "Remove forward-looking operations (center=True, shift(-k), full-sample statistics).",
    "test_fitted_threshold": "Fit parameters on prices.iloc[:split] only and freeze them for the test period.",
    "best_of_n": "Report the deflated Sharpe with the true trial count, or confirm on an untouched hold-out.",
    "label_overlap": "Purge training samples whose label horizon reaches a test fold and add an embargo after it.",
}

CHECKS = ("check_signal_timing", "check_fit_window", "check_purge_embargo", "deflated_sharpe")


def audit_deterministic(ctx: AuditContext) -> tuple[AuditReport, dict]:
    evidence = {name: call_tool(ctx, name) for name in CHECKS}
    evidence["compare_fills"] = call_tool(ctx, "compare_fills")
    findings = []
    for tool_name in CHECKS:
        for issue in evidence[tool_name].get("issues", []):
            findings.append(
                Finding(
                    bias_type=issue,
                    severity="high",
                    evidence=f"{tool_name}: {_short(evidence[tool_name])}",
                    fix=FIXES[issue],
                )
            )
    verdict = "biased" if findings else "clean"
    summary = (
        f"{len(findings)} issue(s) found by deterministic checks."
        if findings
        else "No deterministic check fired. Undisclosed research history cannot be detected this way."
    )
    return AuditReport(strategy=ctx.strategy.name, verdict=verdict, findings=findings, summary=summary), evidence


def _short(d: dict) -> str:
    keep = {k: v for k, v in d.items() if k not in ("folds", "source")}
    s = str(keep)
    return s if len(s) < 400 else s[:400] + "..."
