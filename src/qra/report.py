"""The audit report schema, used by both the deterministic baseline and the agent."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

BIAS_TYPES = (
    "same_bar_fill",
    "current_bar_window",
    "future_leak",
    "test_fitted_threshold",
    "best_of_n",
    "label_overlap",
)

BiasType = Literal[
    "same_bar_fill",
    "current_bar_window",
    "future_leak",
    "test_fitted_threshold",
    "best_of_n",
    "label_overlap",
]


class Finding(BaseModel):
    bias_type: BiasType
    severity: Literal["high", "medium", "low"]
    evidence: str = Field(description="Tool name and the numbers that show the problem.")
    fix: str = Field(description="Concrete code or process change.")


class AuditReport(BaseModel):
    strategy: str
    verdict: Literal["clean", "biased", "inconclusive"]
    findings: list[Finding]
    summary: str

    def bias_types(self) -> set[str]:
        return {f.bias_type for f in self.findings}


# Hand-written JSON schema for structured outputs: every object closed,
# every property required, so the API can enforce it exactly.
REPORT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "strategy": {"type": "string"},
        "verdict": {"type": "string", "enum": ["clean", "biased", "inconclusive"]},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "bias_type": {"type": "string", "enum": list(BIAS_TYPES)},
                    "severity": {"type": "string", "enum": ["high", "medium", "low"]},
                    "evidence": {"type": "string"},
                    "fix": {"type": "string"},
                },
                "required": ["bias_type", "severity", "evidence", "fix"],
                "additionalProperties": False,
            },
        },
        "summary": {"type": "string"},
    },
    "required": ["strategy", "verdict", "findings", "summary"],
    "additionalProperties": False,
}
