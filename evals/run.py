"""Eval harness: score the deterministic baseline (and optionally the agent) on the corpus.

    python evals/run.py            # deterministic baseline, no API key needed
    python evals/run.py --agent    # also run the Claude agent (needs ANTHROPIC_API_KEY)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qra.baseline import audit_deterministic  # noqa: E402
from qra.scoring import score  # noqa: E402
from qra.tools.registry import AuditContext  # noqa: E402


def load_truth(strategies: Path) -> dict[str, set[str]]:
    data = json.loads((strategies / "labels.json").read_text(encoding="utf-8"))
    return {k: set(v) for k, v in data["labels"].items()}


def table(res: dict, title: str) -> str:
    lines = [f"### {title}", "", f"Precision {res['precision']:.2f} | Recall {res['recall']:.2f} | "
             f"Verdict accuracy {res['verdict_accuracy']:.2f} | n = {res['n_strategies']}", "",
             "| Strategy | Truth | Predicted | Missed | False alarms |", "|---|---|---|---|---|"]
    for r in res["rows"]:
        lines.append(f"| {r['strategy']} | {', '.join(r['truth']) or 'clean'} | "
                     f"{', '.join(r['predicted']) or 'clean'} | {', '.join(r['missed']) or '-'} | "
                     f"{', '.join(r['false_alarms']) or '-'} |")
    lines += ["", "| Bias type | TP | FP | FN | Precision | Recall |", "|---|---|---|---|---|---|"]
    for t, c in res["per_type"].items():
        prec = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 1.0
        lines.append(f"| {t} | {c['tp']} | {c['fp']} | {c['fn']} | {prec:.2f} | {c['recall']:.2f} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", action="store_true", help="also run the Claude agent")
    ap.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--strategies", default=str(ROOT / "strategies"))
    ap.add_argument("--out", default=str(ROOT / "evals" / "results"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    strat_dir, out = Path(args.strategies), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    truth = load_truth(strat_dir)

    preds = {}
    for name in sorted(truth):
        report, _ = audit_deterministic(AuditContext.from_path(strat_dir / f"{name}.py", seed=args.seed))
        preds[name] = report.bias_types()
    base = score(preds, truth)
    (out / "baseline.json").write_text(json.dumps(base, indent=2), encoding="utf-8")
    md = table(base, "Deterministic baseline (tools only, no LLM)")

    if args.agent:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("ANTHROPIC_API_KEY not set: skipping the agent eval.")
        else:
            from qra.agent import run_agent

            apreds, runs = {}, []
            for name in sorted(truth):
                ctx = AuditContext.from_path(strat_dir / f"{name}.py", seed=args.seed)
                res = run_agent(ctx, effort=args.effort)
                apreds[name] = res.report.bias_types()
                runs.append({"strategy": name, "report": res.report.model_dump(), "tool_calls": res.tool_calls,
                             "usage": res.usage, "turns": res.turns, "stop_reason": res.stop_reason})
                print(f"{name}: {res.report.verdict} {sorted(apreds[name])} tokens={res.usage}")
            agent_score = score(apreds, truth)
            agent_score["runs"] = runs
            (out / "agent.json").write_text(json.dumps(agent_score, indent=2, default=str), encoding="utf-8")
            md += "\n" + table(agent_score, f"Claude agent (effort={args.effort})")

    (out / "results.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
