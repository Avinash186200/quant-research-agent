"""Score predicted bias types against ground truth."""

from __future__ import annotations

from .report import BIAS_TYPES


def score(predicted: dict[str, set[str]], truth: dict[str, set[str]]) -> dict:
    tp = fp = fn = 0
    per_type = {t: {"tp": 0, "fp": 0, "fn": 0} for t in BIAS_TYPES}
    verdict_correct = 0
    rows = []
    for name, true_set in sorted(truth.items()):
        pred = set(predicted.get(name, set()))
        for t in BIAS_TYPES:
            if t in pred and t in true_set:
                per_type[t]["tp"] += 1
            elif t in pred:
                per_type[t]["fp"] += 1
            elif t in true_set:
                per_type[t]["fn"] += 1
        tp += len(pred & true_set)
        fp += len(pred - true_set)
        fn += len(true_set - pred)
        verdict_correct += int(bool(pred) == bool(true_set))
        rows.append({"strategy": name, "truth": sorted(true_set), "predicted": sorted(pred),
                     "missed": sorted(true_set - pred), "false_alarms": sorted(pred - true_set)})
    return {
        "n_strategies": len(truth),
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "verdict_accuracy": _ratio(verdict_correct, len(truth)),
        "tp": tp, "fp": fp, "fn": fn,
        "per_type": {t: {**c, "recall": _ratio(c["tp"], c["tp"] + c["fn"])} for t, c in per_type.items()},
        "rows": rows,
    }


def _ratio(a: int, b: int) -> float:
    return round(a / b, 4) if b else 1.0
