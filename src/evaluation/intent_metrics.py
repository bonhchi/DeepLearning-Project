"""Standard-library evaluation helpers for intent classification."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Iterable, Mapping, Protocol, Sequence

from src.io_utils import ensure_parent, write_json


LOGGER = logging.getLogger(__name__)


class IntentPredictor(Protocol):
    def predict_batch(self, query_texts: Iterable[str]) -> list[dict[str, str | float]]: ...


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def compute_intent_metrics(
    expected: Sequence[str],
    predicted: Sequence[str],
    *,
    labels: Sequence[str] | None = None,
) -> dict[str, object]:
    """Compute accuracy, macro scores, per-class scores and confusion matrix."""

    if len(expected) != len(predicted):
        raise ValueError("expected and predicted labels must have the same length")
    actual = [str(label) for label in expected]
    guesses = [str(label) for label in predicted]
    observed = set(actual) | set(guesses)
    ordered_labels = (
        list(dict.fromkeys(str(label) for label in labels)) if labels else sorted(observed)
    )
    missing = observed - set(ordered_labels)
    if missing:
        raise ValueError(f"labels does not include observed classes: {sorted(missing)}")

    matrix = [[0 for _ in ordered_labels] for _ in ordered_labels]
    indices = {label: index for index, label in enumerate(ordered_labels)}
    for truth, guess in zip(actual, guesses):
        matrix[indices[truth]][indices[guess]] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    for index, label in enumerate(ordered_labels):
        true_positive = matrix[index][index]
        false_positive = (
            sum(matrix[row][index] for row in range(len(ordered_labels))) - true_positive
        )
        false_negative = sum(matrix[index]) - true_positive
        support = sum(matrix[index])
        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        f1 = _safe_divide(2.0 * precision * recall, precision + recall)
        per_class[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": support,
        }

    class_count = len(ordered_labels)
    macro_precision = _safe_divide(
        sum(float(metrics["precision"]) for metrics in per_class.values()), class_count
    )
    macro_recall = _safe_divide(
        sum(float(metrics["recall"]) for metrics in per_class.values()), class_count
    )
    macro_f1 = _safe_divide(
        sum(float(metrics["f1"]) for metrics in per_class.values()), class_count
    )
    correct = sum(int(truth == guess) for truth, guess in zip(actual, guesses))
    return {
        "sample_count": len(actual),
        "accuracy": round(_safe_divide(correct, len(actual)), 6),
        "macro_precision": round(macro_precision, 6),
        "macro_recall": round(macro_recall, 6),
        "macro_f1": round(macro_f1, 6),
        "labels": ordered_labels,
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


evaluate_intent_predictions = compute_intent_metrics


def evaluate_intent_classifier(
    classifier: IntentPredictor,
    rows: Sequence[Mapping[str, object]],
    *,
    split: str | None = "test",
) -> dict[str, object]:
    """Evaluate a classifier directly from query-dataset rows."""

    selected = [row for row in rows if split is None or str(row.get("split")) == split]
    if not selected:
        raise ValueError(f"No query rows found for split {split!r}")
    texts = [str(row.get("query_text", "")).strip() for row in selected]
    expected = [str(row.get("intent", "")).strip() for row in selected]
    if any(not text for text in texts) or any(not label for label in expected):
        raise ValueError("Evaluation rows require non-empty query_text and intent")
    prediction_rows = classifier.predict_batch(texts)
    predicted = [str(row["intent"]) for row in prediction_rows]
    return compute_intent_metrics(expected, predicted)


def save_intent_metrics(
    metrics: Mapping[str, object],
    json_path: str | Path,
    csv_path: str | Path,
) -> tuple[Path, Path]:
    """Save a complete JSON report and a tidy CSV report, including confusion counts."""

    json_target = Path(json_path)
    csv_target = Path(csv_path)
    write_json(json_target, dict(metrics))
    ensure_parent(csv_target)
    fieldnames = ["section", "label", "predicted_label", "metric", "value", "support"]
    with csv_target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for metric_name in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
            writer.writerow(
                {
                    "section": "summary",
                    "label": "all",
                    "predicted_label": "",
                    "metric": metric_name,
                    "value": metrics.get(metric_name, 0.0),
                    "support": metrics.get("sample_count", 0),
                }
            )
        per_class = metrics.get("per_class", {})
        if isinstance(per_class, Mapping):
            for label, values in per_class.items():
                if not isinstance(values, Mapping):
                    continue
                for metric_name in ("precision", "recall", "f1"):
                    writer.writerow(
                        {
                            "section": "per_class",
                            "label": label,
                            "predicted_label": "",
                            "metric": metric_name,
                            "value": values.get(metric_name, 0.0),
                            "support": values.get("support", 0),
                        }
                    )
        labels = metrics.get("labels", [])
        matrix = metrics.get("confusion_matrix", [])
        if isinstance(labels, Sequence) and isinstance(matrix, Sequence):
            for row_index, label in enumerate(labels):
                if row_index >= len(matrix) or not isinstance(matrix[row_index], Sequence):
                    continue
                for column_index, predicted_label in enumerate(labels):
                    value = (
                        matrix[row_index][column_index]
                        if column_index < len(matrix[row_index])
                        else 0
                    )
                    writer.writerow(
                        {
                            "section": "confusion_matrix",
                            "label": label,
                            "predicted_label": predicted_label,
                            "metric": "count",
                            "value": value,
                            "support": "",
                        }
                    )
    LOGGER.info("Saved intent metrics to %s and %s", json_target, csv_target)
    return json_target, csv_target
