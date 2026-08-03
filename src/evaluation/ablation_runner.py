"""Reproducible ablation runner for intent-aware semantic search."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from src.io_utils import ensure_parent


@dataclass(frozen=True)
class AblationConfig:
    name: str
    use_intent_detection: bool = True
    use_entity_extraction: bool = True
    use_query_rewriting: bool = True
    use_personalization: bool = True


DEFAULT_ABLATIONS = (
    AblationConfig("full"),
    AblationConfig("without_intent_detection", use_intent_detection=False),
    AblationConfig("without_entity_extraction", use_entity_extraction=False),
    AblationConfig("without_query_rewriting", use_query_rewriting=False),
    AblationConfig("without_personalization", use_personalization=False),
)


class AblationRunner:
    def __init__(
        self,
        evaluator: Callable[[AblationConfig], dict],
        metadata: dict | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.metadata = dict(metadata or {})

    def run(self, configurations: tuple[AblationConfig, ...] = DEFAULT_ABLATIONS) -> dict:
        results = []
        for config in configurations:
            metrics = self.evaluator(config)
            results.append({"config": asdict(config), "metrics": metrics})
        metric_names = sorted(
            {
                name
                for row in results
                for name in row["metrics"]
                if name.startswith(("ndcg@", "recall@", "mrr@"))
            }
        )
        return {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": self.metadata,
            "target_metrics": metric_names,
            "results": results,
        }

    def run_and_save(
        self,
        path: str | Path,
        configurations: tuple[AblationConfig, ...] = DEFAULT_ABLATIONS,
    ) -> dict:
        report = self.run(configurations)
        ensure_parent(path)
        Path(path).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return report
