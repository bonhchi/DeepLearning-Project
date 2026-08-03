"""TF-IDF + Logistic Regression baseline for multilingual query intents.

scikit-learn is used when available.  Importing this module never requires it:
``backend='auto'`` falls back to a small standard-library implementation, while
``backend='sklearn'`` raises a clear dependency error if it is unavailable.
"""

from __future__ import annotations

import logging
import math
import pickle
import random
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from src.io_utils import ensure_parent


LOGGER = logging.getLogger(__name__)
TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)
ARTIFACT_VERSION = 1


class IntentClassifierError(RuntimeError):
    """Base exception for intent classifier lifecycle failures."""


class IntentDependencyError(IntentClassifierError):
    """Raised when the explicitly requested optional backend is unavailable."""


class IntentClassifierNotTrainedError(IntentClassifierError):
    """Raised when inference or persistence is requested before training."""


def _sklearn_components() -> tuple[type, type, type]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
    except (ImportError, ModuleNotFoundError) as exc:
        raise IntentDependencyError(
            "scikit-learn is required for backend='sklearn'. Install it with "
            "`python -m pip install scikit-learn`, or use backend='auto'/'python'."
        ) from exc
    return TfidfVectorizer, LogisticRegression, Pipeline


def _tokenize(text: str, ngram_range: tuple[int, int]) -> list[str]:
    words = TOKEN_PATTERN.findall(str(text).casefold())
    tokens: list[str] = []
    minimum, maximum = ngram_range
    for size in range(minimum, maximum + 1):
        tokens.extend(
            " ".join(words[index : index + size])
            for index in range(len(words) - size + 1)
        )
    return tokens


class _PythonTfidfLogisticRegression:
    """Small deterministic multinomial logistic-regression fallback."""

    def __init__(
        self,
        *,
        max_features: int,
        ngram_range: tuple[int, int],
        max_iter: int,
        learning_rate: float,
        l2_strength: float,
        random_state: int,
    ) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.l2_strength = l2_strength
        self.random_state = random_state
        self.vocabulary: dict[str, int] = {}
        self.idf: list[float] = []
        self.classes: list[str] = []
        self.weights: list[list[float]] = []
        self.bias: list[float] = []

    def _fit_vocabulary(self, texts: Sequence[str]) -> None:
        document_frequency: Counter[str] = Counter()
        term_frequency: Counter[str] = Counter()
        for text in texts:
            tokens = _tokenize(text, self.ngram_range)
            term_frequency.update(tokens)
            document_frequency.update(set(tokens))
        ranked = sorted(
            document_frequency,
            key=lambda token: (-document_frequency[token], -term_frequency[token], token),
        )[: self.max_features]
        self.vocabulary = {token: index for index, token in enumerate(ranked)}
        sample_count = len(texts)
        self.idf = [
            math.log((1.0 + sample_count) / (1.0 + document_frequency[token])) + 1.0
            for token in ranked
        ]

    def _transform_one(self, text: str) -> dict[int, float]:
        counts: Counter[int] = Counter()
        for token in _tokenize(text, self.ngram_range):
            index = self.vocabulary.get(token)
            if index is not None:
                counts[index] += 1
        if not counts:
            return {}
        values = {
            index: (1.0 + math.log(count)) * self.idf[index]
            for index, count in counts.items()
        }
        norm = math.sqrt(sum(value * value for value in values.values()))
        return {index: value / norm for index, value in values.items()} if norm else values

    @staticmethod
    def _softmax(logits: Sequence[float]) -> list[float]:
        maximum = max(logits)
        exponentials = [math.exp(value - maximum) for value in logits]
        denominator = sum(exponentials)
        return [value / denominator for value in exponentials]

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> None:
        self.classes = sorted(set(labels))
        if not self.classes:
            raise ValueError("At least one training label is required")
        self._fit_vocabulary(texts)
        class_count = len(self.classes)
        feature_count = len(self.vocabulary)
        self.weights = [[0.0] * feature_count for _ in range(class_count)]
        self.bias = [0.0] * class_count
        if class_count == 1:
            return

        vectors = [self._transform_one(text) for text in texts]
        class_index = {label: index for index, label in enumerate(self.classes)}
        order = list(range(len(texts)))
        rng = random.Random(self.random_state)
        for epoch in range(self.max_iter):
            rng.shuffle(order)
            rate = self.learning_rate / math.sqrt(1.0 + epoch * 0.05)
            for sample_index in order:
                vector = vectors[sample_index]
                logits = [
                    self.bias[index]
                    + sum(self.weights[index][feature] * value for feature, value in vector.items())
                    for index in range(class_count)
                ]
                probabilities = self._softmax(logits)
                target = class_index[labels[sample_index]]
                for index in range(class_count):
                    error = probabilities[index] - float(index == target)
                    self.bias[index] -= rate * error
                    for feature, value in vector.items():
                        weight = self.weights[index][feature]
                        gradient = error * value + self.l2_strength * weight
                        self.weights[index][feature] -= rate * gradient

    def predict_proba(self, texts: Sequence[str]) -> list[list[float]]:
        if len(self.classes) == 1:
            return [[1.0] for _ in texts]
        output: list[list[float]] = []
        for text in texts:
            vector = self._transform_one(text)
            logits = [
                self.bias[index]
                + sum(self.weights[index][feature] * value for feature, value in vector.items())
                for index in range(len(self.classes))
            ]
            output.append(self._softmax(logits))
        return output


class IntentClassifier:
    """Trainable baseline returning ``intent`` and ``confidence`` predictions."""

    def __init__(
        self,
        *,
        backend: str = "auto",
        max_features: int = 10_000,
        ngram_range: tuple[int, int] = (1, 2),
        regularization_c: float = 1.0,
        max_iter: int = 300,
        learning_rate: float = 0.2,
        random_state: int = 42,
    ) -> None:
        if backend not in {"auto", "sklearn", "python"}:
            raise ValueError("backend must be 'auto', 'sklearn', or 'python'")
        if max_features <= 0 or max_iter <= 0 or regularization_c <= 0:
            raise ValueError("max_features, max_iter, and regularization_c must be positive")
        if len(ngram_range) != 2 or ngram_range[0] < 1 or ngram_range[0] > ngram_range[1]:
            raise ValueError("ngram_range must be an increasing positive pair")
        self.requested_backend = backend
        self.backend: str | None = None
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.regularization_c = regularization_c
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.random_state = random_state
        self._model: object | None = None
        self.classes_: list[str] = []

    @property
    def is_trained(self) -> bool:
        return self._model is not None and bool(self.classes_)

    @staticmethod
    def _training_data(
        queries: Sequence[str] | Sequence[Mapping[str, object]],
        labels: Sequence[str] | None,
    ) -> tuple[list[str], list[str]]:
        query_list = list(queries)
        if labels is None:
            if not all(isinstance(row, Mapping) for row in query_list):
                raise ValueError("labels are required when queries are plain strings")
            mapping_rows = [row for row in query_list if isinstance(row, Mapping)]
            texts = [str(row.get("query_text", "")).strip() for row in mapping_rows]
            targets = [str(row.get("intent", "")).strip() for row in mapping_rows]
        else:
            texts = [str(text).strip() for text in query_list]
            targets = [str(label).strip() for label in labels]
        if len(texts) != len(targets):
            raise ValueError("queries and labels must have the same length")
        if not texts or any(not text for text in texts) or any(not label for label in targets):
            raise ValueError("training queries and labels must be non-empty")
        return texts, targets

    def _create_python_model(self) -> _PythonTfidfLogisticRegression:
        return _PythonTfidfLogisticRegression(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            l2_strength=1.0 / self.regularization_c,
            random_state=self.random_state,
        )

    def train(
        self,
        queries: Sequence[str] | Sequence[Mapping[str, object]],
        labels: Sequence[str] | None = None,
    ) -> "IntentClassifier":
        """Fit the classifier from texts+labels or query dataset rows."""

        texts, targets = self._training_data(queries, labels)
        backend = self.requested_backend
        if backend in {"auto", "sklearn"}:
            try:
                vectorizer_type, logistic_type, pipeline_type = _sklearn_components()
            except IntentDependencyError:
                if backend == "sklearn":
                    raise
                LOGGER.warning(
                    "scikit-learn is unavailable; using the standard-library TF-IDF logistic-regression fallback"
                )
            else:
                model = pipeline_type(
                    [
                        (
                            "tfidf",
                            vectorizer_type(
                                max_features=self.max_features,
                                ngram_range=self.ngram_range,
                                lowercase=True,
                                sublinear_tf=True,
                            ),
                        ),
                        (
                            "classifier",
                            logistic_type(
                                C=self.regularization_c,
                                max_iter=self.max_iter,
                                random_state=self.random_state,
                            ),
                        ),
                    ]
                )
                model.fit(texts, targets)
                self._model = model
                self.classes_ = [str(label) for label in model.classes_]
                self.backend = "sklearn"
                LOGGER.info("Trained sklearn intent classifier on %d queries", len(texts))
                return self

        fallback = self._create_python_model()
        fallback.fit(texts, targets)
        self._model = fallback
        self.classes_ = list(fallback.classes)
        self.backend = "python"
        LOGGER.info("Trained Python intent classifier on %d queries", len(texts))
        return self

    fit = train

    def _require_trained(self) -> None:
        if not self.is_trained:
            raise IntentClassifierNotTrainedError(
                "Train or load the intent classifier before inference"
            )

    def _probability_rows(self, texts: Sequence[str]) -> list[list[float]]:
        self._require_trained()
        cleaned = [str(text).strip() for text in texts]
        if any(not text for text in cleaned):
            raise ValueError("query text must not be empty")
        probabilities = self._model.predict_proba(cleaned)  # type: ignore[union-attr]
        return [[float(value) for value in row] for row in probabilities]

    def predict_proba(
        self, query: str | Sequence[str]
    ) -> dict[str, float] | list[dict[str, float]]:
        """Return class probabilities for one query or a batch."""

        is_single = isinstance(query, str)
        texts = [query] if is_single else list(query)
        if not texts:
            self._require_trained()
            return []
        rows = [dict(zip(self.classes_, values)) for values in self._probability_rows(texts)]
        return rows[0] if is_single else rows

    def predict(self, query_text: str) -> dict[str, str | float]:
        """Predict one query and return its label and confidence."""

        probabilities = self.predict_proba(query_text)
        assert isinstance(probabilities, dict)
        intent, confidence = max(probabilities.items(), key=lambda item: (item[1], item[0]))
        return {"intent": intent, "confidence": round(confidence, 6)}

    def predict_batch(self, query_texts: Iterable[str]) -> list[dict[str, str | float]]:
        """Predict multiple queries while preserving their input order."""

        texts = list(query_texts)
        if not texts:
            return []
        probability_rows = self.predict_proba(texts)
        assert isinstance(probability_rows, list)
        predictions: list[dict[str, str | float]] = []
        for probabilities in probability_rows:
            intent, confidence = max(probabilities.items(), key=lambda item: (item[1], item[0]))
            predictions.append({"intent": intent, "confidence": round(confidence, 6)})
        return predictions

    batch_predict = predict_batch

    def save(self, path: str | Path) -> Path:
        """Persist the fitted pipeline. Never load artifacts from untrusted sources."""

        self._require_trained()
        target = Path(path)
        ensure_parent(target)
        artifact = {
            "version": ARTIFACT_VERSION,
            "requested_backend": self.requested_backend,
            "backend": self.backend,
            "max_features": self.max_features,
            "ngram_range": self.ngram_range,
            "regularization_c": self.regularization_c,
            "max_iter": self.max_iter,
            "learning_rate": self.learning_rate,
            "random_state": self.random_state,
            "classes": self.classes_,
            "model": self._model,
        }
        with target.open("wb") as handle:
            pickle.dump(artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)
        LOGGER.info("Saved intent classifier to %s", target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "IntentClassifier":
        """Load an artifact created by :meth:`save`."""

        target = Path(path)
        try:
            with target.open("rb") as handle:
                artifact = pickle.load(handle)
        except ModuleNotFoundError as exc:
            raise IntentDependencyError(
                "This classifier artifact needs an optional dependency that is not installed"
            ) from exc
        if not isinstance(artifact, dict) or artifact.get("version") != ARTIFACT_VERSION:
            raise IntentClassifierError("Unsupported or invalid intent classifier artifact")
        classifier = cls(
            backend=str(artifact["requested_backend"]),
            max_features=int(artifact["max_features"]),
            ngram_range=tuple(artifact["ngram_range"]),
            regularization_c=float(artifact["regularization_c"]),
            max_iter=int(artifact["max_iter"]),
            learning_rate=float(artifact["learning_rate"]),
            random_state=int(artifact["random_state"]),
        )
        classifier.backend = str(artifact["backend"])
        classifier.classes_ = [str(label) for label in artifact["classes"]]
        classifier._model = artifact["model"]
        classifier._require_trained()
        LOGGER.info("Loaded intent classifier from %s", target)
        return classifier
