"""Build a reproducible intent-detection query dataset from product metadata.

The module intentionally has no dependency on the training stack.  It accepts
products as dictionaries, which keeps it usable with both the project's CSV
catalog and small in-memory fixtures.
"""

from __future__ import annotations

import hashlib
import logging
import math
import string
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from src.io_utils import read_csv_rows, write_csv_rows
from src.nlp.intent_taxonomy import SUPPORTED_INTENTS


LOGGER = logging.getLogger(__name__)

QUERY_FIELDS = ["query_id", "query_text", "intent", "category", "source", "split"]
VALID_SPLITS = frozenset({"train", "validation", "test"})


class QueryDatasetValidationError(ValueError):
    """Raised when a query row or a split configuration is invalid."""


@dataclass(frozen=True)
class QueryTemplate:
    """Template used to turn one product metadata row into one query."""

    intent: str
    text: str
    source: str = "product_metadata"


DEFAULT_QUERY_TEMPLATES: tuple[QueryTemplate, ...] = (
    QueryTemplate("product_search", "Tìm {title}"),
    QueryTemplate("product_search", "Find {title}"),
    QueryTemplate("need_based_search", "Tôi cần giải pháp {category} cho nhu cầu như {title}"),
    QueryTemplate("need_based_search", "Best {category} for a use case like {title}"),
    QueryTemplate("similar_product_search", "Tìm sản phẩm tương tự {title}"),
    QueryTemplate("similar_product_search", "Show me products similar to {title}"),
    QueryTemplate("personalized_recommendation", "Dựa trên sở thích của tôi, hãy gợi ý {title}"),
    QueryTemplate("personalized_recommendation", "Recommend {title} based on my preferences"),
    QueryTemplate("availability_check", "{title} còn hàng không?"),
    QueryTemplate("availability_check", "Is {title} available?"),
    QueryTemplate("comparison", "So sánh {title} với các sản phẩm {category} khác"),
    QueryTemplate("comparison", "Compare {title} with other {category} products"),
)


def normalize_query_text(value: object) -> str:
    """Normalize display text without removing Vietnamese accents or casing."""

    text = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(text.split()).strip()


def _deduplication_key(value: object) -> str:
    return normalize_query_text(value).casefold()


def _stable_digest(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()


def _query_id(query_text: str, intent: str) -> str:
    return f"q_{_stable_digest(f'{intent}\0{_deduplication_key(query_text)}')[:16]}"


def make_query_id(query_text: str, intent: str) -> str:
    """Return the stable public query ID used by generated and derived rows."""

    return _query_id(normalize_query_text(query_text), str(intent))


def _coerce_template(template: QueryTemplate | Mapping[str, object]) -> QueryTemplate:
    if isinstance(template, QueryTemplate):
        result = template
    elif isinstance(template, Mapping):
        result = QueryTemplate(
            intent=str(template.get("intent", "")),
            text=str(template.get("text", template.get("template", ""))),
            source=str(template.get("source", "product_metadata")),
        )
    else:
        raise TypeError("Each query template must be a QueryTemplate or mapping")
    if result.intent not in SUPPORTED_INTENTS:
        raise QueryDatasetValidationError(f"Unsupported intent in template: {result.intent!r}")
    if not normalize_query_text(result.text):
        raise QueryDatasetValidationError("Query template text must not be empty")
    if not normalize_query_text(result.source):
        raise QueryDatasetValidationError("Query template source must not be empty")
    return result


def _template_fields(template: str) -> set[str]:
    fields: set[str] = set()
    try:
        for _, field_name, _, _ in string.Formatter().parse(template):
            if field_name:
                fields.add(field_name)
    except ValueError as exc:
        raise QueryDatasetValidationError(f"Invalid query template {template!r}: {exc}") from exc
    return fields


def _metadata_context(product: Mapping[str, object]) -> dict[str, str]:
    """Expose a small, explicit set of safe product fields to templates."""

    category = normalize_query_text(product.get("category"))
    return {
        "product_id": normalize_query_text(product.get("product_id")),
        "title": normalize_query_text(product.get("title")),
        "category": category,
        "brand": normalize_query_text(product.get("brand") or product.get("store")),
        "store": normalize_query_text(product.get("store")),
        "price": normalize_query_text(product.get("price")),
        "features": normalize_query_text(product.get("features")),
    }


def validate_query_row(
    row: Mapping[str, object],
    *,
    allowed_intents: Iterable[str] = SUPPORTED_INTENTS,
    require_split: bool = True,
) -> None:
    """Validate one dataset row and raise a descriptive exception on failure."""

    missing = [field for field in QUERY_FIELDS if field not in row]
    if missing:
        raise QueryDatasetValidationError(f"Query row is missing fields: {', '.join(missing)}")
    for field in ("query_id", "query_text", "intent", "source"):
        if not normalize_query_text(row.get(field)):
            raise QueryDatasetValidationError(f"Query field {field!r} must not be empty")
    intents = set(allowed_intents)
    if str(row.get("intent")) not in intents:
        raise QueryDatasetValidationError(f"Unsupported intent: {row.get('intent')!r}")
    split = str(row.get("split", ""))
    if require_split and split not in VALID_SPLITS:
        raise QueryDatasetValidationError(
            f"Invalid split {split!r}; expected one of {sorted(VALID_SPLITS)}"
        )


def validate_query_dataset(rows: Sequence[Mapping[str, object]]) -> None:
    """Validate schema, values, IDs, and normalized-text uniqueness."""

    seen_ids: set[str] = set()
    seen_queries: set[str] = set()
    for index, row in enumerate(rows):
        try:
            validate_query_row(row)
        except QueryDatasetValidationError as exc:
            raise QueryDatasetValidationError(f"Invalid query row at index {index}: {exc}") from exc
        query_id = str(row["query_id"])
        query_key = _deduplication_key(row["query_text"])
        if query_id in seen_ids:
            raise QueryDatasetValidationError(f"Duplicate query_id: {query_id}")
        if query_key in seen_queries:
            raise QueryDatasetValidationError(
                f"Duplicate normalized query_text: {row['query_text']!r}"
            )
        seen_ids.add(query_id)
        seen_queries.add(query_key)


def deduplicate_queries(rows: Iterable[Mapping[str, object]]) -> list[dict[str, str]]:
    """Keep the first occurrence of each case-insensitive normalized query."""

    unique: list[dict[str, str]] = []
    seen: dict[str, str] = {}
    duplicate_count = 0
    conflict_count = 0
    for row in rows:
        copied = {field: normalize_query_text(row.get(field)) for field in QUERY_FIELDS}
        key = _deduplication_key(copied["query_text"])
        if key in seen:
            duplicate_count += 1
            conflict_count += int(seen[key] != copied["intent"])
            continue
        seen[key] = copied["intent"]
        unique.append(copied)
    if duplicate_count:
        LOGGER.info("Removed %d duplicate generated queries", duplicate_count)
    if conflict_count:
        LOGGER.warning(
            "Dropped %d duplicate query texts carrying conflicting intent labels",
            conflict_count,
        )
    return unique


def _split_counts(size: int, validation_ratio: float, test_ratio: float) -> tuple[int, int]:
    if size <= 0:
        return 0, 0
    validation_count = int(math.floor(size * validation_ratio))
    test_count = int(math.floor(size * test_ratio))
    if size >= 3 and validation_ratio > 0:
        validation_count = max(validation_count, 1)
    if size - validation_count >= 2 and test_ratio > 0:
        test_count = max(test_count, 1)
    while validation_count + test_count >= size:
        if test_count > validation_count and test_count > 0:
            test_count -= 1
        elif validation_count > 0:
            validation_count -= 1
        else:
            break
    return validation_count, test_count


def assign_deterministic_splits(
    rows: Sequence[Mapping[str, object]],
    *,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> list[dict[str, str]]:
    """Assign reproducible, intent-stratified train/validation/test splits."""

    if validation_ratio < 0 or test_ratio < 0 or validation_ratio + test_ratio >= 1:
        raise QueryDatasetValidationError(
            "validation_ratio and test_ratio must be non-negative and sum to less than 1"
        )
    copied_rows = [
        {field: normalize_query_text(row.get(field)) for field in QUERY_FIELDS}
        for row in rows
    ]
    source_groups = sorted({row["source"] for row in copied_rows if row["source"]})
    # Generated datasets encode product provenance in source. Keep every query
    # derived from one product in the same split so title/brand tokens cannot leak
    # across intent train and holdout examples.
    if len(source_groups) >= 3 and all(":" in source for source in source_groups):
        ordered_sources = sorted(
            source_groups,
            key=lambda source: _stable_digest(f"{seed}\0source\0{source}"),
        )
        validation_count, test_count = _split_counts(
            len(ordered_sources), validation_ratio, test_ratio
        )
        split_by_source = {
            source: (
                "test"
                if index < test_count
                else "validation"
                if index < test_count + validation_count
                else "train"
            )
            for index, source in enumerate(ordered_sources)
        }
        for row in copied_rows:
            row["split"] = split_by_source[row["source"]]
        return sorted(copied_rows, key=lambda row: row["query_id"])

    grouped: dict[str, list[dict[str, str]]] = {}
    for copied in copied_rows:
        grouped.setdefault(copied["intent"], []).append(copied)

    output: list[dict[str, str]] = []
    for intent in sorted(grouped):
        intent_rows = sorted(
            grouped[intent],
            key=lambda row: _stable_digest(
                f"{seed}\0{intent}\0{row.get('query_id')}\0{row.get('query_text')}"
            ),
        )
        validation_count, test_count = _split_counts(
            len(intent_rows), validation_ratio, test_ratio
        )
        for index, row in enumerate(intent_rows):
            if index < test_count:
                row["split"] = "test"
            elif index < test_count + validation_count:
                row["split"] = "validation"
            else:
                row["split"] = "train"
            output.append(row)
    return sorted(output, key=lambda row: row["query_id"])


def build_query_dataset(
    products: Iterable[Mapping[str, object]],
    *,
    templates: Iterable[QueryTemplate | Mapping[str, object]] | None = None,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    max_products: int | None = None,
) -> list[dict[str, str]]:
    """Generate, deduplicate, validate, and split intent queries.

    Products are processed in input order.  ``max_products`` is optional so the
    caller, rather than this reusable module, controls dataset size.
    """

    if max_products is not None and max_products <= 0:
        raise QueryDatasetValidationError("max_products must be positive when provided")
    template_source = DEFAULT_QUERY_TEMPLATES if templates is None else templates
    selected_templates = tuple(_coerce_template(template) for template in template_source)
    if not selected_templates:
        raise QueryDatasetValidationError("At least one query template is required")
    generated: list[dict[str, str]] = []
    product_count = 0
    skipped_count = 0
    for product in products:
        if max_products is not None and product_count >= max_products:
            break
        product_count += 1
        context = _metadata_context(product)
        for template in selected_templates:
            required = _template_fields(template.text)
            unknown = required - context.keys()
            if unknown:
                raise QueryDatasetValidationError(
                    f"Template contains unsupported fields: {', '.join(sorted(unknown))}"
                )
            if any(not context[field] for field in required):
                skipped_count += 1
                continue
            query_text = normalize_query_text(template.text.format_map(context))
            if not query_text:
                skipped_count += 1
                continue
            generated.append(
                {
                    "query_id": _query_id(query_text, template.intent),
                    "query_text": query_text,
                    "intent": template.intent,
                    "category": context["category"],
                    "source": f"{template.source}:{context['product_id']}",
                    "split": "",
                }
            )

    unique = deduplicate_queries(generated)
    split_rows = assign_deterministic_splits(
        unique,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    validate_query_dataset(split_rows)
    LOGGER.info(
        "Built %d intent queries from %d products (%d skipped template instances)",
        len(split_rows),
        product_count,
        skipped_count,
    )
    return split_rows


def save_query_dataset(rows: Sequence[Mapping[str, object]], output_path: str | Path) -> int:
    """Validate and save a query dataset with a stable column order."""

    validate_query_dataset(rows)
    count = write_csv_rows(output_path, rows, QUERY_FIELDS)
    LOGGER.info("Saved %d intent queries to %s", count, output_path)
    return count


def load_query_dataset(path: str | Path, *, validate: bool = True) -> list[dict[str, str]]:
    """Load a generated query CSV."""

    rows = [dict(row) for row in read_csv_rows(path)]
    if validate:
        validate_query_dataset(rows)
    return rows


def build_query_dataset_from_csv(
    products_path: str | Path,
    output_path: str | Path,
    **build_options: object,
) -> list[dict[str, str]]:
    """Convenience entry point for CLI code; no project paths are hard-coded."""

    products = read_csv_rows(products_path)
    rows = build_query_dataset(products, **build_options)
    save_query_dataset(rows, output_path)
    return rows
