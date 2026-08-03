# Điểm vào CLI cho hệ gợi ý mua sắm cá nhân hóa đa phương thức.

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

from src.config import ProjectConfig, ensure_project_dirs
from src.data.amazon_reviews import (
    DEFAULT_CATEGORIES,
    SUPPORTED_CATEGORIES,
    METADATA_CATEGORIES,
    first_user_id,
    iter_huggingface_reviews,
    iter_reviews,
)
from src.data.qrels_builder import (
    build_qrels,
    merge_review_queue,
    validate_qrels,
    write_qrels,
)
from src.data.query_dataset_builder import (
    build_query_dataset,
    load_query_dataset,
    make_query_id,
    save_query_dataset,
)
from src.evaluation.ablation_runner import AblationConfig, AblationRunner
from src.evaluation.intent_metrics import evaluate_intent_classifier, save_intent_metrics
from src.evaluation.metrics import artifact_catalog_health, evaluate_recommenders
from src.evaluation.search_metrics import (
    compare_search_configurations,
    evaluate_search_configuration,
    save_search_metrics,
)
from src.feature_extraction.embeddings import (
    DenseTextEncoder,
    TfidfEncoder,
    build_and_save_embeddings,
    product_text,
)
from src.io_utils import parse_int, read_csv_rows, read_jsonl
from src.models.content_based import build_user_profiles, recommend_content
from src.models.popularity import recommend_popular, train_popularity
from src.models.two_tower import TwoTowerModel
from src.personalization.recommender import PersonalizedRecommender
from src.personalization.intent_router import IntentRouter, SEEN_FILTER_INTENTS
from src.personalization.user_profile import UserProfileBuilder
from src.preprocessing.dataset_builder import (
    append_processed_dataset_from_reviews,
    write_processed_dataset,
    write_processed_dataset_from_reviews,
)
from src.preprocessing.metadata_enricher import enrich_product_images
from src.nlp.entity_extractor import EntityExtractor
from src.nlp.intent_classifier import IntentClassifier
from src.nlp.query_rewriter import QueryRewriter
from src.semantic_search.hybrid_search import HybridSearchEngine
from src.semantic_search.catalog import (
    artifact_fingerprint,
    attach_business_context,
    search_catalog_fingerprint,
    validate_search_manifest,
    write_search_manifest,
)
from src.semantic_search.lexical_index import SparseTfidfIndex
from src.semantic_search.vector_index import VectorIndex


def add_data_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--source",
        choices=("local", "huggingface"),
        default="local",
        help="Read a local JSONL file or stream from Hugging Face.",
    )
    parser.add_argument("--raw", default="dataset/Amazon_Fashion.jsonl", help="Local JSONL path.")
    parser.add_argument("--limit", type=int, default=10000, help="Maximum local reviews to read.")
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=SUPPORTED_CATEGORIES,
        default=list(DEFAULT_CATEGORIES),
        help="Amazon categories used with --source huggingface.",
    )
    parser.add_argument(
        "--limit-per-category",
        type=int,
        default=100_000,
        help="Maximum streamed reviews for each Hugging Face category.",
    )


# Tạo giao diện dòng lệnh cho các workflow của dự án.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Intent-aware semantic search and personalized recommendation MVP "
            "from Amazon Reviews 2023."
        )
    )
    parser.add_argument("--project-root", default=".", help="Project root directory.")

    subparsers = parser.add_subparsers(dest="command")

    prepare = subparsers.add_parser("prepare", help="Build processed CSV files and embeddings.")
    add_data_source_arguments(prepare)
    prepare.add_argument(
        "--append",
        action="store_true",
        help="Append reviews to the current processed catalog instead of replacing it.",
    )
    prepare.add_argument("--max-text-features", type=int, default=160, help="TF-IDF vocabulary size.")
    prepare.add_argument("--image-dim", type=int, default=32, help="Pseudo image embedding size.")
    prepare.add_argument("--metadata-dim", type=int, default=32, help="Metadata embedding size.")

    train = subparsers.add_parser("train", help="Train the lightweight two-tower model.")
    train.add_argument("--epochs", type=int, default=3, help="Training epochs.")
    train.add_argument("--negative-samples", type=int, default=2, help="Negative samples per positive.")
    train.add_argument("--dim", type=int, default=48, help="Latent embedding size.")
    train.add_argument("--learning-rate", type=float, default=0.04, help="SGD learning rate.")

    evaluate = subparsers.add_parser("evaluate", help="Evaluate popularity, content, and two-tower.")
    evaluate.add_argument("--top-k", type=int, default=10, help="K for ranking metrics.")

    recommend = subparsers.add_parser("recommend", help="Print Top-K products for a user.")
    recommend.add_argument("--user-id", default=None, help="User id to recommend for.")
    recommend.add_argument("--top-k", type=int, default=10, help="Number of products to print.")
    recommend.add_argument("--query", default="", help="Current shopping need in Vietnamese or English.")

    audit = subparsers.add_parser(
        "audit",
        help="Trace one need-based recommendation and validate ranking invariants.",
    )
    audit.add_argument("--query", required=True, help="Shopping need to trace.")
    audit.add_argument("--user-id", default="", help="Optional user id for personalization.")
    audit.add_argument("--top-k", type=int, default=5, help="Number of results to inspect.")

    prepare_queries = subparsers.add_parser(
        "prepare-queries",
        help="Generate the intent-query dataset and qrels review queue.",
    )
    prepare_queries.add_argument("--category", default="Electronics")
    prepare_queries.add_argument("--max-products", type=int, default=500)
    prepare_queries.add_argument("--validation-ratio", type=float, default=0.1)
    prepare_queries.add_argument("--test-ratio", type=float, default=0.1)
    prepare_queries.add_argument("--seed", type=int, default=42)
    prepare_queries.add_argument("--qrels-per-query", type=int, default=10)

    train_intent = subparsers.add_parser(
        "train-intent", help="Train the TF-IDF + Logistic Regression intent baseline."
    )
    train_intent.add_argument("--backend", choices=("auto", "sklearn", "python"), default="auto")
    train_intent.add_argument("--max-features", type=int, default=10000)
    train_intent.add_argument("--max-iter", type=int, default=300)

    evaluate_intent = subparsers.add_parser(
        "evaluate-intent", help="Evaluate intent detection on a held-out query split."
    )
    evaluate_intent.add_argument("--split", choices=("validation", "test"), default="test")

    index_semantic = subparsers.add_parser(
        "index-semantic", help="Build dense and lexical product vector indices."
    )
    index_semantic.add_argument("--category", default="Electronics")
    index_semantic.add_argument(
        "--max-products", type=int, default=10000,
        help="Maximum products to index; 0 indexes the full selected category.",
    )
    index_semantic.add_argument("--batch-size", type=int, default=64)
    index_semantic.add_argument("--dense-dim", type=int, default=384)
    index_semantic.add_argument("--lexical-features", type=int, default=4096)
    index_semantic.add_argument(
        "--dense-backend",
        choices=("auto", "sentence-transformers", "fallback"),
        default="auto",
    )
    index_semantic.add_argument(
        "--model-name", default=DenseTextEncoder.DEFAULT_MODEL,
    )
    index_semantic.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow Sentence Transformer model download instead of local-only loading.",
    )
    index_semantic.add_argument(
        "--allow-legacy-catalog",
        action="store_true",
        help="Smoke-test only: index a catalog without the leakage-safe dataset manifest.",
    )

    pool_qrels = subparsers.add_parser(
        "pool-qrels",
        help="Pool lexical, dense and intent-aware candidates for manual qrels review.",
    )
    pool_qrels.add_argument("--top-k", type=int, default=10)
    pool_qrels.add_argument("--qrels-per-query", type=int, default=50)

    evaluate_search = subparsers.add_parser(
        "evaluate-search", help="Compare lexical, semantic, intent and personalized search."
    )
    evaluate_search.add_argument("--top-k", type=int, default=5)
    evaluate_search.add_argument(
        "--allow-unreviewed-qrels",
        action="store_true",
        help="Smoke-test only: include automatic validation/test qrels.",
    )

    ablation = subparsers.add_parser(
        "ablation", help="Measure intent/entity/rewrite/personalization contributions."
    )
    ablation.add_argument("--top-k", type=int, default=5)
    ablation.add_argument("--allow-unreviewed-qrels", action="store_true")

    audit_search = subparsers.add_parser(
        "audit-search", help="Trace one intent-aware semantic-search request."
    )
    audit_search.add_argument("--query", required=True)
    audit_search.add_argument("--user-id", default="")
    audit_search.add_argument("--product-id", default="")
    audit_search.add_argument("--top-k", type=int, default=5)
    audit_search.add_argument(
        "--mode", choices=("lexical", "semantic", "hybrid"), default="hybrid"
    )
    audit_search.add_argument(
        "--intent",
        choices=(
            "product_search", "need_based_search", "similar_product_search",
            "personalized_recommendation", "availability_check", "comparison",
        ),
        default=None,
    )

    all_cmd = subparsers.add_parser(
        "all",
        help="Run data, NLP and indexing through the qrels annotation gate.",
    )
    add_data_source_arguments(all_cmd)
    all_cmd.add_argument("--top-k", type=int, default=10, help="K for ranking metrics and recommendations.")
    all_cmd.add_argument("--epochs", type=int, default=3, help="Training epochs.")
    all_cmd.add_argument(
        "--nlp-category", default="",
        help="NLP MVP category; empty selects the largest prepared category.",
    )
    all_cmd.add_argument("--query-products", type=int, default=500)
    all_cmd.add_argument("--index-products", type=int, default=10000)
    all_cmd.add_argument("--allow-unreviewed-qrels", action="store_true")

    enrich_images = subparsers.add_parser(
        "enrich-images",
        help="Stream Amazon item metadata and fill catalog image URLs.",
    )
    enrich_images.add_argument(
        "--categories",
        nargs="+",
        choices=METADATA_CATEGORIES,
        default=None,
        help="Metadata domains; inferred automatically when omitted.",
    )
    enrich_images.add_argument(
        "--metadata-limit-per-category",
        type=int,
        default=0,
        help="Rows scanned per category; 0 scans until targets are found or the file ends.",
    )
    enrich_images.add_argument("--max-text-features", type=int, default=160)
    enrich_images.add_argument("--image-dim", type=int, default=32)
    enrich_images.add_argument("--metadata-dim", type=int, default=32)

    subparsers.add_parser("app", help="Show the Streamlit command.")
    return parser


# Tạo cấu hình dự án từ tham số dòng lệnh.
def make_config(args: argparse.Namespace) -> ProjectConfig:
    root = Path(args.project_root).resolve()
    config = ProjectConfig(project_root=root)
    ensure_project_dirs(config)
    return config


# Sinh dataset đã xử lý và embedding sản phẩm đa phương thức.
def run_prepare(args: argparse.Namespace, config: ProjectConfig) -> dict:
    if args.source == "huggingface":
        reviews = iter_huggingface_reviews(
            categories=args.categories,
            limit_per_category=args.limit_per_category,
            progress_callback=print,
        )
        if getattr(args, "append", False):
            summary = append_processed_dataset_from_reviews(reviews, config.processed_dir)
        else:
            summary = write_processed_dataset_from_reviews(reviews, config.processed_dir)
    else:
        raw_path = Path(args.raw)
        if not raw_path.is_absolute():
            raw_path = config.project_root / raw_path
        if getattr(args, "append", False):
            summary = append_processed_dataset_from_reviews(
                iter_reviews(raw_path, limit=args.limit),
                config.processed_dir,
            )
        else:
            summary = write_processed_dataset(raw_path, config.processed_dir, limit=args.limit)
    products = read_csv_rows(config.products_path)
    embedding_summary = build_and_save_embeddings(
        products=products,
        output_dir=config.embeddings_dir,
        max_text_features=args.max_text_features,
        image_dim=args.image_dim,
        metadata_dim=args.metadata_dim,
    )
    summary.update(embedding_summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


# Train và lưu model retrieval two-tower.
def run_train(args: argparse.Namespace, config: ProjectConfig) -> TwoTowerModel:
    interactions = read_csv_rows(config.interactions_path)
    embeddings = {
        row["product_id"]: row["fused_embedding"]
        for row in read_jsonl(config.product_embeddings_jsonl_path)
    }
    model = TwoTowerModel(dim=args.dim)
    model.fit(
        interactions=interactions,
        product_embeddings=embeddings,
        epochs=args.epochs,
        negative_samples=args.negative_samples,
        learning_rate=args.learning_rate,
    )
    model.save(config.two_tower_model_path)
    print(f"Saved model to {config.two_tower_model_path}")
    return model


# Bổ sung ảnh catalog thật từ item metadata rồi rebuild embedding sản phẩm.
def run_enrich_images(args: argparse.Namespace, config: ProjectConfig) -> dict:
    limit = args.metadata_limit_per_category or None
    summary = enrich_product_images(
        config.products_path,
        categories=args.categories,
        limit_per_category=limit,
        progress_callback=print,
    )
    products = read_csv_rows(config.products_path)
    embedding_summary = build_and_save_embeddings(
        products=products,
        output_dir=config.embeddings_dir,
        max_text_features=args.max_text_features,
        image_dim=args.image_dim,
        metadata_dim=args.metadata_dim,
    )
    summary.update(embedding_summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


# Đánh giá các model gợi ý hiện có và ghi báo cáo.
def run_evaluate(args: argparse.Namespace, config: ProjectConfig) -> dict:
    interactions = read_csv_rows(config.interactions_path)
    products = read_csv_rows(config.products_path)
    product_ids = [row["product_id"] for row in products]
    embeddings = {
        row["product_id"]: row["fused_embedding"]
        for row in read_jsonl(config.product_embeddings_jsonl_path)
    }
    popularity_scores = train_popularity(interactions)
    user_profiles = build_user_profiles(interactions, embeddings)
    model = TwoTowerModel.load(config.two_tower_model_path) if config.two_tower_model_path.exists() else None

    recommenders = {
        "popularity": lambda user_id, seen, k: recommend_popular(popularity_scores, product_ids, seen, k),
        "content_based": lambda user_id, seen, k: recommend_content(
            user_id, user_profiles, embeddings, seen, k
        ),
    }
    if model is not None:
        recommenders["two_tower"] = lambda user_id, seen, k: model.recommend(user_id, seen, k)

    metrics = evaluate_recommenders(
        interactions,
        recommenders,
        top_k=args.top_k,
        catalog_product_ids=set(product_ids),
    )
    catalog_ids = set(product_ids)
    metrics["popularity"].update(
        artifact_catalog_health(set(popularity_scores), catalog_ids)
    )
    metrics["content_based"].update(
        artifact_catalog_health(set(embeddings), catalog_ids)
    )
    if model is not None and "two_tower" in metrics:
        metrics["two_tower"].update(
            artifact_catalog_health(set(model.item_embeddings), catalog_ids)
        )
    config.metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


# In ra gợi ý cá nhân hóa cho một user.
def run_recommend(args: argparse.Namespace, config: ProjectConfig) -> list[dict]:
    user_id = args.user_id or first_user_id(config.users_path)
    recommender = PersonalizedRecommender.from_project(config.project_root)
    if getattr(args, "query", ""):
        recommendations = recommender.recommend_for_need(args.query, user_id, top_k=args.top_k)
    else:
        recommendations = recommender.recommend_for_user(user_id, top_k=args.top_k)
    print(f"User: {user_id}")
    for rank, item in enumerate(recommendations, start=1):
        match_text = (
            f" | match={item['match_percentage']:.1f}%"
            if "match_percentage" in item
            else ""
        )
        print(
            f"{rank:02d}. {item['product_id']} | score={item['score']:.4f}{match_text} | "
            f"{item.get('title', '')[:70]} | {item.get('explanation', '')}"
        )
    return recommendations


# Ghi trace một truy vấn và kiểm tra các invariant quan trọng của serving pipeline.
def run_audit(args: argparse.Namespace, config: ProjectConfig) -> dict:
    recommender = PersonalizedRecommender.from_project(config.project_root, load_embeddings=False)
    trace: dict = {}
    recommendations = recommender.recommend_for_need(
        args.query,
        args.user_id,
        top_k=args.top_k,
        trace=trace,
    )
    result_ids = [item["product_id"] for item in recommendations]
    resolved_categories = set(trace.get("resolved_categories", []))
    score_formula_errors = []
    for item in recommendations:
        breakdown = item.get("score_breakdown", {})
        reconstructed = (
            0.60 * float(breakdown.get("Nhu cầu", 0.0))
            + 0.20 * float(breakdown.get("Cá nhân hóa", 0.0))
            + 0.15 * float(breakdown.get("Chất lượng", 0.0))
            + 0.05 * float(breakdown.get("Phổ biến", 0.0))
        ) / 100.0
        if abs(reconstructed - float(item.get("score", 0.0))) > 0.002:
            score_formula_errors.append(item["product_id"])
    model_items = set(recommender.model.item_embeddings) if recommender.model else set()
    catalog_items = set(recommender.products)
    trace["artifact_health"] = {
        "catalog_items": len(catalog_items),
        "model_items": len(model_items),
        "model_catalog_coverage": round(
            len(model_items & catalog_items) / max(len(catalog_items), 1), 6
        ),
        "model_out_of_catalog_items": len(model_items - catalog_items),
    }
    warnings = list(trace.get("warnings", []))
    if model_items and trace["artifact_health"]["model_catalog_coverage"] < 0.8:
        warnings.append(
            "Two-Tower model covers less than 80% of the current catalog; retrain after data changes."
        )
    if recommendations and trace.get("semantic_match_result_count", 0) == 0:
        warnings.append(
            "Top-K only matches the broad category; no query keyword matched product text."
        )
    trace["warnings"] = warnings
    trace["checks"] = {
        "returned_at_most_top_k": len(result_ids) <= args.top_k,
        "no_duplicate_products": len(result_ids) == len(set(result_ids)),
        "no_seen_item_leakage": trace.get("seen_item_leakage_count", 0) == 0,
        "all_results_exist_in_catalog": all(item_id in catalog_items for item_id in result_ids),
        "all_results_match_resolved_category": all(
            not resolved_categories
            or recommender.products[item_id].get("category") in resolved_categories
            for item_id in result_ids
        ),
        "score_formula_matches_breakdown": not score_formula_errors,
        "model_catalog_coverage_acceptable": (
            not model_items or trace["artifact_health"]["model_catalog_coverage"] >= 0.8
        ),
    }
    trace["score_formula_error_products"] = score_formula_errors
    trace["audit_passed"] = all(trace["checks"].values())
    config.recommendation_audit_path.write_text(
        json.dumps(trace, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(trace, indent=2, ensure_ascii=False))
    print(f"Saved audit log to {config.recommendation_audit_path}")
    return trace


def _select_nlp_products(
    products: list[dict],
    category: str,
    max_products: int,
) -> tuple[str, list[dict]]:
    counts = Counter(str(row.get("category", "")) for row in products if row.get("category"))
    selected_category = category.strip() if category else (counts.most_common(1)[0][0] if counts else "")
    selected = [
        row for row in products if not selected_category or row.get("category") == selected_category
    ]
    if not selected:
        raise ValueError(
            f"Category {selected_category!r} is absent from products.csv; "
            f"available categories: {', '.join(sorted(counts))}"
        )
    if max_products < 0:
        raise ValueError("max_products cannot be negative")
    if max_products:
        selected = selected[:max_products]
    return selected_category, selected


def _load_search_catalog(config: ProjectConfig) -> list[dict]:
    products = read_csv_rows(config.products_path)
    business_rows = (
        read_csv_rows(config.business_context_path)
        if config.business_context_path.exists()
        else []
    )
    return attach_business_context(
        products,
        business_rows,
        availability_threshold=config.availability_inventory_threshold,
    )


def _personalized_users_by_product(
    interactions: list[dict],
    product_ids: set[str],
) -> dict[str, str]:
    """Map holdout targets to users that have a leakage-safe training profile."""

    users_with_train = {
        str(row.get("user_id", ""))
        for row in interactions
        if row.get("split", "train") == "train" and str(row.get("user_id", ""))
    }
    candidates: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for row in interactions:
        product_id = str(row.get("product_id", ""))
        user_id = str(row.get("user_id", ""))
        if (
            product_id in product_ids
            and user_id in users_with_train
            and row.get("split") in {"val", "validation", "test"}
            and parse_int(row.get("label"), 0) == 1
        ):
            candidates[product_id].append(
                (parse_int(row.get("timestamp"), 0), user_id)
            )
    return {
        product_id: sorted(rows, key=lambda item: (-item[0], item[1]))[0][1]
        for product_id, rows in candidates.items()
    }


def _training_context_by_user(
    interactions: list[dict],
    products: list[dict],
) -> dict[str, dict[str, str]]:
    product_by_id = {str(row.get("product_id", "")): row for row in products}
    candidates: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for row in interactions:
        if (
            row.get("split", "train") == "train"
            and parse_int(row.get("label"), 0) == 1
        ):
            candidates[str(row.get("user_id", ""))].append(
                (parse_int(row.get("timestamp"), 0), str(row.get("product_id", "")))
            )
    contexts: dict[str, dict[str, str]] = {}
    for user_id, rows in candidates.items():
        _, product_id = sorted(rows, key=lambda item: (-item[0], item[1]))[0]
        product = product_by_id.get(product_id, {})
        anchor = str(product.get("title") or product.get("category") or "a previous purchase")
        contexts[user_id] = {
            "anchor_title": anchor,
            "anchor_category": str(product.get("category", "")),
            "anchor_brand": str(product.get("brand") or product.get("store") or ""),
        }
    return contexts


def _query_user_id(query: dict) -> str:
    source = str(query.get("source", ""))
    if not source.startswith("user_holdout:"):
        return ""
    parts = source.split(":", 2)
    return parts[1] if len(parts) == 3 else ""


def _query_source_product_id(query: dict) -> str:
    source = str(query.get("source", ""))
    return source.rsplit(":", 1)[-1] if ":" in source else ""


def _annotate_personalized_review_rows(
    review_rows: list[dict],
    queries: list[dict],
    profiles: dict[str, dict],
) -> None:
    query_by_id = {str(row.get("query_id", "")): row for row in queries}
    for review_row in review_rows:
        query = query_by_id.get(str(review_row.get("query_id", "")), {})
        user_id = _query_user_id(query)
        if not user_id:
            continue
        profile = profiles.get(user_id, {})
        review_row["user_id"] = user_id
        review_row["profile_context"] = json.dumps(
            {
                "preferred_categories": profile.get("preferred_categories", []),
                "brands": profile.get("brands", []),
                "price_range": profile.get("price_range", {}),
                "attributes": profile.get("attributes", []),
                "negative_preferences": profile.get("negative_preferences", []),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )


def run_prepare_queries(args: argparse.Namespace, config: ProjectConfig) -> dict:
    products = _load_search_catalog(config)
    category, selected = _select_nlp_products(products, args.category, args.max_products)
    rows = build_query_dataset(
        selected,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    interactions = (
        read_csv_rows(config.interactions_path)
        if config.interactions_path.exists()
        else []
    )
    user_by_product = _personalized_users_by_product(
        interactions, {str(row["product_id"]) for row in selected}
    )
    training_contexts = _training_context_by_user(interactions, products)
    selected_by_id = {str(row["product_id"]): row for row in selected}
    chosen_target: dict[tuple[str, str], str] = {}
    for product in selected:
        product_id = str(product["product_id"])
        user_id = user_by_product.get(product_id, "")
        category_name = str(product.get("category", ""))
        if user_id and user_id in training_contexts:
            chosen_target.setdefault((user_id, category_name), product_id)
    seen_query_texts = {str(row.get("query_text", "")).casefold() for row in rows}
    for row in rows:
        if row.get("intent") != "personalized_recommendation":
            continue
        product_id = str(row.get("source", "")).rsplit(":", 1)[-1]
        user_id = user_by_product.get(product_id)
        product = selected_by_id.get(product_id, {})
        category_name = str(product.get("category", "product"))
        if (
            not user_id
            or chosen_target.get((user_id, category_name)) != product_id
        ):
            continue
        context = training_contexts[user_id]
        anchor_title = context["anchor_title"][:120]
        original_text = str(row.get("query_text", ""))
        if original_text.casefold().startswith("dựa"):
            new_text = (
                f"Dựa trên sản phẩm tôi từng thích {anchor_title}, "
                f"hãy gợi ý một sản phẩm {category_name} phù hợp"
            )
        else:
            new_text = (
                f"Based on my previously liked {anchor_title}, recommend a suitable "
                f"{category_name} product"
            )
        normalized_new = " ".join(new_text.split()).casefold()
        seen_query_texts.discard(original_text.casefold())
        if normalized_new in seen_query_texts:
            seen_query_texts.add(original_text.casefold())
            continue
        seen_query_texts.add(normalized_new)
        # Splits were already grouped by source product. Changing provenance
        # afterward keeps title tokens from leaking across query splits.
        row["query_text"] = " ".join(new_text.split())
        row["query_id"] = make_query_id(row["query_text"], str(row["intent"]))
        row["source"] = f"user_holdout:{user_id}:{product_id}"
    save_query_dataset(rows, config.query_dataset_path)
    config.qrels_pool_manifest_path.unlink(missing_ok=True)
    qrels, review_queue = build_qrels(
        rows,
        selected,
        max_candidates_per_query=args.qrels_per_query,
    )
    profile_builder = UserProfileBuilder()
    review_profiles = {
        user_id: profile_builder.build(user_id, interactions, products).to_dict()
        for user_id in {_query_user_id(row) for row in rows}
        if user_id
    }
    _annotate_personalized_review_rows(review_queue, rows, review_profiles)
    existing_review = (
        read_csv_rows(config.qrels_review_path)
        if config.qrels_review_path.exists()
        else []
    )
    review_queue, preserved_reviews = merge_review_queue(review_queue, existing_review)
    write_qrels(config.qrels_path, qrels)
    write_qrels(config.qrels_review_path, review_queue, review=True)
    summary = {
        "category": category,
        "products_used": len(selected),
        "queries": len(rows),
        "intent_distribution": dict(Counter(row["intent"] for row in rows)),
        "split_distribution": dict(Counter(row["split"] for row in rows)),
        "user_aware_personalized_queries": sum(
            row.get("intent") == "personalized_recommendation"
            and bool(_query_user_id(row))
            for row in rows
        ),
        "qrels": len(qrels),
        "qrels_pending_manual_review": sum(
            str(row.get("reviewed", "false")).casefold()
            not in {"1", "true", "yes"}
            for row in review_queue
        ),
        "manual_reviews_preserved": preserved_reviews,
        "query_dataset_path": str(config.query_dataset_path),
        "qrels_review_path": str(config.qrels_review_path),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def run_train_intent(args: argparse.Namespace, config: ProjectConfig) -> IntentClassifier:
    rows = load_query_dataset(config.query_dataset_path)
    train_rows = [row for row in rows if row.get("split") == "train"]
    if not train_rows:
        raise ValueError("Intent query dataset has no training rows")
    classifier = IntentClassifier(
        backend=args.backend,
        max_features=args.max_features,
        max_iter=args.max_iter,
    ).train(train_rows)
    classifier.save(config.intent_model_path)
    summary = {
        "training_queries": len(train_rows),
        "classes": classifier.classes_,
        "backend": classifier.backend,
        "model_path": str(config.intent_model_path),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return classifier


def run_evaluate_intent(args: argparse.Namespace, config: ProjectConfig) -> dict:
    classifier = IntentClassifier.load(config.intent_model_path)
    rows = load_query_dataset(config.query_dataset_path)
    metrics = evaluate_intent_classifier(classifier, rows, split=args.split)
    save_intent_metrics(
        metrics,
        config.intent_metrics_path,
        config.intent_metrics_csv_path,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


def run_index_semantic(args: argparse.Namespace, config: ProjectConfig) -> dict:
    dataset_manifest: dict = {}
    if config.dataset_manifest_path.exists():
        dataset_manifest = json.loads(
            config.dataset_manifest_path.read_text(encoding="utf-8")
        )
    leakage_safe_catalog = bool(
        dataset_manifest.get("artifact_version") == 1
        and dataset_manifest.get("leakage_safe") is True
    )
    if not leakage_safe_catalog and not getattr(args, "allow_legacy_catalog", False):
        raise ValueError(
            "Processed catalog has no leakage-safe dataset manifest. Re-run prepare "
            "with the current code, or use --allow-legacy-catalog for a smoke test only."
        )
    all_products = _load_search_catalog(config)
    category, products = _select_nlp_products(
        all_products, args.category, args.max_products
    )
    category_product_count = sum(
        str(row.get("category", "")) == category for row in all_products
    )
    if (
        len(products) > config.max_exact_semantic_products
        and importlib.util.find_spec("faiss") is None
    ):
        raise RuntimeError(
            f"Indexing {len(products):,} dense vectors without FAISS is disabled to avoid "
            "excessive memory/latency. Install requirements-nlp.txt or lower --max-products."
        )
    product_ids = [row["product_id"] for row in products]
    documents = [product_text(row) for row in products]
    dense_encoder = DenseTextEncoder(
        model_name=args.model_name,
        dimension=args.dense_dim,
        backend=args.dense_backend,
        local_files_only=not args.allow_model_download,
        cache_enabled=False,
    )
    dense_vectors = dense_encoder.encode(
        documents, batch_size=args.batch_size, use_cache=False
    )
    if dense_vectors and isinstance(dense_vectors[0], (int, float)):
        raise TypeError("Dense encoder returned one vector for a product batch")
    semantic_index = VectorIndex().build(product_ids, dense_vectors)
    # A missing manifest makes concurrent/interrupted rebuilds fail closed.
    config.search_index_manifest_path.unlink(missing_ok=True)
    semantic_index.save(config.semantic_index_path)
    dense_encoder.save(config.dense_encoder_path, include_cache=False)

    lexical_encoder = TfidfEncoder(
        max_features=args.lexical_features,
        cache_enabled=False,
    ).fit(documents)
    lexical_index = SparseTfidfIndex().build(product_ids, documents, lexical_encoder)
    lexical_index.save(config.lexical_index_path)
    lexical_encoder.save(config.lexical_encoder_path, include_cache=False)
    artifact_paths = {
        "semantic_index": config.semantic_index_path,
        "lexical_index": config.lexical_index_path,
        "dense_encoder": config.dense_encoder_path,
        "lexical_encoder": config.lexical_encoder_path,
    }
    write_search_manifest(
        config.search_index_manifest_path,
        {
            "category": category,
            "product_ids": product_ids,
            "catalog_fingerprint": search_catalog_fingerprint(products),
            "artifact_fingerprints": {
                name: artifact_fingerprint(path)
                for name, path in artifact_paths.items()
            },
            "dense_backend": dense_encoder.backend_name,
            "dense_dimension": semantic_index.dimension,
            "lexical_dimension": lexical_index.dimension,
            "availability_inventory_threshold": config.availability_inventory_threshold,
            "leakage_safe_catalog": leakage_safe_catalog,
            "dataset_manifest_fingerprint": (
                artifact_fingerprint(config.dataset_manifest_path)
                if config.dataset_manifest_path.exists()
                else ""
            ),
        },
    )
    summary = {
        "category": category,
        "catalog_products": len(all_products),
        "indexed_products": len(product_ids),
        "category_products": category_product_count,
        "category_coverage": round(
            len(product_ids) / max(category_product_count, 1), 6
        ),
        "catalog_coverage": round(
            len(product_ids) / max(category_product_count, 1), 6
        ),
        "global_catalog_coverage": round(
            len(product_ids) / max(len(all_products), 1), 6
        ),
        "dense_dimension": semantic_index.dimension,
        "dense_backend": dense_encoder.backend_name,
        "vector_backend": semantic_index.backend,
        "lexical_dimension": lexical_index.dimension,
        "lexical_backend": lexical_index.backend,
        "semantic_index_path": str(config.semantic_index_path),
        "manifest_path": str(config.search_index_manifest_path),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def _load_search_resources(config: ProjectConfig) -> dict:
    manifest_fingerprint_before = artifact_fingerprint(
        config.search_index_manifest_path
    )
    semantic_index = VectorIndex.load(config.semantic_index_path)
    lexical_index = SparseTfidfIndex.load(config.lexical_index_path)
    dense_encoder = DenseTextEncoder.load(config.dense_encoder_path)
    _ = dense_encoder.backend_name  # Eagerly verify the persisted model snapshot.
    if dense_encoder.dimension != semantic_index.dimension:
        raise ValueError("Dense encoder and semantic index dimensions do not match")
    lexical_encoder = TfidfEncoder.load(config.lexical_encoder_path)
    classifier = IntentClassifier.load(config.intent_model_path)
    all_products = _load_search_catalog(config)
    product_by_id = {row["product_id"]: row for row in all_products}
    missing = [product_id for product_id in semantic_index.product_ids if product_id not in product_by_id]
    if missing:
        raise ValueError(f"Semantic index contains products absent from catalog: {missing[:5]}")
    indexed_products = [product_by_id[product_id] for product_id in semantic_index.product_ids]
    if lexical_index.product_ids != semantic_index.product_ids:
        raise ValueError("Lexical and semantic indices use different product catalogs")
    loaded_manifest = validate_search_manifest(
        config.search_index_manifest_path,
        indexed_products,
        expected_product_ids=semantic_index.product_ids,
        artifact_paths={
            "semantic_index": config.semantic_index_path,
            "lexical_index": config.lexical_index_path,
            "dense_encoder": config.dense_encoder_path,
            "lexical_encoder": config.lexical_encoder_path,
        },
    )
    expected_dataset_manifest = loaded_manifest.get("dataset_manifest_fingerprint", "")
    actual_dataset_manifest = (
        artifact_fingerprint(config.dataset_manifest_path)
        if config.dataset_manifest_path.exists()
        else ""
    )
    if expected_dataset_manifest != actual_dataset_manifest:
        raise ValueError("Processed dataset manifest changed after indexing; rebuild index")
    if manifest_fingerprint_before != artifact_fingerprint(
        config.search_index_manifest_path
    ):
        raise RuntimeError("Search artifacts changed while they were being loaded; retry")

    extractor = EntityExtractor()
    aware_engine = HybridSearchEngine(
        indexed_products,
        vector_index=semantic_index,
        lexical_encoder=lexical_encoder,
        dense_encoder=dense_encoder,
        intent_classifier=classifier,
        entity_extractor=extractor,
        semantic_weight=config.hybrid_semantic_weight,
        lexical_weight=config.hybrid_lexical_weight,
        candidate_pool_size=config.search_candidate_pool_size,
        currency_rates_to_catalog=config.currency_rates_to_catalog,
    )
    aware_engine.lexical_index = lexical_index
    baseline_engine = HybridSearchEngine(
        indexed_products,
        vector_index=semantic_index,
        lexical_encoder=lexical_encoder,
        dense_encoder=dense_encoder,
        semantic_weight=config.hybrid_semantic_weight,
        lexical_weight=config.hybrid_lexical_weight,
        candidate_pool_size=config.search_candidate_pool_size,
        currency_rates_to_catalog=config.currency_rates_to_catalog,
    )
    baseline_engine.lexical_index = lexical_index
    recommender = PersonalizedRecommender.from_project(
        config.project_root, load_embeddings=False
    )
    rewriter = QueryRewriter(entity_extractor=extractor)
    router = IntentRouter(
        aware_engine,
        intent_classifier=classifier,
        entity_extractor=extractor,
        query_rewriter=rewriter,
        recommender=recommender,
    )
    return {
        "semantic_index": semantic_index,
        "lexical_index": lexical_index,
        "dense_encoder": dense_encoder,
        "lexical_encoder": lexical_encoder,
        "classifier": classifier,
        "extractor": extractor,
        "aware_engine": aware_engine,
        "baseline_engine": baseline_engine,
        "recommender": recommender,
        "router": router,
        "indexed_products": indexed_products,
    }


def run_pool_qrels(args: argparse.Namespace, config: ProjectConfig) -> dict:
    """Build an unbiased review pool from multiple retrieval configurations."""

    if args.top_k <= 0 or args.qrels_per_query <= 0:
        raise ValueError("top_k and qrels_per_query must be positive")
    minimum_pool_size = 4 * args.top_k + 1
    if args.qrels_per_query < minimum_pool_size:
        raise ValueError(
            "qrels_per_query must be at least 4 * top_k + 1 so the union of all "
            f"evaluated configurations is judged (minimum {minimum_pool_size})"
        )
    resources = _load_search_resources(config)
    queries = load_query_dataset(config.query_dataset_path)
    baseline: HybridSearchEngine = resources["baseline_engine"]
    router: IntentRouter = resources["router"]
    profile_cache: dict[str, dict] = {}
    candidate_pools: dict[str, list[str]] = {}
    for query in queries:
        query_id = str(query["query_id"])
        if query.get("split") not in {"val", "validation", "test"}:
            candidate_pools[query_id] = []
            continue
        query_text = str(query["query_text"])
        pooled: list[str] = []
        user_id = (
            _query_user_id(query)
            if query.get("intent") == "personalized_recommendation"
            else ""
        )
        if user_id and user_id not in profile_cache:
            profile_cache[user_id] = _profile_for_user(resources, user_id)
        source_product_id = (
            _query_source_product_id(query)
            if query.get("intent") == "similar_product_search"
            else ""
        )
        configurations = (
            baseline.search(query_text, args.top_k, "lexical"),
            baseline.search(query_text, args.top_k, "semantic"),
            router.route(
                query_text,
                product_id=source_product_id,
                top_k=args.top_k,
                mode="semantic",
            )["results"],
            router.route(
                query_text,
                user_id=user_id,
                user_profile=profile_cache.get(user_id, {}),
                product_id=source_product_id,
                top_k=args.top_k,
                mode="semantic",
            )["results"],
        )
        for rows in configurations:
            for row in rows:
                product_id = str(row.get("product_id", ""))
                if product_id and product_id not in pooled:
                    pooled.append(product_id)
        candidate_pools[query_id] = pooled

    qrels, review_queue = build_qrels(
        queries,
        resources["indexed_products"],
        max_candidates_per_query=args.qrels_per_query,
        candidate_pools=candidate_pools,
    )
    _annotate_personalized_review_rows(review_queue, queries, profile_cache)
    existing_review = (
        read_csv_rows(config.qrels_review_path)
        if config.qrels_review_path.exists()
        else []
    )
    review_queue, preserved_reviews = merge_review_queue(review_queue, existing_review)
    write_qrels(config.qrels_path, qrels)
    write_qrels(config.qrels_review_path, review_queue, review=True)
    pool_manifest = {
        "artifact_version": 1,
        "pool_depth_per_configuration": args.top_k,
        "configurations": [
            "tfidf",
            "dense_semantic",
            "semantic_intent",
            "semantic_intent_personalized",
        ],
        "query_ids": [str(row["query_id"]) for row in queries],
        "qrel_pairs": len(qrels),
        "search_manifest_fingerprint": artifact_fingerprint(
            config.search_index_manifest_path
        ),
    }
    temporary_manifest = config.qrels_pool_manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(pool_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary_manifest.replace(config.qrels_pool_manifest_path)
    pending = sum(
        str(row.get("reviewed", "false")).casefold() not in {"1", "true", "yes"}
        for row in review_queue
    )
    summary = {
        "queries_pooled": sum(bool(rows) for rows in candidate_pools.values()),
        "retrieval_configurations": 4,
        "pool_top_k_per_configuration": args.top_k,
        "qrels": len(qrels),
        "qrels_pending_manual_review": pending,
        "manual_reviews_preserved": preserved_reviews,
        "qrels_review_path": str(config.qrels_review_path),
        "pool_manifest_path": str(config.qrels_pool_manifest_path),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def _profile_for_user(resources: dict, user_id: str) -> dict:
    if not user_id:
        return {}
    recommender: PersonalizedRecommender = resources["recommender"]
    return UserProfileBuilder().build(
        user_id,
        recommender.interactions,
        recommender.products.values(),
    ).to_dict()


def _representative_users_by_category(recommender: PersonalizedRecommender) -> dict[str, str]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for row in recommender.interactions:
        if row.get("split", "train") != "train":
            continue
        product = recommender.products.get(str(row.get("product_id", "")), {})
        category = str(product.get("category", ""))
        user_id = str(row.get("user_id", ""))
        if category and user_id:
            counts[category][user_id] += 1
    return {
        category: users.most_common(1)[0][0]
        for category, users in counts.items()
        if users
    }


def _load_effective_qrels(config: ProjectConfig) -> list[dict]:
    """Overlay human-reviewed holdout judgments on bootstrap qrels."""

    base_rows = read_csv_rows(config.qrels_path)
    by_pair = {
        (str(row.get("query_id", "")), str(row.get("product_id", ""))): dict(row)
        for row in base_rows
    }
    if config.qrels_review_path.exists():
        for reviewed in read_csv_rows(config.qrels_review_path):
            if str(reviewed.get("reviewed", "false")).casefold() not in {
                "1", "true", "yes"
            }:
                continue
            key = (
                str(reviewed.get("query_id", "")),
                str(reviewed.get("product_id", "")),
            )
            by_pair.setdefault(
                key,
                {"query_id": key[0], "product_id": key[1]},
            ).update(
                {
                    "relevance": reviewed.get("relevance", 0),
                    "source": "manual_review",
                    "reviewed": "true",
                }
            )
    rows = list(by_pair.values())
    validate_qrels(rows)
    return rows


def _require_official_qrels_ready(
    queries: list[dict],
    qrels: list[dict],
    *,
    allow_unreviewed: bool,
) -> None:
    if allow_unreviewed:
        return
    holdout_ids = {
        str(row.get("query_id", ""))
        for row in queries
        if row.get("split") in {"val", "validation", "test"}
    }
    qrels_by_query: dict[str, list[dict]] = defaultdict(list)
    for row in qrels:
        query_id = str(row.get("query_id", ""))
        if query_id in holdout_ids:
            qrels_by_query[query_id].append(row)
    missing = sorted(holdout_ids - qrels_by_query.keys())
    pending_queries = sorted(
        query_id
        for query_id, rows in qrels_by_query.items()
        if any(
            str(row.get("reviewed", "false")).casefold()
            not in {"1", "true", "yes"}
            for row in rows
        )
    )
    if missing or pending_queries:
        raise ValueError(
            "Official retrieval evaluation requires a complete reviewed candidate "
            f"pool for every holdout query (missing={len(missing)}, "
            f"pending={len(pending_queries)}). Review data/queries/qrels_review.csv "
            "or pass --allow-unreviewed-qrels for a smoke test only."
        )


def _require_qrels_pool_compatible(
    config: ProjectConfig,
    queries: list[dict],
    *,
    top_k: int,
    allow_unreviewed: bool,
) -> None:
    if allow_unreviewed:
        return
    if not config.qrels_pool_manifest_path.exists():
        raise ValueError(
            "Official retrieval evaluation requires a system candidate pool. "
            "Run pool-qrels, then review qrels_review.csv."
        )
    payload = json.loads(config.qrels_pool_manifest_path.read_text(encoding="utf-8"))
    expected_configurations = {
        "tfidf",
        "dense_semantic",
        "semantic_intent",
        "semantic_intent_personalized",
    }
    if payload.get("artifact_version") != 1 or set(
        payload.get("configurations", [])
    ) != expected_configurations:
        raise ValueError("Qrels pool manifest uses an unsupported configuration set")
    pool_depth = parse_int(payload.get("pool_depth_per_configuration"), 0)
    if top_k > pool_depth:
        raise ValueError(
            f"Evaluation top_k={top_k} exceeds qrels pool depth={pool_depth}; "
            "re-run pool-qrels with a larger --top-k and review new candidates."
        )
    if payload.get("query_ids") != [str(row["query_id"]) for row in queries]:
        raise ValueError("Query dataset changed after qrels pooling; re-run pool-qrels")
    expected_search_manifest = payload.get("search_manifest_fingerprint")
    if expected_search_manifest != artifact_fingerprint(
        config.search_index_manifest_path
    ):
        raise ValueError("Search index changed after qrels pooling; re-run pool-qrels")
    search_manifest = json.loads(
        config.search_index_manifest_path.read_text(encoding="utf-8")
    )
    if search_manifest.get("leakage_safe_catalog") is not True:
        raise ValueError(
            "Official evaluation refuses a legacy/leakage-uncertified catalog; re-run prepare and indexing"
        )


def run_evaluate_search(args: argparse.Namespace, config: ProjectConfig) -> dict:
    queries = load_query_dataset(config.query_dataset_path)
    qrels = _load_effective_qrels(config)
    _require_official_qrels_ready(
        queries,
        qrels,
        allow_unreviewed=args.allow_unreviewed_qrels,
    )
    _require_qrels_pool_compatible(
        config,
        queries,
        top_k=args.top_k,
        allow_unreviewed=args.allow_unreviewed_qrels,
    )
    resources = _load_search_resources(config)
    validate_qrels(
        qrels,
        query_ids={row["query_id"] for row in queries},
        product_ids=set(resources["semantic_index"].product_ids),
    )
    baseline: HybridSearchEngine = resources["baseline_engine"]
    router: IntentRouter = resources["router"]
    profile_cache: dict[str, dict] = {}

    def intent_search(query: str, top_k: int, context: dict) -> list[dict]:
        return router.route(
            query,
            product_id=(
                _query_source_product_id(context)
                if context.get("intent") == "similar_product_search"
                else ""
            ),
            top_k=top_k,
            mode="semantic",
        )["results"]

    def personalized_search(query: str, top_k: int, context: dict) -> list[dict]:
        user_id = (
            _query_user_id(context)
            if context.get("intent") == "personalized_recommendation"
            else ""
        )
        if user_id and user_id not in profile_cache:
            profile_cache[user_id] = _profile_for_user(resources, user_id)
        return router.route(
            query,
            user_id=user_id,
            user_profile=profile_cache.get(user_id, {}),
            product_id=(
                _query_source_product_id(context)
                if context.get("intent") == "similar_product_search"
                else ""
            ),
            top_k=top_k,
            mode="semantic",
        )["results"]

    searchers = {
        "tfidf": lambda query, top_k, context: baseline.search(query, top_k, "lexical"),
        "dense_semantic": lambda query, top_k, context: baseline.search(query, top_k, "semantic"),
        "semantic_intent": intent_search,
        "semantic_intent_personalized": personalized_search,
    }
    metrics = compare_search_configurations(
        queries,
        qrels,
        searchers,
        top_k=args.top_k,
        require_reviewed_holdout=not args.allow_unreviewed_qrels,
    )
    metric_name = f"ndcg@{args.top_k}"
    user_aware_queries = [
        row
        for row in queries
        if row.get("intent") == "personalized_recommendation"
        and bool(_query_user_id(row))
    ]
    user_aware_ids = {str(row["query_id"]) for row in user_aware_queries}
    user_aware_qrels = [
        row for row in qrels if str(row.get("query_id", "")) in user_aware_ids
    ]
    baseline_user_eval = evaluate_search_configuration(
        user_aware_queries,
        user_aware_qrels,
        intent_search,
        top_k=args.top_k,
        require_reviewed_holdout=not args.allow_unreviewed_qrels,
    )
    personalized_user_eval = evaluate_search_configuration(
        user_aware_queries,
        user_aware_qrels,
        personalized_search,
        top_k=args.top_k,
        require_reviewed_holdout=not args.allow_unreviewed_qrels,
    )
    baseline_value = float(baseline_user_eval["aggregate"].get(metric_name, 0.0))
    personalized_value = float(
        personalized_user_eval["aggregate"].get(metric_name, 0.0)
    )
    metrics["semantic_intent_personalized"]["personalization_evaluation"] = {
        "user_aware_holdout_queries": sum(
            row.get("split") in {"val", "validation", "test"}
            for row in user_aware_queries
        ),
        "profile_source": "same-user train interactions only",
        "target_source": "same-user temporal holdout interaction",
        "query_target_title_exposed": False,
        "non_personalized": baseline_user_eval["aggregate"],
        "personalized": personalized_user_eval["aggregate"],
        f"{metric_name}_uplift_vs_non_personalized": round(
            personalized_value - baseline_value, 6
        ),
    }
    save_search_metrics(
        config.search_metrics_path,
        config.search_metrics_csv_path,
        metrics,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


class _EntityBlindRewriter:
    def __init__(self) -> None:
        self.base = QueryRewriter()

    def rewrite(self, query: str, intent: str, entities: dict, profile: dict) -> dict:
        return self.base.rewrite(
            query,
            intent,
            {"_entity_extraction_disabled": {"value": True}},
            profile,
        )


def run_ablation(args: argparse.Namespace, config: ProjectConfig) -> dict:
    queries = load_query_dataset(config.query_dataset_path)
    qrels = _load_effective_qrels(config)
    _require_official_qrels_ready(
        queries,
        qrels,
        allow_unreviewed=args.allow_unreviewed_qrels,
    )
    _require_qrels_pool_compatible(
        config,
        queries,
        top_k=args.top_k,
        allow_unreviewed=args.allow_unreviewed_qrels,
    )
    resources = _load_search_resources(config)
    validate_qrels(
        qrels,
        query_ids={row["query_id"] for row in queries},
        product_ids=set(resources["semantic_index"].product_ids),
    )
    recommender: PersonalizedRecommender = resources["recommender"]
    profile_cache: dict[str, dict] = {}

    def evaluate_config(ablation_config: AblationConfig) -> dict:
        extractor = resources["extractor"] if ablation_config.use_entity_extraction else None
        classifier = resources["classifier"] if ablation_config.use_intent_detection else None
        engine = HybridSearchEngine(
            resources["indexed_products"],
            vector_index=resources["semantic_index"],
            lexical_encoder=resources["lexical_encoder"],
            dense_encoder=resources["dense_encoder"],
            intent_classifier=classifier,
            entity_extractor=extractor,
            semantic_weight=config.hybrid_semantic_weight,
            lexical_weight=config.hybrid_lexical_weight,
            candidate_pool_size=config.search_candidate_pool_size,
            currency_rates_to_catalog=config.currency_rates_to_catalog,
        )
        engine.lexical_index = resources["lexical_index"]
        if ablation_config.use_query_rewriting:
            rewriter = QueryRewriter() if extractor else _EntityBlindRewriter()
        else:
            rewriter = None
        router = IntentRouter(
            engine,
            intent_classifier=classifier,
            entity_extractor=extractor,
            query_rewriter=rewriter,
            recommender=recommender,
        )

        def search(query: str, top_k: int, context: dict) -> list[dict]:
            user_id = ""
            profile: dict = {}
            if (
                ablation_config.use_personalization
                and context.get("intent") == "personalized_recommendation"
            ):
                user_id = _query_user_id(context)
                if user_id and user_id not in profile_cache:
                    profile_cache[user_id] = _profile_for_user(resources, user_id)
                profile = profile_cache.get(user_id, {})
            return router.route(
                query,
                user_id=user_id,
                user_profile=profile,
                product_id=(
                    _query_source_product_id(context)
                    if context.get("intent") == "similar_product_search"
                    else ""
                ),
                top_k=top_k,
                mode="semantic",
                intent=None if ablation_config.use_intent_detection else "product_search",
            )["results"]

        return evaluate_search_configuration(
            queries,
            qrels,
            search,
            top_k=args.top_k,
            require_reviewed_holdout=not args.allow_unreviewed_qrels,
        )["aggregate"]

    report = AblationRunner(
        evaluate_config,
        metadata={
            "top_k": args.top_k,
            "require_reviewed_holdout": not args.allow_unreviewed_qrels,
            "indexed_products": resources["semantic_index"].size,
            "dense_backend": resources["dense_encoder"].backend_name,
            "intent_model_backend": resources["classifier"].backend,
        },
    ).run_and_save(config.ablation_report_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def run_audit_search(args: argparse.Namespace, config: ProjectConfig) -> dict:
    resources = _load_search_resources(config)
    profile = _profile_for_user(resources, args.user_id)
    started_at = perf_counter()
    routed = resources["router"].route(
        args.query,
        user_id=args.user_id,
        user_profile=profile,
        product_id=args.product_id,
        top_k=args.top_k,
        mode=args.mode,
        intent=args.intent,
    )
    elapsed_ms = (perf_counter() - started_at) * 1000.0
    results = routed.get("results", [])
    engine_trace = dict(resources["aware_engine"].last_trace)
    result_ids = [str(row.get("product_id", "")) for row in results]
    indexed_ids = set(resources["semantic_index"].product_ids)
    seen_ids = resources["recommender"].seen_by_user.get(args.user_id, set())
    seen_filter_required = routed["detected_intent"] in SEEN_FILTER_INTENTS
    checks = {
        "nonempty_results": bool(results),
        "returned_at_most_top_k": len(results) <= args.top_k,
        "no_duplicate_products": len(result_ids) == len(set(result_ids)),
        "no_seen_item_leakage": (
            not seen_filter_required or not bool(set(result_ids) & seen_ids)
        ),
        "all_results_exist_in_index": all(product_id in indexed_ids for product_id in result_ids),
    }
    trace = {
        "query": _sanitize_audit_value(routed["original_query"]),
        "detected_intent": routed["detected_intent"],
        "intent_confidence": routed["intent_confidence"],
        "extracted_entities": _sanitize_audit_value(routed["extracted_entities"]),
        "rewritten_query": _sanitize_audit_value(routed["rewritten_query"]),
        "added_preferences": _sanitize_audit_value(routed.get("added_preferences", [])),
        "ignored_preferences": _sanitize_audit_value(routed.get("ignored_preferences", [])),
        "search_mode": routed["search_mode"],
        "result_status": "ok" if results else "no_results_inconclusive",
        "strategy": routed["strategy"],
        "ranking_weights": config.intent_ranking_weights.get(
            routed["detected_intent"], config.intent_ranking_weights["product_search"]
        ),
        "personalization_applied": bool(args.user_id and profile),
        "candidate_count": engine_trace.get("candidate_count", routed["candidate_count"]),
        "candidate_count_after_routing": routed["candidate_count"],
        "filtered_count": engine_trace.get("filtered_count", 0),
        "applied_filters": engine_trace.get(
            "applied_filters",
            list(dict.fromkeys(str(row.get("filter_reason", "")) for row in results)),
        ),
        "returned_count": len(results),
        "latency_ms": round(elapsed_ms, 3),
        "checks": checks,
        "audit_passed": all(checks.values()),
        "top_results": [
            {
                "rank": rank,
                "product_id": row.get("product_id", ""),
                "title": row.get("title", ""),
                "semantic_score": row.get("semantic_score", 0.0),
                "lexical_score": row.get("lexical_score", 0.0),
                "personalization_score": row.get("score_components", {}).get(
                    "user_preference_score", 0.0
                ),
                "final_score": row.get("final_score", row.get("score", 0.0)),
                "score_components": row.get("score_components", {}),
                "ranking_reason": row.get("explanation", ""),
                "available": row.get("available", "unknown"),
                "availability_source": row.get("availability_source", "unknown"),
            }
            for rank, row in enumerate(results, start=1)
        ],
    }
    config.search_audit_path.write_text(
        json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(trace, indent=2, ensure_ascii=False))
    return trace


def _sanitize_audit_value(value: Any) -> Any:
    """Redact common direct identifiers before persisting a search trace."""

    if isinstance(value, dict):
        return {key: _sanitize_audit_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_audit_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_audit_value(item) for item in value)
    if not isinstance(value, str):
        return value
    sanitized = re.sub(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        "[REDACTED_EMAIL]",
        value,
        flags=re.IGNORECASE,
    )
    return re.sub(
        r"(?<!\w)(?:\+?\d[\d .()-]{7,}\d)(?!\w)",
        "[REDACTED_PHONE]",
        sanitized,
    )


# Chạy toàn bộ workflow NLP MVP trên subset có giới hạn.
def run_all(args: argparse.Namespace, config: ProjectConfig) -> None:
    prepare_args = argparse.Namespace(
        source=args.source,
        raw=args.raw,
        limit=args.limit,
        categories=args.categories,
        limit_per_category=args.limit_per_category,
        max_text_features=160,
        image_dim=32,
        metadata_dim=32,
    )
    train_args = argparse.Namespace(epochs=args.epochs, negative_samples=2, dim=48, learning_rate=0.04)
    run_prepare(prepare_args, config)
    query_args = argparse.Namespace(
        category=args.nlp_category,
        max_products=args.query_products,
        validation_ratio=0.1,
        test_ratio=0.1,
        seed=42,
        qrels_per_query=10,
    )
    query_summary = run_prepare_queries(query_args, config)
    if args.index_products and args.index_products < query_summary["products_used"]:
        raise ValueError(
            "--index-products must be 0 (full category) or at least --query-products "
            "so every qrel target exists in the search index"
        )
    run_train_intent(
        argparse.Namespace(backend="auto", max_features=10000, max_iter=300),
        config,
    )
    run_index_semantic(
        argparse.Namespace(
            category=query_summary["category"],
            max_products=args.index_products,
            batch_size=64,
            dense_dim=384,
            lexical_features=4096,
            dense_backend="auto",
            model_name=DenseTextEncoder.DEFAULT_MODEL,
            allow_model_download=False,
            allow_legacy_catalog=False,
        ),
        config,
    )
    run_pool_qrels(
        argparse.Namespace(
            top_k=max(10, args.top_k),
            qrels_per_query=4 * max(10, args.top_k) + 1,
        ),
        config,
    )
    run_train(train_args, config)
    run_evaluate_intent(argparse.Namespace(split="test"), config)
    if not args.allow_unreviewed_qrels:
        print(
            json.dumps(
                {
                    "status": "awaiting_qrels_review",
                    "next_step": (
                        "Review every holdout pair in data/queries/qrels_review.csv, "
                        "then run evaluate-search and ablation."
                    ),
                    "smoke_test_option": "Re-run all with --allow-unreviewed-qrels.",
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    evaluation_args = argparse.Namespace(
        top_k=args.top_k,
        allow_unreviewed_qrels=args.allow_unreviewed_qrels,
    )
    run_evaluate_search(evaluation_args, config)
    run_ablation(evaluation_args, config)


# Điều hướng lệnh CLI tới workflow tương ứng.
def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    config = make_config(args)
    if args.command == "prepare":
        run_prepare(args, config)
    elif args.command == "train":
        run_train(args, config)
    elif args.command == "evaluate":
        run_evaluate(args, config)
    elif args.command == "recommend":
        run_recommend(args, config)
    elif args.command == "audit":
        run_audit(args, config)
    elif args.command == "prepare-queries":
        run_prepare_queries(args, config)
    elif args.command == "train-intent":
        run_train_intent(args, config)
    elif args.command == "evaluate-intent":
        run_evaluate_intent(args, config)
    elif args.command == "index-semantic":
        run_index_semantic(args, config)
    elif args.command == "pool-qrels":
        run_pool_qrels(args, config)
    elif args.command == "evaluate-search":
        run_evaluate_search(args, config)
    elif args.command == "ablation":
        run_ablation(args, config)
    elif args.command == "audit-search":
        run_audit_search(args, config)
    elif args.command == "all":
        run_all(args, config)
    elif args.command == "enrich-images":
        run_enrich_images(args, config)
    elif args.command == "app":
        print("Run: streamlit run src/app/streamlit_app.py")


if __name__ == "__main__":
    main()
