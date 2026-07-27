# Điểm vào CLI cho hệ gợi ý mua sắm cá nhân hóa đa phương thức.

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import ProjectConfig, ensure_project_dirs
from src.data.amazon_reviews import (
    DEFAULT_CATEGORIES,
    SUPPORTED_CATEGORIES,
    METADATA_CATEGORIES,
    first_user_id,
    iter_huggingface_reviews,
    iter_reviews,
)
from src.evaluation.metrics import artifact_catalog_health, evaluate_recommenders
from src.feature_extraction.embeddings import build_and_save_embeddings
from src.io_utils import read_csv_rows, read_jsonl
from src.models.content_based import build_user_profiles, recommend_content
from src.models.popularity import recommend_popular, train_popularity
from src.models.two_tower import TwoTowerModel
from src.personalization.recommender import PersonalizedRecommender
from src.preprocessing.dataset_builder import (
    append_processed_dataset_from_reviews,
    write_processed_dataset,
    write_processed_dataset_from_reviews,
)
from src.preprocessing.metadata_enricher import enrich_product_images


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
        description="Personalized shopping recommendation MVP from Amazon Reviews 2023."
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

    all_cmd = subparsers.add_parser("all", help="Run prepare, train, evaluate, and sample recommend.")
    add_data_source_arguments(all_cmd)
    all_cmd.add_argument("--top-k", type=int, default=10, help="K for ranking metrics and recommendations.")
    all_cmd.add_argument("--epochs", type=int, default=3, help="Training epochs.")

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


# Chạy toàn bộ workflow MVP trên subset có giới hạn.
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
    eval_args = argparse.Namespace(top_k=args.top_k)
    rec_args = argparse.Namespace(user_id=None, top_k=args.top_k, query="")
    run_prepare(prepare_args, config)
    run_train(train_args, config)
    run_evaluate(eval_args, config)
    run_recommend(rec_args, config)


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
    elif args.command == "all":
        run_all(args, config)
    elif args.command == "enrich-images":
        run_enrich_images(args, config)
    elif args.command == "app":
        print("Run: streamlit run src/app/streamlit_app.py")


if __name__ == "__main__":
    main()
