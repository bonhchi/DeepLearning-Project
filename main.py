# Điểm vào CLI cho hệ gợi ý mua sắm cá nhân hóa đa phương thức.

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import ProjectConfig, ensure_project_dirs
from src.data.amazon_reviews import DEFAULT_CATEGORIES, first_user_id, iter_huggingface_reviews
from src.evaluation.metrics import evaluate_recommenders
from src.feature_extraction.embeddings import build_and_save_embeddings
from src.io_utils import read_csv_rows, read_jsonl
from src.models.content_based import build_user_profiles, recommend_content
from src.models.popularity import recommend_popular, train_popularity
from src.models.two_tower import TwoTowerModel
from src.personalization.recommender import PersonalizedRecommender
from src.preprocessing.dataset_builder import (
    write_processed_dataset,
    write_processed_dataset_from_reviews,
)


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
        choices=DEFAULT_CATEGORIES,
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

    all_cmd = subparsers.add_parser("all", help="Run prepare, train, evaluate, and sample recommend.")
    add_data_source_arguments(all_cmd)
    all_cmd.add_argument("--top-k", type=int, default=10, help="K for ranking metrics and recommendations.")
    all_cmd.add_argument("--epochs", type=int, default=3, help="Training epochs.")

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
        summary = write_processed_dataset_from_reviews(reviews, config.processed_dir)
    else:
        raw_path = Path(args.raw)
        if not raw_path.is_absolute():
            raw_path = config.project_root / raw_path
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

    metrics = evaluate_recommenders(interactions, recommenders, top_k=args.top_k)
    config.metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


# In ra gợi ý cá nhân hóa cho một user.
def run_recommend(args: argparse.Namespace, config: ProjectConfig) -> list[dict]:
    user_id = args.user_id or first_user_id(config.users_path)
    recommender = PersonalizedRecommender.from_project(config.project_root)
    recommendations = recommender.recommend_for_user(user_id, top_k=args.top_k)
    print(f"User: {user_id}")
    for rank, item in enumerate(recommendations, start=1):
        print(
            f"{rank:02d}. {item['product_id']} | score={item['score']:.4f} | "
            f"{item.get('title', '')[:70]} | {item.get('explanation', '')}"
        )
    return recommendations


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
    rec_args = argparse.Namespace(user_id=None, top_k=args.top_k)
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
    elif args.command == "all":
        run_all(args, config)
    elif args.command == "app":
        print("Run: streamlit run src/app/streamlit_app.py")


if __name__ == "__main__":
    main()
