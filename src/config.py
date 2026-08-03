# Cấu hình trung tâm của dự án.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_INTENT_RANKING_WEIGHTS: dict[str, dict[str, float]] = {
    "product_search": {
        "semantic_score": 0.35,
        "lexical_score": 0.25,
        "entity_match_score": 0.20,
        "user_preference_score": 0.05,
        "quality_score": 0.08,
        "popularity_score": 0.04,
        "availability_score": 0.03,
    },
    "need_based_search": {
        "semantic_score": 0.40,
        "lexical_score": 0.15,
        "entity_match_score": 0.20,
        "user_preference_score": 0.10,
        "quality_score": 0.08,
        "popularity_score": 0.04,
        "availability_score": 0.03,
    },
    "similar_product_search": {
        "semantic_score": 0.55,
        "lexical_score": 0.15,
        "entity_match_score": 0.10,
        "user_preference_score": 0.05,
        "quality_score": 0.07,
        "popularity_score": 0.03,
        "availability_score": 0.05,
    },
    "personalized_recommendation": {
        "semantic_score": 0.25,
        "lexical_score": 0.10,
        "entity_match_score": 0.10,
        "user_preference_score": 0.35,
        "quality_score": 0.10,
        "popularity_score": 0.05,
        "availability_score": 0.05,
    },
    "availability_check": {
        "semantic_score": 0.20,
        "lexical_score": 0.15,
        "entity_match_score": 0.20,
        "user_preference_score": 0.05,
        "quality_score": 0.05,
        "popularity_score": 0.05,
        "availability_score": 0.30,
    },
    "comparison": {
        "semantic_score": 0.30,
        "lexical_score": 0.20,
        "entity_match_score": 0.15,
        "user_preference_score": 0.10,
        "quality_score": 0.12,
        "popularity_score": 0.08,
        "availability_score": 0.05,
    },
}


# Lưu các đường dẫn quan trọng dùng trong pipeline dự án.
@dataclass
class ProjectConfig:

    project_root: Path
    hybrid_semantic_weight: float = 0.65
    hybrid_lexical_weight: float = 0.35
    search_candidate_pool_size: int = 200
    availability_inventory_threshold: float = 0.30
    max_exact_semantic_products: int = 50_000
    currency_rates_to_catalog: dict[str, float] = field(
        default_factory=lambda: {"USD": 1.0, "VND": 1.0 / 25_000.0}
    )
    intent_ranking_weights: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            intent: dict(weights)
            for intent, weights in DEFAULT_INTENT_RANKING_WEIGHTS.items()
        }
    )

    # Trả về thư mục data gốc.
    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    # Trả về thư mục dữ liệu đã xử lý.
    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    # Trả về thư mục output của embedding.
    @property
    def embeddings_dir(self) -> Path:
        return self.data_dir / "embeddings"

    @property
    def queries_dir(self) -> Path:
        return self.data_dir / "queries"

    # Trả về thư mục artifact output.
    @property
    def outputs_dir(self) -> Path:
        return self.project_root / "outputs"

    # Trả về thư mục chứa model đã train.
    @property
    def models_dir(self) -> Path:
        return self.outputs_dir / "models"

    # Trả về thư mục output báo cáo.
    @property
    def reports_dir(self) -> Path:
        return self.outputs_dir / "reports"

    @property
    def indexes_dir(self) -> Path:
        return self.outputs_dir / "indexes"

    # Trả về đường dẫn CSV users đã sinh.
    @property
    def users_path(self) -> Path:
        return self.processed_dir / "users.csv"

    # Trả về đường dẫn CSV products đã sinh.
    @property
    def products_path(self) -> Path:
        return self.processed_dir / "products.csv"

    # Trả về đường dẫn CSV interactions đã sinh.
    @property
    def interactions_path(self) -> Path:
        return self.processed_dir / "interactions.csv"

    @property
    def business_context_path(self) -> Path:
        return self.processed_dir / "business_context.csv"

    @property
    def dataset_manifest_path(self) -> Path:
        return self.processed_dir / "dataset_manifest.json"

    # Trả về đường dẫn JSONL product embeddings đã sinh.
    @property
    def product_embeddings_jsonl_path(self) -> Path:
        return self.embeddings_dir / "product_embeddings.jsonl"

    # Trả về vocabulary TF-IDF dùng để hiểu nhu cầu nhập vào.
    @property
    def text_vocabulary_path(self) -> Path:
        return self.embeddings_dir / "text_vocabulary.json"

    # Trả về đường dẫn artifact model two-tower đã train.
    @property
    def two_tower_model_path(self) -> Path:
        return self.models_dir / "two_tower_model.json"

    # Trả về đường dẫn báo cáo metric đánh giá.
    @property
    def metrics_path(self) -> Path:
        return self.reports_dir / "metrics.json"

    # Trace chi tiết của lần audit recommendation gần nhất.
    @property
    def recommendation_audit_path(self) -> Path:
        return self.reports_dir / "recommendation_audit.json"

    @property
    def query_dataset_path(self) -> Path:
        return self.queries_dir / "intent_queries.csv"

    @property
    def qrels_path(self) -> Path:
        return self.queries_dir / "qrels.csv"

    @property
    def qrels_review_path(self) -> Path:
        return self.queries_dir / "qrels_review.csv"

    @property
    def qrels_pool_manifest_path(self) -> Path:
        return self.queries_dir / "qrels_pool_manifest.json"

    @property
    def intent_model_path(self) -> Path:
        return self.models_dir / "intent_classifier.pkl"

    @property
    def intent_metrics_path(self) -> Path:
        return self.reports_dir / "intent_metrics.json"

    @property
    def intent_metrics_csv_path(self) -> Path:
        return self.outputs_dir / "tables" / "intent_metrics.csv"

    @property
    def intent_confusion_matrix_path(self) -> Path:
        return self.outputs_dir / "tables" / "intent_confusion_matrix.csv"

    @property
    def semantic_index_path(self) -> Path:
        return self.indexes_dir / "semantic_products"

    @property
    def lexical_index_path(self) -> Path:
        return self.indexes_dir / "lexical_products"

    @property
    def dense_encoder_path(self) -> Path:
        return self.models_dir / "dense_text_encoder.json"

    @property
    def lexical_encoder_path(self) -> Path:
        return self.models_dir / "search_tfidf_encoder.json"

    @property
    def search_index_manifest_path(self) -> Path:
        return self.indexes_dir / "search_manifest.json"

    @property
    def dense_text_cache_path(self) -> Path:
        return self.embeddings_dir / "dense_text_cache.json"

    @property
    def search_metrics_path(self) -> Path:
        return self.reports_dir / "search_metrics.json"

    @property
    def search_metrics_csv_path(self) -> Path:
        return self.outputs_dir / "tables" / "search_metrics.csv"

    @property
    def ablation_report_path(self) -> Path:
        return self.reports_dir / "ablation.json"

    @property
    def search_audit_path(self) -> Path:
        return self.reports_dir / "search_audit.json"


# Tạo các thư mục cần thiết của dự án nếu chưa tồn tại.
def ensure_project_dirs(config: ProjectConfig) -> None:
    for path in [
        config.processed_dir,
        config.embeddings_dir,
        config.queries_dir,
        config.models_dir,
        config.indexes_dir,
        config.outputs_dir / "tables",
        config.outputs_dir / "figures",
        config.reports_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)
