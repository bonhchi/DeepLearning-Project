# Cấu hình trung tâm của dự án.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# Lưu các đường dẫn quan trọng dùng trong pipeline dự án.
@dataclass
class ProjectConfig:

    project_root: Path

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

    # Trả về đường dẫn JSONL product embeddings đã sinh.
    @property
    def product_embeddings_jsonl_path(self) -> Path:
        return self.embeddings_dir / "product_embeddings.jsonl"

    # Trả về đường dẫn artifact model two-tower đã train.
    @property
    def two_tower_model_path(self) -> Path:
        return self.models_dir / "two_tower_model.json"

    # Trả về đường dẫn báo cáo metric đánh giá.
    @property
    def metrics_path(self) -> Path:
        return self.reports_dir / "metrics.json"


# Tạo các thư mục cần thiết của dự án nếu chưa tồn tại.
def ensure_project_dirs(config: ProjectConfig) -> None:
    for path in [
        config.processed_dir,
        config.embeddings_dir,
        config.models_dir,
        config.outputs_dir / "tables",
        config.outputs_dir / "figures",
        config.reports_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)
