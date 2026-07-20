# Các hàm IO nhỏ dùng chung trong dự án.

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Iterator, Any


# Tạo thư mục cha cho file output.
def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


# Serialize giá trị phức tạp cho CSV và giữ nguyên scalar đơn giản.
def json_dumps_safe(value: Any) -> str:
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return str(value)


# Parse chuỗi dạng JSON và trả về mặc định nếu lỗi.
def json_loads_safe(value: Any, default: Any = None) -> Any:
    if value is None or value == "":
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


# Chuyển giá trị sang float, dùng mặc định an toàn khi lỗi.
def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        if value == "" or value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


# Chuyển giá trị sang int, dùng mặc định an toàn khi lỗi.
def parse_int(value: Any, default: int = 0) -> int:
    try:
        if value == "" or value is None:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


# Chuyển các chuỗi truthy phổ biến sang boolean.
def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


# Ghi dictionary ra file CSV và trả về số dòng.
def write_csv_rows(path: str | Path, rows: Iterable[dict], fieldnames: list[str]) -> int:
    ensure_parent(path)
    count = 0
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: json_dumps_safe(row.get(name, "")) for name in fieldnames})
            count += 1
    return count


# Đọc file CSV thành danh sách dictionary.
def read_csv_rows(path: str | Path) -> list[dict]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# Ghi file JSON với format dễ đọc.
def write_json(path: str | Path, data: Any) -> None:
    ensure_parent(path)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# Đọc JSON hoặc trả về mặc định nếu file không tồn tại.
def read_json(path: str | Path, default: Any = None) -> Any:
    target = Path(path)
    if not target.exists():
        return default
    return json.loads(target.read_text(encoding="utf-8"))


# Ghi dictionary ra JSON Lines và trả về số dòng.
def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    ensure_parent(path)
    count = 0
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


# Đọc streaming các dictionary từ file JSON Lines.
def read_jsonl(path: str | Path, limit: int | None = None) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            if line.strip():
                yield json.loads(line)
