# Các phép toán vector Python thuần dùng trong recommender.

from __future__ import annotations

import hashlib
import math
import random
from typing import Iterable


# Chuyển text thành số hash nguyên có tính lặp lại.
def stable_hash_int(value: str, modulo: int | None = None) -> int:
    digest = hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()
    number = int(digest[:16], 16)
    return number % modulo if modulo else number


# Tạo dense vector giả ngẫu nhiên nhưng có tính lặp lại.
def stable_random_vector(key: str, dim: int, scale: float = 1.0) -> list[float]:
    rng = random.Random(stable_hash_int(key))
    return normalize([(rng.random() * 2.0 - 1.0) * scale for _ in range(dim)])


# Tính tích vô hướng của hai dense vector.
def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


# Tính chuẩn Euclid của một vector.
def l2_norm(vector: Iterable[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


# Chuẩn hóa vector về độ dài 1 khi có thể.
def normalize(vector: list[float]) -> list[float]:
    norm = l2_norm(vector)
    if norm == 0.0:
        return list(vector)
    return [value / norm for value in vector]


# Tính độ tương đồng cosine giữa hai dense vector.
def cosine(left: list[float], right: list[float]) -> float:
    left_norm = l2_norm(left)
    right_norm = l2_norm(right)
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot(left, right) / (left_norm * right_norm)


# Tính trung bình nhiều vector thành một vector hồ sơ đã chuẩn hóa.
def average_vectors(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    output = [0.0] * dim
    for vector in vectors:
        for index, value in enumerate(vector[:dim]):
            output[index] += value
    return normalize([value / len(vectors) for value in output])


# Nối nhiều vector modality và chuẩn hóa kết quả.
def concatenate_and_normalize(parts: list[list[float]]) -> list[float]:
    output: list[float] = []
    for part in parts:
        output.extend(part)
    return normalize(output)


# Chiếu vector có độ dài bất kỳ về latent dimension cố định.
def project_vector(vector: list[float], dim: int) -> list[float]:
    output = [0.0] * dim
    if not vector:
        return output
    for index, value in enumerate(vector):
        output[index % dim] += value
    return normalize(output)


# Tính logistic sigmoid theo cách ổn định số học.
def sigmoid(value: float) -> float:
    if value > 35:
        return 1.0
    if value < -35:
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))
