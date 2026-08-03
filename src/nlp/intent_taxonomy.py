"""Canonical six-intent taxonomy for product discovery."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class IntentDefinition:
    label: str
    definition: str
    vietnamese_examples: tuple[str, ...]
    english_examples: tuple[str, ...]
    confusable_with: tuple[str, ...]
    distinction_rule: str

    def to_dict(self) -> dict:
        return asdict(self)


INTENT_TAXONOMY: dict[str, IntentDefinition] = {
    "product_search": IntentDefinition(
        label="product_search",
        definition="Tìm một loại hoặc tên sản phẩm với thuộc tính cụ thể.",
        vietnamese_examples=("Tìm tai nghe Sony màu đen", "Cho tôi xem giày chạy bộ size 42"),
        english_examples=("Find black Sony headphones", "Show running shoes size 9"),
        confusable_with=("need_based_search", "availability_check"),
        distinction_rule="Có loại/tên sản phẩm rõ; không hỏi tồn kho và không chỉ mô tả bài toán sử dụng.",
    ),
    "need_based_search": IntentDefinition(
        label="need_based_search",
        definition="Mô tả mục đích hoặc vấn đề cần giải quyết thay vì một SKU cụ thể.",
        vietnamese_examples=("Tôi cần tai nghe để học ở quán cà phê", "Kem phù hợp da nhạy cảm"),
        english_examples=("Headphones for studying in noisy cafés", "Something for sensitive skin"),
        confusable_with=("product_search", "personalized_recommendation"),
        distinction_rule="Ưu tiên purpose/feature trong câu; không cần tham chiếu lịch sử người dùng.",
    ),
    "similar_product_search": IntentDefinition(
        label="similar_product_search",
        definition="Tìm item thay thế hoặc tương tự một sản phẩm neo.",
        vietnamese_examples=("Tìm sản phẩm giống B08XYZ", "Có mẫu nào tương tự đôi giày này?"),
        english_examples=("Find products similar to B08XYZ", "Show alternatives to this shoe"),
        confusable_with=("comparison", "product_search"),
        distinction_rule="Có sản phẩm neo và yêu cầu tương tự/thay thế, không yêu cầu đặt cạnh để so sánh.",
    ),
    "personalized_recommendation": IntentDefinition(
        label="personalized_recommendation",
        definition="Yêu cầu đề xuất dựa trên sở thích hoặc lịch sử cá nhân.",
        vietnamese_examples=("Gợi ý tai nghe phù hợp với tôi", "Dựa vào lịch sử hãy chọn mỹ phẩm"),
        english_examples=("Recommend headphones for me", "Pick skincare based on my history"),
        confusable_with=("need_based_search", "product_search"),
        distinction_rule="Có tín hiệu rõ như 'cho tôi', 'lịch sử', 'sở thích' hoặc personalized/recommend.",
    ),
    "availability_check": IntentDefinition(
        label="availability_check",
        definition="Kiểm tra sản phẩm hoặc cấu hình có đang bán/còn hàng.",
        vietnamese_examples=("Tai nghe này còn hàng không?", "Có sẵn màu xanh size M không?"),
        english_examples=("Is this headphone in stock?", "Is blue size M available?"),
        confusable_with=("product_search",),
        distinction_rule="Câu hỏi trọng tâm là tồn kho/availability; áp dụng hard filter.",
    ),
    "comparison": IntentDefinition(
        label="comparison",
        definition="Đặt từ hai sản phẩm hoặc phương án cạnh nhau để đánh giá.",
        vietnamese_examples=("So sánh AirPods và Sony XM5", "Mẫu A hay mẫu B tốt hơn?"),
        english_examples=("Compare AirPods with Sony XM5", "Which is better, model A or B?"),
        confusable_with=("similar_product_search",),
        distinction_rule="Có ít nhất hai phương án hoặc động từ compare/so sánh; output là một nhóm so sánh.",
    ),
}

SUPPORTED_INTENTS = tuple(INTENT_TAXONOMY)


def get_intent_definition(label: str) -> IntentDefinition:
    try:
        return INTENT_TAXONOMY[label]
    except KeyError as exc:
        raise ValueError(f"Unknown intent: {label}") from exc


__all__ = ["INTENT_TAXONOMY", "SUPPORTED_INTENTS", "IntentDefinition", "get_intent_definition"]
