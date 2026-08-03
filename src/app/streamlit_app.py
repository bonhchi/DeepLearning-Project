# Demo Streamlit: nhu cầu -> Top-5 -> giải thích -> tương tự -> benchmark.

from __future__ import annotations

import re
from pathlib import Path

from src.config import ProjectConfig
from src.feature_extraction.embeddings import DenseTextEncoder, TfidfEncoder
from src.io_utils import read_csv_rows, read_json
from src.nlp.entity_extractor import EntityExtractor
from src.nlp.intent_classifier import IntentClassifier
from src.nlp.query_rewriter import QueryRewriter
from src.personalization.intent_router import IntentRouter
from src.personalization.recommender import PersonalizedRecommender
from src.personalization.user_profile import UserProfileBuilder
from src.semantic_search.hybrid_search import HybridSearchEngine
from src.semantic_search.catalog import (
    artifact_fingerprint,
    attach_business_context,
    validate_search_manifest,
)
from src.semantic_search.lexical_index import SparseTfidfIndex
from src.semantic_search.vector_index import VectorIndex


SEARCH_OPTIONS = (
    "Intent-Aware Personalized Search",
    "Keyword Search (TF-IDF)",
    "Dense Semantic Search",
    "So sánh cả 3 chế độ",
)


def load_intent_search_stack(
    config: ProjectConfig,
    recommender: PersonalizedRecommender,
) -> dict | None:
    """Load optional NLP artifacts; legacy recommendation remains available."""

    required = (
        config.intent_model_path,
        config.semantic_index_path,
        config.lexical_index_path,
        config.dense_encoder_path,
        config.lexical_encoder_path,
        config.search_index_manifest_path,
    )
    if not all(path.exists() for path in required):
        return None
    manifest_fingerprint_before = artifact_fingerprint(
        config.search_index_manifest_path
    )
    semantic_index = VectorIndex.load(config.semantic_index_path)
    lexical_index = SparseTfidfIndex.load(config.lexical_index_path)
    if semantic_index.product_ids != lexical_index.product_ids:
        raise ValueError("Lexical and semantic index catalogs do not match")
    serving_products = attach_business_context(
        recommender.products.values(),
        read_csv_rows(config.business_context_path)
        if config.business_context_path.exists()
        else [],
        availability_threshold=config.availability_inventory_threshold,
    )
    serving_by_id = {str(row["product_id"]): row for row in serving_products}
    indexed_products = [
        serving_by_id[product_id]
        for product_id in semantic_index.product_ids
        if product_id in serving_by_id
    ]
    if len(indexed_products) != semantic_index.size:
        raise ValueError("Semantic index contains products outside the serving catalog")
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
        raise ValueError("Processed dataset changed after indexing; rebuild search artifacts")
    if manifest_fingerprint_before != artifact_fingerprint(
        config.search_index_manifest_path
    ):
        raise RuntimeError("Search artifacts changed while loading; retry")
    classifier = IntentClassifier.load(config.intent_model_path)
    extractor = EntityExtractor()
    dense_encoder = DenseTextEncoder.load(config.dense_encoder_path)
    _ = dense_encoder.backend_name  # Fail during stack load, not on the first user query.
    if dense_encoder.dimension != semantic_index.dimension:
        raise ValueError("Dense encoder and semantic index dimensions do not match")
    lexical_encoder = TfidfEncoder.load(config.lexical_encoder_path)
    aware = HybridSearchEngine(
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
    aware.lexical_index = lexical_index
    baseline = HybridSearchEngine(
        indexed_products,
        vector_index=semantic_index,
        lexical_encoder=lexical_encoder,
        dense_encoder=dense_encoder,
        semantic_weight=config.hybrid_semantic_weight,
        lexical_weight=config.hybrid_lexical_weight,
        candidate_pool_size=config.search_candidate_pool_size,
        currency_rates_to_catalog=config.currency_rates_to_catalog,
    )
    baseline.lexical_index = lexical_index
    router = IntentRouter(
        aware,
        intent_classifier=classifier,
        entity_extractor=extractor,
        query_rewriter=QueryRewriter(entity_extractor=extractor),
        recommender=recommender,
    )
    return {"baseline": baseline, "router": router}


def _baseline_result_rows(rows: list[dict], label: str) -> list[dict]:
    output = []
    for row in rows:
        item = dict(row)
        score = max(0.0, min(float(item.get("final_score", 0.0)), 1.0))
        item["score"] = score
        item["match_percentage"] = round(score * 100.0, 1)
        item["score_breakdown"] = {
            "Semantic": round(float(item.get("semantic_score", 0.0)) * 100.0, 1),
            "Lexical": round(float(item.get("lexical_score", 0.0)) * 100.0, 1),
        }
        item["explanation"] = f"Kết quả baseline {label}, chưa áp dụng cá nhân hóa."
        output.append(item)
    return output


def execute_search_option(
    stack: dict,
    recommender: PersonalizedRecommender,
    query: str,
    option: str,
    *,
    user_id: str = "",
    user_profile: dict | None = None,
    top_k: int = 5,
) -> tuple[list[dict], dict, dict[str, list[dict]]]:
    """Execute one UI search mode and return results, trace and comparison rows."""

    baseline: HybridSearchEngine = stack["baseline"]
    router: IntentRouter = stack["router"]
    if option == "Keyword Search (TF-IDF)":
        rows = _baseline_result_rows(baseline.search(query, top_k, "lexical"), "TF-IDF")
        return rows, dict(baseline.last_trace), {}
    if option == "Dense Semantic Search":
        rows = _baseline_result_rows(baseline.search(query, top_k, "semantic"), "dense semantic")
        return rows, dict(baseline.last_trace), {}

    routed = router.route(
        query,
        user_id=user_id,
        user_profile=user_profile or {},
        top_k=top_k,
        mode="hybrid",
    )
    aware_rows = routed["results"]
    if option != "So sánh cả 3 chế độ":
        return aware_rows, routed, {}
    keyword = _baseline_result_rows(baseline.search(query, top_k, "lexical"), "TF-IDF")
    semantic = _baseline_result_rows(baseline.search(query, top_k, "semantic"), "dense semantic")
    return aware_rows, routed, {
        "Keyword Search": keyword,
        "Semantic Search": semantic,
        "Intent-Aware Personalized": aware_rows,
    }


def display_image_url(value: object, size: int = 360) -> str:
    """Dùng thumbnail Amazon vừa đủ cho UI thay vì tải ảnh gốc nhiều megapixel."""
    url = str(value or "").strip()
    if not url or not any(
        host in url for host in ("m.media-amazon.com", "images-na.ssl-images-amazon.com")
    ):
        return url
    resized = re.sub(
        r"\._[^/?]+_\.(?=(?:jpe?g|png|webp)(?:\?|$))",
        f"._SL{size}_.",
        url,
        count=1,
        flags=re.IGNORECASE,
    )
    if resized != url:
        return resized
    return re.sub(
        r"\.(?=(?:jpe?g|png|webp)(?:\?|$))",
        f"._SL{size}_.",
        url,
        count=1,
        flags=re.IGNORECASE,
    )


def next_result_page(previous_key: object, current_key: object, current_page: int) -> int:
    """Nhu cầu mới bắt đầu từ Top-5; submit lại cùng nhu cầu lấy nhóm kế tiếp."""
    return max(current_page, 0) + 1 if previous_key == current_key else 0


def product_card(st, item: dict, rank: int | None = None) -> None:
    """Hiển thị một kết quả cùng mức phù hợp và breakdown điểm."""
    with st.container(border=True):
        image_column, detail_column = st.columns([1, 3])
        image_url = item.get("image_url", "")
        if image_url:
            image_column.image(display_image_url(image_url), use_container_width=True)
        else:
            image_column.markdown("🖼️ *Chưa có ảnh* ")

        prefix = f"#{rank} · " if rank is not None else ""
        detail_column.subheader(prefix + item.get("title", item.get("product_id", "")))
        match = float(item.get("match_percentage", item.get("score", 0.0) * 100))
        detail_column.progress(
            max(0.0, min(match / 100.0, 1.0)),
            text=f"Mức độ phù hợp: {match:.1f}%",
        )
        detail_column.caption(
            f"{item.get('category', '')} · ${item.get('price', '')} · "
            f"rating {item.get('average_rating', '')} · ID {item.get('product_id', '')}"
        )
        if "available" in item:
            is_available = str(item.get("available", "")).strip().casefold() in {
                "1", "true", "yes", "available", "in stock", "còn hàng", "con hang"
            }
            stock_label = "Còn hàng" if is_available else "Tạm hết hàng"
            if item.get("availability_source") == "business_context_inventory_proxy":
                stock_label += " (inventory proxy của dataset)"
            detail_column.caption(stock_label)
        detail_column.markdown(str(item.get("explanation", "")))
        breakdown = item.get("score_breakdown")
        if breakdown:
            with detail_column.expander("Xem thành phần điểm"):
                score_columns = st.columns(len(breakdown))
                for column, (label, value) in zip(score_columns, breakdown.items()):
                    column.metric(str(label), f"{float(value):.1f}%")


def benchmark_table(metrics: dict) -> list[dict]:
    """Chuyển metrics JSON thành bảng dễ trình bày trong Streamlit."""
    rows = []
    for model_name, values in metrics.items():
        row = {"Mô hình": model_name, "Users đánh giá": values.get("users_evaluated", 0)}
        row.update({key: value for key, value in values.items() if key != "users_evaluated"})
        rows.append(row)
    return rows


def benchmark_markdown(metrics: dict) -> str:
    """Render benchmark bằng Markdown để app không phụ thuộc Pandas DataFrame."""
    metric_names = sorted(
        {
            key
            for values in metrics.values()
            for key in values
            if key != "users_evaluated"
        }
    )
    headers = ["Mô hình", *metric_names, "Users đánh giá"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---", *(["---:"] * (len(headers) - 1))]) + " |",
    ]
    for model_name, values in metrics.items():
        cells = [model_name]
        cells.extend(f"{float(values.get(name, 0.0)):.4f}" for name in metric_names)
        cells.append(str(values.get("users_evaluated", 0)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise SystemExit("Install Streamlit first: pip install streamlit") from exc

    st.set_page_config(page_title="Shopping Need Recommender", page_icon="🛍️", layout="wide")
    project_root = Path(__file__).resolve().parents[2]
    config = ProjectConfig(project_root=project_root)

    @st.cache_resource
    def load_recommender() -> PersonalizedRecommender:
        # App dùng serving mode nhẹ; train/evaluate vẫn đọc full embedding khi cần.
        return PersonalizedRecommender.from_project(project_root, load_embeddings=False)

    recommender = load_recommender()

    @st.cache_resource
    def load_search_stack() -> dict | None:
        return load_intent_search_stack(config, recommender)

    try:
        search_stack = load_search_stack()
        search_stack_error = ""
    except Exception as exc:  # Artifact mismatch should not break the legacy demo.
        search_stack = None
        search_stack_error = str(exc)
    available_categories = set(recommender.products_by_category)
    if "Electronics" in available_categories:
        default_need = "Tôi cần một sản phẩm điện tử hữu ích, được đánh giá tốt"
    elif "dresses" in available_categories:
        default_need = "Tôi cần một chiếc váy thoải mái, chất lượng tốt"
    else:
        default_need = "I need a useful, highly rated product"
    st.title("🛍️ Gợi ý sản phẩm theo nhu cầu hiện tại")
    st.caption(
        "Top-5 kết hợp mức khớp nhu cầu, lịch sử người dùng, chất lượng và độ phổ biến. "
        "Phần trăm là điểm phù hợp tổng hợp, không phải xác suất mua hàng."
    )

    user_ids = recommender.active_user_ids
    selected_user = st.selectbox(
        "Khách hàng dùng để cá nhân hóa (500 user hoạt động nhất)",
        [""] + user_ids,
        format_func=lambda value: "Không dùng lịch sử cá nhân" if not value else value,
    )
    search_option = st.radio(
        "Chế độ tìm kiếm",
        SEARCH_OPTIONS,
        horizontal=True,
        disabled=search_stack is None,
        help="So sánh lexical, dense semantic và pipeline có intent + cá nhân hóa.",
    )
    if search_stack is None:
        detail = f" ({search_stack_error})" if search_stack_error else ""
        st.info(
            "Chưa có NLP search artifact; app đang dùng recommender cũ. "
            "Chạy prepare-queries → train-intent → index-semantic để bật ba chế độ tìm kiếm."
            + detail
        )
    st.session_state.setdefault("need_input", default_need)
    st.session_state.setdefault("submitted_need", default_need)
    st.session_state.setdefault("result_page", 0)
    with st.form("shopping_need_form", clear_on_submit=False):
        st.text_input(
            "Bạn đang cần sản phẩm gì?",
            key="need_input",
            placeholder="Ví dụ: Tôi cần sản phẩm chăm sóc da, rating tốt...",
        )
        submitted = st.form_submit_button("Tìm 5 sản phẩm phù hợp", type="primary")
        st.caption("Đổi nhu cầu để tìm lại từ đầu; nhấn lại cùng nhu cầu để xem nhóm 5 tiếp theo.")
    if submitted:
        st.session_state["submitted_need"] = st.session_state["need_input"].strip() or default_need
        search_key = (
            st.session_state["submitted_need"].casefold(), selected_user, search_option
        )
        result_page = next_result_page(
            st.session_state.get("last_search_key"),
            search_key,
            int(st.session_state.get("result_page", 0)),
        )
        requested_count = (result_page + 1) * 5
        with st.spinner("Đang xếp hạng sản phẩm..."):
            if search_stack is None:
                ranked = recommender.recommend_for_need(
                    st.session_state["submitted_need"], selected_user, top_k=requested_count
                )
                search_trace = {}
                comparison_rows = {}
            else:
                profile_cache = st.session_state.setdefault("user_profile_cache", {})
                if selected_user and selected_user not in profile_cache:
                    profile_cache[selected_user] = UserProfileBuilder().build(
                        selected_user,
                        recommender.interactions,
                        recommender.products.values(),
                    ).to_dict()
                ranked, search_trace, comparison_rows = execute_search_option(
                    search_stack,
                    recommender,
                    st.session_state["submitted_need"],
                    search_option,
                    user_id=selected_user,
                    user_profile=profile_cache.get(selected_user, {}),
                    top_k=requested_count,
                )
        page_start = result_page * 5
        page_results = ranked[page_start:page_start + 5]
        if not page_results and result_page > 0 and ranked:
            result_page = 0
            page_results = ranked[:5]
            st.session_state["search_notice"] = "Đã hiển thị hết ứng viên; quay lại nhóm Top-5 đầu tiên."
        else:
            st.session_state.pop("search_notice", None)
        st.session_state["result_page"] = result_page
        st.session_state["last_search_key"] = search_key
        st.session_state["recommendations"] = page_results
        st.session_state["recommendation_user"] = selected_user
        st.session_state["search_trace"] = search_trace
        st.session_state["search_comparison"] = comparison_rows
    active_need = st.session_state["submitted_need"]
    _, missing_categories = recommender.resolve_need_categories(active_need)
    if missing_categories:
        st.warning(
            "Dataset hiện tại chưa có category: " + ", ".join(missing_categories) +
            ". Hệ thống sẽ không trả sản phẩm khác loại để tránh gợi ý sai."
        )
    if st.session_state.get("search_notice"):
        st.info(st.session_state["search_notice"])
    if "recommendations" not in st.session_state:
        with st.spinner("Đang chuẩn bị gợi ý đầu tiên..."):
            if search_stack is None:
                st.session_state["recommendations"] = recommender.recommend_for_need(
                    active_need, selected_user, top_k=5
                )
                st.session_state["search_trace"] = {}
            else:
                initial_rows, initial_trace, comparison_rows = execute_search_option(
                    search_stack,
                    recommender,
                    active_need,
                    search_option,
                    user_id=selected_user,
                    top_k=5,
                )
                st.session_state["recommendations"] = initial_rows
                st.session_state["search_trace"] = initial_trace
                st.session_state["search_comparison"] = comparison_rows
        st.session_state["recommendation_user"] = selected_user
    recommendations = st.session_state["recommendations"]

    selected_view = st.radio(
        "Nội dung demo",
        ["1 · Top-5 theo nhu cầu", "2 · Sản phẩm tương tự", "3 · Benchmark"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if selected_view == "1 · Top-5 theo nhu cầu":
        result_page = int(st.session_state.get("result_page", 0))
        page_suffix = f" · nhóm {result_page + 1}" if result_page else ""
        st.subheader(f"5 kết quả tốt nhất cho: “{active_need}”{page_suffix}")
        search_trace = st.session_state.get("search_trace", {})
        if search_trace:
            trace_columns = st.columns(3)
            detected_intent = search_trace.get(
                "detected_intent", search_trace.get("intent", "baseline")
            )
            trace_columns[0].metric("Intent", detected_intent)
            trace_columns[1].metric(
                "Confidence",
                f"{float(search_trace.get('intent_confidence', 0.0)) * 100:.1f}%",
            )
            trace_columns[2].metric(
                "Search mode", search_trace.get("search_mode", search_trace.get("mode", ""))
            )
            rewritten_query = search_trace.get("rewritten_query")
            if rewritten_query and rewritten_query != active_need:
                st.caption(f"Rewritten query: {rewritten_query}")
            entities = search_trace.get(
                "extracted_entities", search_trace.get("entities", {})
            )
            if entities:
                with st.expander("Entity, constraint và trace"):
                    st.json(search_trace)

        comparison_rows = st.session_state.get("search_comparison", {})
        if comparison_rows:
            st.markdown("**So sánh kết quả trên cùng truy vấn**")
            columns = st.columns(len(comparison_rows))
            for column, (label, rows) in zip(columns, comparison_rows.items()):
                column.markdown(f"**{label}**")
                for rank, row in enumerate(rows[:5], start=1):
                    score = float(row.get("match_percentage", 0.0))
                    column.markdown(
                        f"{rank}. {row.get('title', row.get('product_id', ''))} · {score:.1f}%"
                    )
        if not recommendations:
            st.warning("Không tìm thấy ứng viên. Hãy prepare dữ liệu hoặc thử nhu cầu khác.")
        for rank, item in enumerate(recommendations, start=1):
            product_card(st, item, rank=rank)

    elif selected_view == "2 · Sản phẩm tương tự":
        st.subheader("Sản phẩm gần giống về nội dung và công năng")
        st.caption(
            "Serving mode nhẹ kết hợp độ gần của nội dung 55%, metadata 35% và trạng thái ảnh 10%. "
            "Bạn có thể dùng full embedding ở bước benchmark/training, nhưng UI không nạp file JSONL lớn."
        )
        recommended_ids = [item["product_id"] for item in recommendations]
        selectable_ids = recommended_ids or sorted(recommender.products)
        selected_product = st.selectbox(
            "Chọn sản phẩm gốc",
            selectable_ids,
            format_func=lambda product_id: (
                f"{recommender.products.get(product_id, {}).get('title', product_id)} · {product_id}"
            ),
        )
        if selected_product:
            st.markdown("**Sản phẩm đang so sánh**")
            source = dict(recommender.products.get(selected_product, {}))
            source["match_percentage"] = 100.0
            source["explanation"] = "Sản phẩm gốc dùng làm điểm neo cho phép tìm kiếm đa phương thức."
            product_card(st, source)
            if st.button("Tìm 5 sản phẩm tương tự", type="primary"):
                with st.spinner("Đang so sánh sản phẩm cùng công năng..."):
                    st.session_state["similar_product_id"] = selected_product
                    st.session_state["similar_results"] = recommender.similar_products(
                        selected_product, top_k=5
                    )
            similar_results = (
                st.session_state.get("similar_results", [])
                if st.session_state.get("similar_product_id") == selected_product
                else []
            )
            if similar_results:
                st.markdown("**5 sản phẩm tương tự nhất**")
            for rank, item in enumerate(similar_results, start=1):
                product_card(st, item, rank=rank)

    else:
        st.subheader("Benchmark các mô hình hiện có")
        metrics = read_json(config.metrics_path, default={})
        count_columns = st.columns(4)
        count_columns[0].metric("Users", f"{len(recommender.users):,}")
        count_columns[1].metric("Products", f"{len(recommender.products):,}")
        count_columns[2].metric("Interactions", f"{len(recommender.interactions):,}")
        embedding_dim = len(recommender.vocabulary) + 64
        count_columns[3].metric("Embedding dim", embedding_dim)
        if metrics:
            accuracy_keys = ("precision", "recall", "hit_rate", "ndcg", "mrr", "users_evaluated")
            accuracy_metrics = {
                model: {
                    key: value for key, value in values.items()
                    if key.startswith(accuracy_keys)
                }
                for model, values in metrics.items()
            }
            health_metrics = {
                model: {
                    key: value for key, value in values.items()
                    if not key.startswith(accuracy_keys)
                } | {"users_evaluated": values.get("users_evaluated", 0)}
                for model, values in metrics.items()
            }
            st.markdown("**Ranking accuracy**")
            st.markdown(benchmark_markdown(accuracy_metrics))
            if any(health_metrics.values()):
                st.markdown("**Serving health**")
                st.markdown(benchmark_markdown(health_metrics))
        else:
            st.info("Chưa có metrics. Chạy: python3 main.py evaluate --top-k 5")

        search_metrics = read_json(config.search_metrics_path, default={})
        st.markdown("**Intent-aware search benchmark**")
        if search_metrics:
            aggregate_search_metrics = {
                name: result.get("aggregate", {})
                for name, result in search_metrics.items()
            }
            st.markdown(benchmark_markdown(aggregate_search_metrics))
            with st.expander("Metric theo từng intent"):
                st.json(
                    {
                        name: result.get("per_intent", {})
                        for name, result in search_metrics.items()
                    }
                )
        else:
            st.info("Chưa có search metrics. Chạy: python main.py evaluate-search --top-k 5")

        intent_metrics = read_json(config.intent_metrics_path, default={})
        if intent_metrics:
            intent_columns = st.columns(4)
            intent_columns[0].metric("Intent accuracy", f"{float(intent_metrics.get('accuracy', 0.0)) * 100:.1f}%")
            intent_columns[1].metric("Macro precision", f"{float(intent_metrics.get('macro_precision', 0.0)) * 100:.1f}%")
            intent_columns[2].metric("Macro recall", f"{float(intent_metrics.get('macro_recall', 0.0)) * 100:.1f}%")
            intent_columns[3].metric("Macro F1", f"{float(intent_metrics.get('macro_f1', 0.0)) * 100:.1f}%")

        audit = read_json(config.recommendation_audit_path, default={})
        st.markdown("**Recommendation audit gần nhất**")
        if audit:
            audit_columns = st.columns(4)
            audit_columns[0].metric("Trạng thái", "PASS" if audit.get("audit_passed") else "FAIL")
            audit_columns[1].metric("Candidates", audit.get("candidate_count_before_seen_filter", 0))
            audit_columns[2].metric("Shortlist", audit.get("shortlist_total", 0))
            audit_columns[3].metric("Latency", f"{float(audit.get('total_ms', 0.0)):.1f} ms")
            with st.expander("Xem trace và các kiểm tra invariant"):
                st.json(audit)
        else:
            st.info(
                "Chưa có audit log. Chạy: python main.py audit --query \"Tôi cần tai nghe\" --top-k 5"
            )

        search_audit = read_json(config.search_audit_path, default={})
        st.markdown("**Intent-aware search audit gần nhất**")
        if search_audit:
            search_audit_columns = st.columns(4)
            search_audit_columns[0].metric(
                "Trạng thái", "PASS" if search_audit.get("audit_passed") else "FAIL"
            )
            search_audit_columns[1].metric(
                "Intent", search_audit.get("detected_intent", "")
            )
            search_audit_columns[2].metric(
                "Candidates", search_audit.get("candidate_count", 0)
            )
            search_audit_columns[3].metric(
                "Latency", f"{float(search_audit.get('latency_ms', 0.0)):.1f} ms"
            )
            with st.expander("Xem intent, entity, rewrite, filter và ranking trace"):
                st.json(search_audit)
        else:
            st.info(
                "Chưa có search audit. Chạy: python main.py audit-search --query \"tai nghe chống ồn\""
            )

        st.markdown("**Thông số Two-Tower**")
        if recommender.model:
            model_parameters = {
                "latent_dimension": recommender.model.dim,
                "regularization": recommender.model.regularization,
                "trained_users": len(recommender.model.user_embeddings),
                "trained_items": len(recommender.model.item_embeddings),
                **recommender.model.training_config,
            }
            st.json(model_parameters)
        else:
            st.info("Chưa có artifact Two-Tower. Chạy: python3 main.py train --epochs 3")

        st.markdown("**Ý nghĩa metric**")
        st.markdown(
            "Precision@K đo tỷ lệ item đúng trong Top-K; Recall@K đo phần item liên quan được tìm thấy; "
            "Hit Rate@K đo tỷ lệ user có ít nhất một hit; NDCG@K thưởng hit ở thứ hạng cao; "
            "MRR@K đo nghịch đảo thứ hạng của hit đầu tiên."
            " Coverage đo độ phủ catalog; leakage phải bằng 0; out-of-catalog phải bằng 0; "
            "latency là thời gian trung bình cho mỗi user."
        )


if __name__ == "__main__":
    main()
