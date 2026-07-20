# Demo Streamlit cho hệ gợi ý mua sắm cá nhân hóa.

from __future__ import annotations

from pathlib import Path

from src.personalization.recommender import PersonalizedRecommender


# Hiển thị một sản phẩm được gợi ý trong app Streamlit.
def product_card(st, item: dict) -> None:
    with st.container(border=True):
        columns = st.columns([1, 3])
        image_url = item.get("image_url", "")
        if image_url:
            columns[0].image(image_url, use_container_width=True)
        else:
            columns[0].markdown("No image")
        columns[1].subheader(item.get("title", item.get("product_id", "")))
        columns[1].caption(f"{item.get('category', '')} | ${item.get('price', '')} | score {item.get('score', 0):.4f}")
        columns[1].write(item.get("explanation", ""))


# Chạy dashboard gợi ý bằng Streamlit.
def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise SystemExit("Install Streamlit first: pip install streamlit") from exc

    project_root = Path(__file__).resolve().parents[2]
    recommender = PersonalizedRecommender.from_project(project_root)

    st.set_page_config(page_title="Personalized Shopping Recommender", layout="wide")
    st.title("Personalized Shopping Experience")
    user_ids = sorted(recommender.users)
    selected_user = st.selectbox("Customer", user_ids)
    top_k = st.slider("Top-K", 3, 20, 10)

    tab_recs, tab_similar, tab_bundle = st.tabs(["Recommended for You", "Similar Products", "Bundle Suggestion"])
    with tab_recs:
        for item in recommender.recommend_for_user(selected_user, top_k=top_k):
            product_card(st, item)
    with tab_similar:
        product_id = st.selectbox("Product", sorted(recommender.products))
        for item in recommender.similar_products(product_id, top_k=top_k):
            product_card(st, item)
    with tab_bundle:
        for item in recommender.bundle_suggestion(selected_user, top_k=min(top_k, 5)):
            product_card(st, item)


if __name__ == "__main__":
    main()
