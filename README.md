# Personalized Shopping Multimodal Recommender

MVP xử lý [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/), tạo dữ liệu cho user/product/session/business context, tạo embedding text/image/metadata, chạy baseline popularity/content-based, train two-tower nhẹ, đánh giá Top-K và mở demo Streamlit.

Pipeline hỗ trợ hai nguồn mà không thay đổi các bước train/evaluate phía sau:

- `local`: file `dataset/Amazon_Fashion.jsonl` đã có từ trước.
- `huggingface`: đọc streaming trực tiếp từ Hugging Face, không tải trọn bộ category.

Bốn category mặc định là `Automotive`, `Electronics`, `Health_and_Household` và `Beauty_and_Personal_Care`.

## Cấu trúc

```text
data/processed/          users/products/interactions/reviews/sessions/business_context
data/embeddings/         product text/image/metadata/fused embeddings
outputs/models/          two_tower_model.json
outputs/reports/         metrics.json
src/                     source code chính
main.py                  CLI workflow
```

## Cài đặt

```bash
python3 -m pip install -r requirements.txt
```

## Demo Hugging Face nhỏ

Nên chạy smoke test trước. Lệnh dưới chỉ đọc 1.000 review cho mỗi category (tổng tối đa 4.000):

```bash
python3 main.py prepare --source huggingface --limit-per-category 1000
python3 main.py train --epochs 3
python3 main.py evaluate --top-k 10
python3 main.py recommend --top-k 10
```

Khi smoke test thành công, tạo demo khoảng 100.000 review mỗi category:

```bash
python3 main.py prepare \
  --source huggingface \
  --limit-per-category 100000
```

Có thể chỉ chọn một phần category để giảm thời gian/RAM:

```bash
python3 main.py prepare \
  --source huggingface \
  --categories Automotive Electronics \
  --limit-per-category 10000
```

`datasets` mở từng JSONL ở chế độ `streaming=True` và dừng ngay khi đủ giới hạn của category. Dữ liệu chuẩn hóa được ghi vào `data/processed`; code train, evaluate, Streamlit và bất kỳ model local nào đọc CSV này không phụ thuộc Hugging Face hay LM Studio sau bước prepare.

## Chạy toàn bộ bằng dữ liệu local cũ

```bash
python3 main.py all --limit 5000 --top-k 10 --epochs 3
```

Lệnh này tiếp tục đọc `dataset/Amazon_Fashion.jsonl`, tạo dữ liệu xử lý, train model, đánh giá và in thử recommendation cho một user. Đây là hành vi tương thích ngược mặc định.

## Chạy từng bước

```bash
python3 main.py prepare --limit 10000
python3 main.py train --epochs 3
python3 main.py evaluate --top-k 10
python3 main.py recommend --top-k 10
```

## Demo

```bash
pip install -r requirements.txt
streamlit run src/app/streamlit_app.py
```

## Ghi chú dữ liệu

Pipeline hiện đọc phần review thô. Với Hugging Face, `category` lấy trực tiếp từ tên domain; với file Fashion local, category con vẫn được suy luận bằng keyword. Do chưa đọc file item metadata riêng, store/price/business context được sinh ổn định theo `product_id`; image embedding dùng URL trong review nếu có hoặc vector fallback nếu thiếu ảnh.

LM Studio không bắt buộc cho pipeline hiện tại. Nếu dùng model/embedding từ LM Studio sau này, nên giữ bước ingest này độc lập và đọc các CSV đã chuẩn hóa trong `data/processed` qua endpoint OpenAI-compatible của LM Studio.

Mốc 100.000 là giới hạn tối đa, không phải yêu cầu phải tải hết file nguồn. Với máy ít RAM, bắt đầu từ 1.000–10.000 mỗi category rồi tăng dần.
