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

Nếu `Amazon_Fashion.jsonl` đã có ở local và muốn ghép Fashion vào catalog hiện tại
thay vì ghi đè bốn category đã xử lý, chạy từ CMD:

```cmd
.venv\Scripts\python.exe main.py prepare --source local --raw dataset\Amazon_Fashion.jsonl --limit 100000 --append
```

Fashion local được tách thành các nhóm con như `tops`, `dresses`, `shoes`, `bags`.
Ví dụ nhu cầu “áo thun” sẽ tìm trong `tops`; tùy chọn `--append` giữ lại catalog hiện có
và bỏ qua review trùng theo user, sản phẩm và timestamp.

Pipeline mở từng JSONL qua HTTPS streaming bằng Python standard library và dừng ngay khi đủ giới hạn của category. Cách này không import `datasets`, Pandas hay NumPy nên tránh lỗi DLL native trên Windows. Dữ liệu chuẩn hóa được ghi vào `data/processed`; code train, evaluate, Streamlit và bất kỳ model local nào đọc CSV này không phụ thuộc Hugging Face hay LM Studio sau bước prepare.

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

Dashboard được tổ chức theo đúng luồng trình bày:

1. Nhập nhu cầu hiện tại bằng tiếng Việt hoặc tiếng Anh và tùy chọn một user để cá nhân hóa.
2. Hiển thị Top-5 với mức độ phù hợp, lý do chọn và breakdown điểm.
3. Chọn một kết quả để xem 5 sản phẩm cùng nhóm công năng gần nhất theo text/image signal/metadata/fused embedding.
4. Xem benchmark `popularity`, `content_based`, `two_tower`, số lượng dữ liệu và tham số model.

Điểm theo nhu cầu hiện tại dùng trọng số: khớp nhu cầu 60%, cá nhân hóa 20%, chất lượng 15% và phổ biến 5%. Đây là điểm xếp hạng được chuẩn hóa để giải thích, không phải xác suất user sẽ mua hàng.

Cũng có thể demo nhanh bằng CLI:

```bash
python3 main.py recommend \
  --query "Tôi cần tai nghe điện tử được đánh giá tốt" \
  --top-k 5
```

Để cập nhật benchmark và lưu đầy đủ hyperparameter của Two-Tower:

```bash
python3 main.py train --epochs 3 --negative-samples 2 --dim 48 --learning-rate 0.04
python3 main.py evaluate --top-k 5
```

## Bổ sung ảnh catalog thật

Ảnh trong raw review là ảnh tùy chọn do người mua đăng nên thường bị thiếu. Ảnh sản phẩm chính nằm trong item metadata và được ghép bằng `parent_asin`. Với dataset Fashion local hiện tại, chạy trên Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe main.py enrich-images --categories Amazon_Fashion
```

Với bốn domain Hugging Face:

```powershell
.\.venv\Scripts\python.exe main.py enrich-images `
  --categories Automotive Electronics Health_and_Household Beauty_and_Personal_Care
```

Lệnh đọc metadata theo streaming, ưu tiên ảnh biến thể `MAIN` có độ phân giải cao, cập nhật `products.csv`, rồi tự rebuild text/image/metadata/fused embeddings. Quá trình dừng sớm nếu đã tìm thấy toàn bộ `parent_asin`. Một số item nguồn vốn không có metadata hoặc không có ảnh nên độ phủ thực tế có thể thấp hơn 100%; recommender sẽ ưu tiên sản phẩm có ảnh và chỉ dùng item thiếu ảnh khi không đủ Top-K.

## Ghi chú dữ liệu

Lệnh `prepare` mặc định đọc phần review thô. Với Hugging Face, `category` lấy trực tiếp từ tên domain; với file Fashion local, category con vẫn được suy luận bằng keyword. Trước khi chạy `enrich-images`, store/price/business context được sinh ổn định theo `product_id` và ảnh lấy từ review nếu có. Sau enrichment, title/store/price/ảnh được thay bằng item metadata thật khi nguồn cung cấp bản ghi tương ứng.

LM Studio không bắt buộc cho pipeline hiện tại. Nếu dùng model/embedding từ LM Studio sau này, nên giữ bước ingest này độc lập và đọc các CSV đã chuẩn hóa trong `data/processed` qua endpoint OpenAI-compatible của LM Studio.

Mốc 100.000 là giới hạn tối đa, không phải yêu cầu phải tải hết file nguồn. Với máy ít RAM, bắt đầu từ 1.000–10.000 mỗi category rồi tăng dần.
