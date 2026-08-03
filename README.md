# Personalized Shopping Multimodal Recommender

MVP xử lý [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/), tạo dữ liệu cho user/product/session/business context, tạo embedding text/image/metadata, chạy baseline popularity/content-based, train two-tower nhẹ, đánh giá Top-K và mở demo Streamlit.

Project hiện cũng có một pipeline NLP độc lập cho đề tài **Intent-Aware Semantic
Search for Personalized Product Discovery**. Pipeline mới tái sử dụng catalog và
interactions cũ, đồng thời bổ sung intent detection, entity extraction, query
rewriting, dense/lexical retrieval, intent router, personalized ranking, audit và
retrieval evaluation. Các lệnh cũ vẫn được giữ nguyên.

## NLP semantic-search MVP

Sáu intent được hỗ trợ:

| Intent | Chiến lược |
| --- | --- |
| `product_search` | Hybrid retrieval và metadata filter |
| `need_based_search` | Query expansion/rewrite rồi semantic retrieval |
| `similar_product_search` | Item-to-item similarity |
| `personalized_recommendation` | Tăng trọng số user profile |
| `availability_check` | Filter theo stock thật nếu có; hiện demo dùng inventory proxy |
| `comparison` | Trả một nhóm sản phẩm ngắn để so sánh |

Entity extractor hỗ trợ tiếng Việt và tiếng Anh với các trường `category`, `brand`,
`color`, `size`, `material`, `feature`, `purpose`, `min_price` và `max_price`. Ranking
theo intent kết hợp semantic, lexical, entity match, user preference, quality,
popularity và availability; trọng số nằm trong `ProjectConfig`.

`business_context.csv` hiện sinh `inventory_score` xác định từ hash để demo ranking;
đây **không phải tồn kho Amazon thời gian thực**. UI, audit và qrels ghi rõ nguồn
`business_context_inventory_proxy`. Muốn dùng production phải ingest stock thật và
thay `availability_source`. Giá catalog hiện là USD; truy vấn VND dùng tỷ giá cấu hình
trong `ProjectConfig.currency_rates_to_catalog`, không phải tỷ giá live.

### Chạy workflow NLP trên Windows CMD

```cmd
cd /d D:\Backup\Code\DeepLearning
.venv\Scripts\python.exe main.py prepare --source huggingface --limit-per-category 1000
.venv\Scripts\python.exe main.py prepare-queries --category Electronics --max-products 500
.venv\Scripts\python.exe main.py train-intent
.venv\Scripts\python.exe main.py evaluate-intent
.venv\Scripts\python.exe main.py index-semantic --category Electronics --max-products 10000
.venv\Scripts\python.exe main.py pool-qrels --top-k 10 --qrels-per-query 50
.venv\Scripts\python.exe main.py audit-search --query "tai nghe chống ồn dưới $100" --top-k 5
```

Catalog cũ chưa có `data/processed/dataset_manifest.json` phải chạy lại `prepare`
trước khi index chính thức. Manifest xác nhận split theo timeline và product/user
feature chỉ dùng train. `--allow-legacy-catalog` chỉ dành cho smoke test, không được
chấp nhận ở benchmark chính thức.

`prepare-queries` tạo:

- `data/queries/intent_queries.csv`: dataset sáu intent với split train/validation/test.
- `data/queries/qrels.csv`: relevance candidate tự động.
- `data/queries/qrels_review.csv`: hàng validation/test cần review thủ công.

Sau khi có index, `pool-qrels` lấy hợp của đúng bốn cấu hình sẽ được benchmark ở
độ sâu đã chọn. `qrels-per-query` phải đủ chứa `4 × top-k + 1`. Các judgment đã có
`reviewed=true` được giữ lại khi chạy lại `prepare-queries` hoặc `pool-qrels` nếu
cặp query–product vẫn còn.

Benchmark chính thức mặc định chỉ dùng qrels validation/test đã có
`reviewed=true`, nhằm tránh coi nhãn sinh tự động là ground truth. Sau khi review:

Mở `data/queries/qrels_review.csv`, kiểm tra từng cặp query–product, sửa
`relevance` về `0`, `1` hoặc `2` và đổi `reviewed` thành `true`. Evaluation tự động
overlay những hàng đã duyệt lên qrels bootstrap; không cần sửa `qrels.csv`.
Một query chỉ được đưa vào benchmark chính thức khi **toàn bộ** candidate của query
đã được duyệt. Với personalized query, cột `profile_context` cung cấp tín hiệu train
của đúng user để annotator đánh giá.

```cmd
.venv\Scripts\python.exe main.py evaluate-search --top-k 5
.venv\Scripts\python.exe main.py ablation --top-k 5
```

Chỉ để smoke-test pipeline trước khi annotation, có thể dùng
`--allow-unreviewed-qrels`. Báo cáo so sánh bốn cấu hình TF-IDF, dense semantic,
semantic + intent và semantic + intent + personalization được lưu trong
`outputs/reports/search_metrics.json`.
`top-k` lúc evaluate không được lớn hơn độ sâu đã ghi trong
`qrels_pool_manifest.json`. Uplift personalization được tính riêng trên query của
đúng user, với profile từ train và target từ temporal holdout; query không chứa tên
sản phẩm target.

### Backend semantic đầy đủ và fallback offline

Không cài thêm dependency, `IntentClassifier`, `DenseTextEncoder` và `VectorIndex`
vẫn chạy bằng fallback Python thuần để test và phát triển offline. Dense fallback là
feature hashing deterministic, **không phải Sentence Transformer thật**; vector index
fallback dùng exact cosine thay vì FAISS.
Lexical index được lưu dưới dạng sparse inverted index, không tạo ma trận đặc
`số sản phẩm × vocabulary` trong RAM.

Để chạy đúng stack dense semantic + FAISS của đề tài:

```cmd
.venv\Scripts\python.exe -m pip install -r requirements-nlp.txt
.venv\Scripts\python.exe main.py index-semantic --category Electronics --max-products 10000 --dense-backend sentence-transformers --allow-model-download
```

`index-semantic` là tác vụ dài và tạo artifact mới. Có thể đặt `--max-products 0`
để index toàn bộ domain đã chọn khi máy đủ RAM và đã cài FAISS; exact fallback lớn
bị chặn an toàn. Backend và verification vector của model được khóa trong artifact.
Manifest checksum buộc rebuild nếu catalog, encoder hoặc index thay đổi giữa chừng.

Streamlit tự phát hiện NLP artifacts và cho phép chọn hoặc so sánh trên cùng truy vấn:

- Keyword Search (TF-IDF).
- Dense Semantic Search.
- Intent-Aware Personalized Search.

Nếu artifacts chưa có hoặc không đồng bộ với catalog, app vẫn chạy recommender cũ và
hiển thị hướng dẫn tạo lại index.

Pipeline hỗ trợ hai nguồn mà không thay đổi các bước train/evaluate phía sau:

- `local`: file `dataset/Amazon_Fashion.jsonl` đã có từ trước.
- `huggingface`: đọc streaming trực tiếp từ Hugging Face, không tải trọn bộ category.

Bốn category mặc định là `Automotive`, `Electronics`, `Health_and_Household` và `Beauty_and_Personal_Care`.

## Cấu trúc

```text
data/processed/          tables + leakage/provenance dataset_manifest.json
data/embeddings/         product text/image/metadata/fused embeddings
data/queries/            intent query dataset, qrels và annotation queue
outputs/models/          two_tower_model.json
outputs/indexes/         sparse lexical, semantic index và checksum manifest
outputs/reports/         recommendation, intent, search và ablation metrics
src/nlp/                 intent taxonomy/classifier, entity extraction, query rewriting
src/semantic_search/     encoder-facing hybrid retrieval và vector index
src/                     source code chính
main.py                  CLI workflow
```

## Cài đặt

```bash
python3 -m pip install -r requirements.txt
```

## Thiết lập trên macOS

Project chạy được trên cả Mac Apple Silicon (M1/M2/M3/M4) và Mac Intel. Pipeline
ingest/training hiện dùng Python standard library và Streamlit, không phụ thuộc NumPy,
Pandas hay `datasets`, nên không cần môi trường Conda.

### 1. Cài Python

Yêu cầu Python 3.10 trở lên. Kiểm tra trước:

```bash
python3 --version
```

Nếu chưa có Python phù hợp và đã cài [Homebrew](https://brew.sh/):

```bash
brew install python@3.12
python3.12 --version
```

Chỉ khi `pip install` báo lỗi compiler/toolchain, cài Command Line Tools của Apple:

```bash
xcode-select --install
```

### 2. Tạo môi trường và cài dependency

Mở Terminal, vào thư mục source rồi chạy:

```bash
cd /duong-dan/DeepLearning
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Mỗi lần mở Terminal mới, kích hoạt lại môi trường bằng:

```bash
cd /duong-dan/DeepLearning
source .venv/bin/activate
```

### 3. Chạy demo và CLI trên Mac

```bash
python -m streamlit run src/app/streamlit_app.py
```

Sau đó mở URL Streamlit in ra trong Terminal, thông thường là
`http://localhost:8501`.

Các lệnh workflow tương đương Windows CMD:

```bash
python main.py audit --query "Tôi cần tai nghe điện tử chất lượng tốt" --top-k 5
python main.py recommend --query "Tôi cần áo thun thoải mái" --top-k 5
python main.py train --epochs 3
python main.py evaluate --top-k 5
```

`train` và `evaluate` cần đọc embedding lớn, vì vậy hãy để trống dung lượng đĩa và
chạy chúng trong Terminal riêng. Dùng `Ctrl+C` để dừng tác vụ đang chạy.

### 4. Chuyển hoặc tạo dữ liệu trên Mac

Thư mục `data/` và `dataset/` bị Git ignore vì rất lớn; clone source không tự mang
theo catalog, embedding hay file Fashion JSONL. Có hai lựa chọn:

1. Copy các thư mục `data/` và, nếu dùng Fashion local, `dataset/Amazon_Fashion.jsonl`
   từ máy Windows/ổ cứng ngoài sang đúng thư mục project trên Mac. Có thể chạy app ngay
   nếu đã copy cả `data/processed`, `data/embeddings` và `outputs/models`.
2. Tạo một subset mới trực tiếp trên Mac qua Hugging Face streaming:

   ```bash
   python main.py prepare --source huggingface --limit-per-category 1000
   python main.py train --epochs 3
   python main.py evaluate --top-k 5
   ```

Để kiểm tra dữ liệu/model sau khi copy hoặc prepare, luôn chạy audit trước:

```bash
python main.py audit --query "Tôi cần tai nghe điện tử chất lượng tốt" --top-k 5
```

Nếu audit báo `model_catalog_coverage` thấp, catalog đã thay đổi nhưng Two-Tower chưa
được train lại; chạy `python main.py train --epochs 3` trước khi dùng benchmark làm kết luận.

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

## Chạy toàn bộ pipeline NLP

```bash
python3 main.py all --limit 5000 --query-products 300 --index-products 5000 --top-k 5 --epochs 3
```

`all` chạy theo thứ tự `prepare → prepare-queries → train-intent → index-semantic
→ pool-qrels → train → evaluate-intent`. Mặc định lệnh dừng tại trạng thái
`awaiting_qrels_review`; sau khi duyệt toàn bộ `qrels_review.csv`, chạy:

```bash
python3 main.py evaluate-search --top-k 5
python3 main.py ablation --top-k 5
```

Đây là workflow dài và ghi lại processed data, embedding, model, index và report.
Với nguồn local Fashion, để trống `--nlp-category` sẽ tự chọn category lớn nhất vừa
prepare. Thêm `--allow-unreviewed-qrels` chỉ khi cần smoke-test; khi đó `all` mới chạy
tiếp `evaluate-search → ablation`, và kết quả không được dùng làm benchmark chính thức.

Các lệnh cũ `prepare`, `train`, `evaluate`, `recommend` và `audit` vẫn hoạt động độc lập.

## Chạy từng bước

```bash
python3 main.py prepare --limit 10000
python3 main.py train --epochs 3
python3 main.py evaluate --top-k 10
python3 main.py recommend --top-k 10
```

## Audit và benchmark thuật toán

Ghi trace chi tiết cho một nhu cầu: category được nhận diện, số candidate, shortlist,
thành phần điểm, thời gian từng pha và các invariant:

```cmd
.venv\Scripts\python.exe main.py audit --query "Tôi cần tai nghe điện tử chất lượng tốt" --top-k 5
```

Log được lưu tại `outputs/reports/recommendation_audit.json` và hiển thị trong tab
Benchmark. Chạy lại benchmark mở rộng sau khi prepare/train:

```cmd
.venv\Scripts\python.exe main.py evaluate --top-k 5
```

Ngoài Precision/Recall/NDCG/MRR, báo cáo còn có catalog coverage, độ dài danh sách,
duplicate rate, seen-item leakage, out-of-catalog rate, latency và độ phủ artifact.
Leakage, duplicate và out-of-catalog nên bằng `0`; artifact coverage nên gần `1`.

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

Split interaction dùng cutoff thời gian theo từng user: mọi event sau validation/test
cutoff đều không được quay lại train. Text/rating/feature lấy từ review chỉ aggregate
trên train; metadata Amazon độc lập được giữ qua field-level provenance khi append.
Nếu append làm split thay đổi, catalog behavioral fields được rebuild lại toàn bộ.

LM Studio không bắt buộc cho pipeline hiện tại. Nếu dùng model/embedding từ LM Studio sau này, nên giữ bước ingest này độc lập và đọc các CSV đã chuẩn hóa trong `data/processed` qua endpoint OpenAI-compatible của LM Studio.

Mốc 100.000 là giới hạn tối đa, không phải yêu cầu phải tải hết file nguồn. Với máy ít RAM, bắt đầu từ 1.000–10.000 mỗi category rồi tăng dần.
