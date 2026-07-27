# Báo cáo tiến độ Amazon Multimodal Recommender

Ngày cập nhật: 2026-07-22

## 1. Trạng thái tổng quan

Project đã đạt mức MVP demo chạy local: đọc dữ liệu, tạo catalog và interactions, nhận nhu cầu tiếng Việt, trả Top-5 có giải thích và ảnh nếu có, tìm sản phẩm tương tự, audit truy vấn và hiển thị benchmark trong Streamlit.

Phần chưa hoàn tất chủ yếu nằm ở chất lượng artifact offline: Two-Tower hiện chưa được train lại sau khi catalog tăng lên 307.637 sản phẩm, và `metrics.json` vẫn là benchmark cũ.

| Hạng mục | Trạng thái | Bằng chứng |
| --- | --- | --- |
| Streaming Amazon Reviews 2023 | Hoàn thành | Không phụ thuộc Pandas/NumPy/HF datasets |
| Bốn category chính | Hoàn thành | Automotive, Electronics, Beauty, Health |
| Fashion local append | Hoàn thành | Có 11 nhóm Fashion, gồm 5.091 sản phẩm `tops` |
| Need-based Top-5 | Hoàn thành | Có category resolution, synonym tiếng Việt và score breakdown |
| Session tìm nhiều lần | Hoàn thành | Nhu cầu mới reset; cùng nhu cầu lấy nhóm 5 tiếp theo |
| Ảnh sản phẩm | Một phần | 33.218/307.637 sản phẩm có ảnh, tương đương 10,8% |
| Similar products | Hoàn thành cho demo | Serving mode dùng text + metadata + trạng thái ảnh |
| Two-Tower | Cần train lại | Model chỉ có 3.877 item, phủ 1,26% catalog |
| Benchmark mở rộng | Code hoàn thành, chưa chạy lại | Đã thêm coverage, leakage, latency và artifact health |
| Multimodal thị giác thật | Chưa hoàn thành | Image signal hiện chưa phải CLIP/ViT embedding từ pixel |
| Streamlit demo | Hoàn thành ở mức MVP | Top-5, similar products, benchmark và audit trace |

## 2. Dữ liệu hiện tại

| Artifact | Số dòng/đối tượng | Kích thước gần đúng |
| --- | ---: | ---: |
| Users | 90.918 | 8,8 MB |
| Products | 307.637 | 136,6 MB |
| Interactions | 499.727 | 64,0 MB |
| Sessions | 336.740 | 50,2 MB |
| Product embeddings | 307.637 | 1,22 GB |
| Two-Tower items | 3.877 | 4,6 MB model JSON |

Phân bố catalog:

| Category | Products |
| --- | ---: |
| Automotive | 64.415 |
| Electronics | 59.429 |
| Beauty_and_Personal_Care | 57.316 |
| Health_and_Household | 49.675 |
| fashion_accessories | 43.001 |
| jewelry | 8.349 |
| dresses | 7.071 |
| tops | 5.091 |
| shoes | 3.025 |
| bags | 2.783 |
| bottoms | 2.024 |
| eyewear | 1.707 |
| watches | 1.563 |
| outerwear | 1.101 |
| socks | 1.087 |

## 3. Kiểm thử và audit

- 28/28 unit test pass ngày 2026-07-22.
- Audit truy vấn `Tôi cần tai nghe điện tử chất lượng tốt`:
  - 59.429 ứng viên Electronics.
  - Shortlist 1.000 sản phẩm.
  - Trả đủ 5 sản phẩm, không trùng lặp.
  - Không có seen-item leakage hoặc item ngoài catalog.
  - 5/5 kết quả khớp trực tiếp headphone/earbuds.
  - Ranking mất khoảng 370 ms sau khi recommender đã được load.
  - Công thức tổng điểm khớp score breakdown.
  - Audit tổng thể FAIL vì model catalog coverage chỉ đạt 1,26%.

Log gần nhất: `outputs/reports/recommendation_audit.json`.

## 4. Benchmark hiện tại

`outputs/reports/metrics.json` vẫn là kết quả cũ trên 213 user đánh giá:

| Model | Precision@5 | Recall@5 | NDCG@5 | MRR@5 |
| --- | ---: | ---: | ---: | ---: |
| Popularity | 0 | 0 | 0 | 0 |
| Content-based | 0,002817 | 0,014085 | 0,005980 | 0,003443 |
| Two-Tower | 0 | 0 | 0 | 0 |

Không nên dùng bảng này làm kết luận cuối vì model và benchmark chưa đồng bộ với catalog sau append Fashion. Code evaluation mới đã có thêm:

- Catalog coverage.
- Artifact catalog coverage.
- Average recommendation list size.
- Duplicate rate.
- Seen-item leakage rate.
- Out-of-catalog rate.
- Relevant-item catalog coverage.
- Latency trung bình mỗi user.

## 5. Việc ưu tiên tiếp theo

1. Train lại Two-Tower trên embedding/catalog hiện tại.
2. Chạy lại evaluate để cập nhật benchmark mở rộng.
3. Chạy audit với ít nhất một user có lịch sử và các nhu cầu Electronics, Beauty, Health, Automotive, Fashion.
4. Tăng image coverage từ 10,8%, ưu tiên sản phẩm có khả năng xuất hiện trong demo.
5. Nếu cần khẳng định multimodal thật, thay URL-hash image signal bằng embedding pixel từ CLIP/ViT trên subset ảnh local.
6. Bổ sung benchmark theo category và kịch bản cold-start nếu dùng cho báo cáo học thuật.

## 6. Lệnh xác nhận tiếp theo trên Windows CMD

Các lệnh train/evaluate dưới đây là tác vụ dài và sẽ cập nhật artifact/report:

```cmd
cd /d D:\Backup\Code\DeepLearning
.venv\Scripts\python.exe main.py train --epochs 3
.venv\Scripts\python.exe main.py evaluate --top-k 5
.venv\Scripts\python.exe main.py audit --query "Tôi cần tai nghe điện tử chất lượng tốt" --top-k 5
.venv\Scripts\python.exe -m streamlit run src\app\streamlit_app.py
```

## 7. Skill vận hành

Đã tạo skill cá nhân `amazon-recommender-ops` tại:

`~/.codex/skills/amazon-recommender-ops`

Skill hướng dẫn Codex kiểm tra project theo thứ tự dữ liệu → serving audit → artifact coverage → benchmark, tránh kết luận sai chỉ từ metric accuracy.
