# Nhật ký thay đổi

## 2026-10-12 — gộp task ghi dấu hoàn tất vào `bat_app`

- DAG bớt task `ghi_dau_hoan_tat`: `bat_app` tự ghi `artifacts/_pipeline_done` rồi mới bật
  app. Chuỗi task còn `dung_app → s1 … s5 → bat_app`.

## 2026-10-11 — rút gọn pipeline, chỉ giữ code chạy

Đợt này gồm 4 commit (`364daa5` → `ff3f3f9`): 32 file đổi, thêm 1.593 dòng, bỏ 7.567 dòng.
**Đặc trưng đưa vào model không đổi**, trừ 3 bảng bị bỏ ở mục "Đặc trưng" bên dưới.

### Cách chạy

- App **không tự chạy pipeline** nữa. Khi chưa có kết quả, app chờ và báo
  `chưa có kết quả — app CHỜ`. Pipeline chạy bằng DAG `pipeline_khuyen_nghi` trên Airflow
  (http://localhost:8080): `dung_app → s1 … s5 → ghi_dau_hoan_tat → bat_app`.
- Tên compose project đổi thành `trade_hold` (image `trade_hold-app`, network
  `trade_hold_default`). Tên container vẫn là `vv-*`, nên container cũ của project
  `personalized_stock_recommendation` phải `docker rm` trước khi `docker compose up`.
- Airflow gắn thẳng `docker.sock`, bỏ container `vv-docker-proxy`.
- Chỉ còn nhịp ngày: bỏ `DECISION_FREQ` week/month.

### Đặc trưng và mã nguồn

- Bỏ 3 bảng đặc trưng ít tác dụng: `cust_phs_quality`, `cust_skill`,
  `cust_recent_sector_seq` (đóng góp < 2,1% cho model MUA, < 1% cho model danh mục).
- Bỏ các cột được tạo ra nhưng không nơi nào dùng (`is_outlier`, `fwd_ret`, `y_stock`,
  `candidate_rank`, `realized_pnl_pct`…), và không ghi bảng `idx` nữa.
- S1 viết lại bằng pandas, mỗi bảng một hàm; kết quả khớp bản SQL cũ.
- S2–S4 tách thành hàm nhỏ; S4 gộp phần tính điểm "khách giống nhau" vào `peer_scores()`.
- S5 bỏ audit MarketScore, quét `MARKET_WEIGHT`, sai số theo cụm (592 → 385 dòng).

### Đã bỏ

- Bộ kiểm thử và chạy thử: `scripts/verify.sh`, `verify_app.py`,
  `metric_guard.py`, `validate.py`, `compare_*.py`, `s3_graph_sql.py`, `SAMPLE_PCT`,
  `ABLATE_FEATURES`, cấu hình pytest (`pyproject.toml`, `requirements-dev.txt`).
- Các task kiểm tra của DAG: `kiem_tra_du_lieu`, `kiem_tra_he_thong`, `chot_chan_chi_so`.
- File thử nghiệm ở gốc: `test_code.ipynb`, `candidate_stocks_phs_skill.csv`.

### Tài nguyên — cần đọc trước khi chạy

| Thành phần | Trước | Nay |
|---|---|---|
| DuckDB `memory_limit` / luồng ([config.py](../src/config.py)) | 3 GB / 4 | **7 GB / 8** (ghi cứng, không đặt qua `.env`) |
| Container task của DAG | — | trần 10 GB |
| App / Airflow | 5 GB / 2 GB | 7 GB / 3 GB |
| Neo4j heap / pagecache | 1,5 GB / 0,5 GB | 2 GB / 1 GB |

**Docker cần khoảng 14–16 GB RAM.** Trên Windows, WSL2 mặc định được 50% RAM máy nên
thường đủ. Trên macOS, Docker Desktop cấp một mức cố định (có máy chỉ 7,75 GB), và khi đó
bước `s4_features` bị kill với exit 137 ngay sau khi bắt đầu. Cách sửa: Docker Desktop →
Settings → Resources → Memory → 14–16 GB → Apply & restart, rồi chạy lại DAG từ `s4_features`.

### Tài liệu

- Thêm thư mục [docs/](README.md): giải thích từng bước S1–S5.
