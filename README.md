# Khuyến nghị cổ phiếu cá nhân hoá

PHS quyết định **mã nào** có tín hiệu (mua / chốt lời / cắt lỗ). Hệ thống học từ lịch sử
giao dịch để biết **khách nào hợp với mã nào**, tại từng phiên `t`.

| Nhánh | Ứng viên | Kết quả |
|---|---|---|
| **MUA** | Mã PHS vừa khuyến nghị `BUY` đúng ngày `t` | Top-`TOPK` mã hợp khách nhất |
| **DANH MỤC** | Mã khách đang giữ | `BÁN` / `CÓ THỂ MUA` / `GIỮ` / `CHƯA CÓ TÍN HIỆU` |

- Nhánh MUA: `điểm cuối = 60% MarketScore + 40% điểm hành vi`. MarketScore là dư địa từ giá
  hiện tại tới giá mục tiêu của PHS. Cả hai điểm được chuẩn hoá về [0, 1].
- Nhánh DANH MỤC: tín hiệu PHS xếp trước (`SELL > BUY > HOLD > không có`). Model chỉ sắp
  thứ tự các mã cùng tín hiệu và không lật được tín hiệu PHS.

## Pipeline

```
data/*.csv → S1 → S2 → S3 → S4 → S5 → app (Gradio)
```

| Bước | File | Làm gì |
|---|---|---|
| S1 | [src/s1_etl.py](src/s1_etl.py) | Đọc CSV, làm sạch, ghép khuyến nghị PHS, tính đặc trưng giá |
| S2 | [src/s2_state.py](src/s2_state.py) | Dựng lại vị thế từ giao dịch; trạng thái, phong cách, kỳ hạn của khách tại từng `t` |
| S3 | [src/s3_graph.py](src/s3_graph.py) | Neo4j: mã hay được mua cùng, độ phổ biến, khách tương tự |
| S4 | [src/s4_features.py](src/s4_features.py) | Ghép thành 2 bảng học: `train_buy`, `train_portfolio` |
| S5 | [src/s5_train.py](src/s5_train.py) | Huấn luyện 2 model LightGBM, đánh giá walk-forward theo quý |

Mỗi bước được giải thích kỹ trong [docs/](docs/README.md); các thay đổi gần đây ghi ở
[docs/CHANGELOG.md](docs/CHANGELOG.md).

Mọi đặc trưng tại `t` chỉ dùng dữ liệu trước `t`. Train và test cách nhau `HORIZON`
phiên, và không chia ngẫu nhiên.

## Dữ liệu (`data/`)

| File | Nội dung |
|---|---|
| `securities_master.csv` | Mã, sàn, ngành ICB |
| `stock_market_daily.csv` | Giá, khối lượng theo ngày |
| `market_index_daily.csv` | VNINDEX và các chỉ số |
| `customer_profile.csv` | Loại khách, ngày mở tài khoản |
| `customer_transactions_raw.csv` | Lịch sử BUY/SELL của khách |
| `candidate_stocks_phs_skill.csv` | Khuyến nghị PHS: `BUY`, `TAKE_PROFIT`, `CUT_LOSS` |

## Chạy

**RAM:** Docker cần khoảng **14–16 GB**, vì riêng DuckDB đã dùng 7 GB và 8 luồng.
- Windows: WSL2 mặc định được 50% RAM máy. Thiếu thì đặt `memory=16GB` trong `%USERPROFILE%\.wslconfig`, rồi chạy `wsl --shutdown`.
- macOS: vào Docker Desktop → Settings → Resources → Memory. Nếu thiếu RAM, bước `s4_features` sẽ chết với exit 137.

```bash
cp .env.example .env        # điền mật khẩu Neo4j và Airflow (≥ 8 ký tự)
docker compose up -d --build
docker compose exec airflow airflow dags trigger pipeline_khuyen_nghi
```

| Dịch vụ | Địa chỉ |
|---|---|
| App | http://localhost:7860 |
| Airflow | http://localhost:8080 |
| Neo4j | http://localhost:7474 |

DAG `pipeline_khuyen_nghi` chạy `dung_app → s1 … s5 → bat_app`, mỗi bước
trong một container riêng. Trong lúc chạy, app tạm dừng vì DuckDB chỉ cho một tiến trình
ghi; khi tất cả các bước xong thì app tự bật lại. Nếu hỏng ở bước nào, bấm *Clear* bước đó
để chạy lại từ đó.

## Tham số chính ([src/config.py](src/config.py))

| Tham số | Giá trị | Ý nghĩa |
|---|---:|---|
| `HORIZON` | 20 | Số phiên nhìn tới để gắn nhãn mua/bán |
| `TOPK` | 10 | Số mã MUA hiển thị |
| `MARKET_WEIGHT` | 0,60 | Trọng số MarketScore trong điểm MUA |
| `BEHAVIOR_HALF_LIFE` | 90 | Sau bấy nhiêu ngày, trọng số của một lệnh cũ giảm một nửa |
| `COOC_WINDOW` | 180 | Cửa sổ ngày tính quan hệ đồ thị |
| `DUCKDB_MEMORY` / `DUCKDB_THREADS` | 7GB / 8 | RAM và số luồng DuckDB |

Sửa tham số trong `config.py` rồi chạy lại DAG. `.env` chỉ chứa đường dẫn và mật khẩu.

## Kết quả (nhịp ngày, 11/10/2026)

| Nhánh | AUC | HitRate@1 | Ngẫu nhiên |
|---|---:|---:|---:|
| DANH MỤC — model hành vi | 0,767 | 0,505 | 0,421 |
| DANH MỤC — điểm cuối (PHS trước) | | 0,496 | 0,421 |
| MUA | 0,763 | — | — |

- Nhánh MUA không đo được xếp hạng: mỗi ngày PHS gần như chỉ mở 1 mã BUY mới, nên mỗi khách
  chỉ có 1 ứng viên.
- Đặt tín hiệu PHS lên trước làm HitRate@1 giảm nhẹ (0,505 → 0,496), vì khách hầu như không
  bán theo lệnh đóng của PHS. Đây là quy tắc nghiệp vụ, không phải lựa chọn tối ưu.
- Đây là thước đo mức phù hợp với hành vi khách, không bảo đảm lợi nhuận và không phải tư vấn
  đầu tư.
