# S5 — Huấn luyện và đánh giá

File code: [src/s5_train.py](../src/s5_train.py)

Huấn luyện 2 model **LightGBM**:

| Model | Học từ | Dự đoán |
|---|---|---|
| `buy` | `train_buy` | xác suất khách **mua** một mã PHS vừa khuyến nghị |
| `portfolio` | `train_portfolio` | xác suất khách **bán** một mã đang giữ |

Model hành vi **không** được thấy điểm của PHS (`BUY_RESEARCH_COLS`,
`PORTFOLIO_RESEARCH_COLS` bị loại) — hai phần độc lập rồi mới ghép ở điểm cuối.

## 1. Đọc dữ liệu (`load_table`)

- Chỉ lấy dòng đã đủ nhãn (`label_complete = 1`).
- Nhánh danh mục chỉ lấy **mỗi phiên thứ 5** (8,7 triệu → ~1,7 triệu dòng): các ngày liền nhau
  gần như giống hệt nhau.
- Số thực `float32`, chuỗi `category` → ~0,5 GB thay vì 2 GB.
- Sắp cố định theo (t, khách, mã); thêm `g` = mã nhóm (khách, ngày), `q` = quý.

## 2. Đánh giá "đi tới" theo quý (`walk_forward`)

Không chia ngẫu nhiên — giả lập đúng ngoài đời: học quá khứ, kiểm tra tương lai.

```
Q1 Q2 Q3 Q4 │ Q5          ← học Q1–Q4, kiểm tra Q5
Q1 Q2 Q3 Q4 Q5 │ Q6       ← học Q1–Q5, kiểm tra Q6 ...
```

- Cần ≥ 4 quý để học.
- **Khoảng đệm 20 phiên** (`embargoed`): dòng train chỉ dùng nếu nhãn của nó đã kết thúc trước
  ngày đầu quý test.
- **Số vòng học** (`pick_rounds`): tách quý cuối của tập train làm kiểm tra phụ, học tối đa 400
  vòng, dừng sớm nếu 30 vòng liền không tiến bộ.

## 3. Điểm cuối (`final_score`)

**MUA** — `0,6 × market_score_norm + 0,4 × behavior`:

| Mã | PHS | model | điểm cuối |
|---|---|---|---|
| FPT | 1,0 | 0,2 | **0,68** |
| HPG | 0,5 | 0,9 | **0,66** |

**DANH MỤC** — `research_sell_signal + 0,49 × behavior`. Các bậc PHS cách nhau ≥ 0,5 mà phần
model tối đa 0,49 → **model không lật được thứ tự PHS**, chỉ xếp các mã cùng bậc (ví dụ 3 mã
đều HOLD → nhắc bán mã nào trước).

## 4. Thước đo (`ranking_metrics`)

Mỗi nhóm (khách, ngày) có vài ứng viên được xếp hạng:

| Thước đo | Ý nghĩa |
|---|---|
| `AUC` | 0,5 = đoán bừa, 1 = hoàn hảo |
| `hit@1` | mã xếp đầu có đúng là mã khách mua/bán không |
| `ndcg@3`, `mrr` | mã đúng đứng càng cao điểm càng cao |
| `precision@1` | như `hit@1` nhưng tính trên **mọi** nhóm |
| `excess@1` | mã xếp đầu thắng VNINDEX bao nhiêu |
| `random_hit@1` | `hit@1` nếu xếp ngẫu nhiên — mốc thấp nhất |

`hit@k`, `ndcg`, `mrr` chỉ tính trên nhóm **có cả mã đúng lẫn mã sai** — nhóm 1 ứng viên xếp
kiểu gì cũng đúng, tính vào sẽ đẹp giả tạo. Mỗi thước đo tính cho 3 cách: `baseline` (chỉ
PHS), `behavior` (chỉ model), `final` (ghép).

## 5. Lưu (`save`)

Model cuối học trên **toàn bộ** dữ liệu, số vòng = trung vị các lượt walk-forward.

| File trong `artifacts/` | Nội dung |
|---|---|
| `model_buy.txt`, `model_portfolio.txt` | model — app nạp để chấm điểm |
| `meta_*.json` | danh sách đặc trưng, giá trị cột phân loại, số vòng |
| `importance_*.csv` | độ quan trọng từng đặc trưng |
| `eval_*.csv` | kết quả từng quý |
| `summary.json` | tóm tắt — app hiển thị ở tab "Hiệu quả model" |

## Kết quả

**Danh mục:** AUC 0,767 · 146.904 nhóm · HitRate@1: ngẫu nhiên 0,421 · chỉ PHS 0,422 ·
**model 0,505** · điểm cuối 0,495. Model chọn đúng mã khách sẽ bán ~50% so với 42% ngẫu nhiên;
mạnh nhất là kỳ hạn đầu tư (`short_ratio`, `investment_horizon`, `long_ratio`).

**MUA:** AUC 0,763, nhưng **không nhóm nào xếp hạng được** — mỗi ngày PHS gần như chỉ mở 1
khuyến nghị nên mỗi khách chỉ có 1 ứng viên. Mạnh nhất là `peer_buy_score`.

**Trước:** [S4](s4_features.md) · **Về:** [Tổng quan](README.md)
