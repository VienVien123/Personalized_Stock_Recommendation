# S4 — Ghép đặc trưng thành bảng học

File code: [src/s4_features.py](../src/s4_features.py)

Ghép các bảng của S1–S3 thành 2 bảng cho model:

| Bảng | Mỗi dòng là | Nhãn `y = 1` khi |
|---|---|---|
| `train_buy` | (t, khách, mã PHS vừa khuyến nghị MUA) | khách **mua** mã đó trong 20 phiên tới |
| `train_portfolio` | (t, khách, mã khách đang giữ) | khách **bán** mã đó trong 20 phiên tới |

```python
build_labels(con)                          # future_buy, future_sell (CHỈ là nhãn)
build_stock_at_t(con)                      # trạng thái mã + tín hiệu PHS tại t
build_graph_features(con)                  # _cooc, _peer, _recent
build_buy_candidates(con)                  # cand_pool
peer_scores(con, "_peer_buy", "cand_pool") # người giống bạn vừa làm gì (mã PHS)
peer_scores(con, "_peer_port", "pos")      # ... (mã đang giữ)
build_train_buy(con)
build_train_portfolio(con)
drop_temp_tables(con)
```

## Nhãn và trạng thái mã

- `future_buy` / `future_sell`: khách nào mua/bán mã nào trong phiên `t+1 … t+20`. Đây là
  **tương lai** → chỉ làm nhãn.
- `stock_at_t`: ghép `sd` + `sec` + `research` đúng ngày t; bỏ mã có < 5 phiên giao dịch.
  `label_complete = 0` ở các ngày cuối chưa đủ 20 phiên → không dùng để học, app vẫn chấm điểm.

## Đặc trưng đồ thị

- `_cooc`: mã ứng viên hay được mua cùng các mã khách đang giữ đến mức nào
  (giữ 25 mã đồng mua mạnh nhất mỗi mã).
- `peer_scores` — **bước thứ 3** trên đồ thị:

```
Bạn ─giống─▶ P (0,4) ─vừa MUA─▶ FPT
Bạn ─giống─▶ Q (0,3) ─vừa MUA─▶ FPT     peer_buy_score = 0,7   n_peer_buyers = 2
Bạn ─giống─▶ R (0,5) ─vừa BÁN─▶ FPT     peer_sell_score = 0,5
```

## Ứng viên MUA (`cand_pool`)

**Mỗi khách đã mở tài khoản trước t × mỗi mã PHS vừa khuyến nghị MUA hôm đó.** Khách chưa giao
dịch lần nào vẫn có mặt (gợi ý theo điểm PHS). Kèm điểm chuẩn hoá 0 → 1:

| Cột | Cách tính |
|---|---|
| `market_score_norm` | `market_score` so với min/max cùng ngày |
| `cooc_score_norm`, `buyer_pop_norm` | đồng mua, độ phổ biến (chia cho max) |
| `graph_score` | 0,75 × đồng mua + 0,25 × phổ biến |

## `train_buy` — 3 điểm tự tính

```
historical_preference = 0,4 × tỷ trọng tiền đã mua mã này
                      + 0,3 × tỷ trọng số lệnh mua mã này
                      + 0,3 × độ gần đây (mua hôm qua ≈ 1, 90 ngày trước = 0,5)

suitability_score = trung bình của: icb_share
                  + EXP(−|vol_20 mã − vol_20 khách thích| / 0,03)
                  + EXP(−|ret_20 mã − ret_20 khách thích| / 0,15)
                  + EXP(−|thanh khoản mã − thanh khoản khách thích| / 5)

personalization_score = 0,5 × graph_score + 0,3 × suitability + 0,2 × historical_preference
```

`EXP(−|lệch|/scale)`: lệch 0 → 1 (rất hợp), lệch càng lớn → càng gần 0.

## `train_portfolio`

Tín hiệu PHS thành số `research_sell_signal`: SELL **1,5** · BUY 0,5 · HOLD 0 · không có −0,5.
Không vào model — S5 dùng làm "khung", model chỉ xếp trong cùng bậc.

| Đặc trưng riêng | Ý nghĩa |
|---|---|
| `unreal_pnl_pct` | đang lãi/lỗ bao nhiêu |
| `hold_days_vs_typical` | đã giữ / thời gian khách thường giữ |
| `pnl_vs_win_habit`, `pnl_vs_loss_habit` | lãi hiện tại so với mức khách hay chốt lời / cắt lỗ |
| `weight_in_port` | mã chiếm bao nhiêu % danh mục |
| `*_rank_in_port` | hạng của mã **so với các mã khác cùng danh mục** (lãi, tuổi, tỷ trọng, đà giá, độ gần đây) |

## Phần dùng chung

Cột và phép JOIN hành vi khách giống nhau ở hai bảng nằm trong `customer_columns(k)` và
`customer_joins(k)` — thêm đặc trưng khách cho cả hai model thì chỉ sửa ở đây.

**Trước:** [S3](s3_graph.md) · **Tiếp theo:** [S5](s5_train.md)
