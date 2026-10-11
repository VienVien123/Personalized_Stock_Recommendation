# S2 — Trạng thái khách hàng tại từng ngày

File code: [src/s2_state.py](../src/s2_state.py)

Với **mỗi mốc t** và **mỗi khách**, S2 trả lời: "đứng ở ngày t, chỉ nhìn lịch sử **trước** t,
khách này là người thế nào?" — đang giữ gì, lãi/lỗ bao nhiêu, giữ lâu hay ngắn, hay mua
ngành nào, thích mã kiểu gì.

`main()` đọc như mục lục — mỗi hàm tạo một (vài) bảng:

```python
build_hold_episodes(con, txn)  # vòng mua -> bán hết          (pandas)
build_positions(con, txn)      # đang giữ gì tại t             (pandas + SQL)
build_portfolio_summary(con)   # tổng quan danh mục
build_trade_history(con)       # thói quen giao dịch
build_horizon(con)             # ngắn hạn hay dài hạn
build_sector_taste(con)        # khẩu vị ngành
build_recent_buys(con)         # chuỗi mua gần nhất
build_style(con)               # phong cách: đà giá, biến động, thanh khoản
build_stock_history(con)       # lịch sử từng cặp khách x mã
```

Hai bước đầu chỉ làm trên ~490 nghìn giao dịch nên dùng pandas. Phần còn lại tạo bảng tới
**52 triệu dòng** nên dùng SQL DuckDB, chạy theo lô mốc (`create_by_date_chunks`) với dạng:

```sql
FROM decision_dates dd
JOIN txn x ON x.d < dd.t     -- chỉ giao dịch TRƯỚC ngày t
GROUP BY dd.t, x.customer_id
```

## `hold_episode` — vòng nắm giữ

Một **vòng** = từ lúc mua (khi đang không có mã đó) tới lúc **bán hết**.

```
lệnh       đang giữ sau lệnh   mở vòng mới?        vòng
MUA 100    100                 có (trước đó 0)     1
MUA 50     150                 không               1
BÁN 150    0                   không               1   ← close_d
MUA 200    200                 có (trước đó 0)     2   ← còn mở
```

Cột: `open_d`, `close_d` (trống nếu còn giữ), `first_sell_d`, `buy_value`, `sell_value`.

## `pos_state`, `pos` — đang giữ gì tại t

`position_states()` đi qua từng lệnh:

- **MUA:** giá vốn = bình quân gia quyền, **cộng phí 0,15%** (`BUY_FEE`);
- **BÁN:** giảm số lượng, giá vốn giữ nguyên; bán hết thì về 0.

```
MUA 100 @ 20.000 → giá vốn 20.030       SL 100
MUA 100 @ 22.000 → giá vốn 21.031,5     SL 200
BÁN  50          → giá vốn 21.031,5     SL 150
```

`pos` (~8,9 triệu dòng) = (t, khách, mã đang giữ): `qty`, `avg_cost`, `mv` (giá trị theo giá
đóng cửa gần nhất), `unreal_pnl_pct` (lãi/lỗ đang treo), `pos_hold_days`, `pos_has_sold`.

## `port_pit` — tổng quan danh mục

`n_positions`, `port_value`, `avg_pnl_pct`, `port_hhi` (1 = dồn hết vào một mã, gần 0 = dàn trải).

## `cust_pit` — thói quen giao dịch

| Cột | Ý nghĩa |
|---|---|
| `n_trades`, `n_buys`, `n_sells`, `n_stocks` | số lệnh, lệnh mua/bán, số mã đã giao dịch |
| `avg_trade_value`, `days_since_first_trade`, `days_since_last_trade` | |
| `n_buys_recent`, `n_sells_recent`, `n_active_days_recent` | như trên trong **90 ngày gần nhất** — khách "nóng" hay "nguội" |
| `phs_follow_rate` | tỷ lệ lệnh mua rơi đúng lúc PHS đang khuyến nghị mã đó |

## `cust_horizon` — kỳ hạn đầu tư

Bảng **quan trọng nhất** cho model danh mục (~73% sức dự báo). Mỗi vòng nắm giữ là một bằng chứng:

| Vòng | Xếp vào |
|---|---|
| bán hết trong ≤ 30 ngày | ngắn |
| bán hết sau 30–90 ngày | trung bình |
| bán hết sau > 90 ngày, hoặc đang giữ > 90 ngày chưa bán bớt | dài |

```
≥ 3 bằng chứng, ≥ 50% ngắn (dài < 50%)  → SHORT
≥ 3 bằng chứng, ≥ 50% dài  (ngắn < 50%) → LONG
ngắn + trung bình + dài < 3            → UNRATED
còn lại: nhóm nhiều nhất               → LEAN_SHORT / MEDIUM / LEAN_LONG
```

Kèm thói quen chốt lời/cắt lỗ: `median_hold_days`, `win_rate`, `avg_win_ret` (khách thường chốt
ở +x%), `avg_loss_ret` (thường cắt ở −y%), `short_ratio`, `mid_ratio`, `long_ratio`.

## `cust_sector_recent` — khẩu vị ngành

Tỷ trọng mỗi ngành trong các lệnh mua: toàn lịch sử (`icb_share_all`), 30 / 90 ngày gần nhất,
và độ lệch `icb_recent_shift_*` = gần đây − toàn bộ (dương = đang mua ngành này nhiều hơn thói quen).

## `cust_recent_stock_seq` — chuỗi mua gần nhất

`stock_in_last_buy`, `stock_in_last_3_buys`, `stock_in_last_5_buys`: mã có nằm trong 1/3/5 lệnh
mua gần nhất không; `days_since_last_buy_stock`.

## `cust_style` — phong cách đầu tư

Lúc khách mua, mã đó đang thế nào? Trung bình có trọng số → `preferred_ret_20` (thích đuổi đà
hay bắt đáy), `preferred_vol_20` (thích mã biến động mạnh?), `preferred_log_turnover` (thích
thanh khoản cao?). Lệnh càng nhiều tiền càng nặng ký; cứ 90 ngày trọng số **giảm một nửa**.

## `cust_stock_pit` — khách × mã

Đã mua/bán mã này mấy lần, lần cuối cách đây bao lâu, lệnh cuối là MUA hay BÁN (`last_side`).

**Trước:** [S1](s1_etl.md) · **Tiếp theo:** [S3](s3_graph.md)
