# S1 — Đọc và làm sạch dữ liệu

File code: [src/s1_etl.py](../src/s1_etl.py)

Đọc 6 file CSV trong `data/`, sửa chỗ "bẩn", ghi thành bảng trong DuckDB. Dữ liệu nhỏ (dưới
1 triệu dòng mỗi bảng) nên viết bằng **pandas**, mỗi bảng **một hàm**:

```
securities_master.csv          ─▶ load_securities()     ─▶ sec
stock_market_daily.csv         ─▶ load_prices()         ─▶ px
                                  trading_calendar()    ─▶ daycal, decision_dates
market_index_daily.csv         ─▶ load_index()          (VNINDEX, chỉ dùng trong S1)
customer_profile.csv           ─▶ load_profiles()       ─▶ prof
customer_transactions_raw.csv  ─▶ load_transactions()   ─▶ txn
candidate_stocks_phs_skill.csv ─▶ pair_phs_calls()      ─▶ phs_calls
                                  research_signals()    ─▶ research
                     px + idx  ─▶ price_features()      ─▶ sd
```

`main()` gọi lần lượt các hàm rồi ghi bằng `write_table()` (cột ngày pandas → kiểu `DATE` để
S2–S5 viết bằng SQL đọc được).

## 2 vấn đề của dữ liệu gốc

| Vấn đề | Cách xử lý |
|---|---|
| Ký tự BOM ở đầu file | Đọc với `encoding="utf-8-sig"` |
| File giá ghi nghìn đồng (`25.3`), file giao dịch ghi đồng (`25300`) | Nhân giá trong file giá với **1000** → mọi giá là **VND** |

## Các bảng

| Bảng | Một dòng là | Ghi chú |
|---|---|---|
| `sec` | một mã | sàn, ngành ICB (`icb_code` giữ dạng chuỗi) |
| `px` | (mã, ngày) | giá đóng cửa × 1000; bỏ dòng giá ≤ 0 (khối lượng chỉ dùng để tính `sd`) |
| `daycal` | một phiên | số thứ tự phiên `dn` = 1, 2, 3… |
| `decision_dates` | một mốc `t` | = mọi phiên |
| `prof` | một khách | loại khách, ngày mở tài khoản |
| `txn` | một lệnh | `value = qty × price` |
| `phs_calls` | một khuyến nghị PHS | mã, ngày mở, ngày đóng (giá mục tiêu / cắt lỗ nằm ở `research`) |
| `research` | (ngày, mã) | tín hiệu BUY / SELL / HOLD tại ngày đó |
| `sd` | (mã, phiên) | đặc trưng giá + kết quả 20 phiên tới |

### `phs_calls` — ghép lệnh mở với lệnh đóng

File PHS là **nhật ký sự kiện**: `BUY` mở khuyến nghị, `TAKE_PROFIT`/`CUT_LOSS` đóng nó.
**BUY thứ k của một mã được đóng bởi lệnh đóng thứ k của mã đó:**

```
FPT BUY 05/03/2021  ─┐
FPT TAKE_PROFIT 20/04/2021 ─┘ → FPT mở 05/03, đóng 20/04 (chốt lời)
FPT BUY 10/06/2021  ─┐
FPT CUT_LOSS 01/07/2021 ─┘    → FPT mở 10/06, đóng 01/07 (cắt lỗ)
```

Vùng giá `"20.0 - 20.5"` lấy điểm giữa (hàm `mid_price`) rồi × 1000.

### `research` — tín hiệu tại từng ngày t

| Tín hiệu | Khi nào |
|---|---|
| `BUY` | PHS mở khuyến nghị **đúng ngày t** (chỉ một ngày, hôm sau thành HOLD) |
| `SELL` | PHS chốt lời / cắt lỗ **đúng ngày t** |
| `HOLD` | khuyến nghị cũ còn mở, hôm nay không có gì mới |

BUY có thêm `market_score` = dư địa tới giá mục tiêu (%): giá 100.000, mục tiêu 120.000 →
`20`.

### `sd` — đặc trưng giá

| Cột | Ý nghĩa |
|---|---|
| `ret_5`, `ret_20`, `ret_60` | giá đã tăng/giảm bao nhiêu sau 5 / 20 / 60 phiên |
| `vol_20` | độ dao động 20 phiên |
| `up_ratio_20` | tỷ lệ phiên tăng trong 20 phiên |
| `vol_ratio`, `turnover` | khối lượng so với trung bình 20 phiên; giá trị giao dịch |
| `n_sessions_20` | số phiên mã thật sự có giao dịch |
| `excess_fwd` | **tương lai**: mã vượt VNINDEX bao nhiêu trong 20 phiên tới — chỉ để **chấm điểm** ở S5, không đưa vào model |

Cửa sổ tính theo **số phiên `dn`**, không theo số dòng: mã nghỉ giao dịch vài phiên thì "20
phiên trước" vẫn đúng là 20 phiên thị trường.

**Cách code tính cửa sổ (không lặp từng mã):** xoay giá thành bảng rộng — mỗi hàng một
phiên `dn`, mỗi cột một mã, ô trống là phiên mã không giao dịch — nên "k phiên trước" chỉ là
dịch k hàng:

- `ago(k)` → giá k phiên trước; phiên đó mã nghỉ thì lấy phiên có giá kế sau;
- `ahead(prices, k)` → giá sau k phiên; phiên đó không có giá thì lấy phiên gần nhất trước;
- `last20(x)` → cửa sổ từ `dn − 20` tới `dn`, bỏ qua các phiên mã không giao dịch.

**Tiếp theo:** [S2](s2_state.md)
