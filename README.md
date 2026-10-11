# Hệ thống khuyến nghị cổ phiếu cá nhân hóa

Project kết hợp **tín hiệu nghiên cứu cổ phiếu** với **hành vi giao dịch lịch sử của
khách hàng** để tạo khuyến nghị phù hợp cho từng người tại thời điểm `t`.

Hệ thống không tự thay thế bộ phận research để dự đoán cổ phiếu nào sinh lời. Khuyến
nghị của PHS quyết định mã nào vừa có `BUY` mới, mã nào đang `HOLD` do BUY cũ còn mở,
và mã nào vừa được chốt lời/cắt lỗ (`SELL`);
model hành vi chỉ xếp mức độ phù hợp của các tín hiệu đó với từng khách hàng.

## 1. Bài toán cần giải quyết

Tại mỗi mốc quyết định `t` — mỗi phiên giao dịch — hệ thống trả lời hai câu hỏi:

1. Trong các mã research vừa đánh giá `BUY` đúng ngày `t`, mã nào phù hợp nhất với khách hàng?
2. Với các mã khách đang sở hữu, mã nào nên `BÁN`, `CÓ THỂ MUA`, `GIỮ` hoặc `CHƯA CÓ TÍN HIỆU`?

Hai nhánh được tách riêng:

| Nhánh | Candidate | Kết quả |
|---|---|---|
| Mua mới | Chỉ mã PHS mới khuyến nghị `BUY` đúng ngày `t` | Top-K mã phù hợp nhất |
| Danh mục | Chỉ mã khách đang sở hữu | BÁN, CÓ THỂ MUA, GIỮ hoặc CHƯA CÓ TÍN HIỆU |

Nguyên tắc quan trọng:

```text
Research quyết định chất lượng/tín hiệu đầu tư.
Model hành vi quyết định mức độ phù hợp với từng khách.
```

## 2. Kiến trúc tổng thể

```text
6 file CSV
   │
   ▼
[S1] ETL + đặc trưng thị trường trong DuckDB
   │
   ▼
[S2] Trạng thái khách hàng point-in-time
   │
   ▼
[S3] Neo4j: Customer – TRADED – Stock – Sector
   │
   ▼
[S4] Candidate research BUY + vị thế danh mục
   │
   ▼
[S5] LightGBM hành vi BUY/SELL + walk-forward
   │
   ▼
Gradio: TOP-K mua mới + hành động cho danh mục
```

- **DuckDB** lưu dữ liệu đã làm sạch, feature và bảng huấn luyện.
- **Neo4j** lưu quan hệ khách–cổ phiếu–ngành, tạo đặc trưng graph và tìm **khách tương tự**
  bằng phép duyệt Khách → Mã ← Khách.
- **LightGBM** học hành vi khách hàng từ các feature dạng bảng.
- **Gradio** hiển thị kết quả khuyến nghị.

LightGBM không chạy trực tiếp bên trong Neo4j. Neo4j tạo các con số như độ đồng mua,
số người mua/bán, giá trị giao dịch và danh sách khách tương tự; sau đó các cột này được
đưa vào LightGBM.

## 3. Dữ liệu đầu vào

Thư mục `data/` gồm:

| File | Vai trò |
|---|---|
| `securities_master.csv` | Danh mục mã, sàn và ngành ICB |
| `stock_market_daily.csv` | Giá và khối lượng cổ phiếu theo ngày |
| `market_index_daily.csv` | Dữ liệu VNINDEX và các chỉ số |
| `customer_profile.csv` | Loại khách và ngày mở tài khoản |
| `customer_transactions_raw.csv` | Lịch sử BUY/SELL của khách |
| `candidate_stocks_phs_skill.csv` | Nhật ký khuyến nghị PHS: mở `BUY`, đóng bằng `TAKE_PROFIT`/`CUT_LOSS` |

Hồ sơ khách không còn cột khẩu vị rủi ro và kỳ hạn tự khai. Kỳ hạn đầu tư được suy ra
từ hành vi nắm giữ thực tế (mục 6.4).

**Vị thế được dựng lại từ giao dịch.** S2 tính số lượng và giá vốn bình quân (có phí mua
`BUY_FEE` = 0,15%) sau từng lệnh, nên biết được danh mục của khách tại bất kỳ ngày nào chứ
không chỉ cuối tháng. Vì vậy không cần file snapshot danh mục (`customer_holdings.csv`).

Các cột của file khuyến nghị PHS:

| Cột | Ý nghĩa |
|---|---|
| `symbol` | Mã cổ phiếu |
| `recommendationType` | `BUY` (mở khuyến nghị), `TAKE_PROFIT` hoặc `CUT_LOSS` (đóng) |
| `recommendationDate` | Ngày ra khuyến nghị, dạng `dd/mm/yyyy` |
| `recommendationPrice` | Giá hoặc vùng giá (nghìn VND), ví dụ `20.0 - 20.5` |
| `targetPrice`, `cutLossPrice` | Giá mục tiêu và giá cắt lỗ (nghìn VND) |
| `realizedProfitLoss` | Lãi/lỗ thực hiện, chỉ có ở dòng đóng |

S1 ghép lệnh `BUY` thứ k của một mã với lệnh đóng thứ k của mã đó thành bảng
`phs_calls`, rồi dựng tín hiệu tại từng mốc `t`:

| Tín hiệu tại `t` | Điều kiện |
|---|---|
| `BUY` | PHS mở khuyến nghị mua mới đúng ngày `t` |
| `HOLD` | BUY cũ còn mở, nhưng PHS không có tín hiệu mới cho mã đó trong ngày `t` |
| `SELL` | PHS chốt lời/cắt lỗ đúng ngày `t` |
| Không có tín hiệu | Các trường hợp còn lại |

BUY cũ không được lặp lại thành khuyến nghị mua mới ở các ngày sau; nếu khách đang nắm mã đó,
nhánh danh mục sẽ hiển thị `GIỮ` cho tới khi PHS có tín hiệu mới (`BUY`, `TAKE_PROFIT` hoặc
`CUT_LOSS`). Nếu một ngày không có BUY mới, nhánh mua mới không tự gợi ý mã.

## 4. MarketScore được dùng như thế nào?

### 4.1 MarketScore là dư địa tới giá mục tiêu

PHS không cung cấp điểm số sẵn, nên project tính:

```text
MarketScore(s, t) = 100 × (TargetPrice(s) / Close(s, t) - 1)
```

Tức là phần trăm dư địa tăng còn lại từ giá đóng cửa tại `t` tới giá mục tiêu. Công thức
chỉ dùng giá tại `t` nên không rò rỉ tương lai. Chất lượng sinh lời của kết quả cuối vẫn
phụ thuộc vào chất lượng khuyến nghị PHS đầu vào.

### 4.2 Lọc candidate

Nhánh mua mới chỉ nhận mã thỏa mãn:

```text
recommendation(t, stock) = BUY
```

Nếu một mã không nằm trong danh sách `BUY` tại `t`, model hành vi không được phép đưa mã
đó vào danh sách mua, dù khách từng giao dịch mã đó nhiều lần.

### 4.3 Chuẩn hóa MarketScore

MarketScore được chuẩn hóa Min-Max riêng tại từng thời điểm:

```text
MarketScoreNorm(s, t)
    = (MarketScore(s, t) - MinScore(t))
      / (MaxScore(t) - MinScore(t))
```

Kết quả nằm trong `[0, 1]`. Nếu toàn bộ mã tại `t` có cùng điểm, mẫu số bằng 0 và code
gán `MarketScoreNorm = 0` để tránh lỗi chia cho 0.

### 4.4 Công thức điểm mua cuối

LightGBM trả về xác suất hành vi mua. Xác suất này tiếp tục được chuẩn hóa Min-Max trong
từng nhóm `khách hàng × thời điểm`:

```text
BehaviorScoreNorm(c, s, t)
    = (P_buy(c, s, t) - MinP(c, t))
      / (MaxP(c, t) - MinP(c, t))
```

Điểm xếp hạng cuối:

```text
FinalScore(c, s, t)
    = MARKET_WEIGHT × MarketScoreNorm(s, t)
      + (1 - MARKET_WEIGHT) × BehaviorScoreNorm(c, s, t)
```

Mặc định `MARKET_WEIGHT = 0.60`:

```text
FinalScore = 60% MarketScoreNorm + 40% BehaviorScoreNorm
```

### 4.5 Ví dụ tính điểm

Giả sử research có ba mã BUY:

| Mã | MarketScore | MarketScoreNorm | Xác suất model | BehaviorScoreNorm |
|---|---:|---:|---:|---:|
| FPT | 85 | 1,00 | 0,32 | 0,00 |
| MBB | 75 | 0,50 | 0,70 | 1,00 |
| HPG | 65 | 0,00 | 0,55 | 0,61 |

Với trọng số 60/40:

```text
FPT = 0,60 × 1,00 + 0,40 × 0,00 = 0,600
MBB = 0,60 × 0,50 + 0,40 × 1,00 = 0,700
HPG = 0,60 × 0,00 + 0,40 × 0,61 = 0,244
```

Thứ hạng cuối là `MBB → FPT → HPG`. MBB có MarketScore thấp hơn FPT nhưng phù hợp với
hành vi khách hơn, nên được đưa lên đầu. Tuy nhiên HPG vẫn chỉ được xét vì research đã
đánh giá HPG là `BUY`.

## 5. Các đặc trưng thị trường

Tại bước `s1_etl.py`, hệ thống tính:

### Momentum

```text
ret_N(t) = Price(t) / Price(t-N) - 1
```

Gồm:

- `ret_5`: lợi suất 5 phiên;
- `ret_20`: lợi suất 20 phiên;
- `ret_60`: lợi suất 60 phiên.

### Biến động 20 phiên

```text
r1(t)   = Price(t) / Price(t-1) - 1
vol_20  = StdDev(r1 trong 20 phiên gần nhất)
```

### Tỷ lệ phiên tăng

```text
up_ratio_20 = số phiên có r1 > 0 / số phiên quan sát trong cửa sổ 20 phiên
```

### Khối lượng và thanh khoản

```text
vol_ratio = Volume(t) / AverageVolume20(t)
turnover  = Volume(t) × Price(t)
```

Project cũng tính `fwd_ret`, `idx_fwd_ret` và `excess_fwd` để audit kết quả trong tương
lai. Các cột tương lai này bị chặn khỏi feature của LightGBM, không được dùng khi dự đoán.

## 6. Hệ thống học sở thích khách hàng thế nào?

Mọi feature hành vi tại `t` chỉ dùng giao dịch có ngày `< t`.

### 6.1 Trọng số giảm theo thời gian

Lệnh mua càng gần hiện tại có trọng số càng lớn:

```text
w = transaction_value × 0,5 ^ (days_ago / BEHAVIOR_HALF_LIFE)
```

Mặc định `BEHAVIOR_HALF_LIFE = 90` ngày. Một lệnh cách 90 ngày còn một nửa trọng số;
một lệnh cách 180 ngày còn một phần tư trọng số.

Từ đó tính phong cách trung bình có trọng số của khách:

```text
preferred_ret_20       = Σ(ret_20 × w) / Σw
preferred_vol_20       = Σ(vol_20 × w) / Σw
preferred_log_turnover = Σ(log(turnover) × w) / Σw
```

### 6.2 Mức yêu thích một mã trong lịch sử

```text
HistoricalPreference
  = 40% × (giá trị mua mã / tổng giá trị mua của khách)
  + 30% × (số lần mua mã / tổng số lần mua của khách)
  + 30% × 0,5 ^ (số ngày từ lần mua gần nhất / half-life)
```

Điểm cao khi khách từng mua mã với giá trị lớn, mua nhiều lần hoặc vừa mua gần đây.

Các trọng số đặt tay của mục 6.2, 6.3 và 7 được gom về một chỗ trong `src/config.py`
(`HISTORY_PREF_WEIGHTS`, `STYLE_SCALES`, `GRAPH_SCORE_WEIGHTS`, `PERSONALIZATION_WEIGHTS`,
`RESEARCH_SELL_SIGNAL`). Chúng chỉ tạo ra đặc trưng cho model hành vi, không quyết định
trực tiếp điểm cuối.

### 6.3 Mức phù hợp phong cách

```text
SuitabilityScore = (
    SectorShare
    + exp(-|vol_20 - preferred_vol_20| / 0,03)
    + exp(-|ret_20 - preferred_ret_20| / 0,15)
    + exp(-|log_turnover - preferred_log_turnover| / 5)
) / 4
```

Điểm cao khi mã hiện tại gần với ngành, mức biến động, momentum và thanh khoản mà khách
thường lựa chọn.

Ngoài ra model còn nhận:

- số lệnh BUY/SELL và tổng giá trị giao dịch;
- số mã từng giao dịch;
- lần giao dịch/mua/bán gần nhất;
- khách đang nắm mã hay chưa;
- số vị thế, giá trị và lãi/lỗ danh mục;
- loại khách và tuổi tài khoản.

### 6.4 Kỳ hạn đầu tư suy ra từ thời gian nắm giữ

S2 tách lịch sử giao dịch thành các **vòng nắm giữ** (`hold_episode`): một vòng bắt đầu
khi khách mua lúc đang không nắm mã đó và kết thúc khi bán hết. Tại mốc `t`, mỗi vòng đã
bắt đầu trước `t` là một bằng chứng:

| Vòng nắm giữ | Bằng chứng |
|---|---|
| Mua → bán hết trong ≤ `SHORT_HOLD_DAYS` (30 ngày) | NGẮN HẠN |
| Giữ 1–3 tháng | Chưa kết luận ở tầng 1, là nhóm "trung hạn" ở tầng 2 |
| Vị thế còn quá mới, hoặc giữ > 90 ngày nhưng đã bán bớt | Chưa kết luận |
| Giữ > `LONG_HOLD_DAYS` (90 ngày) rồi bán hết, hoặc vẫn đang giữ > 90 ngày và chưa bán bớt lần nào | DÀI HẠN |

```text
short_ratio = số bằng chứng ngắn hạn / tổng số bằng chứng
mid_ratio   = số vòng giữ 1–3 tháng  / tổng số bằng chứng
long_ratio  = số bằng chứng dài hạn  / tổng số bằng chứng
```

Mẫu số là toàn bộ bằng chứng, nên các vòng "chưa kết luận" kéo các tỷ lệ xuống. Một mã
được mua bán nhiều vòng thì mỗi vòng là một bằng chứng riêng.

Nhãn được xếp theo hai tầng:

| Tầng | Điều kiện | Nhãn |
|---|---|---|
| 1 — rõ ràng | `short_ratio ≥ HOLD_EVIDENCE_THRESHOLD` | `SHORT` |
| 1 — rõ ràng | `long_ratio ≥ HOLD_EVIDENCE_THRESHOLD` | `LONG` |
| 2 — thiên về | Nhóm ngắn hạn lớn nhất trong ba nhóm ngắn / 1–3 tháng / dài | `LEAN_SHORT` |
| 2 — thiên về | Nhóm 1–3 tháng lớn nhất, hoặc ngắn và dài ngang nhau | `MEDIUM` |
| 2 — thiên về | Nhóm dài hạn lớn nhất | `LEAN_LONG` |
| — | Ít hơn `HOLD_MIN_EVIDENCE` vòng đã kết luận được | `UNRATED` |

Tầng 2 tồn tại vì phần lớn khách không đạt ngưỡng tầng 1 không hề thiếu dữ liệu: họ có
trung vị 73 vòng nắm giữ nhưng hành vi pha trộn. Nếu chỉ dùng tầng 1, 56% khách ở mốc
cuối bị xếp "chưa đánh giá được"; với tầng 2 con số này còn 1,9%.

Nhãn được tính lại tại từng mốc `t` nên một khách có thể đổi nhóm theo thời gian (88% giữ
nguyên nhãn giữa hai mốc cuối tháng liên tiếp). Kiểm tra ngoài mẫu (nhãn tại `t`, hành vi sau `t`):

| Kỳ hạn tại `t` | Số khách ở mốc cuối | Giữ trung vị của vòng mở sau `t` | Bán vị thế trong 20 phiên tới |
|---|---:|---:|---:|
| `SHORT` | 387 | 22 ngày | 75,4% |
| `LEAN_SHORT` | 306 | 40 ngày | 51,7% |
| `MEDIUM` | 478 | 51 ngày | 44,5% |
| `LEAN_LONG` | 302 | 96 ngày | 24,7% |
| `LONG` | 488 | 135 ngày | 15,2% |
| `UNRATED` | 38 | 51 ngày | 36,8% |

### 6.5 Các đặc trưng khách hàng bổ sung

| Đặc trưng | Ý nghĩa |
|---|---|
| `median_hold_days` | Thời gian giữ trung vị của các vòng đã đóng |
| `win_rate`, `avg_realized_ret` | Tỷ lệ vòng có lãi và lãi/lỗ thực hiện trung bình |
| `avg_win_ret`, `avg_loss_ret` | Mức lãi khách thường chốt, mức lỗ khách thường chịu trước khi cắt |
| `risk_appetite_vol` | Biến động trung bình của các mã khách chọn mua, thay cho `risk_level` tự khai |
| `n_buys_recent`, `n_sells_recent`, `n_active_days_recent` | Nhịp giao dịch trong `RECENT_WINDOW` ngày gần nhất |
| `phs_follow_rate` | Tỷ lệ lệnh mua rơi vào lúc PHS đang mở khuyến nghị cho đúng mã đó |
| `icb_share_30d`, `icb_share_90d`, `icb_recent_shift_30d`, `icb_recent_shift_90d` | Khẩu vị ngành gần đây so với toàn lịch sử mua của khách |
| `stock_in_last_3_buys`, `stock_in_last_5_buys` | Mã đang xét có nằm trong các lệnh BUY gần nhất của khách không |
| `port_hhi` | Độ tập trung danh mục (1 = dồn vào một mã) |
| `pos_hold_days`, `pos_has_sold` | Vị thế đã giữ bao lâu, đã từng bán bớt chưa |
| `hold_days_vs_typical` | Tuổi vị thế so với thời gian khách thường giữ |
| `pnl_vs_win_habit`, `pnl_vs_loss_habit` | Lãi/lỗ đang treo so với mức khách thường chốt lời/cắt lỗ |

Ba đặc trưng cuối chỉ có ở nhánh danh mục. So với khi bỏ toàn bộ nhóm đặc trưng mới
(3 quý kiểm tra), AUC nhánh danh mục tăng từ 0,760 lên 0,771 và nhánh BUY từ 0,753 lên
0,757.

## 7. Đặc trưng graph từ Neo4j

### Node và edge

- `Customer`: khách hàng;
- `Stock`: cổ phiếu;
- `Sector`: ngành ICB;
- `Customer -[:TRADED]-> Stock`: giao dịch có `d`, `side`, `qty`, `value`;
- `Stock -[:IN_SECTOR]-> Sector`: quan hệ mã–ngành.

Đặc trưng đồ thị được tính tại lát cắt cuối mỗi tháng rồi gán cho mọi mốc quyết định bằng
lát cắt gần nhất `≤ t`. Trong cửa sổ `COOC_WINDOW` ngày trước lát cắt, hệ thống tính:

- `cooc_score`: số khách cùng mua hai mã;
- `n_buyers`, `buy_value`: độ phổ biến phía mua;
- `n_sellers`, `sell_value`: độ phổ biến phía bán.

Các điểm được chuẩn hóa và ghép:

```text
GraphScore = 75% × CoocScoreNorm + 25% × BuyerPopularityNorm
```

Một điểm cá nhân hóa sơ bộ cũng được tạo làm feature cho model:

```text
PersonalizationScore
    = 50% × GraphScore
      + 30% × SuitabilityScore
      + 20% × HistoricalPreference
```

### Khách tương tự

Đây là phần dùng đồ thị đúng nghĩa: một phép duyệt nhiều bước mà bảng phẳng khó diễn đạt.

```text
(Khách A) -[MUA]-> (Mã) <-[MUA]- (Khách B)        bước 1-2: ai giống ai (Neo4j)
(Khách B) -[vừa MUA / BÁN]-> (Mã X)               bước 3: họ vừa làm gì (S4)
```

1. Tại mỗi lát cắt cuối tháng, Neo4j tính độ giống Jaccard giữa mọi cặp khách theo rổ mã
   đã mua trong `COOC_WINDOW` ngày: `số mã mua chung / số mã của cả hai`. Hai khách phải
   có ít nhất `PEER_MIN_SHARED` mã chung; mỗi khách giữ `PEER_TOPK` người giống nhất.
2. S4 nối thêm bước thứ ba: trong `RECENT_WINDOW` ngày ngay trước `t`, các khách tương tự
   đó vừa mua / bán mã nào.

| Đặc trưng | Ý nghĩa |
|---|---|
| `n_peer_buyers`, `peer_buy_score` | Số khách tương tự vừa mua mã này, và tổng độ giống của họ |
| `n_peer_sellers`, `peer_sell_score` | Tương tự cho phía bán |

Khác với đặc trưng cấp khách (kỳ hạn, nhịp giao dịch), các cột này khác nhau giữa từng mã
của cùng một khách, nên giúp xếp hạng bên trong rổ của khách đó. Cả hai nhánh đều dùng,
và app hiển thị số khách tương tự vừa mua / vừa bán bên cạnh từng mã.

### Đồ thị luôn khớp dữ liệu

S3 lưu một **dấu vân tay** (băm nội dung toàn bộ giao dịch, danh mục mã và hồ sơ khách) lên
node `Meta` sau khi nạp xong. Lần chạy sau chỉ bỏ qua bước nạp khi dấu vân tay trùng khớp;
nếu dữ liệu đổi — kể cả khi số dòng giữ nguyên — hoặc lần nạp trước bị ngắt giữa chừng, đồ
thị được xoá theo lô và nạp lại.

Đây chưa phải `BehaviorScore` cuối. `BehaviorScore` cuối là xác suất do LightGBM học từ
PersonalizationScore cùng toàn bộ feature hành vi, thị trường, graph và hồ sơ khách.

## 8. Model BUY học gì?

Candidate huấn luyện là từng bộ `(t, customer, research BUY stock)`.

Nhãn:

```text
y_buy = 1 nếu khách mua mã đó trong HORIZON phiên sau t
y_buy = 0 nếu không mua
```

Mặc định `HORIZON = 20` phiên. Các cột `market_score`, `candidate_rank`,
`market_score_norm`, `target_price` và `cut_loss_price` bị loại khỏi model hành vi để tránh model trộn tín hiệu research với
sở thích khách. Chúng chỉ được ghép lại sau khi LightGBM dự đoán.

Model dùng mục tiêu `binary` và **không** cân bằng lại lớp, nên đầu ra là xác suất thật
(khả năng khách mua mã đó trong `HORIZON` phiên tới) và được hiển thị trực tiếp trong app.

Số vòng boosting không đặt tay: mỗi lần huấn luyện dùng dừng sớm trên quý cuối của tập
train, và model cuối dùng trung vị số vòng của các quý đã đánh giá.

## 9. Khuyến nghị BÁN / MUA / GIỮ

Nhánh danh mục chỉ xét mã khách đang sở hữu tại `t`. Action do research quyết định:

| Tín hiệu PHS tại `t` | Action hiển thị | Điểm tầng |
|---|---|---:|
| `SELL` (vừa chốt lời/cắt lỗ) | `BÁN` | 1,5 |
| `BUY` (mua mới đúng ngày) | `CÓ THỂ MUA` | 0,5 |
| `HOLD` (BUY cũ còn mở, chưa có tín hiệu mới) | `GIỮ` | 0,0 |
| Không có tín hiệu | `CHƯA CÓ TÍN HIỆU` | -0,5 |

Nhật ký PHS không có dòng `HOLD`; hệ thống tự suy ra `HOLD` từ BUY cũ còn hiệu lực.

Model portfolio học:

```text
y_sell = 1 nếu khách bán mã đang nắm trong HORIZON phiên sau t
```

Feature gồm lịch sử BUY/SELL, thời gian giữ, lãi/lỗ, tỷ trọng vị thế, số vị thế, hồ sơ
khách, trạng thái thị trường và hành vi của khách tương tự. Tín hiệu research và
MarketScore bị loại khỏi model hành vi bán.

Vì câu hỏi là "trong danh mục của khách này, mã nào dễ bị bán nhất", model còn nhận vị trí
của từng mã **so với các mã khác trong cùng danh mục**:

| Đặc trưng | Ý nghĩa |
|---|---|
| `pnl_vs_port_avg`, `pnl_rank_in_port` | Lãi/lỗ của mã so với trung bình và thứ hạng trong danh mục |
| `hold_days_rank_in_port` | Mã được giữ lâu hay mới so với các mã còn lại |
| `weight_rank_in_port` | Vị thế lớn hay nhỏ trong danh mục |
| `ret_20_rank_in_port` | Đà giá 20 phiên so với các mã còn lại |
| `recency_rank_in_port` | Mã vừa được giao dịch gần đây hơn các mã còn lại hay không |

Thứ tự trên giao diện luôn là:

```text
SELL > BUY > chưa có tín hiệu
```

Trong cùng một tầng, xác suất bán do model dự đoán được dùng để sắp mã nào cần xem trước. Model
không được phép đổi một tín hiệu `SELL` thành `BUY`, hoặc ngược lại.

## 10. Chống rò rỉ dữ liệu tương lai

Các biện pháp đang áp dụng:

1. Feature hành vi chỉ dùng giao dịch có `transaction_date < t`.
2. Truy vấn Neo4j cũng dùng `r.d < t`.
3. Không sử dụng `portfolio_value` và `cash_balance` cuối kỳ làm feature quá khứ.
4. Tay nghề lịch sử chỉ dùng lệnh đã hoàn tất đủ 20 phiên trước `t`.
5. `fwd_ret`, `excess_fwd` và nhãn tương lai không nằm trong feature model.
6. Mốc cuối dữ liệu chưa đủ `HORIZON` có `label_complete = 0` và bị loại khỏi train/test.
7. Đánh giá theo thời gian, không chia train/test ngẫu nhiên.
8. Có khoảng đệm `HORIZON` phiên giữa train và test: một dòng train chỉ được dùng khi cửa
   sổ nhãn của nó đã kết thúc trước mốc test đầu tiên. Nhãn của các mốc liền nhau chồng
   lên nhau, nên thiếu khoảng đệm này là rò rỉ.
9. Vị thế tại `t` được dựng từ các lệnh tới hết ngày `t`, không dùng snapshot tương lai.

Các nguyên tắc trên được kiểm tra tự động (mục 16): test rò rỉ cắt bỏ toàn bộ dữ liệu sau
một ngày `T`, chạy lại pipeline, và yêu cầu mọi đặc trưng tại các mốc `t ≤ T` giữ nguyên.

Những mốc `label_complete = 0` vẫn được giữ để app chấm điểm hiện tại; chúng chỉ không
được dùng làm nhãn huấn luyện.

## 11. Đánh giá walk-forward

Model được đánh giá mở rộng theo quý, có khoảng đệm `HORIZON` phiên giữa train và test:

```text
Quý 1..4 → test quý 5
Quý 1..5 → test quý 6
Quý 1..6 → test quý 7
...
```

### Cách đo

Mỗi **nhóm** là một cặp (khách, mốc). Thước đo xếp hạng chỉ tính trên nhóm *có thể xếp
sai*: có ít nhất một mã đúng và một mã sai. Nhóm chỉ có một ứng viên, hoặc mọi ứng viên
đều đúng, thì cách xếp nào cũng đạt 1,0 nên bị loại. Số tổng hợp là **gộp mọi nhóm của
mọi quý**, không phải trung bình các quý.

- `HitRate@1`: tỷ lệ nhóm mà mã xếp đầu đúng là mã khách mua/bán;
- `NDCG@3`, `MRR`: chất lượng thứ tự, mã đúng càng ở trên điểm càng cao;
- `Precision@1`: tính trên **mọi** nhóm — mã đứng đầu thực sự được khách hành động;
- `random HitRate@1`: kết quả kỳ vọng nếu chọn ngẫu nhiên, là mốc so sánh thấp nhất;
- `AUC`: khả năng phân biệt hành động có/không trên toàn bộ dòng.

Các mức K cố định là 1 và 3 (`EVAL_KS`), không phụ thuộc `TOPK` hiển thị trong app.

### Kết quả lần chạy gần nhất (nhịp ngày, 11/10/2026)

| Nhánh | Cách chấm điểm | HitRate@1 | NDCG@3 | MRR | Precision@1 |
|---|---|---:|---:|---:|---:|
| Danh mục | Chọn ngẫu nhiên | 0,421 | — | — | — |
| Danh mục | Chỉ tín hiệu PHS | 0,422 | 0,497 | 0,626 | 0,441 |
| Danh mục | Chỉ model hành vi | 0,505 | 0,557 | 0,684 | 0,501 |
| Danh mục | Điểm cuối (PHS trước, hành vi phá hoà) | 0,496 | 0,554 | 0,679 | 0,494 |

- Danh mục: AUC 0,767, 12 quý test, 146.904 nhóm xếp hạng được.
- MUA: AUC 0,763, 10 quý test, nhưng **không có nhóm nào xếp hạng được**: ở nhịp ngày, mỗi
  ngày PHS gần như chỉ mở một khuyến nghị MUA mới, nên mỗi khách chỉ có một ứng viên. Chỉ còn
  Precision@1 = 1,55%.

Đây là các thước đo phù hợp hành vi, không phải bằng chứng đảm bảo lợi nhuận.

## 12. Pipeline theo từng file

| Bước | File | Chức năng chính |
|---|---|---|
| S1 | `src/s1_etl.py` | CSV → DuckDB, chuẩn hóa giá, mốc quyết định, tín hiệu PHS, market feature |
| S2 | `src/s2_state.py` | Dựng lại vị thế từ giao dịch, trạng thái và sở thích khách tại từng `t` |
| S3 | `src/s3_graph.py` | Nạp Neo4j, tính đồng mua, độ phổ biến BUY/SELL và khách tương tự |
| S4 | `src/s4_features.py` | Hard-filter research BUY, tạo bảng BUY và portfolio |
| S5 | `src/s5_train.py` | Huấn luyện LightGBM và đánh giá walk-forward |
| App | `src/app.py` | Ghép điểm, hiển thị khuyến nghị bằng Gradio |
| Orchestrator | `src/pipeline.py` | Chạy tuần tự S1 → S5 |
| Cấu hình | `src/config.py` | Tham số bài toán, trọng số đặt tay; đường dẫn và Neo4j đọc từ `.env` |



## 13. Cách chạy bằng Docker

### Yêu cầu

- Docker Desktop được cấp ~18 GB RAM (Neo4j ~3,5 GB, pipeline tối đa 10 GB, Airflow 3 GB).
  WSL2 mặc định chỉ cấp 50% RAM máy; tăng bằng file `%USERPROFILE%\.wslconfig`:
  ```ini
  [wsl2]
  memory=18GB
  processors=16
  ```
  rồi chạy `wsl --shutdown` và mở lại Docker Desktop;
- đủ 6 file CSV (mục 3) trong thư mục `data/`.

### Tạo `.env`

```cmd
copy .env.example .env
```

Điền mật khẩu (ít nhất 8 ký tự). `.env` đã được `.gitignore`, không bị push lên Git.

```env
NEO4J_AUTH=neo4j/mat_khau_cua_ban
NEO4J_PASSWORD=mat_khau_cua_ban
AIRFLOW_ADMIN_USER=admin
AIRFLOW_ADMIN_PASSWORD=mat_khau_cua_ban
```

### Khởi động và chạy pipeline

Chạy từ thư mục gốc của dự án (Airflow cần `${PWD}` để gắn thư mục vào các task):

```cmd
docker compose up -d --build
docker compose exec airflow airflow dags trigger pipeline_khuyen_nghi
```

Hoặc bấm chạy DAG `pipeline_khuyen_nghi` trên giao diện Airflow. Pipeline mất khoảng 20 phút;
trong lúc đó app dừng, xong thì app tự bật lại. Chưa có kết quả thì app đứng chờ.

- Ứng dụng: <http://localhost:7860>
- Airflow: <http://localhost:8080>
- Neo4j Browser: <http://localhost:7474>

### DAG `pipeline_khuyen_nghi`

```text
dung_app -> s1_etl -> s2_state -> s3_graph -> s4_features -> s5_train
         -> ghi_dau_hoan_tat -> bat_app
```

- Mỗi bước chạy trong một container riêng tạo từ image của app; hỏng ở đâu thì bấm *Clear*
  task đó để chạy lại từ bước hỏng.
- App bị dừng trong lúc chạy (DuckDB chỉ cho một tiến trình ghi) và chỉ bật lại khi mọi
  bước xong; hỏng giữa chừng thì app nằm yên, không phục vụ kết quả dở dang.
- Airflow gắn thẳng `docker.sock` để điều khiển Docker (toàn quyền với Docker — không mở
  cổng 8080 ra Internet).

## 14. Tham số cấu hình

Tham số là hằng số trong `src/config.py`; muốn đổi thì sửa file đó rồi chạy lại pipeline.

| Tham số | Giá trị | Ý nghĩa |
|---|---:|---|
| `HORIZON` | 20 | Số phiên tương lai dùng để tạo nhãn hành vi |
| `TOPK` | 10 | Số mã mua mới hiển thị trên app |
| `MARKET_WEIGHT` | 0,60 | Trọng số MarketScore trong điểm mua cuối |
| `BEHAVIOR_HALF_LIFE` | 90 | Sau bấy nhiêu ngày, trọng số một lệnh cũ giảm một nửa |
| `RECENT_WINDOW` | 90 | Cửa sổ ngày đo nhịp giao dịch gần đây |
| `BUY_FEE` | 0,0015 | Phí mua tính vào giá vốn khi dựng lại vị thế |
| `SHORT_HOLD_DAYS` / `LONG_HOLD_DAYS` | 30 / 90 | Ngưỡng bằng chứng ngắn hạn / dài hạn |
| `HOLD_EVIDENCE_THRESHOLD` | 0,50 | Tỷ lệ bằng chứng tối thiểu để xếp `SHORT`/`LONG` |
| `HOLD_MIN_EVIDENCE` | 3 | Số vòng nắm giữ tối thiểu trước khi kết luận |
| `COOC_WINDOW` | 180 | Cửa sổ ngày tính quan hệ graph |
| `COOC_TOPN` | 25 | Số quan hệ đồng mua mạnh giữ lại mỗi mã |
| `PEER_TOPK` / `PEER_MIN_SHARED` | 20 / 2 | Số khách tương tự giữ lại; số mã chung tối thiểu |
| `PORTFOLIO_TRAIN_STRIDE` | 5 | Model danh mục học trên mỗi phiên thứ 5 (giới hạn RAM) |
| `DUCKDB_MEMORY` / `DUCKDB_THREADS` | 3GB / 4 | Giới hạn RAM và số luồng DuckDB |

## 15. Cấu trúc repository

```text
data/                    dữ liệu CSV đầu vào
src/                     pipeline (s1..s5), app, cấu hình
airflow/dags/            DAG chạy pipeline
artifacts/               DuckDB/model sinh tự động, không commit
docker-compose.yml       Neo4j + app + Airflow
Dockerfile               image chạy Python
requirements.txt         dependency runtime
requirements-dev.txt     Ruff (lint + format)
.env.example             mẫu cấu hình, không chứa mật khẩu thật
```

Kiểm tra định dạng code: `ruff check src airflow && ruff format --check src airflow`.

## 16. Giới hạn và lưu ý sử dụng
- Model học khả năng phù hợp/hành vi, không bảo đảm cổ phiếu sẽ sinh lời.
- Chất lượng đầu tư phụ thuộc trực tiếp vào khuyến nghị PHS đầu vào.
- Nếu mã đang nắm không có tín hiệu PHS tại `t`, hệ thống trả `CHƯA CÓ TÍN HIỆU` thay vì
  tự suy diễn BUY/SELL.
- Rổ BUY của PHS rất nhỏ; ở các mốc không có khuyến nghị nào mới, nhánh mua mới
  không trả kết quả. Chỉ 0,1% lệnh mua thật của khách rơi vào rổ, nên nhánh MUA trả lời
  câu hỏi "trong vài mã PHS đang khuyến nghị, mã nào hợp khách nhất" chứ không dự đoán
  khách sẽ mua gì.
- Khách hầu như không bán theo lệnh đóng của PHS: xếp tín hiệu PHS lên trước làm HitRate@1
  nhánh danh mục giảm từ 0,505 xuống 0,496. Thứ tự này là quy tắc nghiệp vụ, không phải
  lựa chọn tối ưu theo hành vi.
- Trong danh mục của một khách, model chỉ đoán đúng mã bị bán đầu tiên ở 50% số nhóm so
  với 42% khi chọn ngẫu nhiên. Phần lớn sức dự báo đến từ việc phân biệt khách hay bán với
  khách ít bán, không phải phân biệt các mã của cùng một khách.
- Backtest thực tế cần bổ sung phí giao dịch, thuế, trượt giá, thanh khoản và giới hạn
  tỷ trọng theo mã/ngành.
- Kết quả của hệ thống chỉ phục vụ nghiên cứu và hỗ trợ quyết định, không phải cam kết
  lợi nhuận hay tư vấn đầu tư tự động.

