# KẾ HOẠCH TRIỂN KHAI DỰ ÁN PERSONALIZED STOCK RECOMMENDATION

## 1. Mục tiêu dự án

Mục tiêu của hệ thống là xây dựng mô hình khuyến nghị cổ phiếu cá nhân hóa cho từng khách hàng.

Tại mỗi ngày \(t\), hệ thống research bên ngoài đã cung cấp một tập cổ phiếu ứng viên:

$$
C_t
$$

Hệ thống recommender không tìm cổ phiếu trên toàn thị trường mà chỉ thực hiện:

$$
\text{Personalized Ranking}(u, C_t)
$$

với \(u\) là khách hàng.

Kết quả cuối cùng là Top-K cổ phiếu phù hợp nhất với từng khách hàng:

$$
TopK(u,t)
$$

Bài toán cần trả lời câu hỏi:

> Trong các cổ phiếu đã được hệ thống research đánh giá là tiềm năng, cổ phiếu nào phù hợp nhất với hành vi và phong cách đầu tư của từng khách hàng?

---

# 2. Dữ liệu sử dụng

Dự án hiện có 7 bảng chính.

| Dataset                         | Vai trò                                      |
| ------------------------------- | -------------------------------------------- |
| `customer_transactions_raw.csv` | Lịch sử BUY/SELL của khách hàng              |
| `customer_holdings.csv`         | Danh mục khách đang nắm giữ tại các snapshot |
| `customer_profile.csv`          | Hồ sơ khách hàng                             |
| `candidate_stocks_research.csv` | Candidate stocks tại từng ngày               |
| `securities_master.csv`         | Ngành, sàn, thông tin mã cổ phiếu            |
| `stock_market_daily.csv`        | OHLCV từng cổ phiếu                          |
| `market_index_daily.csv`        | VNINDEX, VN30, HNXINDEX, UPCOMINDEX          |

Khoảng thời gian chính:

$$
2019 \rightarrow 2022
$$

---

# 3. Kiến trúc tổng thể

```text
Raw Data
   │
   ▼
Data Cleaning
   │
   ▼
Market Feature Engineering
   │
   ▼
Temporal Alignment
   │
   ├───────────────┐
   ▼               ▼
Customer       Stock/Market
Behavior       Characteristics
   │               │
   └───────┬───────┘
           ▼
 Customer Investment Style
           │
           ▼
 Customer-Stock Preference
           │
           ▼
 Candidate Stocks at t
           │
           ▼
 Personalized Ranking
           │
           ▼
         Top-K
           │
           ▼
 Temporal Backtesting
```

---

# PHASE 1 — DATA FOUNDATION

## Step 1. Data Cleaning & Normalization

### Mục tiêu

Tạo một bộ dữ liệu sạch, nhất quán và có thể join an toàn giữa các bảng.

### Công việc

Chuẩn hóa:

```text
customer_id
stock_code
exchange
date
side
```

Xử lý:

```text
duplicate
missing values
invalid price
invalid quantity
OHLC anomaly
invalid customer
invalid stock
```

Chuẩn hóa đơn vị giá cổ phiếu:

$$
MarketPrice_{VND}=MarketPrice\times1000
$$

Trong khi dữ liệu index giữ nguyên đơn vị điểm.

Kiểm tra:

$$
TransactionDate \ge AccountOpenDate
$$

và:

$$
HoldingSnapshotDate \ge AccountOpenDate
$$

### Future leakage

Không sử dụng trực tiếp các trường:

```text
portfolio_value
cash_balance
```

trong historical backtest nếu đây là snapshot cuối kỳ.

Tạo:

```text
customer_profile_static
```

chỉ chứa thông tin không thay đổi theo thời gian.

### Output

```text
cleaned/
    customer_transactions_clean
    customer_holdings_clean
    customer_profile_static
    candidate_clean
    securities_master_clean
    stock_market_daily_clean
    market_index_daily_clean
```

---

# PHASE 2 — MARKET REPRESENTATION

## Step 2. Market Feature Engineering

### Mục tiêu

Biến OHLCV thô thành các đặc điểm mô tả trạng thái của từng cổ phiếu.

### Return

Tính:

$$
Return_{1d}
$$

$$
Return_{5d}
$$

$$
Return_{20d}
$$

$$
Return_{60d}
$$

với:

$$
Return_{n}
=
\frac{P_t}{P_{t-n}}-1
$$

---

### Volatility

Tính:

```text
volatility_5d
volatility_20d
volatility_60d
```

Ví dụ:

$$
Volatility_{20}
=
Std(Return_{1d,t-19:t})
$$

Mục đích là mô tả mức độ biến động/rủi ro của cổ phiếu.

---

### Liquidity

Tính:

```text
avg_volume_5d
avg_volume_20d
volume_ratio_20d
```

Trong đó:

$$
VolumeRatio_{20}
=
\frac{Volume_t}{AvgVolume_{20}}
$$

---

### Trend

Tính:

```text
MA20
MA60
price_vs_ma20
price_vs_ma60
```

Ví dụ:

$$
PriceVsMA20
=
\frac{Close_t}{MA20}-1
$$

---

### Market context

Từ `market_index_daily` tính:

```text
market_return_1d
market_return_5d
market_return_20d
market_return_60d
market_volatility_20d
```

Benchmark:

```text
HOSE  → VNINDEX
HNX   → HNXINDEX
UPCOM → UPCOMINDEX
```

---

### Relative performance

Tính:

$$
ExcessReturn
=
StockReturn-MarketReturn
$$

và:

$$
VolatilityVsMarket
=
\frac{StockVolatility}
{MarketVolatility}
$$

### Output

```text
market_features_daily
```

Mỗi dòng:

```text
date
stock_code
return_20d
volatility_20d
avg_volume_20d
volume_ratio_20d
price_vs_ma20
market_return_20d
excess_return_20d
...
```

---

# PHASE 3 — TEMPORAL ALIGNMENT

## Step 3. Xây dựng nguyên tắc thời gian

Đây là bước chống future leakage.

Tại ngày recommendation:

$$
t
$$

chỉ được sử dụng thông tin:

$$
information < t
$$

Nếu không biết candidate được publish trước hay sau phiên giao dịch, sử dụng:

$$
MarketFeatureDate < ResearchDate
$$

thay vì:

$$
MarketFeatureDate = ResearchDate
$$

### Transactions

Chỉ dùng:

$$
TradeDate<t
$$

### Holdings

Dùng snapshot gần nhất:

$$
SnapshotDate<t
$$

### Market

Dùng phiên gần nhất trước ngày research:

$$
MarketDate<t
$$

### Candidate

Dùng:

$$
CandidateDate=t
$$

---

# PHASE 4 — CUSTOMER REPRESENTATION

## Step 4. Customer Behavior Features

Từ lịch sử giao dịch trước thời điểm \(t\), tạo:

```text
total_orders
buy_orders
sell_orders
buy_ratio
total_buy_value
total_sell_value
avg_order_value
unique_stocks
active_days
days_since_last_trade
trades_per_month
```

### Ví dụ

$$
BuyRatio
=
\frac{BuyOrders}
{BuyOrders+SellOrders}
$$

Các feature này mô tả mức độ hoạt động của nhà đầu tư.

---

## Step 5. Portfolio Behavior

Từ holdings gần nhất trước \(t\), tạo:

```text
number_of_holdings
portfolio_concentration
top1_holding_share
top3_holding_share
sector_concentration
```

Ví dụ:

$$
Top1Share
=
\frac{LargestHoldingValue}
{TotalHoldingValue}
$$

Mục đích:

> xác định khách hàng có xu hướng tập trung hay đa dạng hóa danh mục.

---

# PHASE 5 — INVESTMENT STYLE

## Step 6. Sector Preference

Từ transaction kết hợp với `securities_master`, tính:

$$
SectorPreference(u,s)
=
\frac{BuyValue_{u,s}}
{TotalBuyValue_u}
$$

Ví dụ:

```text
Customer A

Banking       45%
Technology    25%
Securities    20%
Others        10%
```

---

## Step 7. Market-characteristic Preference

Ghép mỗi BUY transaction với market features tại thời điểm trước giao dịch.

Sau đó tính:

```text
preferred_volatility
preferred_momentum
preferred_liquidity
preferred_excess_return
preferred_price_vs_ma20
```

Ví dụ:

$$
PreferredVolatility_u
=
\frac{
\sum TransactionValue\times Volatility
}{
\sum TransactionValue
}
$$

Tương tự cho momentum và liquidity.

Mục tiêu:

> Không chỉ biết khách thích mã nào, mà biết khách thường chọn cổ phiếu có đặc điểm như thế nào.

---

# PHASE 6 — CUSTOMER-STOCK PREFERENCE

## Step 8. Historical Preference Score

Với customer \(u\) và stock \(i\), tính:

### Value Share

$$
ValueShare_{u,i}
=
\frac{BuyValue_{u,i}}
{TotalBuyValue_u}
$$

### Frequency Share

$$
FrequencyShare_{u,i}
=
\frac{BuyCount_{u,i}}
{TotalBuyCount_u}
$$

### Recency

$$
Recency_{u,i,t}
=
e^{-\lambda DaysSinceLastBuy}
$$

### Preference Score

Khởi đầu:

$$
PreferenceScore
=
0.4ValueShare
+
0.3FrequencyShare
+
0.3Recency
$$

Weights sẽ được tune sau.

### Output

```text
customer_id
stock_code
value_share
frequency_share
recency
preference_score
```

---

# PHASE 7 — DEFINE CANDIDATE SET

## Step 9. Xác định stocks được phép recommend

Với bài toán recommend cổ phiếu để mua:

$$
C_t=
\{i:Recommendation_i=BUY\}
$$

Bộ dữ liệu hiện tại có khoảng:

```text
30 BUY candidates / research date
```

Mỗi model chỉ được ranking các mã trong:

$$
C_t
$$

Không recommend mã ngoài candidate.

Có thể làm experiment phụ:

```text
BUY only
vs
BUY + HOLD
```

---

# PHASE 8 — COLLABORATIVE FILTERING

## Step 10. Tạo Customer × Stock Matrix

Matrix:

```text
             AAA    ACB    FPT    MBB
CUS001       0.8    0.1    0.3    0.7
CUS002       0.0    0.9    0.6    0.4
...
```

Giá trị:

$$
PreferenceScore(u,i)
$$

---

## Step 11. Item-Based KNN

Tính similarity giữa hai cổ phiếu bằng customer preference.

Cosine similarity:

$$
sim(i,j)
=
\frac{v_i\cdot v_j}
{\|v_i\|\|v_j\|}
$$

Mục tiêu:

> Nếu khách thích ACB và những người thích ACB cũng thường thích MBB thì MBB có thể phù hợp với khách, ngay cả khi khách chưa từng mua MBB.

Output:

$$
CFScore(u,i)
$$

---

# PHASE 9 — CUSTOMER-STOCK SUITABILITY

## Step 12. Sector Match

$$
SectorMatch(u,i)
=
SectorPreference_u[Sector_i]
$$

---

## Step 13. Volatility Match

Ví dụ:

$$
VolatilityMatch
=
e^{-
\frac{
|Volatility_i-PreferredVolatility_u|
}{\sigma_v}
}
$$

---

## Step 14. Momentum Match

$$
MomentumMatch
=
e^{-
\frac{
|Momentum_i-PreferredMomentum_u|
}{\sigma_m}
}
$$

---

## Step 15. Liquidity Match

Tương tự:

$$
LiquidityMatch
=
f(
Liquidity_i,
PreferredLiquidity_u
)
$$

---

## Suitability Score

$$
SuitabilityScore
=
w_1SectorMatch
+
w_2VolatilityMatch
+
w_3MomentumMatch
+
w_4LiquidityMatch
$$

---

# PHASE 10 — BUILD TRAINING SAMPLES

## Step 16. Tạo một row cho mỗi Customer–Candidate

Đơn vị training:

$$
(customer,date,candidate\ stock)
$$

Ví dụ:

```text
CUS001
2022-08-01
MBB
```

Feature:

```text
market_score
candidate_rank

historical_preference
CF_score

sector_match
volatility_match
momentum_match
liquidity_match

customer trading features
customer portfolio features

stock market features
```

---

# PHASE 11 — GROUND TRUTH

## Step 17. Xác định khách mua gì sau recommendation

Recommendation tại ngày:

$$
t
$$

Quan sát BUY trong:

$$
t+1 \rightarrow t+5
$$

trading days.

Có thể experiment thêm:

```text
1 day
5 days
10 days
20 days
```

Nhưng bắt đầu với:

$$
h=5
$$

---

Eligible ground truth:

$$
G_{u,t}
=
FutureBuy_{u,t+1:t+5}
\cap C_t
$$

Candidate được mua:

$$
label=1
$$

Candidate không được mua:

$$
label=0
$$

---

# PHASE 12 — BASELINE MODELS

## Step 18. MarketScore Baseline

$$
Score=MarketScore
$$

Mọi khách gần như cùng ranking.

Đây là baseline quan trọng nhất.

---

## Step 19. Popularity Baseline

Ranking theo lịch sử số người mua candidate.

---

## Step 20. Historical Preference Baseline

$$
Score=PreferenceScore(u,i)
$$

---

## Step 21. Style Match Baseline

$$
Score=SuitabilityScore(u,i)
$$

Mục tiêu là biết từng nhóm signal riêng lẻ có hiệu quả hay không.

---

# PHASE 13 — HYBRID RECOMMENDER

## Step 22. Personalization Score

Kết hợp:

$$
PersonalizationScore
=
w_{CF}CFScore
+
w_{style}SuitabilityScore
+
w_{pref}HistoricalPreference
$$

---

## Step 23. Final Score

Test hai hướng.

### Model A — Pure personalization

$$
FinalScore
=
PersonalizationScore
$$

Candidate research chỉ là hard filter.

### Model B — Hybrid Research + Personalization

$$
FinalScore
=
\alpha MarketScore
+
(1-\alpha)PersonalizationScore
$$

Ví dụ test:

```text
α = 0
α = 0.2
α = 0.4
α = 0.6
α = 0.8
α = 1
```

Chọn bằng validation set.

---

# PHASE 14 — TEMPORAL SPLIT

Không random split.

Đề xuất:

```text
TRAIN
2019-01-01 → 2021-12-31

VALIDATION
2022-01-01 → 2022-06-30

TEST
2022-07-01 → 2022-12-30
```

Hoặc rolling backtest sau khi pipeline ổn.

---

# PHASE 15 — EVALUATION

## Step 24. Candidate Coverage

Trước tiên đo:

$$
CandidateCoverage
=
\frac{|FutureBuy\cap Candidate|}
{|FutureBuy|}
$$

Metric này đo chất lượng giới hạn candidate, không phải personalization.

---

## Step 25. Precision@K

$$
Precision@K
=
\frac{RelevantRecommended}
{K}
$$

---

## Step 26. Recall@K

$$
Recall@K
=
\frac{RelevantRecommended}
{EligibleFutureBuys}
$$

---

## Step 27. HitRate@K

Khách có ít nhất một recommendation đúng hay không.

---

## Step 28. NDCG@K

Đây là metric chính cho ranking.

Stock đúng ở vị trí 1 được thưởng cao hơn stock đúng ở vị trí 5.

Đánh giá:

```text
NDCG@3
NDCG@5
NDCG@10
```

---

## Step 29. MAP@K

Sử dụng thêm để đánh giá chất lượng nhiều relevant stocks.

---

# PHASE 16 — MODEL COMPARISON

Tạo bảng:

| Model                 | Precision@5 | Recall@5 | HitRate@5 | NDCG@5 |
| --------------------- | ----------: | -------: | --------: | -----: |
| MarketScore           |             |          |           |        |
| Popularity            |             |          |           |        |
| Historical Preference |             |          |           |        |
| Style Match           |             |          |           |        |
| Item-KNN              |             |          |           |        |
| Hybrid                |             |          |           |        |

Mục tiêu:

> kiểm tra personalized recommender có thực sự tốt hơn ranking research ban đầu hay không.

---

# PHASE 17 — ABLATION STUDY

Chạy:

```text
Full model
Without CF
Without Sector
Without Volatility
Without Momentum
Without Liquidity
Without Historical Preference
Without MarketScore
```

Để xác định thành phần nào đóng góp nhiều nhất.

---

# PHASE 18 — CUSTOMER SEGMENTATION

Clustering không dùng làm model chính.

Có thể dùng các feature:

```text
trading_frequency
avg_order_value
sector_concentration
portfolio_concentration
preferred_volatility
preferred_momentum
preferred_liquidity
```

Sau đó phân nhóm investor để:

```text
EDA
model interpretation
cold-start fallback
```

Ví dụ có thể xuất hiện các nhóm:

```text
low-volatility investors
momentum investors
high-turnover traders
sector-focused investors
diversified investors
```

Tên cluster chỉ được đặt sau khi phân tích dữ liệu thực tế.

---

# PHASE 19 — COLD START

## Customer nhiều history

Dùng:

```text
CF
+
Investment Style
+
Historical Preference
```

## Customer ít history

Dùng:

```text
Risk profile
+
Sector preference
+
Style
+
MarketScore
```

## Customer mới hoàn toàn

Dùng:

```text
MarketScore
+
global candidate popularity
+
risk_level
+
investment_horizon
```

---

# PHASE 20 — EXPLAINABILITY

Mỗi recommendation nên trả được lý do.

Ví dụ:

```text
MBB được xếp hạng cao vì:

- khách có tỷ trọng giao dịch ngành ngân hàng cao
- volatility của MBB gần với mức khách thường mua
- liquidity phù hợp
- MBB tương đồng với ACB/TCB mà khách từng giao dịch
- MBB đang nằm trong candidate BUY
```

---

# PHASE 21 — OUTPUT CUỐI CÙNG

Recommendation table:

```text
research_date
customer_id
stock_code

rank

market_score
cf_score
historical_preference
sector_match
volatility_match
momentum_match
liquidity_match

personalization_score
final_score

reason
```

Kết quả cuối:

```text
Top-5 / Top-10 stocks
per customer
per research date
```

---

# LỘ TRÌNH TRIỂN KHAI THỰC TẾ

## Giai đoạn 1 — Data preparation

Hoàn thành:

```text
Step 1 Clean data
Step 2 Market features
Step 3 Temporal alignment
```

Deliverable:

```text
clean datasets
market_features_daily
```

---

## Giai đoạn 2 — Customer modeling

Thực hiện:

```text
Customer behavior features
Holdings features
Sector preference
Investment style
Historical preference
```

Deliverable:

```text
customer_features_at_t
customer_stock_preferences
```

---

## Giai đoạn 3 — Baseline

Xây:

```text
MarketScore baseline
Popularity baseline
Historical preference baseline
Style match baseline
```

Deliverable:

```text
baseline evaluation table
```

---

## Giai đoạn 4 — Collaborative filtering

Xây:

```text
Customer × Stock matrix
Item similarity
Item-KNN
CFScore
```

Deliverable:

```text
CF recommendation score
```

---

## Giai đoạn 5 — Hybrid recommender

Kết hợp:

```text
CF
+
Investment Style
+
Historical Preference
+
MarketScore
```

Deliverable:

```text
final personalized score
```

---

## Giai đoạn 6 — Backtest

Chạy temporal evaluation.

Deliverable:

```text
Precision@K
Recall@K
NDCG@K
MAP@K
HitRate@K
CandidateCoverage
```

---

## Giai đoạn 7 — Research experiments

Thực hiện:

```text
weight tuning
different future horizons
BUY vs BUY+HOLD candidates
ablation study
customer segments
cold-start
```

---

# THỨ TỰ CODE TỪ THỜI ĐIỂM HIỆN TẠI

Hiện tại đang ở:

```text
[✓] Step 1 Clean data

[✓/đang làm] Step 2 Market features

[ ] Step 3 Temporal alignment

[ ] Step 4 Customer behavior features

[ ] Step 5 Customer investment style

[ ] Step 6 Customer-stock PreferenceScore

[ ] Step 7 Candidate BUY filtering

[ ] Step 8 Training/evaluation samples

[ ] Step 9 Baselines

[ ] Step 10 Item-based CF

[ ] Step 11 Suitability model

[ ] Step 12 Hybrid model

[ ] Step 13 Temporal backtest

[ ] Step 14 Metrics

[ ] Step 15 Ablation

[ ] Step 16 Cold-start

[ ] Step 17 Explainability
```