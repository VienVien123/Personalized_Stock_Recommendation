# Pipeline khuyến nghị — giải thích từng bước

| Bước | File code | Tài liệu | Làm gì |
|---|---|---|---|
| S1 | [src/s1_etl.py](../src/s1_etl.py) | [s1_etl.md](s1_etl.md) | Đọc 6 file CSV, làm sạch, ghi vào database |
| S2 | [src/s2_state.py](../src/s2_state.py) | [s2_state.md](s2_state.md) | Tại mỗi ngày, mô tả "khách này đang như thế nào" |
| S3 | [src/s3_graph.py](../src/s3_graph.py) | [s3_graph.md](s3_graph.md) | Đồ thị Khách – Mã: mã hay được mua cùng, khách giống nhau |
| S4 | [src/s4_features.py](../src/s4_features.py) | [s4_features.md](s4_features.md) | Ghép mọi thứ thành 2 bảng học: MUA và DANH MỤC |
| S5 | [src/s5_train.py](../src/s5_train.py) | [s5_train.md](s5_train.md) | Huấn luyện 2 model, đánh giá, lưu kết quả |

## Bài toán

PHS ra khuyến nghị "mua mã A", "chốt lời mã B", "cắt lỗ mã C". Hệ thống trả lời:

1. **Nhánh MUA:** hôm nay PHS khuyến nghị mua vài mã — với **từng khách**, mã nào hợp nhất?
2. **Nhánh DANH MỤC:** khách đang giữ nhiều mã — mã nào nên được nhắc **bán** trước?

PHS quyết định **mã nào** có tín hiệu; model chỉ học **khách nào hợp với mã nào**.

## Luồng dữ liệu

```
data/*.csv ─▶ S1 ─▶ S2 ─▶ S3 ─▶ S4 ─▶ S5 ─▶ app (cổng 7860)
              │     │     │     │     └ model_*.txt, summary.json
              │     │     │     └ train_buy, train_portfolio
              │     │     └ graph_pop, graph_cooc, graph_peer (Neo4j)
              │     └ pos, cust_* (trạng thái khách tại từng ngày)
              └ px, txn, research, sd (dữ liệu sạch + đặc trưng giá)
```

Mọi bảng nằm trong **một file** `artifacts/vienvien.duckdb`; bước sau đọc bảng của bước
trước, nên phải chạy đúng thứ tự S1 → S5.

## 5 khái niệm cần nắm

1. **Phiên `dn`** — mỗi ngày sàn mở cửa là một phiên, đánh số 1, 2, 3… "20 phiên sau" là
   `dn + 20`, khỏi lo cuối tuần, ngày lễ.
2. **Mốc `t`** — ngày hệ thống "đứng ở đó" để khuyến nghị. Mỗi phiên là một mốc (~1.000 mốc).
3. **Point-in-time** — tính gì "tại ngày t" thì chỉ dùng dữ liệu **trước t**. Điều kiện
   `x.d < dd.t` xuất hiện khắp nơi chính là quy tắc này.
4. **Rò rỉ tương lai** — lỡ dùng dữ liệu sau t thì model trông rất giỏi khi kiểm tra nhưng vô
   dụng ngoài đời. Mọi bước đều được thiết kế để chặn chuyện này.
5. **`HORIZON = 20`** — nhãn nhìn tới 20 phiên: "khách có mua/bán mã này trong 20 phiên tới?".

## Chia lô theo mốc (`create_by_date_chunks`)

Nhiều bảng có dạng "với **mỗi** mốc t, gom **toàn bộ** lịch sử trước t". 1.000 mốc × 490.000
giao dịch làm một lần sẽ tạo hàng trăm triệu dòng tạm và hết RAM. Hàm này trong
[src/config.py](../src/config.py) chia mốc thành từng lô **50 mốc** (`DATE_CHUNK`), tính xong lô
này mới sang lô kế, rồi nối lại — kết quả y hệt, chỉ tốn ít RAM hơn.

## Tài nguyên

Docker được cấp 18 GB (file `%USERPROFILE%\.wslconfig`). Mỗi bước chạy trong container giới
hạn 10 GB; DuckDB dùng 7 GB và 8 luồng.

## Chạy

Bấm DAG `pipeline_khuyen_nghi` trên Airflow (<http://localhost:8080>):

```
dung_app -> s1_etl -> s2_state -> s3_graph -> s4_features -> s5_train -> ghi_dau_hoan_tat -> bat_app
```

Mỗi bước là một task riêng: hỏng ở đâu thì bấm *Clear* task đó để chạy lại từ bước hỏng.
