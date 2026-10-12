# S3 — Đồ thị Khách – Mã (Neo4j)

File code: [src/s3_graph.py](../src/s3_graph.py)

S2 nhìn từng khách riêng lẻ; S3 nhìn **quan hệ** giữa khách và mã — giống "người mua sản phẩm
này cũng mua…":

| Bảng | Trả lời |
|---|---|
| `graph_pop` | mã nào đang được **nhiều người** mua/bán? |
| `graph_cooc` | mã nào hay được mua **cùng nhau**? |
| `graph_peer` | khách nào **giống** khách nào? |

## Đồ thị

```
(Khách A) ─TRADED─▶ (FPT)
                      ▲
(Khách B) ─TRADED─────┘
```

Mỗi **lệnh** là một cạnh `TRADED` mang ngày `d`. Truy vấn tại ngày t chỉ đi qua cạnh `r.d < t`,
nên **một** đồ thị dùng được cho mọi mốc mà không nhìn thấy tương lai.

## Nạp đồ thị (`load_graph`)

1. `create_schema` — ràng buộc duy nhất cho node, chỉ mục theo ngày trên cạnh.
2. **Dấu vân tay** (`data_fingerprint`): băm nội dung `txn`, `sec`, `prof`. Khớp dấu đã lưu →
   **bỏ qua** nạp (tiết kiệm thời gian). Không khớp → xoá sạch rồi nạp lại.
3. `load_nodes` — Khách, Mã; `load_trades` — các cạnh `TRADED`, lô 10.000 lệnh.
4. Nạp xong mới ghi dấu vân tay — nạp dở thì lần sau tự nạp lại.

## Rút đặc trưng (`extract_features`)

Tính tại **ngày cuối mỗi tháng** (48 lát cắt), mỗi lát nhìn **180 ngày** trước đó. Mỗi ngày t
dùng lát cắt **gần nhất ≤ t** (bảng `dd_q`). `query_slice` chạy 3 truy vấn của một lát;
2 lát chạy song song (`GRAPH_WORKERS`).

### Đồng mua — `graph_cooc`

```
Khách 1, 2, 3 đều mua FPT và CMG  → co(FPT, CMG) = 3
Chỉ khách 4 mua FPT và VNM        → co(FPT, VNM) = 1 → bỏ (< 3)
```

Giữ cặp có ≥ 3 khách cùng mua, ghi cả hai chiều.

### Khách tương tự — `graph_peer` (Jaccard)

```
C mua: FPT, CMG, VNM, HPG      P mua: FPT, CMG, MWG
chung = 2 → Jaccard = 2 / (4 + 3 − 2) = 0,4
```

Cần ≥ 2 mã chung; mỗi khách giữ 20 người giống nhất. S4 đi thêm bước thứ 3: "người giống bạn
vừa mua/bán gì?" — đặc trưng **mạnh nhất** của model MUA.

**Trước:** [S2](s2_state.md) · **Tiếp theo:** [S4](s4_features.md)
