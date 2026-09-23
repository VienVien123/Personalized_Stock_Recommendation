"""
Bước 3 — Neo4j
Nạp đồ thị Khách - Mã - Ngành, rồi rút ra đặc trưng đồ thị THEO LÁT CẮT THỜI GIAN.

Nguyên tắc chống rò rỉ (quan trọng nhất của bước này):
  Cạnh TRADED mang sẵn thuộc tính ngày `d`. Mọi truy vấn đặc trưng đều có
  `WHERE r.d < $t`, nên đặc trưng tại t chỉ nhìn thấy lịch sử trước t.
  Không cần dựng lại đồ thị 48 lần, không cần embedding.

Đặc trưng lấy ra:
  1. cooc(a, b) — số khách cùng mua cả mã a và mã b trong cửa sổ gần đây
  2. pop(a)     — số khách đã mua mã a trong cửa sổ gần đây
Tính tại mốc QUÝ (16 mốc) rồi gán tới các mốc tháng bằng as-of join.
"""
from neo4j import GraphDatabase
import pandas as pd
from config import (connect, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
                     COOC_WINDOW, log)

BATCH = 10_000


def load_graph(drv, con):
    log("s3", "tạo ràng buộc & chỉ mục...")
    with drv.session() as s:
        s.run("CREATE CONSTRAINT cust_id IF NOT EXISTS "
              "FOR (c:Customer) REQUIRE c.id IS UNIQUE")
        s.run("CREATE CONSTRAINT stock_code IF NOT EXISTS "
              "FOR (x:Stock) REQUIRE x.code IS UNIQUE")
        s.run("CREATE CONSTRAINT sector_code IF NOT EXISTS "
              "FOR (g:Sector) REQUIRE g.code IS UNIQUE")
        s.run("CREATE INDEX traded_date IF NOT EXISTS "
              "FOR ()-[r:TRADED]-() ON (r.d)")

        n = s.run("MATCH ()-[r:TRADED]->() RETURN count(r) AS n").single()["n"]
        expected = con.execute("SELECT COUNT(*) FROM txn").fetchone()[0]
        if n == expected:
            log("s3", f"đồ thị đã có {n:,} cạnh TRADED — bỏ qua bước nạp.")
            return
        if n > 0:
            # Một lần chạy bị ngắt có thể để lại graph nạp dở. Không được coi
            # "có ít nhất một cạnh" là hoàn tất vì sẽ làm mất dữ liệu các năm sau.
            log("s3", f"graph đang nạp dở ({n:,}/{expected:,} cạnh) — làm sạch cạnh TRADED.")
            s.run("MATCH ()-[r:TRADED]->() DELETE r").consume()

        log("s3", "nạp node Khách / Mã / Ngành...")
        cust = con.execute("SELECT customer_id, customer_type, risk_level, "
                           "investment_horizon FROM prof").fetchall()
        s.run("UNWIND $rows AS r MERGE (c:Customer {id: r[0]}) "
              "SET c.type=r[1], c.risk=r[2], c.horizon=r[3]", rows=cust)

        stk = con.execute("SELECT stock_code, exchange, icb_code, icb_name "
                          "FROM sec").fetchall()
        s.run("UNWIND $rows AS r MERGE (x:Stock {code: r[0]}) "
              "SET x.exchange=r[1], x.icb=r[2]", rows=stk)

        sect = con.execute("SELECT DISTINCT icb_code, icb_name FROM sec").fetchall()
        s.run("UNWIND $rows AS r MERGE (g:Sector {code: r[0]}) SET g.name=r[1]",
              rows=sect)
        s.run("""UNWIND $rows AS r
                 MATCH (x:Stock {code:r[0]}), (g:Sector {code:r[1]})
                 MERGE (x)-[:IN_SECTOR]->(g)""",
              rows=con.execute("SELECT stock_code, icb_code FROM sec").fetchall())

        log("s3", "nạp cạnh TRADED (có mốc thời gian)...")
        rows = con.execute("""SELECT customer_id, stock_code, CAST(d AS VARCHAR),
                                     side, qty, value FROM txn ORDER BY d""").fetchall()
        for i in range(0, len(rows), BATCH):
            chunk = [{"cid": r[0], "sc": r[1], "d": r[2], "side": r[3],
                      "qty": r[4], "val": r[5]} for r in rows[i:i + BATCH]]
            s.run("""UNWIND $rows AS r
                     MATCH (c:Customer {id:r.cid}), (x:Stock {code:r.sc})
                     CREATE (c)-[:TRADED {d:r.d, side:r.side, qty:r.qty,
                                          value:r.val}]->(x)""", rows=chunk)
            if (i // BATCH) % 10 == 0:
                log("s3", f"   ... {i + len(chunk):,}/{len(rows):,} cạnh")
        log("s3", f"nạp xong {len(rows):,} cạnh.")


def extract_features(drv, con):
    """Tính đặc trưng đồ thị tại từng mốc quý — luôn có WHERE r.d < t."""
    quarters = [r[0] for r in con.execute("""
        SELECT MAX(t) FROM decision_dates
        GROUP BY DATE_TRUNC('quarter', t) ORDER BY 1""").fetchall()]
    log("s3", f"tính đặc trưng đồ thị tại {len(quarters)} mốc quý "
              f"(cửa sổ {COOC_WINDOW} ngày)...")

    cooc_rows, pop_rows = [], []
    with drv.session() as s:
        for q in quarters:
            t  = str(q)
            t0 = str(q - __import__("datetime").timedelta(days=COOC_WINDOW))

            pop = s.run("""
                MATCH (c:Customer)-[r:TRADED]->(x:Stock)
                WHERE r.d > $t0 AND r.d < $t
                RETURN x.code AS code,
                       count(DISTINCT CASE WHEN r.side='BUY' THEN c END) AS n_buyers,
                       sum(CASE WHEN r.side='BUY' THEN r.value ELSE 0 END) AS buy_value,
                       count(DISTINCT CASE WHEN r.side='SELL' THEN c END) AS n_sellers,
                       sum(CASE WHEN r.side='SELL' THEN r.value ELSE 0 END) AS sell_value""",
                t0=t0, t=t).data()
            pop_rows += [(t, p["code"], p["n_buyers"], p["buy_value"],
                          p["n_sellers"], p["sell_value"]) for p in pop]

            co = s.run("""
                MATCH (c:Customer)-[r1:TRADED]->(a:Stock)
                WHERE r1.side='BUY' AND r1.d > $t0 AND r1.d < $t
                MATCH (c)-[r2:TRADED]->(b:Stock)
                WHERE r2.side='BUY' AND r2.d > $t0 AND r2.d < $t
                  AND elementId(a) < elementId(b)
                WITH a.code AS a, b.code AS b, count(DISTINCT c) AS co
                WHERE co >= 3
                RETURN a, b, co""", t0=t0, t=t).data()
            # đồng mua là quan hệ hai chiều -> ghi cả 2 hướng cho dễ join
            for r in co:
                cooc_rows.append((t, r["a"], r["b"], r["co"]))
                cooc_rows.append((t, r["b"], r["a"], r["co"]))
            log("s3", f"   {t}: {len(pop)} mã, {len(co):,} cặp đồng mua")

    # Đăng ký DataFrame rồi bulk-copy vào DuckDB. executemany từng dòng với hơn
    # 600 nghìn cặp đồng mua từng làm bước này im lặng hàng chục phút.
    pop_df = pd.DataFrame(pop_rows, columns=[
        "qt", "stock_code", "n_buyers", "buy_value", "n_sellers", "sell_value"
    ])
    cooc_df = pd.DataFrame(cooc_rows, columns=["qt", "a", "b", "co"])
    con.register("_graph_pop_df", pop_df)
    con.register("_graph_cooc_df", cooc_df)
    con.execute("""
        CREATE OR REPLACE TABLE graph_pop AS
        SELECT CAST(qt AS DATE) AS qt, CAST(stock_code AS VARCHAR) AS stock_code,
               CAST(n_buyers AS BIGINT) AS n_buyers, CAST(buy_value AS DOUBLE) AS buy_value,
               CAST(n_sellers AS BIGINT) AS n_sellers, CAST(sell_value AS DOUBLE) AS sell_value
        FROM _graph_pop_df
    """)
    con.execute("""
        CREATE OR REPLACE TABLE graph_cooc AS
        SELECT CAST(qt AS DATE) AS qt, CAST(a AS VARCHAR) AS a,
               CAST(b AS VARCHAR) AS b, CAST(co AS BIGINT) AS co
        FROM _graph_cooc_df
    """)
    con.unregister("_graph_pop_df")
    con.unregister("_graph_cooc_df")

    # gán mốc quý gần nhất <= t cho từng mốc tháng (as-of, giữ point-in-time)
    con.execute("""
    CREATE OR REPLACE TABLE dd_q AS
    SELECT d.t, (SELECT MAX(qt) FROM graph_pop g WHERE g.qt <= d.t) AS qt
    FROM decision_dates d;
    """)
    log("s3", f"graph_pop {len(pop_rows):,} dòng | graph_cooc {len(cooc_rows):,} dòng")


def main():
    con = connect()
    drv = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    drv.verify_connectivity()
    load_graph(drv, con)
    extract_features(drv, con)
    drv.close()
    con.close()
    log("s3", "xong.")


if __name__ == "__main__":
    main()
