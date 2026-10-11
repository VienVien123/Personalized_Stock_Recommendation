"""
Bước 4 — Ghép tín hiệu nghiên cứu với hành vi khách hàng.

Nhánh MUA
---------
Candidate là các mã PHS vừa mở khuyến nghị BUY đúng ngày t. Model không tự phát
minh một rổ cổ phiếu khác, và BUY cũ không được lặp lại thành mua mới ở ngày sau.

Nhánh DANH MỤC
--------------
Candidate là đúng các mã khách đang sở hữu. Tín hiệu của PHS (BUY mới, HOLD khi
BUY cũ còn mở, hoặc chốt lời/cắt lỗ -> SELL) quyết định hành động nghiệp vụ; model học xác suất
khách bán để xếp mức ưu tiên, không thay thế tín hiệu nghiên cứu.

Khách tương tự (Neo4j)
----------------------
s3 cho biết ai giống ai (Khách -> Mã <- Khách). Ở đây nối thêm bước thứ ba: các
khách tương tự vừa MUA / BÁN mã nào trong RECENT_WINDOW ngày trước t. Đây là tín
hiệu khác nhau giữa các mã của CÙNG một khách, nên giúp xếp hạng trong từng khách.

Mọi đặc trưng hành vi chỉ dùng giao dịch trước t. Future BUY/SELL chỉ là nhãn.
"""

from config import (
    BEHAVIOR_HALF_LIFE,
    COOC_TOPN,
    GRAPH_SCORE_WEIGHTS,
    HISTORY_PREF_WEIGHTS,
    HORIZON,
    PERSONALIZATION_WEIGHTS,
    RECENT_WINDOW,
    RESEARCH_SELL_SIGNAL,
    STYLE_SCALES,
    connect,
    create_by_date_chunks,
    log,
)

# Các bảng theo mốc t mà cả hai bảng train đều ghép vào (xem create_by_date_chunks).
PIT_JOINS = (
    "stock_at_t",
    "port_pit",
    "cust_pit",
    "cust_skill",
    "cust_style",
    "cust_horizon",
    "cust_phs_quality",
    "cust_sector_recent",
    "cust_recent_stock_seq",
    "cust_recent_sector_seq",
    "cust_stock_pit",
)


def customer_columns(k: str) -> str:
    """Đặc trưng hành vi khách mà train_buy và train_portfolio cùng dùng.

    `k` là bí danh của bảng gốc (có t, customer_id, stock_code); cần đi kèm
    customer_joins(k) và một bảng stock_at_t bí danh `s` đã JOIN trước đó.
    """
    return f"""
           pr.customer_type,
           COALESCE(ch.investment_horizon, 'UNRATED') AS investment_horizon,
           ch.n_evidence AS n_hold_evidence,
           ch.short_ratio, ch.mid_ratio, ch.long_ratio,
           ch.median_hold_days, ch.win_rate, ch.avg_realized_ret,
           ch.avg_win_ret, ch.avg_loss_ret,
           st.preferred_vol_20 AS risk_appetite_vol,
           cu.n_buys_recent, cu.n_sells_recent, cu.n_active_days_recent,
           cu.phs_follow_rate,
           pq.phs_scored_buys, pq.phs_hit_rate, pq.phs_avg_excess,
           pq.phs_recent_scored_buys, pq.phs_recent_hit_rate,
           pq.phs_recent_avg_excess,
           csr.icb_share_all, csr.icb_share_30d, csr.icb_share_90d,
           csr.icb_value_share_90d, csr.icb_recent_shift_30d,
           csr.icb_recent_shift_90d,
           COALESCE(rss.stock_in_last_buy, 0) AS stock_in_last_buy,
           COALESCE(rss.stock_in_last_3_buys, 0) AS stock_in_last_3_buys,
           COALESCE(rss.stock_in_last_5_buys, 0) AS stock_in_last_5_buys,
           rss.days_since_last_buy_stock,
           COALESCE(rsec.same_sector_as_last_buy, 0) AS same_sector_as_last_buy,
           COALESCE(rsec.same_sector_last_3_buys, 0) AS same_sector_last_3_buys,
           COALESCE(rsec.same_sector_last_5_buys, 0) AS same_sector_last_5_buys,
           rsec.days_since_last_buy_same_sector,
           DATE_DIFF('day', pr.open_date, {k}.t) AS account_age,
           cu.n_trades, cu.n_stocks, cu.n_buys, cu.n_sells"""


def customer_joins(k: str) -> str:
    """Các bảng của S2 mà customer_columns(k) đọc, ghép đúng tại mốc t của dòng."""
    return f"""
    JOIN prof pr ON pr.customer_id = {k}.customer_id
    LEFT JOIN cust_pit cu ON cu.t = {k}.t AND cu.customer_id = {k}.customer_id
    LEFT JOIN cust_skill sk ON sk.t = {k}.t AND sk.customer_id = {k}.customer_id
    LEFT JOIN cust_style st ON st.t = {k}.t AND st.customer_id = {k}.customer_id
    LEFT JOIN cust_horizon ch ON ch.t = {k}.t AND ch.customer_id = {k}.customer_id
    LEFT JOIN cust_phs_quality pq ON pq.t = {k}.t AND pq.customer_id = {k}.customer_id
    LEFT JOIN port_pit pp ON pp.t = {k}.t AND pp.customer_id = {k}.customer_id
    LEFT JOIN cust_sector_recent csr ON csr.t = {k}.t AND csr.customer_id = {k}.customer_id
                                    AND csr.icb_code = s.icb_code
    LEFT JOIN cust_recent_sector_seq rsec ON rsec.t = {k}.t
                                         AND rsec.customer_id = {k}.customer_id
                                         AND rsec.icb_code = s.icb_code
    LEFT JOIN cust_recent_stock_seq rss ON rss.t = {k}.t AND rss.customer_id = {k}.customer_id
                                       AND rss.stock_code = {k}.stock_code
    LEFT JOIN cust_stock_pit csp ON csp.t = {k}.t AND csp.customer_id = {k}.customer_id
                                AND csp.stock_code = {k}.stock_code"""


def build_labels(con):
    """future_buy, future_sell — khách nào mua / bán mã nào trong HORIZON phiên tới.
    Đây là tương lai: CHỈ dùng làm nhãn y, không bao giờ làm đặc trưng."""
    for side, table in (("BUY", "future_buy"), ("SELL", "future_sell")):
        con.execute(f"""
        CREATE OR REPLACE TABLE {table} AS
        SELECT DISTINCT dd.t, x.customer_id, x.stock_code
        FROM decision_dates dd
        JOIN txn x ON x.side = '{side}'
        JOIN daycal k ON k.d = x.d
        WHERE k.dn > dd.dn AND k.dn <= dd.dn + {HORIZON};
        """)


def build_stock_at_t(con):
    """stock_at_t — đặc trưng giá, sàn, ngành và tín hiệu PHS của mỗi mã tại đúng t.
    label_complete = 0 ở các mốc cuối chưa đủ HORIZON phiên để biết nhãn."""
    con.execute(f"""
    CREATE OR REPLACE TABLE stock_at_t AS
    SELECT s.d AS t, s.dn, s.stock_code, s.c AS price,
           s.ret_5, s.ret_20, s.ret_60, s.vol_20, s.up_ratio_20, s.vol_ratio,
           LN(GREATEST(s.turnover, 1)) AS log_turnover,
           s.y_stock, s.excess_fwd,
           sc.exchange, sc.icb_code,
           r.market_score, r.candidate_rank,
           r.recommendation AS research_recommendation,
           r.call_type AS research_call_type,
           r.target_price, r.cut_loss_price,
           CASE WHEN r.stock_code IS NULL THEN 0 ELSE 1 END AS in_research,
           CASE WHEN s.dn + {HORIZON} <= (SELECT MAX(dn) FROM daycal)
                THEN 1 ELSE 0 END AS label_complete
    FROM sd s
    JOIN decision_dates dd ON dd.t = s.d
    JOIN sec sc ON sc.stock_code = s.stock_code
    LEFT JOIN research r ON r.stock_code = s.stock_code AND r.d = s.d
    WHERE s.n_sessions_20 >= 5;
    """)


def build_graph_features(con):
    """Bảng tạm từ đồ thị S3, lấy lát cắt gần nhất <= t:
    _cooc      mã ứng viên hay được mua cùng các mã khách đang giữ đến mức nào
    _pop       độ phổ biến của mã
    _peer_buy  khách tương tự vừa mua / bán mã PHS vừa khuyến nghị
    _peer_port khách tương tự vừa mua / bán mã khách đang giữ
    """
    log("s4", "chuẩn bị đặc trưng graph BUY/SELL...")
    con.execute(f"""
    CREATE OR REPLACE TABLE _cooc_top AS
    SELECT qt, a, b, co FROM (
        SELECT qt, a, b, co,
               ROW_NUMBER() OVER (PARTITION BY qt, a ORDER BY co DESC, b) AS rn
        FROM graph_cooc
    ) WHERE rn <= {COOC_TOPN};

    CREATE OR REPLACE TABLE _cooc AS
    SELECT p.t, p.customer_id, g.b AS stock_code, SUM(g.co) AS cooc_score
    FROM pos p
    JOIN dd_q q      ON q.t = p.t
    JOIN _cooc_top g ON g.qt = q.qt AND g.a = p.stock_code
    GROUP BY 1, 2, 3;

    CREATE OR REPLACE TABLE _pop AS
    SELECT q.t, g.stock_code, g.n_buyers, g.buy_value,
           g.n_sellers, g.sell_value
    FROM dd_q q JOIN graph_pop g ON g.qt = q.qt;

    -- khách tương tự của từng khách tại t (lát cắt đồ thị gần nhất <= t)
    CREATE OR REPLACE TABLE _peer AS
    SELECT q.t, g.customer_id, g.peer_id, g.sim
    FROM dd_q q JOIN graph_peer g ON g.qt = q.qt;

    -- ai vừa mua / bán mã nào trong RECENT_WINDOW ngày ngay trước t
    CREATE OR REPLACE TABLE _recent AS
    SELECT dd.t, x.customer_id, x.stock_code,
           MAX(CASE WHEN x.side = 'BUY'  THEN 1 ELSE 0 END) AS bought,
           MAX(CASE WHEN x.side = 'SELL' THEN 1 ELSE 0 END) AS sold
    FROM decision_dates dd
    JOIN txn x ON x.d < dd.t AND x.d >= dd.t - {RECENT_WINDOW}
    GROUP BY 1, 2, 3;

    -- bước thứ 3 của đường đi, chỉ tính cho các mã PHS vừa BUY mới tại t
    CREATE OR REPLACE TABLE _peer_buy AS
    SELECT p.t, p.customer_id, r.stock_code,
           SUM(p.sim * r.bought) AS peer_buy_score, SUM(r.bought) AS n_peer_buyers,
           SUM(p.sim * r.sold)   AS peer_sell_score, SUM(r.sold)  AS n_peer_sellers
    FROM _peer p
    JOIN _recent r ON r.t = p.t AND r.customer_id = p.peer_id
    JOIN stock_at_t s ON s.t = r.t AND s.stock_code = r.stock_code
                     AND s.research_recommendation = 'BUY'
    GROUP BY 1, 2, 3;

    -- ... và cho các mã khách đang nắm tại t
    CREATE OR REPLACE TABLE _peer_port AS
    SELECT ps.t, ps.customer_id, ps.stock_code,
           SUM(p.sim * r.bought) AS peer_buy_score, SUM(r.bought) AS n_peer_buyers,
           SUM(p.sim * r.sold)   AS peer_sell_score, SUM(r.sold)  AS n_peer_sellers
    FROM pos ps
    JOIN _peer p ON p.t = ps.t AND p.customer_id = ps.customer_id
    JOIN _recent r ON r.t = ps.t AND r.customer_id = p.peer_id
                  AND r.stock_code = ps.stock_code
    GROUP BY 1, 2, 3;
    """)


def build_buy_candidates(con):
    """cand_pool — mỗi khách x mỗi mã PHS vừa khuyến nghị MUA tại t, kèm điểm đã chuẩn
    hoá về [0, 1]. Model không tự phát minh mã nào ngoài danh sách của PHS.

    Mọi tài khoản đã mở trước t đều nhận candidate BUY. Khách chưa có lịch sử sẽ
    cold-start bằng MarketScore thay vì biến mất khỏi hệ thống (kỳ hạn của các
    khách này là UNRATED vì chưa có bằng chứng nắm giữ nào).
    """
    con.execute("""
    CREATE OR REPLACE TABLE cust_universe AS
    SELECT dd.t, p.customer_id
    FROM decision_dates dd
    JOIN prof p ON p.open_date < dd.t;
    """)
    log("s4", "dựng candidate BUY từ danh sách nghiên cứu tại t...")
    con.execute(f"""
    CREATE OR REPLACE TABLE cand_pool AS
    WITH raw AS (
        SELECT u.t, u.customer_id, s.stock_code,
               COALESCE(c.cooc_score, 0) AS cooc_score,
               COALESCE(cs.icb_share_all, 0) AS icb_share,
               COALESCE(p.n_buyers, 0) AS n_buyers,
               COALESCE(p.buy_value, 0) AS graph_buy_value,
               COALESCE(p.n_sellers, 0) AS n_sellers,
               COALESCE(p.sell_value, 0) AS graph_sell_value,
               s.market_score, s.candidate_rank, s.label_complete, s.dn,
               s.target_price, s.cut_loss_price
        FROM cust_universe u
        JOIN stock_at_t s ON s.t = u.t AND s.research_recommendation = 'BUY'
        LEFT JOIN _cooc c ON c.t = u.t AND c.customer_id = u.customer_id
                         AND c.stock_code = s.stock_code
        LEFT JOIN cust_sector_recent cs ON cs.t = u.t AND cs.customer_id = u.customer_id
                                       AND cs.icb_code = s.icb_code
        LEFT JOIN _pop p ON p.t = u.t AND p.stock_code = s.stock_code
    ), norm AS (
        SELECT *,
          COALESCE((market_score - MIN(market_score) OVER (PARTITION BY t)) /
                   NULLIF(MAX(market_score) OVER (PARTITION BY t) -
                          MIN(market_score) OVER (PARTITION BY t), 0), 0) AS market_score_norm,
          COALESCE(cooc_score /
                   NULLIF(MAX(cooc_score) OVER (PARTITION BY t, customer_id), 0), 0)
                   AS cooc_score_norm,
          COALESCE(n_buyers /
                   NULLIF(MAX(n_buyers) OVER (PARTITION BY t), 0), 0)
                   AS buyer_pop_norm
        FROM raw
    )
    SELECT *,
           {GRAPH_SCORE_WEIGHTS["cooc"]} * cooc_score_norm +
           {GRAPH_SCORE_WEIGHTS["buyer_pop"]} * buyer_pop_norm AS graph_score,
           COUNT(*) OVER (PARTITION BY t, customer_id) AS basket_size
    FROM norm;
    """)

    buy_cov = con.execute("""
        SELECT ROUND(100.0 * AVG(CASE WHEN c.stock_code IS NULL THEN 0 ELSE 1 END), 1)
        FROM future_buy f
        LEFT JOIN cand_pool c ON c.t = f.t AND c.customer_id = f.customer_id
                             AND c.stock_code = f.stock_code
    """).fetchone()[0]
    log("s4", f"độ phủ candidate BUY trên lệnh mua tương lai: {buy_cov}%")


def build_train_buy(con):
    """train_buy — mỗi dòng là một ứng viên MUA; y = 1 nếu khách mua mã đó trong
    HORIZON phiên tới. Thêm 3 điểm tự tính: historical_preference (khách từng thích
    mã này?), suitability_score (mã hợp gu khách?), personalization_score (tổng hợp).

    Ghép với các bảng hàng chục triệu dòng của S2 nên dựng theo từng lô mốc (như S2)
    để phép ghép không vượt trần bộ nhớ ở nhịp ngày.
    """
    log("s4", "dựng train_buy: MarketScore + graph + style + BUY/SELL history...")
    hw, pw = HISTORY_PREF_WEIGHTS, PERSONALIZATION_WEIGHTS
    # mã gần với phong cách khách thích -> 1, càng lệch càng về 0; thiếu dữ liệu -> 0
    style = [
        f"COALESCE(EXP(-ABS(s.{c} - st.preferred_{c}) / {scale}), 0)"
        for c, scale in STYLE_SCALES.items()
    ]
    create_by_date_chunks(
        con,
        f"""
    CREATE OR REPLACE TABLE train_buy AS
    WITH b AS (
        SELECT cp.*, s.ret_5, s.ret_20, s.ret_60, s.vol_20, s.up_ratio_20,
               s.vol_ratio, s.log_turnover, s.exchange, s.icb_code,
               {hw["value_share"]} * COALESCE(csp.buy_value / NULLIF(cu.buy_value, 0), 0) +
               {hw["count_share"]} * COALESCE(csp.n_buys_stock * 1.0 / NULLIF(cu.n_buys, 0), 0) +
               {hw["recency"]} * CASE WHEN csp.days_since_buy IS NULL THEN 0 ELSE
                    POWER(0.5, csp.days_since_buy * 1.0 / {BEHAVIOR_HALF_LIFE}) END
                    AS historical_preference,
               (cp.icb_share + {" + ".join(style)}) / 4.0 AS suitability_score,
               -- khách tương tự vừa mua / bán chính mã này
               COALESCE(pb.peer_buy_score, 0)  AS peer_buy_score,
               COALESCE(pb.n_peer_buyers, 0)   AS n_peer_buyers,
               COALESCE(pb.peer_sell_score, 0) AS peer_sell_score,
               COALESCE(pb.n_peer_sellers, 0)  AS n_peer_sellers,
               {customer_columns("cp")},
               cu.avg_trade_value, cu.days_since_first_trade, cu.days_since_last_trade,
               sk.hist_hit_rate, pp.n_positions, pp.port_hhi,
               LN(GREATEST(pp.port_value, 1)) AS log_port_value, pp.avg_pnl_pct,
               COALESCE(csp.n_trades_stock, 0) AS n_trades_stock,
               COALESCE(csp.n_buys_stock, 0) AS n_buys_stock,
               COALESCE(csp.n_sells_stock, 0) AS n_sells_stock,
               csp.days_since_stock, csp.days_since_buy, csp.days_since_sell,
               csp.last_side,
               CASE WHEN ps.stock_code IS NULL THEN 0 ELSE 1 END AS is_holding,
               CASE WHEN fb.stock_code IS NULL THEN 0 ELSE 1 END AS y,
               s.excess_fwd
        FROM cand_pool cp
        JOIN decision_dates dd ON dd.t = cp.t
        JOIN stock_at_t s ON s.t = cp.t AND s.stock_code = cp.stock_code
        {customer_joins("cp")}
        LEFT JOIN pos ps ON ps.t = cp.t AND ps.customer_id = cp.customer_id
                        AND ps.stock_code = cp.stock_code
        LEFT JOIN _peer_buy pb ON pb.t = cp.t AND pb.customer_id = cp.customer_id
                              AND pb.stock_code = cp.stock_code
        LEFT JOIN future_buy fb ON fb.t = cp.t AND fb.customer_id = cp.customer_id
                               AND fb.stock_code = cp.stock_code
    )
    SELECT *,
           {pw["graph"]} * graph_score + {pw["suitability"]} * suitability_score +
           {pw["history"]} * historical_preference AS personalization_score
    FROM b;
    """,
        slice_tables=(*PIT_JOINS, "pos", "_peer_buy", "future_buy"),
    )


def build_train_portfolio(con):
    """train_portfolio — mỗi dòng là một mã khách đang giữ; y = 1 nếu khách bán mã đó
    trong HORIZON phiên tới. Model học hành vi BÁN; tín hiệu PHS (research_sell_signal)
    mới quyết định hành động, model chỉ xếp thứ tự bên trong cùng một bậc."""
    log("s4", "dựng train_portfolio cho các mã khách đang sở hữu...")
    rs = RESEARCH_SELL_SIGNAL
    create_by_date_chunks(
        con,
        f"""
    CREATE OR REPLACE TABLE train_portfolio AS
    WITH b AS (
    SELECT p.t, p.dn, p.customer_id, p.stock_code,
           s.ret_5, s.ret_20, s.ret_60, s.vol_20, s.up_ratio_20, s.vol_ratio,
           s.log_turnover, s.market_score, s.candidate_rank,
           s.research_recommendation, s.research_call_type,
           s.in_research, s.label_complete,
           -- SELL = vừa chốt lời/cắt lỗ, BUY = mới mở, HOLD = BUY cũ còn mở
           CASE s.research_recommendation WHEN 'SELL' THEN {rs["SELL"]}
                WHEN 'BUY' THEN {rs["BUY"]} WHEN 'HOLD' THEN {rs["HOLD"]}
                ELSE {rs["NONE"]} END
                AS research_sell_signal,
           s.exchange, s.icb_code,
           p.unreal_pnl_pct,
           -- tuổi vị thế so với thời gian khách thường giữ một mã
           p.pos_hold_days, p.pos_has_sold,
           p.pos_hold_days / NULLIF(ch.median_hold_days, 0) AS hold_days_vs_typical,
           -- lãi/lỗ đang treo so với mức khách thường chốt lời / cắt lỗ
           p.unreal_pnl_pct - ch.avg_win_ret  AS pnl_vs_win_habit,
           p.unreal_pnl_pct - ch.avg_loss_ret AS pnl_vs_loss_habit,
           p.mv / NULLIF(pp.port_value, 0) AS weight_in_port,
           LN(GREATEST(p.mv, 1)) AS log_position_value,
           -- khách tương tự vừa bán / mua chính mã này
           COALESCE(pe.peer_sell_score, 0) AS peer_sell_score,
           COALESCE(pe.n_peer_sellers, 0)  AS n_peer_sellers,
           COALESCE(pe.peer_buy_score, 0)  AS peer_buy_score,
           COALESCE(pe.n_peer_buyers, 0)   AS n_peer_buyers,
           csp.days_since_stock, csp.days_since_buy, csp.days_since_sell,
           COALESCE(csp.n_trades_stock, 0) AS n_trades_stock,
           COALESCE(csp.n_buys_stock, 0) AS n_buys_stock,
           COALESCE(csp.n_sells_stock, 0) AS n_sells_stock,
           csp.last_side,
           {customer_columns("p")},
           sk.hist_hit_rate, pp.n_positions, pp.port_hhi,
           LN(GREATEST(pp.port_value, 1)) AS log_port_value, pp.avg_pnl_pct,
           CASE WHEN fs.stock_code IS NULL THEN 0 ELSE 1 END AS y,
           s.excess_fwd
    FROM pos p
    JOIN decision_dates dd ON dd.t = p.t
    JOIN stock_at_t s ON s.t = p.t AND s.stock_code = p.stock_code
    {customer_joins("p")}
    LEFT JOIN _peer_port pe ON pe.t = p.t AND pe.customer_id = p.customer_id
                           AND pe.stock_code = p.stock_code
    LEFT JOIN future_sell fs ON fs.t = p.t AND fs.customer_id = p.customer_id
                            AND fs.stock_code = p.stock_code
    )
    -- Vị trí của mã SO VỚI các mã khác trong cùng danh mục. Đặc trưng cấp khách
    -- (kỳ hạn, nhịp giao dịch) giống nhau cho mọi mã của một khách nên không xếp
    -- hạng được trong danh mục; các cột dưới đây thì có.
    SELECT *,
           unreal_pnl_pct - avg_pnl_pct AS pnl_vs_port_avg,
           PERCENT_RANK() OVER (PARTITION BY t, customer_id
                                ORDER BY unreal_pnl_pct)     AS pnl_rank_in_port,
           PERCENT_RANK() OVER (PARTITION BY t, customer_id
                                ORDER BY pos_hold_days)      AS hold_days_rank_in_port,
           PERCENT_RANK() OVER (PARTITION BY t, customer_id
                                ORDER BY log_position_value) AS weight_rank_in_port,
           PERCENT_RANK() OVER (PARTITION BY t, customer_id
                                ORDER BY ret_20)             AS ret_20_rank_in_port,
           PERCENT_RANK() OVER (PARTITION BY t, customer_id
                                ORDER BY days_since_stock DESC)
                                                             AS recency_rank_in_port
    FROM b;
    """,
        slice_tables=(*PIT_JOINS, "_peer_port", "future_sell"),
    )


def report(con):
    """Số dòng, số dòng đủ nhãn và tỷ lệ nhãn dương của hai bảng train."""
    for tb in ["train_buy", "train_portfolio"]:
        n, n_labeled, pos_rate = con.execute(f"""
            SELECT COUNT(*), COUNT_IF(label_complete = 1),
                   ROUND(100.0*AVG(y) FILTER (WHERE label_complete = 1), 2)
            FROM {tb}
        """).fetchone()
        log(
            "s4",
            f"  {tb:16s} {n:>9,} dòng | {n_labeled:>9,} đủ nhãn | "
            f"tỷ lệ hành động tương lai {pos_rate}%",
        )


def drop_temp_tables(con):
    """Bảng trung gian chỉ phục vụ việc dựng train_buy / train_portfolio; giữ lại
    chỉ làm file DuckDB phình ra."""
    for tmp in (
        "_cooc",
        "_cooc_top",
        "_pop",
        "_peer",
        "_recent",
        "_peer_buy",
        "_peer_port",
        "cand_pool",
        "cust_universe",
        "future_buy",
        "future_sell",
    ):
        con.execute(f"DROP TABLE IF EXISTS {tmp};")


def main():
    con = connect()
    build_labels(con)
    build_stock_at_t(con)
    build_graph_features(con)
    build_buy_candidates(con)
    build_train_buy(con)
    build_train_portfolio(con)
    report(con)
    drop_temp_tables(con)
    con.close()
    log("s4", "xong.")


if __name__ == "__main__":
    main()
