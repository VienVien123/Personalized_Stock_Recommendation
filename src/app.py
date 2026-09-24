"""Giao diện recommendation từ research signal + hành vi khách hàng."""

import json

import gradio as gr
import lightgbm as lgb
import numpy as np
import pandas as pd

import pipeline
from config import GRADIO_PORT, MARKET_WEIGHT, OUT_DIR, TOPK, connect, log


def load_model(name):
    model = lgb.Booster(model_file=f"{OUT_DIR}/model_{name}.txt")
    with open(f"{OUT_DIR}/meta_{name}.json", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta


def score(df, model, meta):
    x = df.copy()
    for c in meta["cats"]:
        x[c] = pd.Categorical(
            x[c].astype("string").fillna("UNKNOWN"),
            categories=meta["categories"][c],
        )
    for c in meta["features"]:
        if c not in meta["cats"]:
            x[c] = pd.to_numeric(x[c], errors="coerce").astype("float32")
    return model.predict(x[meta["features"]])


def minmax(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or not np.isfinite(values).any():
        return np.zeros(len(values))
    lo, hi = np.nanmin(values), np.nanmax(values)
    if hi - lo <= 1e-12:
        return np.zeros(len(values))
    return np.nan_to_num((values - lo) / (hi - lo))


def research_action(value):
    return {
        "SELL": "BÁN",
        "HOLD": "GIỮ",
        "BUY": "CÓ THỂ MUA THÊM",
    }.get(value, "CHƯA CÓ TÍN HIỆU")


def build():
    con = connect(read_only=True)
    buy_model, buy_meta = load_model("buy")
    portfolio_model, portfolio_meta = load_model("portfolio")
    dates = [
        str(r[0])
        for r in con.execute("SELECT t FROM decision_dates ORDER BY t").fetchall()
    ]
    customers = [
        r[0]
        for r in con.execute(
            "SELECT customer_id FROM prof ORDER BY customer_id"
        ).fetchall()
    ]
    names = (
        con.execute("SELECT stock_code, company_name, icb_name FROM sec")
        .df()
        .set_index("stock_code")
    )
    with open(f"{OUT_DIR}/summary.json", encoding="utf-8") as f:
        summary = json.load(f)

    def decorate(df):
        return df.join(names, on="stock_code")

    def recommend(customer_id, t):
        info = con.execute(
            "SELECT customer_type, risk_level, investment_horizon, open_date "
            "FROM prof WHERE customer_id = ?",
            [customer_id],
        ).fetchone()
        header = (
            f"**{customer_id}** — {info[0]} · khẩu vị {info[1]} · "
            f"kỳ hạn {info[2]} · mở TK {info[3]}  \n*Thời điểm: {t}*"
        )

        # BUY: hard-filter đã được thực hiện ở train_buy, mọi dòng đều là research BUY.
        buy_rows = con.execute(
            "SELECT * FROM train_buy WHERE customer_id = ? AND t = ?",
            [customer_id, t],
        ).df()
        if buy_rows.empty:
            buy_table = pd.DataFrame(
                {"Thông báo": ["Không có candidate BUY tại mốc này"]}
            )
        else:
            buy_rows["behavior_score"] = minmax(score(buy_rows, buy_model, buy_meta))
            buy_rows["final_score"] = (
                MARKET_WEIGHT * buy_rows["market_score_norm"].fillna(0)
                + (1 - MARKET_WEIGHT) * buy_rows["behavior_score"]
            )
            buy_rows = decorate(
                buy_rows.sort_values("final_score", ascending=False).head(TOPK)
            )
            buy_table = pd.DataFrame(
                {
                    "Mã": buy_rows.stock_code,
                    "Công ty": buy_rows.company_name,
                    "Ngành": buy_rows.icb_name,
                    "Tín hiệu": "BUY",
                    "MarketScore": buy_rows.market_score.round(2),
                    "Điểm hành vi": buy_rows.behavior_score.round(4),
                    "Điểm cuối": buy_rows.final_score.round(4),
                    "Graph": buy_rows.graph_score.round(4),
                    "Style": buy_rows.suitability_score.round(4),
                    "Lịch sử đúng mã": buy_rows.historical_preference.round(4),
                    "Đang nắm": buy_rows.is_holding.map({1: "có", 0: ""}),
                }
            )

        # PORTFOLIO: action do research quyết định; model chỉ cho biết mức ưu tiên
        # hành vi bán để sắp xếp các vị thế cùng tín hiệu.
        portfolio_rows = con.execute(
            "SELECT * FROM train_portfolio WHERE customer_id = ? AND t = ?",
            [customer_id, t],
        ).df()
        if portfolio_rows.empty:
            portfolio_table = pd.DataFrame(
                {"Thông báo": ["Khách không có vị thế tại mốc này"]}
            )
        else:
            portfolio_rows["sell_behavior_score"] = minmax(
                score(portfolio_rows, portfolio_model, portfolio_meta)
            )
            portfolio_rows["action"] = portfolio_rows.research_recommendation.map(
                research_action
            )
            priority = {"BÁN": 0, "GIỮ": 1, "CÓ THỂ MUA THÊM": 2, "CHƯA CÓ TÍN HIỆU": 3}
            portfolio_rows["action_priority"] = portfolio_rows.action.map(priority)
            portfolio_rows = decorate(
                portfolio_rows.sort_values(
                    ["action_priority", "sell_behavior_score"], ascending=[True, False]
                )
            )
            portfolio_table = pd.DataFrame(
                {
                    "Mã": portfolio_rows.stock_code,
                    "Công ty": portfolio_rows.company_name,
                    "Tín hiệu nghiên cứu": portfolio_rows.research_recommendation.fillna(
                        "—"
                    ),
                    "Khuyến nghị": portfolio_rows.action,
                    "Khả năng khách bán": portfolio_rows.sell_behavior_score.round(4),
                    "MarketScore": portfolio_rows.market_score.round(2),
                    "Lãi/lỗ hiện tại": (100 * portfolio_rows.unreal_pnl_pct)
                    .round(1)
                    .astype(str)
                    + "%",
                    "Tỷ trọng": (100 * portfolio_rows.weight_in_port)
                    .round(1)
                    .astype(str)
                    + "%",
                }
            )
        return header, buy_table, portfolio_table

    perf = "### Hiệu quả walk-forward theo quý\n\n"
    perf += (
        "Nhánh BUY chỉ xét mã research `BUY`. Điểm cuối = "
        f"**{MARKET_WEIGHT:.0%} MarketScore + {1 - MARKET_WEIGHT:.0%} hành vi**.\n\n"
    )
    perf += "| Model | AUC hành vi | NDCG | Recall | Precision | HitRate | Baseline NDCG | Số quý |\n"
    perf += "|---|---:|---:|---:|---:|---:|---:|---:|\n"
    for key in ("buy", "portfolio"):
        value = summary.get(key)
        if not value:
            continue
        perf += (
            f"| {key} | {value['auc']} | {value['ndcg']} | {value['recall']} | "
            f"{value['precision']} | {value['hitrate']} | "
            f"{value.get('baseline_ndcg', '—')} | {value['n_folds']} |\n"
        )
    perf += (
        "\nCác metric trên đo mức phù hợp hành vi. MarketScore/BUY-HOLD-SELL là "
        "tín hiệu đầu tư đầu vào từ bảng nghiên cứu, không phải do model hành vi tự tạo."
    )

    with gr.Blocks(title="Khuyến nghị cổ phiếu cá nhân hóa") as ui:
        gr.Markdown(
            "# Khuyến nghị từ tín hiệu nghiên cứu + hành vi khách hàng\n"
            "Chỉ xếp hạng mã **BUY** có sẵn; danh mục dùng tín hiệu **BUY/HOLD/SELL**."
        )
        with gr.Tab("Khuyến nghị"):
            with gr.Row():
                customer = gr.Dropdown(
                    customers, value=customers[0], label="Khách hàng"
                )
                date = gr.Dropdown(dates, value=dates[-1], label="Thời điểm t")
                button = gr.Button("Gợi ý", variant="primary")
            heading = gr.Markdown()
            gr.Markdown(f"### TOP-{TOPK} mã BUY phù hợp nhất")
            buy_output = gr.Dataframe(interactive=False, wrap=True)
            gr.Markdown("### Danh mục hiện tại — tín hiệu GIỮ/BÁN/MUA THÊM")
            portfolio_output = gr.Dataframe(interactive=False, wrap=True)
            button.click(
                recommend,
                [customer, date],
                [heading, buy_output, portfolio_output],
            )
        with gr.Tab("Hiệu quả model"):
            gr.Markdown(perf)
    return ui


if __name__ == "__main__":
    pipeline.run()
    log("app", f"khởi động Gradio ở cổng {GRADIO_PORT}")
    build().launch(server_name="0.0.0.0", server_port=GRADIO_PORT, show_api=False)
