"""Bước 5 — Huấn luyện model hành vi cho BUY và SELL.

Danh sách research đã quyết định mã nào có tín hiệu BUY/HOLD/SELL. Hai model ở
đây chỉ học mức phù hợp với từng khách:

* BUY: xác suất khách mua một mã trong rổ research BUY.
* PORTFOLIO: xác suất khách bán một mã đang sở hữu.

Điểm BUY cuối = MarketScore (chất lượng từ research) + BehaviorScore (model).
Đánh giá walk-forward theo quý, không chia ngẫu nhiên.
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import connect, OUT_DIR, TOPK, MARKET_WEIGHT, log


CATS = [
    "customer_type", "risk_level", "investment_horizon", "exchange", "icb_code",
    "last_side", "research_recommendation",
]
BASE_IDS = [
    "t", "customer_id", "stock_code", "y", "excess_fwd", "label_complete", "q",
]

PARAMS = dict(
    objective="binary", metric="auc", learning_rate=0.05,
    num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
    bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    verbose=-1, num_threads=0, seed=42,
)
ROUNDS = 300


def prep(df, extra_drop=()):
    """Ép kiểu ổn định và tách feature khỏi id/nhãn."""
    blocked = set(BASE_IDS) | set(extra_drop)
    feats = [c for c in df.columns if c not in blocked]
    for c in CATS:
        if c in feats:
            df[c] = df[c].astype("string").fillna("UNKNOWN").astype("category")
    for c in feats:
        if c not in CATS:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    return df, feats


def normalize_within_group(df, score):
    """Đưa score về [0,1] riêng trong mỗi khách/ngày để ghép MarketScore."""
    z = pd.Series(np.asarray(score, dtype=float), index=df.index)
    lo = z.groupby([df["t"], df["customer_id"]]).transform("min")
    hi = z.groupby([df["t"], df["customer_id"]]).transform("max")
    den = hi - lo
    return ((z - lo) / den.where(den > 1e-12, 1.0)).fillna(0.0).to_numpy()


def ranking_metrics(df, score, k):
    """Metric hành vi theo từng khách/ngày và audit lợi suất của top-k."""
    d = df[["t", "customer_id", "y", "excess_fwd"]].copy()
    d["score"] = np.asarray(score, dtype=float)
    d["rk"] = d.groupby(["t", "customer_id"])["score"].rank(
        ascending=False, method="first"
    )
    top = d[d["rk"] <= k]
    precision = float(top["y"].mean()) if len(top) else np.nan
    excess = float(top["excess_fwd"].mean()) if len(top) else np.nan

    total_pos = d.groupby(["t", "customer_id"])["y"].sum()
    top_pos = top.groupby(["t", "customer_id"])["y"].sum()
    positive_groups = total_pos[total_pos > 0]
    if positive_groups.empty:
        recall = hitrate = ndcg = np.nan
    else:
        hits = top_pos.reindex(positive_groups.index, fill_value=0)
        recall = float((hits / positive_groups).mean())
        hitrate = float((hits > 0).mean())

        ndcgs = []
        for key in positive_groups.index:
            g = d[(d["t"] == key[0]) & (d["customer_id"] == key[1])].nsmallest(k, "rk")
            dcg = sum(float(y) / np.log2(i + 2) for i, y in enumerate(g["y"]))
            ideal_n = min(int(positive_groups.loc[key]), k)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_n))
            ndcgs.append(dcg / idcg if idcg else 0.0)
        ndcg = float(np.mean(ndcgs))

    return dict(precision=precision, recall=recall, hitrate=hitrate,
                ndcg=ndcg, excess=excess)


def _fit(train, feats, rounds=ROUNDS):
    pos = float(train["y"].sum())
    neg = float(len(train) - pos)
    params = PARAMS.copy()
    if pos > 0:
        params["scale_pos_weight"] = min(neg / pos, 50.0)
    return lgb.train(params, lgb.Dataset(train[feats], train["y"]),
                     num_boost_round=rounds)


def walk_forward(df, feats, name, baseline_col=None, hybrid=False,
                 research_first=False):
    df["q"] = pd.PeriodIndex(pd.to_datetime(df["t"]), freq="Q")
    quarters = sorted(df["q"].unique())
    rows = []

    for i in range(4, len(quarters)):
        train = df[df.q < quarters[i]]
        test = df[df.q == quarters[i]]
        if test.empty or train.y.nunique() < 2 or test.y.nunique() < 2:
            continue

        model = _fit(train, feats)
        raw = model.predict(test[feats])
        behavior = normalize_within_group(test, raw)
        behavior_m = ranking_metrics(test, behavior, TOPK)

        row = dict(
            quy=str(quarters[i]), n_test=len(test),
            auc=round(float(roc_auc_score(test.y, raw)), 4),
            behavior_precision=behavior_m["precision"],
            behavior_recall=behavior_m["recall"],
            behavior_hitrate=behavior_m["hitrate"],
            behavior_ndcg=behavior_m["ndcg"],
        )

        final_score = behavior
        if baseline_col and baseline_col in test:
            baseline = pd.to_numeric(test[baseline_col], errors="coerce").fillna(0).to_numpy()
            base_m = ranking_metrics(test, baseline, TOPK)
            row.update({f"baseline_{k}": v for k, v in base_m.items()})
            if hybrid:
                final_score = MARKET_WEIGHT * baseline + (1 - MARKET_WEIGHT) * behavior
            elif research_first:
                # SELL > HOLD > BUY > chưa có tín hiệu; behavior chỉ phá hòa bên
                # trong cùng một tầng và không thể lật quyết định của research.
                final_score = baseline + 0.49 * behavior

        final_m = ranking_metrics(test, final_score, TOPK)
        row.update({f"final_{k}": v for k, v in final_m.items()})
        rows.append(row)
        log("s5", f"[{name}] {quarters[i]} AUC={row['auc']:.3f} | "
                  f"NDCG@{TOPK}={final_m['ndcg']:.3f} | "
                  f"Recall@{TOPK}={final_m['recall']:.3f}")

    return pd.DataFrame(rows)


def save(model, df, feats, name):
    model.save_model(f"{OUT_DIR}/model_{name}.txt")
    meta = dict(
        features=feats,
        cats=[c for c in CATS if c in feats],
        categories={c: [str(v) for v in df[c].cat.categories]
                    for c in CATS if c in feats},
    )
    with open(f"{OUT_DIR}/meta_{name}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    imp = pd.DataFrame({
        "feature": feats,
        "gain": model.feature_importance("gain"),
    }).sort_values("gain", ascending=False)
    imp.to_csv(f"{OUT_DIR}/importance_{name}.csv", index=False)
    return imp


def train_one(con, table, name, extra_drop=(), baseline_col=None, hybrid=False,
              research_first=False):
    log("s5", f"=== MODEL {name.upper()} ===")
    # Giữ các mốc mới nhất trong bảng feature để app vẫn chấm điểm được, nhưng
    # tuyệt đối không dùng chúng làm nhãn 0 khi chưa đi hết HORIZON phiên.
    df = con.execute(
        f"SELECT * FROM {table} WHERE label_complete = 1"
    ).df()
    df, feats = prep(df, extra_drop)
    log("s5", f"{len(df):,} dòng | {len(feats)} đặc trưng | "
              f"nhãn dương {100*df.y.mean():.2f}%")

    evaluation = walk_forward(
        df.copy(), feats, name, baseline_col, hybrid, research_first
    )
    if not evaluation.empty:
        evaluation.to_csv(f"{OUT_DIR}/eval_{name}.csv", index=False)
        log("s5", f"[{name}] TB AUC={evaluation.auc.mean():.4f} | "
                  f"NDCG@{TOPK}={evaluation.final_ndcg.mean():.4f} | "
                  f"Recall@{TOPK}={evaluation.final_recall.mean():.4f}")

    final_model = _fit(df, feats, rounds=400)
    importance = save(final_model, df, feats, name)
    log("s5", f"[{name}] 8 đặc trưng mạnh nhất: "
              f"{', '.join(importance.head(8).feature)}")
    return evaluation


def main():
    con = connect()
    # MarketScore bị loại khỏi behavior model để hai thành phần độc lập; nó chỉ
    # được ghép lại bằng MARKET_WEIGHT khi xếp hạng.
    buy_drop = ["market_score", "candidate_rank", "market_score_norm",
                "candidate_rk", "initial_hybrid_score"]
    ev_buy = train_one(
        con, "train_buy", "buy", extra_drop=buy_drop,
        baseline_col="market_score_norm", hybrid=True,
    )
    ev_portfolio = train_one(
        con, "train_portfolio", "portfolio",
        extra_drop=["market_score", "candidate_rank", "research_recommendation",
                    "in_research", "research_sell_signal"],
        baseline_col="research_sell_signal", hybrid=False, research_first=True,
    )
    con.close()

    summary = {"market_weight": MARKET_WEIGHT}
    for name, ev in [("buy", ev_buy), ("portfolio", ev_portfolio)]:
        if ev is not None and not ev.empty:
            summary[name] = dict(
                auc=round(float(ev.auc.mean()), 4),
                ndcg=round(float(ev.final_ndcg.mean()), 4),
                recall=round(float(ev.final_recall.mean()), 4),
                precision=round(float(ev.final_precision.mean()), 4),
                hitrate=round(float(ev.final_hitrate.mean()), 4),
                n_folds=int(len(ev)),
            )
            if "baseline_ndcg" in ev:
                summary[name]["baseline_ndcg"] = round(float(ev.baseline_ndcg.mean()), 4)

    with open(f"{OUT_DIR}/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log("s5", f"tóm tắt: {summary}")
    log("s5", "xong.")


if __name__ == "__main__":
    main()
