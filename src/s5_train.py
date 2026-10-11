"""Bước 5 — Huấn luyện model hành vi cho BUY và SELL.

Danh sách research đã quyết định mã nào có tín hiệu BUY/SELL. Hai model ở đây chỉ
học mức phù hợp với từng khách:

* BUY: xác suất khách mua một mã trong rổ research BUY.
* PORTFOLIO: xác suất khách bán một mã đang sở hữu.

Cách đánh giá (walk-forward theo quý, không chia ngẫu nhiên):
  * Có khoảng đệm HORIZON phiên giữa train và test: một dòng train chỉ được dùng
    khi cửa sổ nhãn của nó đã kết thúc trước mốc test đầu tiên.
  * Số vòng boosting chọn bằng dừng sớm trên quý train cuối, không đặt tay.
  * Thước đo xếp hạng chỉ tính trên nhóm (khách, mốc) CÓ THỂ xếp sai: có ít nhất
    một mã đúng và một mã sai. Nhóm chỉ có 1 ứng viên luôn đạt 1,0 nên bị loại.
"""

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import (
    EVAL_KS,
    HORIZON,
    MARKET_WEIGHT,
    OUT_DIR,
    PORTFOLIO_TRAIN_STRIDE,
    RESEARCH_TIEBREAK,
    connect,
    log,
)

CATS = [
    "customer_type",
    "investment_horizon",
    "exchange",
    "icb_code",
    "last_side",
    "research_recommendation",
]
# cột định danh / nhãn, không phải đặc trưng
BASE_IDS = ["t", "dn", "customer_id", "stock_code", "y", "excess_fwd"]
BASE_IDS += ["label_complete", "q", "g"]
# Cột của PHS bị loại khỏi model hành vi (chỉ dùng khi ghép điểm cuối).
BUY_RESEARCH_COLS = [
    "market_score",
    "candidate_rank",
    "market_score_norm",
    "target_price",
    "cut_loss_price",
]
PORTFOLIO_RESEARCH_COLS = [
    "market_score",
    "candidate_rank",
    "research_recommendation",
    "research_call_type",
    "in_research",
    "research_sell_signal",
]

PARAMS = {
    "objective": "binary",  # cho xác suất thật để hiển thị trên app
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "num_threads": 0,
    "seed": 42,
}
MAX_ROUNDS = 400
EARLY_STOP = 30
MIN_ROUNDS = 20
DEFAULT_ROUNDS = 300
MIN_TRAIN_QUARTERS = 4
REPORT_KEYS = ("hit@1", "ndcg@3", "mrr", "precision@1", "excess@1")


# ---------------------------------------------------------------- dữ liệu
def load_table(table, stride=1):
    """Đọc bảng train rồi đóng DuckDB ngay để nhường RAM cho LightGBM.

    Chỉ lấy dòng đã đủ nhãn, và mỗi mốc thứ `stride`. Việc nặng làm trong DuckDB:
      * sắp xếp cố định theo (t, khách, mã) — nếu không, khi các mã bằng điểm nhau,
        thứ hạng có thể vô tình "nhìn thấy" nhãn qua thứ tự dòng;
      * đánh số nhóm (khách, mốc) thành cột `g`;
      * số thực trả về dạng FLOAT 4 byte, chuỗi dạng category -> ~0,5 GB thay vì 2 GB.
    """
    con = connect(read_only=True)
    try:
        select = ["CAST(t AS TIMESTAMP) AS t"]
        for name, dtype, *_ in con.execute(f"DESCRIBE {table}").fetchall():
            if name in ("t", "customer_id", "stock_code"):
                continue
            if dtype == "VARCHAR" or name == "dn":  # dn: số nguyên để chia train/test
                select.append(f'"{name}"')
            else:
                select.append(f'CAST("{name}" AS FLOAT) AS "{name}"')
        table_arrow = con.execute(f"""
            SELECT {", ".join(select)},
                   CAST(DENSE_RANK() OVER (ORDER BY t, customer_id) - 1 AS INTEGER) AS g
            FROM {table} WHERE label_complete = 1 AND dn % {int(stride)} = 0
            ORDER BY t, customer_id, stock_code""").arrow()
    finally:
        con.close()
    df = table_arrow.to_pandas(
        strings_to_categorical=True, split_blocks=True, self_destruct=True
    )
    del table_arrow
    df["q"] = pd.PeriodIndex(df["t"], freq="Q")
    return df


def prep(df, extra_drop=()):
    """Tách feature khỏi id/nhãn và chốt thứ tự nhãn của các cột phân loại."""
    blocked = set(BASE_IDS) | set(extra_drop)
    feats = [c for c in df.columns if c not in blocked]
    for c in CATS:
        if c in feats:
            col = df[c].astype("category").cat.rename_categories(str)
            if col.isna().any():
                col = col.cat.set_categories(
                    list(dict.fromkeys([*col.cat.categories, "UNKNOWN"]))
                ).fillna("UNKNOWN")
            # thứ tự nhãn cố định -> mã số của nhãn không phụ thuộc thứ tự dòng
            df[c] = col.cat.reorder_categories(sorted(col.cat.categories))
    return df, feats


# ---------------------------------------------------------------- huấn luyện
def fit(train, feats, rounds=None, valid=None):
    """Huấn luyện. Có `valid` thì dừng sớm; không thì chạy đúng `rounds` vòng."""
    dtrain = lgb.Dataset(train[feats], train["y"])
    if valid is None:
        return lgb.train(PARAMS, dtrain, num_boost_round=rounds)
    return lgb.train(
        PARAMS,
        dtrain,
        num_boost_round=MAX_ROUNDS,
        valid_sets=[lgb.Dataset(valid[feats], valid["y"], reference=dtrain)],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)],
    )


def embargoed(df, before_dn):
    """Chỉ giữ dòng mà cửa sổ nhãn HORIZON phiên đã kết thúc trước `before_dn`."""
    return df[df["dn"] + HORIZON <= before_dn]


def pick_rounds(train, feats):
    """Chọn số vòng boosting bằng dừng sớm trên quý cuối của tập train."""
    last_q = train["q"].max()
    valid = train[train["q"] == last_q]
    part = embargoed(train[train["q"] < last_q], valid["dn"].min())
    if part["y"].nunique() < 2 or valid["y"].nunique() < 2:
        return DEFAULT_ROUNDS
    model = fit(part, feats, valid=valid)
    return max(int(model.best_iteration or DEFAULT_ROUNDS), MIN_ROUNDS)


def normalize_within_group(g, score):
    """Đưa score về [0,1] riêng trong mỗi khách/mốc để ghép với điểm PHS."""
    z = pd.Series(np.asarray(score, dtype=float))
    by = z.groupby(np.asarray(g))
    lo, hi = by.transform("min"), by.transform("max")
    den = hi - lo
    return ((z - lo) / den.where(den > 1e-12, 1.0)).fillna(0.0).to_numpy()


def walk_forward(df, feats, name, baseline_col):
    """Học trên các quý trước, dự đoán quý sau; trả về dự đoán ngoài mẫu + từng quý."""
    quarters = sorted(df["q"].unique())
    parts, folds = [], []
    for quarter in quarters[MIN_TRAIN_QUARTERS:]:
        test = df[df["q"] == quarter]
        train = embargoed(df[df["q"] < quarter], test["dn"].min())
        if train["y"].nunique() < 2 or test["y"].nunique() < 2:
            continue

        rounds = pick_rounds(train, feats)
        raw = fit(train, feats, rounds=rounds).predict(test[feats])

        part = test[["t", "q", "g", "y", "excess_fwd"]].copy()
        part["behavior"] = normalize_within_group(test["g"], raw)
        part["baseline"] = test[baseline_col].fillna(0).to_numpy()
        parts.append(part)
        auc = float(roc_auc_score(test["y"], raw))
        folds.append({"quy": str(quarter), "rounds": rounds, "auc": round(auc, 4)})
        log(
            "s5",
            f"[{name}] {quarter} train={len(train):,} test={len(test):,} "
            f"vòng={rounds} AUC={auc:.3f}",
        )
    oof = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return oof, folds


# ---------------------------------------------------------------- đánh giá
def final_score(oof, kind):
    if kind == "hybrid":  # nhánh MUA: trộn điểm PHS và điểm hành vi
        return MARKET_WEIGHT * oof["baseline"] + (1 - MARKET_WEIGHT) * oof["behavior"]
    # nhánh danh mục: SELL > BUY > HOLD > chưa có tín hiệu; behavior chỉ phá hoà
    # bên trong cùng một bậc và không thể lật quyết định của research.
    return oof["baseline"] + RESEARCH_TIEBREAK * oof["behavior"]


def ranking_metrics(oof, score, ks=EVAL_KS):
    """Thước đo xếp hạng theo từng nhóm (khách, mốc).

    * precision@1, excess@1: trên MỌI nhóm — mã đứng đầu có đúng không, lời không.
    * hit@k, recall@k, ndcg@k, mrr: chỉ trên nhóm có cả mã đúng lẫn mã sai.
    * random_hit@1: hit@1 nếu chọn ngẫu nhiên — mốc so sánh thấp nhất.
    """
    d = pd.DataFrame(
        {
            "g": oof["g"].to_numpy(),
            "y": oof["y"].to_numpy(dtype=float),
            "x": oof["excess_fwd"].to_numpy(dtype=float),
            "s": np.asarray(score, dtype=float),
        }
    )
    by = d.groupby("g", sort=False)
    d["rk"] = by["s"].rank(ascending=False, method="first")
    d["n"] = by["y"].transform("size")
    d["npos"] = by["y"].transform("sum")

    top1 = d[d["rk"] == 1]
    out = {
        "n_groups_all": len(top1),
        "precision@1": float(top1["y"].mean()) if len(top1) else np.nan,
        "excess@1": float(top1["x"].mean()) if len(top1) else np.nan,
    }

    ev = d[(d["npos"] > 0) & (d["npos"] < d["n"])]
    first_pos = ev[ev["y"] == 1].groupby("g")["rk"].min()
    out["n_groups"] = len(first_pos)
    if first_pos.empty:
        names = ["random_hit@1", "mrr"]
        names += [f"{m}@{k}" for k in ks for m in ("hit", "recall", "ndcg")]
        return out | dict.fromkeys(names, np.nan)

    head = ev.groupby("g")[["npos", "n"]].first().reindex(first_pos.index)
    out["random_hit@1"] = float((head["npos"] / head["n"]).mean())
    out["mrr"] = float((1.0 / first_pos).mean())
    for k in ks:
        topk = ev[ev["rk"] <= k]
        hits = topk.groupby("g")["y"].sum().reindex(first_pos.index, fill_value=0)
        gain = topk["y"] / np.log2(topk["rk"] + 1)
        dcg = gain.groupby(topk["g"]).sum().reindex(first_pos.index, fill_value=0)
        ideal = np.cumsum(1.0 / np.log2(np.arange(k) + 2))
        idcg = ideal[np.minimum(head["npos"].to_numpy(), k).astype(int) - 1]
        out[f"hit@{k}"] = float((first_pos <= k).mean())
        out[f"recall@{k}"] = float((hits / head["npos"]).mean())
        out[f"ndcg@{k}"] = float((dcg / idcg).mean())
    return out


def evaluate(oof, folds, kind):
    """Bảng theo quý + số tổng hợp gộp mọi nhóm, cho 3 cách chấm điểm."""
    variants = {
        "final": final_score(oof, kind),
        "behavior": oof["behavior"],
        "baseline": oof["baseline"],
    }
    rows = []
    for fold in folds:
        mask = (oof["q"].astype(str) == fold["quy"]).to_numpy()
        row = dict(fold, n_test=int(mask.sum()))
        for vname, score in variants.items():
            m = ranking_metrics(oof[mask], score[mask])
            row["n_groups"] = m["n_groups"]
            row["random_hit@1"] = m["random_hit@1"]
            row |= {f"{vname}_{key}": m[key] for key in REPORT_KEYS}
        rows.append(row)
    pooled = {v: ranking_metrics(oof, score) for v, score in variants.items()}
    return pd.DataFrame(rows), pooled


def rounded(d, digits=4):
    """Làm tròn số thực, bỏ NaN / vô cực để ghi JSON."""
    return {
        k: round(v, digits) if isinstance(v, float) else v
        for k, v in d.items()
        if not (isinstance(v, float) and not np.isfinite(v))
    }


# ---------------------------------------------------------------- lưu
def save(model, df, feats, name, rounds):
    model.save_model(f"{OUT_DIR}/model_{name}.txt")
    cats = [c for c in CATS if c in feats]
    meta = {
        "features": feats,
        "cats": cats,
        "categories": {c: [str(v) for v in df[c].cat.categories] for c in cats},
        "rounds": rounds,
    }
    with open(f"{OUT_DIR}/meta_{name}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    imp = pd.DataFrame({"feature": feats, "gain": model.feature_importance("gain")})
    imp = imp.sort_values("gain", ascending=False)
    imp.to_csv(f"{OUT_DIR}/importance_{name}.csv", index=False)
    return imp


def train_one(table, name, kind, research_cols, baseline_col, stride=1):
    log("s5", f"=== MODEL {name.upper()} ===")
    df, feats = prep(load_table(table, stride), research_cols)
    log(
        "s5",
        f"{len(df):,} dòng | {len(feats)} đặc trưng | "
        f"nhãn dương {100 * df.y.mean():.2f}%",
    )

    oof, folds = walk_forward(df, feats, name, baseline_col)
    rounds = int(np.median([f["rounds"] for f in folds])) if folds else DEFAULT_ROUNDS
    result = None
    if folds:
        evaluation, pooled = evaluate(oof, folds, kind)
        evaluation.to_csv(f"{OUT_DIR}/eval_{name}.csv", index=False)
        f, b, r = pooled["final"], pooled["behavior"], pooled["baseline"]
        result = {
            "auc": round(float(np.mean([x["auc"] for x in folds])), 4),
            "n_folds": len(folds),
            "n_groups": f["n_groups"],
            "n_groups_all": f["n_groups_all"],
            "random_hit@1": round(f["random_hit@1"], 4),
            "rounds": rounds,
            **{v: rounded(m) for v, m in pooled.items()},
        }
        log(
            "s5",
            f"[{name}] {f['n_groups']:,} nhóm xếp hạng được | HitRate@1: "
            f"ngẫu nhiên {f['random_hit@1']:.3f} | baseline {r['hit@1']:.3f} | "
            f"hành vi {b['hit@1']:.3f} | điểm cuối {f['hit@1']:.3f}",
        )

    importance = save(fit(df, feats, rounds=rounds), df, feats, name, rounds)
    log(
        "s5",
        f"[{name}] model cuối {rounds} vòng | 8 đặc trưng mạnh nhất: "
        f"{', '.join(importance.head(8).feature)}",
    )
    return result


def main():
    summary = {
        "market_weight": MARKET_WEIGHT,
        "buy": train_one(
            "train_buy",
            "buy",
            kind="hybrid",
            research_cols=BUY_RESEARCH_COLS,
            baseline_col="market_score_norm",
        ),
        "portfolio": train_one(
            "train_portfolio",
            "portfolio",
            kind="research_first",
            research_cols=PORTFOLIO_RESEARCH_COLS,
            baseline_col="research_sell_signal",
            stride=PORTFOLIO_TRAIN_STRIDE,
        ),
    }
    with open(f"{OUT_DIR}/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log("s5", "xong.")


if __name__ == "__main__":
    main()
