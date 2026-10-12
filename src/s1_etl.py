"""Bước 1 — đọc 6 file CSV, làm sạch, ghi vào DuckDB. Mỗi bảng một hàm (pandas).

File có BOM UTF-8 -> đọc bằng utf-8-sig; file giá tính nghìn VND -> nhân 1000.
"""

import numpy as np
import pandas as pd

from config import (
    DATA_DIR,
    HORIZON,
    connect,
    log,
    write_table,
)


def read_csv(name: str, **kwargs) -> pd.DataFrame:
    # round_trip: đọc số thực chính xác tới chữ số cuối, giống DuckDB
    return pd.read_csv(
        f"{DATA_DIR}/{name}",
        encoding="utf-8-sig",
        float_precision="round_trip",
        **kwargs,
    )


# ---------------------------------------------------------------- danh mục, giá
def load_securities() -> pd.DataFrame:
    sec = read_csv("securities_master.csv", dtype=str)
    return sec[["stock_code", "company_name", "exchange", "icb_code", "icb_name"]]


def load_prices() -> pd.DataFrame:
    """Giá cổ phiếu theo ngày; file tính bằng nghìn VND -> đổi sang VND."""
    raw = read_csv(
        "stock_market_daily.csv", dtype={"stock_code": str}, parse_dates=["date"]
    )
    raw = raw[raw["close"] > 0]
    return pd.DataFrame(
        {
            "d": raw["date"],
            "stock_code": raw["stock_code"],
            "c": raw["close"] * 1000,
            "v": raw["volume"].astype("int64"),
        }
    )


def load_index() -> pd.DataFrame:
    raw = read_csv(
        "market_index_daily.csv", dtype={"index_code": str}, parse_dates=["date"]
    )
    return raw.rename(columns={"date": "d", "close": "c"})[["d", "index_code", "c"]]


def trading_calendar(px: pd.DataFrame) -> pd.DataFrame:
    """Mọi ngày có giao dịch, đánh số phiên dn = 1, 2, 3, ..."""
    days = np.sort(px["d"].unique())
    return pd.DataFrame({"d": days, "dn": np.arange(1, len(days) + 1)})


# ---------------------------------------------------------------- khách hàng
def load_transactions() -> pd.DataFrame:
    """Lệnh mua/bán của khách (giá đã là VND)."""
    text = ["transaction_id", "customer_id", "stock_code", "side"]
    raw = read_csv(
        "customer_transactions_raw.csv",
        dtype=dict.fromkeys([*text, "execution_time"], str),
        parse_dates=["trade_date"],
    )
    txn = pd.DataFrame(
        {
            "transaction_id": raw["transaction_id"],
            "customer_id": raw["customer_id"],
            "stock_code": raw["stock_code"],
            "side": raw["side"],
            "qty": raw["quantity"].astype("int64"),
            "price": raw["price"].astype(float),
            "d": raw["trade_date"],
            "ts": raw["execution_time"],
        }
    )
    txn["value"] = txn["qty"] * txn["price"]
    return txn


def load_profiles() -> pd.DataFrame:
    """Hồ sơ khách: chỉ còn loại khách + ngày mở tài khoản. Kỳ hạn đầu tư không còn
    được khai báo sẵn mà suy ra từ hành vi nắm giữ thực tế ở s2 (cust_horizon)."""
    raw = read_csv(
        "customer_profile.csv",
        dtype={"customer_id": str, "customer_type": str},
        parse_dates=["account_open_date"],
    )
    return raw.rename(columns={"account_open_date": "open_date"})[
        ["customer_id", "customer_type", "open_date"]
    ]


# ---------------------------------------------------------------- khuyến nghị PHS
def mid_price(text: str) -> float:
    """Vùng giá "20.0 - 20.5" -> điểm giữa 20.25; một số "21" -> 21."""
    parts = [float(x) for x in str(text).split("-")]
    return sum(parts) / len(parts)


def pair_phs_calls() -> pd.DataFrame:
    """Ghép lệnh MỞ với lệnh ĐÓNG của nhật ký khuyến nghị PHS.

    File gốc là nhật ký sự kiện: BUY mở một khuyến nghị, TAKE_PROFIT/CUT_LOSS đóng
    nó. Lệnh BUY thứ k của một mã được đóng bởi lệnh đóng thứ k của mã đó.
    Giá trong file tính theo nghìn VND.
    """
    raw = read_csv("candidate_stocks_phs_skill.csv", dtype=str)
    ev = pd.DataFrame(
        {
            "stock_code": raw["symbol"].str.strip().str.upper(),
            "call_type": raw["recommendationType"].str.strip().str.upper(),
            "d": pd.to_datetime(
                raw["recommendationDate"].str.strip(), format="%d/%m/%Y"
            ),
            "price": raw["recommendationPrice"].map(mid_price) * 1000,
            "target_price": raw["targetPrice"].astype(float) * 1000,
            "cut_loss_price": raw["cutLossPrice"].astype(float) * 1000,
        }
    )

    def numbered(events):
        """Đánh số k = 0, 1, 2, ... theo thời gian trong từng mã."""
        events = events.sort_values(["stock_code", "d", "target_price"])
        return events.assign(k=events.groupby("stock_code").cumcount())

    opens = numbered(ev[ev["call_type"] == "BUY"])
    closes = numbered(ev[ev["call_type"].isin(["TAKE_PROFIT", "CUT_LOSS"])])
    calls = opens.merge(
        closes[["stock_code", "k", "d", "call_type"]],
        on=["stock_code", "k"],
        how="left",
        suffixes=("", "_close"),
    )
    # lệnh đóng phải đến sau lệnh mở, nếu không coi như chưa đóng
    bad = calls["d_close"] < calls["d"]
    calls.loc[bad, ["d_close", "call_type_close"]] = np.nan

    return pd.DataFrame(
        {
            "stock_code": calls["stock_code"],
            "open_d": calls["d"],
            "close_d": calls["d_close"],
            "close_type": calls["call_type_close"],
            "entry_price": calls["price"],
            "target_price": calls["target_price"],
            "cut_loss_price": calls["cut_loss_price"],
        }
    )


def research_signals(dates, calls, px) -> pd.DataFrame:
    """Tín hiệu nghiên cứu tại từng mốc t (point-in-time), mỗi (t, mã) một tín hiệu:
      BUY  : PHS mở khuyến nghị mới đúng ngày t.
      SELL : PHS chốt lời/cắt lỗ đúng ngày t (BUY cùng ngày được ưu tiên).
      HOLD : khuyến nghị BUY cũ còn mở, chưa có tín hiệu mới trong ngày.
    Không lặp lại BUY cũ thành khuyến nghị mua mới ở các ngày sau.
    """
    t = dates[["t"]]
    key = ["t", "stock_code"]

    buy = t.merge(calls, left_on="t", right_on="open_d")
    buy = buy.merge(
        px[["stock_code", "d", "c"]].rename(columns={"d": "t"}), on=key, how="left"
    ).drop_duplicates(key)
    # MarketScore = dư địa tăng (%) từ giá đóng cửa ngày t tới giá mục tiêu
    price_now = buy["c"].fillna(buy["entry_price"]).replace(0, np.nan)
    buy["market_score"] = 100.0 * (buy["target_price"] / price_now - 1)
    buy = buy.assign(recommendation="BUY", call_type="BUY")

    sell = t.merge(calls, left_on="t", right_on="close_d").drop_duplicates(key)
    sell = sell.assign(recommendation="SELL", call_type=sell["close_type"])

    hold = t.merge(calls, how="cross")
    still_open = (hold["open_d"] < hold["t"]) & (
        hold["close_d"].isna() | (hold["close_d"] > hold["t"])
    )
    hold = hold[still_open].sort_values("open_d", ascending=False)
    hold = without(without(hold, buy, key), sell, key).drop_duplicates(key)
    hold = hold.assign(recommendation="HOLD", call_type="HOLD")

    out = pd.concat([buy, without(sell, buy, key), hold], ignore_index=True)
    columns = ["d", "stock_code", "market_score", "recommendation", "call_type"]
    return out.rename(columns={"t": "d"})[[*columns, "target_price", "cut_loss_price"]]


def without(df, other, key):
    """Các dòng của df mà khoá `key` không xuất hiện trong other."""
    hit = df[key].merge(
        other[key].drop_duplicates(), on=key, how="left", indicator=True
    )
    return df[(hit["_merge"] == "left_only").to_numpy()]


# ---------------------------------------------------------------- đặc trưng giá
def price_features(px, idx, daycal) -> pd.DataFrame:
    """Đặc trưng giá theo phiên + excess_fwd (mã vượt VNINDEX bao nhiêu trong HORIZON
    phiên tới — chỉ để chấm điểm ở S5, không đưa vào model).
    Cửa sổ tính theo số phiên thị trường (dn), không theo số dòng, nên không bị lệch
    khi một mã nghỉ giao dịch.
    """
    df = px[["stock_code", "d", "c", "v"]].merge(daycal, on="d")
    df = df.sort_values(["stock_code", "dn"], ignore_index=True)
    # bảng rộng: mỗi hàng một phiên, mỗi cột một mã; ô trống = mã không giao dịch phiên đó
    c = df.pivot(index="dn", columns="stock_code", values="c").reindex(daycal["dn"])
    v = df.pivot(index="dn", columns="stock_code", values="v").reindex(daycal["dn"])
    traded = c.notna()

    def ago(k):
        """Giá k phiên trước; phiên đó mã nghỉ thì lấy phiên có giá kế sau."""
        nxt = c.bfill()
        return nxt.shift(k).fillna(nxt.iloc[0])

    def ahead(prices, k):
        """Giá sau k phiên; phiên đó không có giá thì lấy phiên có giá gần nhất trước."""
        last = prices.ffill()
        return last.shift(-k).fillna(last.iloc[-1])

    def last20(values):
        """Cửa sổ từ dn-20 tới dn, bỏ qua các phiên mã không giao dịch."""
        return values.rolling(21, min_periods=1)

    r1 = c / c.ffill().shift(1) - 1  # so với lần giao dịch liền trước của mã
    vni = idx[idx["index_code"] == "VNINDEX"].merge(daycal, on="d")
    vni = vni.set_index("dn")["c"].reindex(daycal["dn"])

    wide = {
        "ret_5": c / ago(5) - 1,
        "ret_20": c / ago(20) - 1,
        "ret_60": c / ago(60) - 1,
        "vol_20": r1.rolling(21, min_periods=2).std(),
        "up_ratio_20": last20((r1 > 0).where(traded)).mean(),
        "n_sessions_20": last20(traded).sum().astype("int64"),
        "vol_ratio": v / last20(v).mean().replace(0, np.nan),
        "turnover": v * c,
        "excess_fwd": (ahead(c, HORIZON) / c).sub(ahead(vni, HORIZON) / vni, axis=0),
    }
    # đưa về dạng dài: mỗi dòng một (mã, phiên) có giao dịch
    row, col = df["dn"].to_numpy() - 1, c.columns.get_indexer(df["stock_code"])
    out = df[["stock_code", "d", "dn"]].assign(
        **{name: w.to_numpy()[row, col] for name, w in wide.items()}
    )
    # phiên thiếu VNINDEX thì không tính được excess_fwd -> bỏ
    return out[out["dn"].isin(vni.dropna().index)].reset_index(drop=True)


# ---------------------------------------------------------------- chạy
def main():
    log("s1", "đọc CSV thô...")
    sec = load_securities()
    px = load_prices()
    idx = load_index()
    daycal = trading_calendar(px)
    dates = daycal.rename(columns={"d": "t"})  # mỗi phiên là một mốc t
    txn = load_transactions()
    prof = load_profiles()
    calls = pair_phs_calls()
    research = research_signals(dates, calls, px)

    log("s1", "tính đặc trưng giá theo phiên (đà giá, biến động, thanh khoản)...")
    sd = price_features(px, idx, daycal)

    tables = {
        "sec": (sec, None),
        "px": (px[["d", "stock_code", "c"]], None),  # S2 chỉ cần giá đóng cửa
        "daycal": (daycal, None),
        "txn": (txn, {"ts": "TIME"}),
        "prof": (prof, None),
        "decision_dates": (dates, None),
        "phs_calls": (calls[["stock_code", "open_d", "close_d"]], None),
        "research": (research, None),
        "sd": (sd, None),
    }
    con = connect()
    for name, (df, casts) in tables.items():
        write_table(con, name, df, casts)
    con.close()
    log("s1", "xong.")


if __name__ == "__main__":
    main()
