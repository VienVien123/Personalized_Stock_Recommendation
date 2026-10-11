"""
Bước 1 — ETL
Đọc 7 file CSV thô (6 file dữ liệu + nhật ký khuyến nghị PHS) -> làm sạch -> chuẩn hoá đơn vị -> ghi vào DuckDB.

Xử lý sẵn 3 cái bẫy của bộ dữ liệu:
  1. BOM UTF-8 ở đầu 3 file  -> đọc bằng encoding utf-8-sig
  2. Đơn vị giá khác nhau    -> quy hết về VND (bảng giá x 1000)
  3. Giá lệnh ngoài biên độ  -> gắn cờ is_outlier, KHÔNG xoá

Dữ liệu ở bước này nhỏ (dưới 1 triệu dòng mỗi bảng) nên xử lý bằng pandas cho dễ
đọc; mỗi bảng là một hàm. Các bước sau làm việc với bảng hàng chục triệu dòng nên
vẫn dùng SQL của DuckDB.
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
    columns = ["stock_code", "company_name", "exchange", "instrument_type"]
    return sec[[*columns, "icb_code", "icb_name"]]


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
            "o": raw["open"] * 1000,
            "h": raw["high"] * 1000,
            "l": raw["low"] * 1000,
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
def load_transactions(px: pd.DataFrame) -> pd.DataFrame:
    """Lệnh mua/bán của khách (giá đã là VND) + cờ giá nằm ngoài biên độ phiên."""
    text = ["transaction_id", "customer_id", "stock_code", "side", "exchange"]
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
            "exchange": raw["exchange"],
        }
    )
    txn["value"] = txn["qty"] * txn["price"]

    # cờ bất thường: giá lệnh nằm ngoài khoảng thấp nhất - cao nhất của phiên
    day = txn[["stock_code", "d"]].merge(
        px[["stock_code", "d", "h", "l"]], on=["stock_code", "d"], how="left"
    )
    outside = (txn["price"].to_numpy() > day["h"].to_numpy()) | (
        txn["price"].to_numpy() < day["l"].to_numpy()
    )
    txn["is_outlier"] = pd.array(outside, dtype="boolean")
    txn.loc[day["h"].isna().to_numpy(), "is_outlier"] = pd.NA  # không có giá phiên
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
    pnl = raw["realizedProfitLoss"].str.strip().replace("", np.nan).str.replace("%", "")
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
            "realized_pnl_pct": pnl.astype(float),
        }
    )

    def numbered(events):
        """Đánh số k = 0, 1, 2, ... theo thời gian trong từng mã."""
        events = events.sort_values(["stock_code", "d", "target_price"])
        return events.assign(k=events.groupby("stock_code").cumcount())

    opens = numbered(ev[ev["call_type"] == "BUY"])
    closes = numbered(ev[ev["call_type"].isin(["TAKE_PROFIT", "CUT_LOSS"])])
    calls = opens.merge(
        closes[["stock_code", "k", "d", "call_type", "realized_pnl_pct"]],
        on=["stock_code", "k"],
        how="left",
        suffixes=("", "_close"),
    )
    # lệnh đóng phải đến sau lệnh mở, nếu không coi như chưa đóng
    bad = calls["d_close"] < calls["d"]
    calls.loc[bad, ["d_close", "call_type_close", "realized_pnl_pct_close"]] = np.nan

    return pd.DataFrame(
        {
            "stock_code": calls["stock_code"],
            "open_d": calls["d"],
            "close_d": calls["d_close"],
            "close_type": calls["call_type_close"],
            "entry_price": calls["price"],
            "target_price": calls["target_price"],
            "cut_loss_price": calls["cut_loss_price"],
            "realized_pnl_pct": calls["realized_pnl_pct_close"],
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
    buy["candidate_rank"] = buy.groupby("t")["market_score"].rank(
        method="min", ascending=False
    )
    buy = buy.assign(recommendation="BUY", call_type="BUY", call_date=buy["open_d"])

    sell = t.merge(calls, left_on="t", right_on="close_d").drop_duplicates(key)
    sell = sell.assign(
        recommendation="SELL", call_type=sell["close_type"], call_date=sell["close_d"]
    )

    hold = t.merge(calls, how="cross")
    still_open = (hold["open_d"] < hold["t"]) & (
        hold["close_d"].isna() | (hold["close_d"] > hold["t"])
    )
    hold = hold[still_open].sort_values("open_d", ascending=False)
    hold = without(without(hold, buy, key), sell, key).drop_duplicates(key)
    hold = hold.assign(
        recommendation="HOLD", call_type="HOLD", call_date=hold["open_d"]
    )

    out = pd.concat([buy, without(sell, buy, key), hold], ignore_index=True)
    columns = ["d", "stock_code", "market_score", "recommendation", "candidate_rank"]
    columns += [
        "call_type",
        "call_date",
        "entry_price",
        "target_price",
        "cut_loss_price",
    ]
    return out.rename(columns={"t": "d"})[columns]


def without(df, other, key):
    """Các dòng của df mà khoá `key` không xuất hiện trong other."""
    hit = df[key].merge(
        other[key].drop_duplicates(), on=key, how="left", indicator=True
    )
    return df[(hit["_merge"] == "left_only").to_numpy()]


# ---------------------------------------------------------------- đặc trưng giá
def window_start(key: np.ndarray, back: int) -> np.ndarray:
    """Vị trí dòng đầu tiên có số phiên >= dn - back (cùng mã). `key` đã sắp tăng."""
    return np.searchsorted(key, key - back, side="left")


def window_end(key: np.ndarray, ahead: int) -> np.ndarray:
    """Vị trí dòng cuối cùng có số phiên <= dn + ahead (cùng mã)."""
    return np.searchsorted(key, key + ahead, side="right") - 1


def price_features(px, idx, daycal) -> pd.DataFrame:
    """Đặc trưng giá theo phiên + kết quả lợi suất tương lai để audit.

    y_stock/excess_fwd KHÔNG đi vào model hành vi. Chúng chỉ được dùng để đo kết quả
    và tính tay nghề lịch sử khi cửa sổ 20 phiên đã hoàn tất trước t.
    Cửa sổ tính theo số phiên thị trường (dn), không theo số dòng, nên không bị lệch
    khi một mã nghỉ giao dịch.
    """
    df = px[["stock_code", "d", "c", "v"]].merge(daycal, on="d")
    df = df.sort_values(["stock_code", "dn"], ignore_index=True)
    # khoá (mã, dn) tăng dần: tìm biên cửa sổ bằng tìm kiếm nhị phân, không lặp từng mã
    stock_id = df["stock_code"].astype("category").cat.codes.to_numpy().astype(np.int64)
    key = stock_id * 1_000_000 + df["dn"].to_numpy()
    c = df["c"].to_numpy()
    df["r1"] = df["c"] / df.groupby("stock_code")["c"].shift(1) - 1

    # tổng trong cửa sổ 20 phiên = tổng dồn tới dòng hiện tại - tổng dồn trước đầu cửa sổ
    start20 = window_start(key, 20)

    def sum20(values: pd.Series) -> np.ndarray:
        incl = values.groupby(df["stock_code"]).cumsum().to_numpy()
        excl = incl - values.to_numpy()
        return incl - excl[start20]

    r1 = df["r1"].fillna(0.0)
    n = np.arange(len(df)) - start20 + 1  # số phiên có giao dịch trong cửa sổ
    n_r1 = sum20(df["r1"].notna().astype(float))
    s1, s2 = sum20(r1), sum20(r1**2)
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (s2 - s1**2 / n_r1) / (n_r1 - 1)
    vol_20 = np.where(n_r1 >= 2, np.sqrt(np.clip(var, 0, None)), np.nan)

    # VNINDEX: giá hôm nay và sau HORIZON phiên
    vni = idx[idx["index_code"] == "VNINDEX"].merge(daycal, on="d").sort_values("dn")
    vni_key = vni["dn"].to_numpy()
    vni = pd.DataFrame(
        {
            "dn": vni_key,
            "idx_c": vni["c"].to_numpy(),
            "idx_c_fwd": vni["c"].to_numpy()[window_end(vni_key, HORIZON)],
        }
    )

    df = df.assign(
        c_5a=c[window_start(key, 5)],
        c_20a=c[start20],
        c_60a=c[window_start(key, 60)],
        c_fwd=c[window_end(key, HORIZON)],
        vol_20=vol_20,
        v_avg20=sum20(df["v"].astype(float)) / n,
        up_ratio_20=sum20((df["r1"] > 0).astype(float)) / n,
        n_sessions_20=n,
    ).merge(vni, on="dn")

    stock_growth = df["c_fwd"] / df["c"]
    index_growth = df["idx_c_fwd"] / df["idx_c"]
    return pd.DataFrame(
        {
            "stock_code": df["stock_code"],
            "d": df["d"],
            "dn": df["dn"],
            "c": df["c"],
            "v": df["v"],
            "ret_5": df["c"] / df["c_5a"] - 1,
            "ret_20": df["c"] / df["c_20a"] - 1,
            "ret_60": df["c"] / df["c_60a"] - 1,
            "vol_20": df["vol_20"],
            "up_ratio_20": df["up_ratio_20"],
            "n_sessions_20": df["n_sessions_20"],
            "vol_ratio": df["v"] / df["v_avg20"].replace(0, np.nan),
            "turnover": df["v"] * df["c"],
            "fwd_ret": stock_growth - 1,
            "idx_fwd_ret": index_growth - 1,
            "excess_fwd": stock_growth - index_growth,
            # kết quả audit: mã này có thắng VNINDEX trong HORIZON phiên tới không?
            "y_stock": (stock_growth > index_growth).astype("int32"),
        }
    )


# ---------------------------------------------------------------- chạy
def main():
    log("s1", "đọc CSV thô...")
    sec = load_securities()
    px = load_prices()
    idx = load_index()
    daycal = trading_calendar(px)
    dates = daycal.rename(columns={"d": "t"})  # mỗi phiên là một mốc t
    txn = load_transactions(px)
    prof = load_profiles()
    calls = pair_phs_calls()
    research = research_signals(dates, calls, px)

    log("s1", "tính đặc trưng giá theo phiên (đà giá, biến động, thanh khoản)...")
    sd = price_features(px, idx, daycal)

    tables = {
        "sec": (sec, None),
        "px": (px, None),
        "idx": (idx, None),
        "daycal": (daycal, None),
        "txn": (txn, {"ts": "TIME"}),
        "prof": (prof, None),
        "decision_dates": (dates, None),
        "phs_calls": (calls, None),
        "research": (research, {"candidate_rank": "INTEGER"}),
        "sd": (sd, None),
    }
    con = connect()
    for name, (df, casts) in tables.items():
        write_table(con, name, df, casts)
    con.execute("CREATE INDEX IF NOT EXISTS sd_idx ON sd(stock_code, dn);")
    con.close()
    log("s1", "xong.")


if __name__ == "__main__":
    main()
