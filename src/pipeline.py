"""Chạy toàn bộ pipeline: ETL -> trạng thái -> đồ thị -> đặc trưng -> huấn luyện."""
import os, sys, time
import s1_etl, s2_state, s3_graph, s4_features, s5_train
from config import DATA_DIR, OUT_DIR, log

DONE = os.path.join(OUT_DIR, "_pipeline_done")
PIPELINE_VERSION = "research-behavior-v2"

REQUIRED = [
    "securities_master.csv",
    "stock_market_daily.csv",
    "market_index_daily.csv",
    "customer_profile.csv",
    "customer_transactions_raw.csv",
    "customer_holdings.csv",
    "candidate_stocks_research.csv",
]


def check_data():
    """Báo sớm và rõ nếu thiếu file CSV, thay vì để lỗi khó hiểu ở giữa pipeline."""
    missing = [f for f in REQUIRED if not os.path.exists(os.path.join(DATA_DIR, f))]
    if not missing:
        return
    log("pipeline", f"THIẾU {len(missing)}/{len(REQUIRED)} file dữ liệu trong thư mục data/:")
    for f in missing:
        log("pipeline", f"    - {f}")
    log("pipeline", "Đặt đủ 7 file CSV vào thư mục data/ rồi chạy lại:")
    log("pipeline", "    docker compose restart app")
    sys.exit(1)


def run(force: bool = False):
    if os.path.exists(DONE) and not force:
        try:
            with open(DONE, encoding="utf-8") as f:
                done_version = f.read().strip().split("|")[0]
        except OSError:
            done_version = ""
        if done_version == PIPELINE_VERSION:
            log("pipeline", "đã chạy đúng phiên bản hiện tại — bỏ qua "
                            "(xoá artifacts/_pipeline_done để chạy lại).")
            return
        log("pipeline", "artifacts thuộc phiên bản cũ — tự chạy lại pipeline.")
    check_data()
    t0 = time.time()
    for step in (s1_etl, s2_state, s3_graph, s4_features, s5_train):
        s = time.time()
        step.main()
        log("pipeline", f"{step.__name__} xong sau {time.time()-s:.0f}s")
    with open(DONE, "w", encoding="utf-8") as f:
        f.write(f"{PIPELINE_VERSION}|{time.time()}")
    log("pipeline", f"HOÀN TẤT sau {(time.time()-t0)/60:.0f} phút")


if __name__ == "__main__":
    run(force="--force" in sys.argv)
