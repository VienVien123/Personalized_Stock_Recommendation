"""Chạy pipeline: ETL -> trạng thái -> đồ thị -> đặc trưng -> huấn luyện.

python pipeline.py                chạy cả 5 bước
python pipeline.py --step TÊN     chạy đúng một bước (Airflow dùng cách này)
python pipeline.py --clear-done | --mark-done
"""

import os
import sys
import time

import s1_etl
import s2_state
import s3_graph
import s4_features
import s5_train
from config import OUT_DIR, log

STEPS = {
    "s1_etl": s1_etl,
    "s2_state": s2_state,
    "s3_graph": s3_graph,
    "s4_features": s4_features,
    "s5_train": s5_train,
}
# Dấu hoàn tất: app chỉ khởi động khi file này có đúng phiên bản dưới đây.
DONE = os.path.join(OUT_DIR, "_pipeline_done")
PIPELINE_VERSION = "phs-v6-day"


def is_done() -> bool:
    try:
        with open(DONE, encoding="utf-8") as f:
            return f.read().strip().split("|")[0] == PIPELINE_VERSION
    except OSError:
        return False


def wait_until_done(poll_seconds: int = 15):
    """App gọi hàm này lúc khởi động: CHỜ pipeline chạy xong chứ không tự chạy."""
    if not is_done():
        log("pipeline", "chưa có kết quả — app CHỜ. Bấm DAG pipeline_khuyen_nghi.")
    while not is_done():
        time.sleep(poll_seconds)


def mark_done():
    with open(DONE, "w", encoding="utf-8") as f:
        f.write(f"{PIPELINE_VERSION}|{time.time()}")


def clear_done():
    """Xoá dấu trước khi chạy lại: lần chạy hỏng giữa chừng không bị coi là hợp lệ."""
    if os.path.exists(DONE):
        os.remove(DONE)


def run_step(name: str):
    start = time.time()
    STEPS[name].main()
    log("pipeline", f"{name} xong sau {time.time() - start:.0f}s")


def main(argv):
    if "--clear-done" in argv:
        clear_done()
    elif "--mark-done" in argv:
        mark_done()
    elif "--step" in argv:
        run_step(argv[argv.index("--step") + 1])
    else:
        clear_done()
        for name in STEPS:
            run_step(name)
        mark_done()


if __name__ == "__main__":
    main(sys.argv[1:])
