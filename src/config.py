"""Cấu hình dùng chung. Tham số là hằng số cố định — muốn đổi thì sửa file này.
Chỉ đường dẫn và tài khoản Neo4j đọc từ biến môi trường."""

import os
import re

DATA_DIR = os.getenv("DATA_DIR", "/app/data")
OUT_DIR = os.getenv("OUT_DIR", "/app/artifacts")
DUCKDB_PATH = os.getenv("DUCKDB_PATH", "/app/artifacts/vienvien.duckdb")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "changeme")  # giá trị thật nằm trong .env

# ---------- Bài toán ----------
# Mỗi phiên giao dịch là một mốc ra quyết định t.
HORIZON = 20  # số phiên nhìn tới để gắn nhãn
TOPK = 10  # số mã BUY hiển thị trên app
# Điểm cuối nhánh MUA = MARKET_WEIGHT * điểm PHS + (1 - MARKET_WEIGHT) * điểm hành vi
MARKET_WEIGHT = 0.60
BEHAVIOR_HALF_LIFE = 90  # sau bấy nhiêu ngày, trọng số một lệnh cũ giảm một nửa
RECENT_WINDOW = 90  # cửa sổ (ngày) đo mức độ giao dịch gần đây của khách
BUY_FEE = 0.0015  # phí mua tính vào giá vốn — khớp avg_cost trong customer_holdings.csv

# Kỳ hạn đầu tư suy ra từ thời gian nắm giữ thực tế (xem s2_state.build_horizon)
SHORT_HOLD_DAYS = 30  # bán hết trong <= 1 tháng -> bằng chứng ngắn hạn
LONG_HOLD_DAYS = 90  # giữ > 3 tháng -> bằng chứng dài hạn
HOLD_EVIDENCE_THRESHOLD = 0.50  # tỷ lệ bằng chứng tối thiểu để kết luận SHORT / LONG
HOLD_MIN_EVIDENCE = 3  # số vòng mua-bán tối thiểu trước khi dám kết luận

# ---------- Đồ thị (S3) ----------
COOC_WINDOW = 180  # mỗi lát cắt nhìn bấy nhiêu ngày giao dịch trước đó
COOC_TOPN = 25  # số mã đồng mua mạnh nhất giữ lại cho mỗi mã
PEER_TOPK = 20  # số khách tương tự giữ lại cho mỗi khách
PEER_MIN_SHARED = 2  # số mã mua chung tối thiểu để hai khách được coi là giống
GRAPH_WORKERS = 4  # số lát cắt gửi tới Neo4j cùng lúc

# ---------- Trọng số đặt tay của các điểm thành phần (chỉ là ĐẶC TRƯNG cho model) ----------
GRAPH_SCORE_WEIGHTS = {"cooc": 0.75, "buyer_pop": 0.25}
HISTORY_PREF_WEIGHTS = {"value_share": 0.40, "count_share": 0.30, "recency": 0.30}
PERSONALIZATION_WEIGHTS = {"graph": 0.50, "suitability": 0.30, "history": 0.20}
# độ rộng (mẫu số trong EXP(-|lệch| / scale)) khi so mã với phong cách của khách
STYLE_SCALES = {"vol_20": 0.03, "ret_20": 0.15, "log_turnover": 5.0}
# Nhánh danh mục: bậc của tín hiệu PHS; behavior chỉ phá hoà trong cùng bậc
# (RESEARCH_TIEBREAK < 0.5 nên không lật được thứ tự giữa hai bậc).
RESEARCH_SELL_SIGNAL = {"SELL": 1.5, "BUY": 0.5, "HOLD": 0.0, "NONE": -0.5}
RESEARCH_TIEBREAK = 0.49

# ---------- Huấn luyện ----------
EVAL_KS = (1, 3)  # các mức K để đo xếp hạng
# Model danh mục chỉ học trên mỗi phiên thứ 5: bảng train_portfolio ~8,7 triệu dòng,
# nạp hết vượt trần 5 GB RAM, mà các phiên liền nhau gần như lặp lại nhau.
PORTFOLIO_TRAIN_STRIDE = 5

# ---------- Tài nguyên ----------
GRADIO_PORT = 7860
# Container pipeline có 10 GB RAM: DuckDB dùng 7 GB + phần thêm theo từng luồng.
DUCKDB_MEMORY = "7GB"
DUCKDB_THREADS = 8
# S2/S4 ghép "mọi mốc x lịch sử trước mốc" theo lô bấy nhiêu mốc, không thì hết RAM.
DATE_CHUNK = 50
# Chỗ DuckDB tràn dữ liệu ra đĩa — trong container, không trong artifacts/ (thư mục
# gắn từ Windows ghi chậm và từng làm s2 hỏng với "Cannot allocate memory").
TMP_DIR = os.getenv("DUCKDB_TEMP_DIR", "/tmp/duckdb_spill")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)


def connect(read_only: bool = False):
    """Mở DuckDB với giới hạn bộ nhớ và cho phép tràn ra đĩa.
    Container chỉ có vài GB RAM nên bắt buộc, nếu không sẽ bị OOM (exit 137)."""
    import duckdb

    con = duckdb.connect(DUCKDB_PATH, read_only=read_only)
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY}'")
    con.execute(f"SET threads={DUCKDB_THREADS}")
    con.execute(f"SET temp_directory='{TMP_DIR}'")
    con.execute("SET preserve_insertion_order=false")
    return con


def write_table(con, name: str, df, casts: dict | None = None) -> None:
    """Ghi một DataFrame thành bảng DuckDB, giữ nguyên thứ tự cột.

    Cột ngày của pandas (datetime64) được lưu thành DATE; `casts` ép kiểu thêm cho
    từng cột khi kiểu pandas không khớp kiểu mong muốn (ví dụ {"ts": "TIME"}).
    """
    import pandas as pd

    casts = dict(casts or {})
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            casts.setdefault(c, "DATE")
    cols = ", ".join(
        f'CAST("{c}" AS {casts[c]}) AS "{c}"' if c in casts else f'"{c}"'
        for c in df.columns
    )
    con.register("_write_df", df)
    con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT {cols} FROM _write_df")
    con.unregister("_write_df")


_CREATE = re.compile(r"\s*CREATE OR REPLACE TABLE (\w+) AS", re.IGNORECASE)


def create_by_date_chunks(
    con, sql: str, chunk: int = DATE_CHUNK, slice_tables: tuple = ()
) -> None:
    """Dựng một bảng "mọi mốc x lịch sử trước mốc" theo từng lô DATE_CHUNK mốc.

    Các bảng này ghép decision_dates với toàn bộ giao dịch có ngày < t. Ở nhịp
    ngày (khoảng 1.000 mốc) một lệnh duy nhất tạo ra hàng trăm triệu dòng trung
    gian và container bị kill vì hết RAM (mã 137). Mọi phép gộp / cửa sổ trong
    các lệnh đó đều theo từng t, nên chia mốc thành lô cho kết quả y hệt.

    `sql` phải có dạng `CREATE OR REPLACE TABLE x AS SELECT ... decision_dates dd ...`.
    `slice_tables`: các bảng có cột t được ghép trong `sql`. Mỗi lô chỉ đọc phần
    của bảng đó thuộc các mốc trong lô; không thì DuckDB dựng bảng băm trên cả
    bảng (hàng chục triệu dòng ở nhịp ngày) và vượt trần bộ nhớ.
    """
    m = _CREATE.match(sql)
    if not m or "decision_dates dd" not in sql:
        raise ValueError("create_by_date_chunks cần CREATE OR REPLACE TABLE ... dd")
    table = m.group(1)
    select = sql[m.end() :].replace("decision_dates dd", "_dd dd")
    for name in slice_tables:
        select, n_sub = re.subn(rf"\bJOIN\s+{name}\b", f"JOIN _dd_{name}", select)
        if not n_sub:
            raise ValueError(f"không thấy 'JOIN {name}' trong lệnh dựng {table}")
    n = con.execute("SELECT COUNT(*) FROM decision_dates").fetchone()[0]
    for i, lo in enumerate(range(0, max(n, 1), chunk)):
        con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _dd AS
        SELECT * FROM decision_dates ORDER BY t LIMIT {chunk} OFFSET {lo}""")
        for name in slice_tables:
            con.execute(f"""
            CREATE OR REPLACE TEMP VIEW _dd_{name} AS
            SELECT * FROM {name} WHERE t IN (SELECT t FROM _dd)""")
        head = (
            f"CREATE OR REPLACE TABLE {table} AS" if i == 0 else f"INSERT INTO {table}"
        )
        con.execute(f"{head} {select}")
    for name in slice_tables:
        con.execute(f"DROP VIEW IF EXISTS _dd_{name}")
    con.execute("DROP TABLE IF EXISTS _dd")


def log(step: str, msg: str) -> None:
    print(f"[{step}] {msg}", flush=True)
