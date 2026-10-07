#!/usr/bin/env bash
# Khởi động Airflow với tài khoản quản trị lấy từ .env:
#   AIRFLOW_ADMIN_USER      (mặc định: admin)
#   AIRFLOW_ADMIN_PASSWORD  (bắt buộc)
# Đổi mật khẩu trong .env rồi `docker compose up -d airflow` là mật khẩu được cập nhật.
set -euo pipefail

AIRFLOW_ADMIN_USER="${AIRFLOW_ADMIN_USER:-admin}"
if [ -z "${AIRFLOW_ADMIN_PASSWORD:-}" ]; then
    echo "THIẾU AIRFLOW_ADMIN_PASSWORD trong .env — thêm dòng sau rồi chạy lại:" >&2
    echo "    AIRFLOW_ADMIN_PASSWORD=mat_khau_cua_ban" >&2
    sleep 30   # tránh khởi động lại dồn dập trong lúc chờ sửa .env
    exit 1
fi

airflow db migrate
# Mọi task chạy container của DAG dùng chung pool 1 chỗ (xem vv_dags.py).
airflow pools set docker_tasks 1 "Mỗi lúc một container: RAM của Docker có hạn" >/dev/null

if airflow users list -o plain 2>/dev/null | awk 'NR > 1 {print $2}' | grep -qx "$AIRFLOW_ADMIN_USER"; then
    airflow users reset-password -u "$AIRFLOW_ADMIN_USER" -p "$AIRFLOW_ADMIN_PASSWORD"
else
    airflow users create --role Admin \
        --username "$AIRFLOW_ADMIN_USER" --password "$AIRFLOW_ADMIN_PASSWORD" \
        --firstname Admin --lastname VV --email "${AIRFLOW_ADMIN_EMAIL:-admin@example.com}"
fi

[ "${1:-}" = "--init-only" ] && exit 0

# File PID nằm trong volume airflow_home nên còn lại sau khi container bị tắt đột
# ngột (Docker restart, máy tắt). Gunicorn thấy file cũ sẽ báo "Already running" và
# không lên, container khởi động lại mãi. Lúc này chưa có tiến trình nào nên xoá được.
rm -f "${AIRFLOW_HOME:-/opt/airflow}"/airflow-*.pid

# Container của task chỉ được xoá khi task kết thúc. Airflow bị tắt giữa chừng thì
# container đó chạy tiếp không ai quản, giữ khoá DuckDB và làm lượt sau hỏng ngay
# ở s1 ("Could not set lock"). Lúc này scheduler chưa chạy nên chưa có task hợp lệ
# nào: mọi container của DockerOperator còn sót lại đều là mồ côi.
python - <<'PY' || echo "không dọn được container mồ côi (bỏ qua)" >&2
import os

import docker

image = os.getenv("PIPELINE_IMAGE", "personalized_stock_recommendation-app")
app = os.getenv("APP_CONTAINER", "vv-app")
client = docker.DockerClient(base_url=os.getenv("DOCKER_URL", "tcp://docker-proxy:2375"))
for c in client.containers.list(all=True, filters={"ancestor": image}):
    # Container có nhãn compose là app hoặc `docker compose run` (verify.sh) — không
    # phải của Airflow, không đụng tới.
    if c.name != app and "com.docker.compose.project" not in c.labels:
        print(f"xoá container mồ côi {c.name}: {c.attrs['Config']['Cmd']}")
        c.remove(force=True)
PY

# Chạy cả hai và thoát ngay khi MỘT trong hai dừng, để Docker khởi động lại container.
# Nếu chỉ `exec` webserver thì scheduler chết âm thầm: giao diện vẫn lên nhưng không
# DAG nào chạy nữa.
airflow scheduler &
airflow webserver --port 8080 &
trap 'kill $(jobs -p) 2>/dev/null' TERM INT
wait -n || true
echo "một tiến trình Airflow đã dừng — thoát để container được khởi động lại" >&2
kill $(jobs -p) 2>/dev/null || true
exit 1
