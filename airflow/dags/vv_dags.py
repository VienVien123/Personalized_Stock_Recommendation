"""DAG pipeline_khuyen_nghi (bấm chạy tay):

    dung_app -> s1..s5 -> bat_app

Mỗi bước chạy trong một container riêng tạo từ image của app (DockerOperator), nên
Airflow không cần cài thư viện của dự án.
"""

import os
import re

import docker
import pendulum
from airflow import DAG
from airflow.models.baseoperator import chain
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

HOST_DIR = os.environ["HOST_PROJECT_DIR"]  # đường dẫn dự án TRÊN MÁY CHỦ
IMAGE = os.getenv("PIPELINE_IMAGE", "trade_hold-app")
NETWORK = os.getenv("PIPELINE_NETWORK", "trade_hold_default")
APP_CONTAINER = os.getenv("APP_CONTAINER", "vv-app")
MOUNTS = [
    Mount(source=f"{HOST_DIR}/data", target="/app/data", type="bind", read_only=True),
    Mount(source=f"{HOST_DIR}/src", target="/app/src", type="bind"),
    Mount(source=f"{HOST_DIR}/artifacts", target="/app/artifacts", type="bind"),
]
STEPS = ("s1_etl", "s2_state", "s3_graph", "s4_features", "s5_train")


def project_env():
    """Chuyển cho task các biến mà .env.example khai báo (trừ tài khoản Airflow)."""
    with open("/opt/project/.env.example", encoding="utf-8") as f:
        names = re.findall(r"^([A-Z][A-Z0-9_]*)=", f.read(), flags=re.M)
    return {
        k: os.environ[k]
        for k in names
        if k in os.environ and not k.startswith("AIRFLOW_")
    }


def pipeline_task(task_id, args):
    """Task chạy `python pipeline.py <args>` trong container tạm từ image của app."""
    return DockerOperator(
        task_id=task_id,
        image=IMAGE,
        command=["sh", "-c", f"python -u /app/src/pipeline.py {args}"],
        network_mode=NETWORK,
        mounts=MOUNTS,
        environment=project_env(),
        mount_tmp_dir=False,
        auto_remove="force",
        mem_limit="10g",
        # pool 1 chỗ (tạo trong start.sh): mỗi lúc chỉ một container chạy
        pool="docker_tasks",
    )


def set_done_marker(client, flag):
    """Airflow chỉ gắn dự án ở chế độ chỉ-đọc nên nhờ một container tạm ghi/xoá dấu."""
    client.containers.run(
        IMAGE,
        ["python", "-u", "/app/src/pipeline.py", flag],
        mounts=MOUNTS,
        remove=True,
    )


def stop_app():
    """Dừng app (nó đang giữ file DuckDB) rồi xoá dấu hoàn tất."""
    client = docker.from_env()
    client.containers.get(APP_CONTAINER).stop(timeout=30)
    set_done_marker(client, "--clear-done")


def start_app():
    """Ghi dấu hoàn tất rồi bật app — app chỉ phục vụ khi thấy dấu này."""
    client = docker.from_env()
    set_done_marker(client, "--mark-done")
    client.containers.get(APP_CONTAINER).start()


with DAG(
    dag_id="pipeline_khuyen_nghi",
    description="Chạy lại toàn bộ pipeline khuyến nghị",
    schedule=None,
    start_date=pendulum.datetime(2026, 10, 1, tz="Asia/Ho_Chi_Minh"),
    catchup=False,
    max_active_runs=1,
    tags=["pipeline"],
):
    chain(
        PythonOperator(task_id="dung_app", python_callable=stop_app),
        *[pipeline_task(name, f"--step {name}") for name in STEPS],
        # chỉ bật lại app khi mọi bước đều xong: hỏng giữa chừng thì app nằm yên
        PythonOperator(task_id="bat_app", python_callable=start_app),
    )
