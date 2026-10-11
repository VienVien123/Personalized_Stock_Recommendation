"""DAG Airflow của dự án
  pipeline_khuyen_nghi  bấm chạy tay: kiểm tra dữ liệu -> S1..S5 từng bước -> bật lại
                        app -> kiểm tra hệ thống thật -> chốt chặn chỉ số.

Airflow không chứa thư viện của dự án. Mỗi task chạy trong một container riêng tạo
từ image của app (DockerOperator), nên môi trường giống hệt lúc chạy tay bằng
scripts/verify.sh.
"""

import os
import re

import pendulum
from airflow import DAG
from airflow.models.baseoperator import chain
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

HOST_DIR = os.environ["HOST_PROJECT_DIR"]
PROJECT_IN_AIRFLOW = "/opt/project"  # cùng thư mục đó, gắn chỉ-đọc vào Airflow
IMAGE = os.getenv("PIPELINE_IMAGE", "trade_hold-app")
NETWORK = os.getenv("PIPELINE_NETWORK", "trade_hold_default")
APP_CONTAINER = os.getenv("APP_CONTAINER", "vv-app")
DOCKER_POOL = "docker_tasks"
START = pendulum.datetime(2026, 10, 1, tz="Asia/Ho_Chi_Minh")


def project_env():
    """Chuyển tiếp cho task đúng các biến mà .env.example khai báo.

    Giá trị lấy từ môi trường của container Airflow (docker-compose nạp .env vào
    đó), nên thêm một tham số mới vào .env.example là task tự nhận được.
    """
    names = []
    try:
        with open(f"{PROJECT_IN_AIRFLOW}/.env.example", encoding="utf-8") as f:
            names = re.findall(r"^([A-Z][A-Z0-9_]*)=", f.read(), flags=re.M)
    except OSError:
        pass
    # Tài khoản Airflow chỉ dành cho Airflow, không đưa vào container của task.
    return {
        k: os.environ[k]
        for k in names
        if k in os.environ and not k.startswith("AIRFLOW_")
    }


def in_app_image(task_id, command, mounts, **kwargs):
    return DockerOperator(
        task_id=task_id,
        image=IMAGE,
        command=["sh", "-c", command],
        network_mode=NETWORK,
        mounts=mounts,
        environment=project_env() | kwargs.pop("environment", {}),
        mount_tmp_dir=False,
        auto_remove="force",
        mem_limit="5g",
        # Pool 1 chỗ (tạo trong start.sh): mỗi lúc chỉ một container chạy, để hai
        # lượt pipeline không cùng dùng gần hết RAM của Docker.
        pool=DOCKER_POOL,
        **kwargs,
    )

def set_app_running(running: bool):
    """Dừng / bật container app qua Docker API.

    Phải dừng app trước khi chạy pipeline: app đang giữ file DuckDB, tiến trình
    khác không mở để ghi được.
    """
    import docker
    container = docker.from_env().containers.get(APP_CONTAINER)
    if running:
        container.start()
    else:
        container.stop(timeout=30)
    print(f"{APP_CONTAINER}: {'đã bật' if running else 'đã dừng'}")


def stop_app_and_clear():
    """Dừng app rồi xoá dấu hoàn tất (artifacts/_pipeline_done).

    Airflow chỉ gắn dự án ở chế độ chỉ-đọc nên nhờ một container tạm từ image app
    xoá file; lỗi thì containers.run ném ContainerError và task đỏ.
    """
    import docker

    set_app_running(False)
    docker.from_env().containers.run(
        IMAGE,
        ["sh", "-c", pipeline_cmd("--clear-done")],
        mounts=APP_MOUNTS,
        network_mode=NETWORK,
        remove=True,
    )
    print("đã xoá dấu hoàn tất")


APP_MOUNTS = [
    Mount(source=f"{HOST_DIR}/data", target="/app/data", type="bind", read_only=True),
    Mount(source=f"{HOST_DIR}/src", target="/app/src", type="bind"),
    Mount(source=f"{HOST_DIR}/artifacts", target="/app/artifacts", type="bind"),
    Mount(
        source=f"{HOST_DIR}/scripts", target="/app/scripts", type="bind", read_only=True
    ),
]


def pipeline_cmd(args):
    return f"python -u /app/src/pipeline.py {args}"


with DAG(
    dag_id="pipeline_khuyen_nghi",
    description="Chạy lại toàn bộ pipeline khuyến nghị rồi kiểm tra kết quả",
    schedule=None,
    start_date=START,
    catchup=False,
    max_active_runs=1,
    tags=["pipeline"],
) as pipeline:
    # check_data kiểm tra dữ liệu đầu vào
    # missing: kiểm tra file đủ chưa
    # validate_inputs kiểm tra nôi dung file
    check_data = in_app_image(
        "kiem_tra_du_lieu", pipeline_cmd("--check-data"), APP_MOUNTS
    )

    stop_app = PythonOperator(
        task_id="dung_app",
        python_callable=stop_app_and_clear,
    )

    steps = [
        in_app_image(name, pipeline_cmd(f"--step {name}"), APP_MOUNTS)
        for name in ("s1_etl", "s2_state", "s3_graph", "s4_features", "s5_train")
    ]
    mark = in_app_image("ghi_dau_hoan_tat", pipeline_cmd("--mark-done"), APP_MOUNTS)
    # Chỉ bật lại app khi mọi bước đều xong; pipeline hỏng thì app nằm yên để
    # không phục vụ kết quả dở dang.
    start_app = PythonOperator(
        task_id="bat_app", python_callable=set_app_running, op_kwargs={"running": True}
    )
    verify = in_app_image(
        "kiem_tra_he_thong",
        "python -u /app/scripts/verify_app.py",
        APP_MOUNTS,
        environment={
            "VERIFY_APP_URL": f"http://{APP_CONTAINER}:7860/",
            "VERIFY_APP_WAIT": "300",
        },
    )
    guard = in_app_image(
        "chot_chan_chi_so",
        "python -u /app/scripts/metric_guard.py /app/artifacts",
        APP_MOUNTS,
    )

    chain(check_data, stop_app, *steps, mark, start_app, verify, guard)
