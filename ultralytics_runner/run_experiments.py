"""
2026.02.11.

ultralyrics_runhner/run_experiments.py

============================================================
실험 목적
============================================================
이 스크립트는 sampling 전략 및 비율(top-k)에 따라 생성된
여러 Ultralytics dataset YAML 실험을 자동으로 학습·평가하기 위한
실험 관리용 러너이다.

각 실험은 다음을 검증하는 것을 목표로 한다.

- 동일한 모델, 동일한 학습 설정 하에서
- frame sampling 전략(uniform, random, k-center, MMR 등)과
- sampling 비율(top01, top03, top05, top10 등)에 따른
- 성능 변화(mAP50, mAP50:95)를 정량적으로 비교

즉, "어떤 프레임을 얼마나 뽑는 것이 가장 효율적인가?"를
체계적으로 평가하기 위한 실험 파이프라인이다.


============================================================
실험 단위 (Experiment Definition)
============================================================
하나의 실험(experiment)은 다음 3가지 조합으로 정의된다.

- dataset version: 예) soldier_eo_v1.2
- sampling method: v1_uniform, v2_random, v3_kcenter, v5_centroid, v6_mmr
- sampling ratio: top01, top03, top05, top10

각 실험은 아래 경로에 YAML 파일로 정의되어 있어야 한다.

    dataset_root/
      splits/
        experiments/
          soldier_eo_v1.2/
            top01_v1_uniform/
              soldier_eo_v1.2_top01_v1_uniform.yaml
            top03_v3_kcenter/
              soldier_eo_v1.2_top03_v3_kcenter.yaml
            ...


============================================================
스크립트가 수행하는 작업
============================================================
본 스크립트는 위 experiments 폴더 하위의 모든 dataset YAML을 순회하며
각 실험에 대해 아래 과정을 자동 수행한다.

1. 학습 (train)
   - train_custom.py를 그대로 호출
   - 각 실험은 독립적인 runs/detect/<experiment_id>/ 디렉토리에 저장됨

2. 평가 (evaluation)
   동일한 best.pt 모델에 대해 3가지 평가를 수행한다.

   (1) val
       - 실험 YAML에 정의된 val split
       - sampling 전략에 따라 서브샘플링된 validation 프레임

   (2) test
       - 실험 YAML에 정의된 test split
       - sampling 전략에 따라 서브샘플링된 test 프레임

   (3) test_manual_gt
       - 기준 데이터셋 YAML(soldier_eo_v1.2.yaml)의 test split 사용
       - 사람이 직접 어노테이션한 manual GT 프레임
       - sampling 전략과 무관한 "절대 기준 성능" 비교용

3. 결과 저장
   - 각 split별 mAP50, mAP50:95를 CSV 파일로 누적 저장


============================================================
결과 파일 설명
============================================================
실험 결과는 다음 CSV 파일 하나로 관리된다.

    splits/experiments/soldier_eo_v1.2/yolo26n_results.csv

CSV 컬럼 의미:

- experiment_id
    예) soldier_eo_v1.2_top03_v1_uniform

- split
    val | test | test_manual_gt

- map50
    IoU=0.5 기준 mAP

- map50_95
    IoU=0.5:0.95 평균 mAP (COCO metric)


============================================================
결과 해석 가이드
============================================================
- val:
    학습 중 사용되는 검증 성능
    → early stopping 및 학습 안정성 판단

- test:
    sampling 전략이 적용된 test 시퀀스 기준 성능
    → "이 sampling 전략으로 실제 inference 시 성능이 얼마나 나오는가?"

- test_manual_gt:
    sampling 전략과 무관한 동일한 GT 프레임 기준 성능
    → sampling 방법 간의 '공정한 성능 비교 기준점'

일반적으로 논문/리포트에서는
- test_manual_gt 성능을 주 비교 지표로 사용하고
- test 성능은 실제 deployment 시나리오 참고용으로 활용하는 것을 권장한다.


============================================================
설계 원칙
============================================================
- train_custom.py는 수정하지 않는다.
- 실험 제어 로직은 모두 외부 runner(run_experiments.py)에서 수행한다.
- dataset YAML은 실험 단위로 관리하여 재현성과 확장성을 보장한다.
- CSV는 "실험 결과의 단일 진실 소스(single source of truth)"로 사용한다.


============================================================
실행 방법
============================================================
1) 모든 실험 YAML 생성
2) 아래 명령 실행

    python ultralytics_runner/run_experiments.py | tee experiments.log

============================================================
확장 가능성
============================================================
- sampling 전략 / 비율 추가 시 YAML만 추가하면 자동 반영
- plot 스크립트와 연동하여 성능 비교 시각화 가능
- 실패한 실험만 재실행(resume)하는 구조로 확장 가능
"""

import subprocess
import csv
import sys
from pathlib import Path
from datetime import datetime

from ultralytics import YOLO
from logger import get_logger

logger = get_logger("run_experiments")

# --------------------------------------------------
# Config
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # ultralytics/
TRAIN_SCRIPT = PROJECT_ROOT / "ultralytics_runner" / "train_custom.py"

DATASET_ROOT = Path("/data/dataset_2025/soldier_2025")
EXPERIMENTS_ROOT = DATASET_ROOT / "splits" / "experiments" / "soldier_eo_v1.2"

BASE_DATASET_YAML = DATASET_ROOT / "datasets" / "soldier_eo_v1.2.yaml"

MODEL_PATH = "yolo26n.pt"

RESULT_CSV = EXPERIMENTS_ROOT / f"{Path(MODEL_PATH).name}_results.csv"


# --------------------------------------------------
# Utils
# --------------------------------------------------
def run_cmd(cmd: list[str], *, exp_id: str | None = None):
    prefix = f"[{exp_id}] " if exp_id else ""

    cmd_str = " ".join(cmd)
    logger.info(prefix + f"[RUN] {cmd_str}")
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,  # 모든 실행 로그가 Python logger로 귀속
            stderr=subprocess.STDOUT,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        # 여러 실험 돌릴 때 어느 실험에서 죽었는지 식별 가능
        logger.error(prefix + f"[FAILED] {cmd_str}")
        raise


def append_csv(row: dict):
    exists = RESULT_CSV.exists()
    with open(RESULT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["experiment_id", "split", "map50", "map50_95"]
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_val(model_path: Path, data_yaml: Path, split: str):
    model = YOLO(model_path)
    metrics = model.val(
        data=data_yaml,
        split=split,
        device="0",
        batch=256,
        imgsz=640,
        half=True,
        save_json=False,
        verbose=False,
    )
    return metrics.box.map50, metrics.box.map


import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def parse_experiment_id(exp_id: str):
    """
    example:
    soldier_eo_v1.2_top03_v1_uniform
    """
    # topXX → percentage
    m = re.search(r"top(\d+)", exp_id)
    if not m:
        return None, None

    percentage = int(m.group(1))

    # sampling method
    exp_id_lower = exp_id.lower()
    if "uniform" in exp_id_lower:
        method = "Uniform"
    elif "random" in exp_id_lower:
        method = "Random"
    elif "kcenter" in exp_id_lower or "k_center" in exp_id_lower:
        method = "K-Center"
    elif "centroid" in exp_id_lower:
        method = "Centroid"
    elif "mmr" in exp_id_lower:
        method = "MMR"
    else:
        return None, None

    return percentage, method


def plot_results():

    df = pd.read_csv(RESULT_CSV)

    # parse experiment_id
    parsed = df["experiment_id"].apply(parse_experiment_id)
    df["percentage"] = parsed.apply(lambda x: x[0])
    df["method"] = parsed.apply(lambda x: x[1])

    df = df.dropna(subset=["percentage", "method"])

    splits = ["val", "test", "test_manual_gt"]
    methods = ["Uniform", "Random", "K-Center", "Centroid", "MMR"]

    markers = {
        "Uniform": "o",
        "Random": "s",
        "K-Center": "^",
        "Centroid": "D",
        "MMR": "x",
    }

    # ---------- map50 ----------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for ax, split in zip(axes, splits):
        sdf = df[df["split"] == split]

        for method in methods:
            mdf = sdf[sdf["method"] == method].sort_values("percentage")
            if mdf.empty:
                continue

            ax.plot(
                mdf["percentage"],
                mdf["map50"],
                marker=markers[method],
                markersize=3,
                linestyle="-",
                label=method,
            )

        ax.set_title(f"map50 / {split}")
        ax.set_xlabel("Sampling Percentage (%)")
        ax.grid(True)

    axes[0].set_ylabel("mAP@0.5")
    axes[0].legend()

    fig.tight_layout()
    fig_path_map50 = EXPERIMENTS_ROOT / "map50_by_split.png"
    fig.savefig(fig_path_map50, dpi=150)
    plt.close(fig)
    logger.info(f"[PLOT] {fig_path_map50}")

    # ---------- map50_95 ----------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for ax, split in zip(axes, splits):
        sdf = df[df["split"] == split]

        for method in methods:
            mdf = sdf[sdf["method"] == method].sort_values("percentage")
            if mdf.empty:
                continue

            ax.plot(
                mdf["percentage"],
                mdf["map50_95"],
                marker=markers[method],
                markersize=3,
                linestyle="-",
                label=method,
            )

        ax.set_title(f"map50-95 / {split}")
        ax.set_xlabel("Sampling Percentage (%)")
        ax.grid(True)

    axes[0].set_ylabel("mAP@0.5:0.95")
    axes[0].legend()

    fig.tight_layout()
    fig_path_map50_95 = EXPERIMENTS_ROOT / "map50_95_by_split.png"
    fig.savefig(fig_path_map50_95, dpi=150)
    plt.close(fig)
    logger.info(f"[PLOT] {fig_path_map50_95}")


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    yaml_files = sorted(EXPERIMENTS_ROOT.rglob("*.yaml"))

    logger.info(f"Found {len(yaml_files)} experiments")

    for yaml_path in yaml_files:
        experiment_id = yaml_path.stem
        logger.info("=" * 80)
        logger.info(f"[EXPERIMENT] {experiment_id}")

        # -----------------------------
        # Train
        # -----------------------------
        run_cmd(
            [
                sys.executable,
                str(TRAIN_SCRIPT),
                "--model",
                MODEL_PATH,
                "--data",
                str(yaml_path),
            ]
        )

        # 최신 best.pt 자동 추론
        runs_dir = PROJECT_ROOT / "runs" / "detect" / experiment_id
        best_pt = max(runs_dir.rglob("best.pt"), key=lambda p: p.stat().st_mtime)

        # -----------------------------
        # Val (sampled)
        # -----------------------------
        map50, map5095 = run_val(best_pt, yaml_path, split="val")
        append_csv(
            dict(
                experiment_id=experiment_id,
                split="val",
                map50=round(map50, 4),
                map50_95=round(map5095, 4),
            )
        )
        logger.info(f"[VAL] map50={map50:.4f}, map50-95={map5095:.4f}")

        # -----------------------------
        # Test (sampled)
        # -----------------------------
        map50, map5095 = run_val(best_pt, yaml_path, split="test")
        append_csv(
            dict(
                experiment_id=experiment_id,
                split="test",
                map50=round(map50, 4),
                map50_95=round(map5095, 4),
            )
        )
        logger.info(f"[TEST] map50={map50:.4f}, map50-95={map5095:.4f}")

        # -----------------------------
        # Test (manual GT)
        # -----------------------------
        map50, map5095 = run_val(best_pt, BASE_DATASET_YAML, split="test")
        append_csv(
            dict(
                experiment_id=experiment_id,
                split="test_manual_gt",
                map50=round(map50, 4),
                map50_95=round(map5095, 4),
            )
        )
        logger.info(f"[TEST_MANUAL_GT] map50={map50:.4f}, map50-95={map5095:.4f}")

    logger.info("ALL EXPERIMENTS DONE")


if __name__ == "__main__":

    main()

    plot_results()
