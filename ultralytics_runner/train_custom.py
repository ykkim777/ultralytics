"""
2026.02.03
Ultralytics YOLO custom training script

Run:
    python ultralytics_runner/train_custom.py

Purpose:
- Quick sanity check with coco8
- Full-scale training on soldier EO dataset
"""

import argparse
import datetime
from pathlib import Path

from yaml import dump
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

# ======================================================
# Global constants & environment settings
# ======================================================
runs_dir = Path(SETTINGS["runs_dir"])  # yolo settings runs_dir


# ======================================================
# Argument parsing
# ======================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sanity", action="store_true", help="Run quick sanity check with coco8"
    )
    return parser.parse_args()


args = parse_args()

# ======================================================
# User inputs (experiment identity)
# ======================================================
model_path = "yolo26n.pt"
data_yaml = "missing_person_2025_v3_kcenter.yaml"  # "soldier_eo_v1.2.yaml"

model_name = Path(model_path).stem
dataset_name = Path(data_yaml).stem

# ======================================================
# Common hyperparameters
# ======================================================
epochs = 150
patience = 30
val_split = "test"  # val | test | train

imgsz = 640
cache = "disk"
batch = 256
workers = 16
optimizer = "auto"

half = True
save_json = True

device_train = "0,1,2,3"  # "cpu"
device_val = "0"
exist_ok = True


# ======================================================
# Derived naming & experiment IDs
# ======================================================
device_name = "ddp" if device_train == "0,1,2,3" else f"gpu{device_train}"
precision = "fp16" if half else "fp32"

# project: 데이터셋 단위
project = dataset_name
date = datetime.datetime.now().strftime("%Y%m%d")

# name: 실험 단위
base_name = f"{model_name}_e{epochs}_b{batch}_{device_name}_{date}"
val_name = f"{base_name}_eval-{val_split}_{precision}_best"

# ======================================================
# Train / Val config dicts
# ======================================================
train_cfg = dict(
    # ----------------------------
    # Dataset & Input
    # ----------------------------
    data=data_yaml,
    imgsz=imgsz,  # standard YOLO input resolution
    cache=cache,  # cache images/labels in disk .cache 파일 생성 후 항상 동일 입력으로 실험 재현성
    # ----------------------------
    # Training Schedule
    # ----------------------------
    epochs=epochs,  # sufficient upper bound for convergence
    patience=patience,  # early stopping based on validation mAP
    # ----------------------------
    # Performance & Resources
    # ----------------------------
    batch=batch,  # adjust based on GPU memory (RTX 3090 OK)
    workers=workers,  # number of dataloader workers (≈ CPU cores)
    # ----------------------------
    # Optimization
    # ----------------------------
    optimizer=optimizer,  # let Ultralytics choose optimizer & LR
    # ----------------------------
    # Device
    # ----------------------------
    device=device_train,  # explicit GPU selection
    exist_ok=exist_ok,
)

val_cfg = dict(
    data=data_yaml,
    split=val_split,
    imgsz=imgsz,
    batch=batch,
    workers=workers,
    device=device_val,
    half=half,
    save_json=save_json,
    exist_ok=exist_ok,
)


# ======================================================
# Model Initialization
# ======================================================
# Load a pretrained YOLO model (recommended for fine-tuning)
# - Backbone + head initialized from COCO-pretrained weights
model = YOLO(model_path)
# print(model.model.model[0].conv.in_channels)

# ======================================================
# Sanity check block (early exit)
#    - 목적: 코드 / 환경 / GPU 정상 동작 확인
# ======================================================
if args.sanity:
    print("[SANITY CHECK] Running coco8 quick test")

    model.train(
        data="coco8.yaml",
        epochs=3,
        imgsz=640,
        project="coco8",
        name="yolo_sanity_train",
        exist_ok=True,
    )

    metrics = model.val(
        data="coco8.yaml",
        project="coco8",
        name="yolo_sanity_val",
        save_json=True,
        exist_ok=True,
    )

    # Example metric access:
    # print(metrics.box.map)      # mAP50-95
    # print(metrics.box.map50)    # mAP50

    print("[SANITY CHECK DONE]")
    exit(0)

# ======================================================
# Training
#    - 목적: 대규모 데이터 기반 본 학습
# ======================================================
train_results = model.train(
    **train_cfg,
    # ----------------------------
    # Experiment Management
    # ----------------------------
    project=project,
    name=base_name,
)

print(f"[TRAIN DONE] {project}/{base_name}")


# ======================================================
# Save experiment meta
# ======================================================
exp_cfg = dict(
    model=model_path,
    dataset=data_yaml,
    epochs=epochs,
    batch=batch,
    optimizer=optimizer,
    val_split=val_split,
    precision=precision,
    device_train=device_train,
    device_val=device_val,
)

exp_dir = runs_dir / "detect" / project / base_name
exp_dir.mkdir(parents=True, exist_ok=True)

meta_path = exp_dir / "experiment_meta.yaml"
meta_path.write_text(dump(exp_cfg, sort_keys=False))

print(f"[META SAVED] {meta_path}")

# ======================================================
# Validation / Test
# ======================================================
metrics = model.val(
    **val_cfg,
    project=project,
    name=val_name,
)

print(f"[VAL DONE] {project}/{val_name}")
