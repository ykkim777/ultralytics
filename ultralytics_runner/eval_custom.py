"""
2026.02.03
Ultralytics YOLO custom evaluation script

Run:
    python ultralytics_runner/eval_custom.py

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
# CLI args
# ======================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained weight (e.g. best.pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Dataset yaml",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
    )
    return parser.parse_args()


args = parse_args()

model_path = Path(args.model)
data_yaml = args.data

model_name = model_path.stem  # best
exp_name = model_path.parents[1].name  # yolo26n_e150_b256_ddp_20260205
dataset_name = Path(data_yaml).stem

# ======================================================
# Val config dicts
# ======================================================
val_cfg = dict(
    data=data_yaml,
    split=args.split,
    imgsz=640,
    batch=128,
    workers=16,
    device="0",
    half=True,
    save_json=True,
    exist_ok=True,
)


# ======================================================
# Project / name
# ======================================================
project = dataset_name
val_name = f"{exp_name}_eval-{args.split}"

# ======================================================
# Model Initialization
# ======================================================
# Load a pretrained YOLO model (recommended for fine-tuning)
# - Backbone + head initialized from COCO-pretrained weights
model = YOLO(model_path)
# print(model.model.model[0].conv.in_channels)

# ======================================================
# Validation / Test
# ======================================================
metrics = model.val(
    **val_cfg,
    project=project,
    name=val_name,
)

print(f"[VAL DONE] {project}/{val_name}")

# from ultralytics import YOLO

# import os
# from ultralytics.utils import colorstr, LOGGER

# # Load model
# model = YOLO("yolo11n.pt")

# LOGGER.info(colorstr("red", "bold", "missing_person_eo_v1.0"))
# metrics = model.val(
#     data="missing_person_eo_v1.0.yaml",
#     imgsz=640,
#     batch=16,
#     conf=0.25,
#     iou=0.5,
#     device="3",
# )

# LOGGER.info(colorstr("red", "bold", "missing_person_ir_v1.0"))
# metrics = model.val(
#     data="missing_person_ir_v1.0.yaml",
#     imgsz=640,
#     batch=16,
#     conf=0.25,
#     iou=0.5,
#     device="3",
# )

# LOGGER.info(colorstr("red", "bold", "soldier_eo_v0.55"))
# metrics = model.val(
#     data="soldier_eo_v0.55.yaml",
#     imgsz=640,
#     batch=16,
#     conf=0.25,
#     iou=0.5,
#     device="3",
# )

# LOGGER.info(colorstr("red", "bold", "soldier_ir_v0.4"))
# metrics = model.val(
#     data="soldier_ir_v0.4.yaml",
#     imgsz=640,
#     batch=16,
#     conf=0.25,
#     iou=0.5,
#     device="3",
# )

# # Val
# LOGGER.info(colorstr("red", "bold", "crowd human"))
# metrics = model.val(
#     data="crowd_human.yaml",
#     imgsz=640,
#     batch=16,
#     conf=0.25,
#     iou=0.5,
#     device="3",
# )

# LOGGER.info(colorstr("red", "bold", "coco"))
# metrics = model.val(
#     data="coco.yaml", imgsz=640, batch=16, conf=0.25, iou=0.5, device="3"
# )
