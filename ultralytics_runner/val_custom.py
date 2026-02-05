from ultralytics import YOLO

import os
from ultralytics.utils import colorstr, LOGGER

# Load model
model = YOLO("yolo11n.pt")

LOGGER.info(colorstr("red", "bold", "missing_person_eo_v1.0"))
metrics = model.val(
    data="missing_person_eo_v1.0.yaml",
    imgsz=640,
    batch=16,
    conf=0.25,
    iou=0.5,
    device="3",
)

LOGGER.info(colorstr("red", "bold", "missing_person_ir_v1.0"))
metrics = model.val(
    data="missing_person_ir_v1.0.yaml",
    imgsz=640,
    batch=16,
    conf=0.25,
    iou=0.5,
    device="3",
)

LOGGER.info(colorstr("red", "bold", "soldier_eo_v0.55"))
metrics = model.val(
    data="soldier_eo_v0.55.yaml",
    imgsz=640,
    batch=16,
    conf=0.25,
    iou=0.5,
    device="3",
)

LOGGER.info(colorstr("red", "bold", "soldier_ir_v0.4"))
metrics = model.val(
    data="soldier_ir_v0.4.yaml",
    imgsz=640,
    batch=16,
    conf=0.25,
    iou=0.5,
    device="3",
)

# Val
LOGGER.info(colorstr("red", "bold", "crowd human"))
metrics = model.val(
    data="crowd_human.yaml",
    imgsz=640,
    batch=16,
    conf=0.25,
    iou=0.5,
    device="3",
)

LOGGER.info(colorstr("red", "bold", "coco"))
metrics = model.val(
    data="coco.yaml", imgsz=640, batch=16, conf=0.25, iou=0.5, device="3"
)
