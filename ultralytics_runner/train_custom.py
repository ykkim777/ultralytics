"""
2026.02.03
Ultralytics YOLO custom training script

Run:
    python ultralytics_runner/train_custom.py

Purpose:
- Quick sanity check with coco8
- Full-scale training on soldier EO dataset
"""

from ultralytics import YOLO

# ======================================================
# Model Initialization
# ======================================================
# Load a pretrained YOLO model (recommended for fine-tuning)
# - Backbone + head initialized from COCO-pretrained weights
model = YOLO("yolo26n.pt")
# print(model.model.model[0].conv.in_channels)

# ======================================================
# 1. TUTORIAL / SANITY CHECK (coco8)
#    - 목적: 코드 / 환경 / GPU 정상 동작 확인
# ======================================================
# results = model.train(
#     data="coco8.yaml",
#     epochs=3,  # very small epochs for quick test
#     project="coco8",  # group experiments by dataset
#     name="train_sanity",  # experiment name
#     exist_ok=True,
# )

# metrics = model.val(
#     project="coco8",
#     name="val_sanity",
#     exist_ok=True,
#     save_json=True,  # COCO-format metrics export
# )

# Example metric access:
# print(metrics.box.map)      # mAP50-95
# print(metrics.box.map50)    # mAP50


# ======================================================
# 2. CUSTOM TRAINING (Soldier EO Dataset)
#    - 목적: 대규모 데이터 기반 본 학습
# ======================================================
results = model.train(
    # ----------------------------
    # Dataset & Input
    # ----------------------------
    data="soldier_eo_v1.2.yaml",
    imgsz=640,  # standard YOLO input resolution
    cache="disk",  # cache images/labels in disk .cache 파일 생성 후 항상 동일 입력으로 실험 재현성
    # ----------------------------
    # Training Schedule
    # ----------------------------
    epochs=150,  # sufficient upper bound for convergence
    patience=30,  # early stopping based on validation mAP
    # ----------------------------
    # Performance & Resources
    # ----------------------------
    batch=256,  # adjust based on GPU memory (RTX 3090 OK)
    workers=16,  # number of dataloader workers (≈ CPU cores)
    # ----------------------------
    # Optimization
    # ----------------------------
    optimizer="auto",  # let Ultralytics choose optimizer & LR
    # ----------------------------
    # Device
    # ----------------------------
    device="0,1,2,3",  # explicit GPU selection
    # ----------------------------
    # Experiment Management
    # ----------------------------
    project="soldier_eo_v1.2",  # dataset-level experiment grouping
    name="train_e200_b256_ddp",  # training configuration identifier
    exist_ok=True,
)

metrics = model.val(
    data="soldier_eo_v1.2.yaml",  # train과 동일한 val set 명시
    imgsz=640,  # train과 동일 (mAP 비교 안정성)
    batch=128,  # val은 메모리 여유 ↑, 속도 ↑
    device="0",  # 단일 GPU (분산 불필요)
    workers=16,  # IO 병목 방지
    half=True,  # FP16 (속도 ↑)
    save_json=True,  # COCO eval
    project="soldier_eo_v1.2",
    name="val_e200",
    exist_ok=True,
)
