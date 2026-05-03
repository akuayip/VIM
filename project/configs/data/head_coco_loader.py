"""
configs/data/head_coco_loader.py
Dataloader config untuk dataset kepala manusia format COCO.
"""

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data.datasets import register_coco_instances
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.data.dataset_mapper import DatasetMapper
from omegaconf import OmegaConf

# ── Path dataset ─────────────────────────────────────────────
TRAIN_JSON = "/home/marh/Desktop/head_detection_project (1)/head_detection_project/project/data/train/_annotations.coco_filtered.json"
TRAIN_IMG  = "/home/marh/Desktop/head_detection_project (1)/head_detection_project/project/data/train"
VAL_JSON   = "/home/marh/Desktop/head_detection_project (1)/head_detection_project/project/data/valid/_annotations.coco_filtered.json"
VAL_IMG    = "/home/marh/Desktop/head_detection_project (1)/head_detection_project/project/data/valid"

# Register dataset
register_coco_instances("head_train", {}, TRAIN_JSON, TRAIN_IMG)
register_coco_instances("head_val",   {}, VAL_JSON,   VAL_IMG)

image_size = 1024

# ── Dataloader sebagai DictConfig (bukan dict biasa) ─────────
dataloader = OmegaConf.create({})

dataloader.train = L(build_detection_train_loader)(
    dataset=L(__import__("detectron2.data", fromlist=["DatasetCatalog"]).DatasetCatalog.get)(
        name="head_train"
    ),
    mapper=L(DatasetMapper)(
        is_train=True,
        augmentations=[
            L(T.RandomFlip)(horizontal=True),
            L(T.ResizeScale)(
                min_scale=0.1,
                max_scale=2.0,
                target_height=image_size,
                target_width=image_size,
            ),
            L(T.FixedSizeCrop)(
                crop_size=(image_size, image_size),
                pad=False,
            ),
        ],
        image_format="RGB",
        use_instance_mask=False,
    ),
    total_batch_size=16,
    num_workers=4,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(__import__("detectron2.data", fromlist=["DatasetCatalog"]).DatasetCatalog.get)(
        name="head_val"
    ),
    mapper=L(DatasetMapper)(
        is_train=False,
        augmentations=[
            L(T.ResizeShortestEdge)(
                short_edge_length=image_size,
                max_size=image_size,
            ),
        ],
        image_format="RGB",
        use_instance_mask=False,
    ),
    num_workers=4,
)

dataloader.evaluator_type = "coco"
