"""
configs/models/faster_rcnn_vimdet.py

Model config: Faster R-CNN + Vim backbone + FPN
- Deteksi kepala manusia (1 class)
- NO mask head (detection only)
- Backbone Vim di-freeze, neck + head trainable
"""

from functools import partial
import torch.nn as nn

from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.modeling.meta_arch import GeneralizedRCNN
from detectron2.modeling.anchor_generator import DefaultAnchorGenerator
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator import RPN, StandardRPNHead
from detectron2.modeling.roi_heads import (
    StandardROIHeads,
    FastRCNNOutputLayers,
    FastRCNNConvFCHead,
)
from detectron2.modeling import VisionMambaDet, SimpleFeaturePyramid
from detectron2.modeling.backbone.fpn import LastLevelMaxPool

# ── Vim backbone config ──────────────────────────────────────────────────────
# Vim-Tiny: embed_dim=192, depth=24
# Vim-Small: embed_dim=384, depth=24
# Vim-Base : embed_dim=768, depth=24
embed_dim = 192   # Vim-Tiny (ganti 384/768 untuk Small/Base)
depth     = 24
dp        = 0.1   # drop path rate

model = L(GeneralizedRCNN)(
    backbone=L(SimpleFeaturePyramid)(
        net=L(VisionMambaDet)(
            img_size=1024,
            patch_size=16,
            embed_dim=embed_dim,
            depth=depth,
            drop_path_rate=dp,
            out_feature="last_feat",
            last_layer_process="add",
            bimamba_type="v2",
            rms_norm=True,
            residual_in_fp32=True,
            fused_add_norm=True,
            if_abs_pos_embed=True,
            if_rope=True,
            if_rope_residual=True,
            pt_hw_seq_len=14,
            if_cls_token=False,
            # pretrained diisi di config training
            pretrained=None,
        ),
        in_feature="${.net.out_feature}",
        out_channels=256,
        scale_factors=(4.0, 2.0, 1.0, 0.5),
        top_block=L(LastLevelMaxPool)(),
        norm="LN",
        square_pad=1024,
    ),

    proposal_generator=L(RPN)(
        in_features=["p2", "p3", "p4", "p5", "p6"],
        head=L(StandardRPNHead)(in_channels=256, num_anchors=3),
        anchor_generator=L(DefaultAnchorGenerator)(
            sizes=[[32], [64], [128], [256], [512]],
            aspect_ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
            offset=0.0,
        ),
        anchor_matcher=L(Matcher)(
            thresholds=[0.3, 0.7],
            labels=[0, -1, 1],
            allow_low_quality_matches=True,
        ),
        box2box_transform=L(Box2BoxTransform)(weights=[1.0, 1.0, 1.0, 1.0]),
        batch_size_per_image=256,
        positive_fraction=0.5,
        pre_nms_topk=(2000, 1000),
        post_nms_topk=(1000, 1000),
        nms_thresh=0.7,
    ),

    roi_heads=L(StandardROIHeads)(
        num_classes=1,           # hanya 1 class: kepala manusia
        batch_size_per_image=512,
        positive_fraction=0.25,
        proposal_matcher=L(Matcher)(
            thresholds=[0.5],
            labels=[0, 1],
            allow_low_quality_matches=False,
        ),
        box_in_features=["p2", "p3", "p4", "p5"],
        box_pooler=L(ROIPooler)(
            output_size=7,
            scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        ),
        box_head=L(FastRCNNConvFCHead)(
            input_shape=ShapeSpec(channels=256, height=7, width=7),
            conv_dims=[256, 256, 256, 256],   # 4 conv layers
            fc_dims=[1024],
            conv_norm="LN",
        ),
        box_predictor=L(FastRCNNOutputLayers)(
            input_shape=ShapeSpec(channels=1024),
            test_score_thresh=0.05,
            box2box_transform=L(Box2BoxTransform)(weights=(10, 10, 5, 5)),
            num_classes="${..num_classes}",
        ),
        # ── TIDAK ADA mask_in_features / mask_pooler / mask_head ──────────
        # Detection only, no segmentation mask
    ),

    pixel_mean=[123.675, 116.28, 103.53],   # ImageNet RGB mean * 255
    pixel_std=[58.395, 57.12, 57.375],
    input_format="RGB",
)
