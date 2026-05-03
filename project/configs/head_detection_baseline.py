"""
configs/head_detection_baseline.py

Config training baseline (default OFAT):
  lr=0.001, batch_size=16, optimizer=AdamW, epochs=100

Jalankan dari folder project/:
  python tools/train_net.py \
      --config-file configs/head_detection_baseline.py \
      --num-gpus 1
"""

from functools import partial
import torch

from fvcore.common.param_scheduler import MultiStepParamScheduler
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from detectron2.solver.build import get_default_optimizer_params
from detectron2.modeling.backbone.vim import get_vim_lr_decay_rate

from configs.models.faster_rcnn_vimdet import model
from configs.data.head_coco_loader import dataloader

# ── Pretrained Vim ───────────────────────────────────────────
PRETRAINED_VIM = "checkpoints/vim_tiny_pretrained.pth"

# ── Hitung max_iter ──────────────────────────────────────────
# Ganti NUM_TRAIN_IMAGES sesuai jumlah gambar training kamu
NUM_TRAIN_IMAGES = 1271
EPOCHS           = 100
BATCH_SIZE       = 4
DEPTH            = 24
ITERS_PER_EPOCH  = NUM_TRAIN_IMAGES // BATCH_SIZE
MAX_ITER         = ITERS_PER_EPOCH * EPOCHS
WARMUP_ITERS     = ITERS_PER_EPOCH * 5

# ── Model ────────────────────────────────────────────────────
model.backbone.net.pretrained = PRETRAINED_VIM
model.roi_heads.num_classes   = 1

# ── Train ────────────────────────────────────────────────────
train = dict(
    output_dir      = "./outputs/baseline",
    init_checkpoint = PRETRAINED_VIM,
    max_iter        = MAX_ITER,
    amp             = dict(enabled=True),
    ddp             = dict(
        broadcast_buffers      = False,
        find_unused_parameters = False,
        fp16_compression       = True,
    ),
    checkpointer    = dict(period=ITERS_PER_EPOCH, max_to_keep=5),
    eval_period     = ITERS_PER_EPOCH,
    log_period      = 20,
    device          = "cuda",
    freeze_backbone = True,
)

# ── Dataloader batch size ────────────────────────────────────
# Akses langsung ke field LazyCall, bukan .train
dataloader.train.total_batch_size = BATCH_SIZE

# ── LR Scheduler ─────────────────────────────────────────────
lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values     = [1.0, 0.1, 0.01],
        milestones = [int(MAX_ITER * 0.889), int(MAX_ITER * 0.963)],
        num_updates= MAX_ITER,
    ),
    warmup_length = WARMUP_ITERS / MAX_ITER,
    warmup_factor = 0.001,
)

# ── Optimizer: AdamW ─────────────────────────────────────────
optimizer = L(torch.optim.AdamW)(
    params=L(get_default_optimizer_params)(
        base_lr          = "${..lr}",
        weight_decay_norm= 0.0,
        lr_factor_func   = partial(
            get_vim_lr_decay_rate,
            num_layers    = DEPTH,
            lr_decay_rate = 0.0,    # backbone lr = 0 (frozen)
        ),
        overrides = {"pos_embed": {"weight_decay": 0.0}},
    ),
    lr           = 0.0001,
    betas        = (0.9, 0.999),
    weight_decay = 1e-4,
)
