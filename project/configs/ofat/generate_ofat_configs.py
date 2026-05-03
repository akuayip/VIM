"""
configs/ofat/generate_ofat_configs.py

Script untuk generate semua config file OFAT secara otomatis.
Jalankan sekali sebelum training:

  python configs/ofat/generate_ofat_configs.py

Akan menghasilkan file config di configs/ofat/:
  - baseline.py
  - lr_0_01.py
  - lr_0_005.py
  - bs_8.py
  - bs_32.py
  - opt_sgd.py
  - opt_rmsprop.py
  - ep_150.py
  - ep_200.py
"""

import os

# ── OFAT settings ─────────────────────────────────────────────────────────────
BASELINE = dict(
    lr=0.001, batch_size=16, optimizer="adamw", epochs=100
)

OFAT_FACTORS = {
    "lr"         : [0.01, 0.001, 0.005],
    "batch_size" : [8, 16, 32],
    "optimizer"  : ["adamw", "sgd", "rmsprop"],
    "epochs"     : [100, 150, 200],
}

# ── Template config ───────────────────────────────────────────────────────────
TEMPLATE = '''"""
Auto-generated OFAT config: {run_name}
  lr={lr}, batch_size={batch_size}, optimizer={optimizer}, epochs={epochs}
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

PRETRAINED_VIM   = "checkpoints/vim_tiny_pretrained.pth"
NUM_TRAIN_IMAGES = 5000
EPOCHS           = {epochs}
BATCH_SIZE       = {batch_size}
DEPTH            = 24
ITERS_PER_EPOCH  = NUM_TRAIN_IMAGES // BATCH_SIZE
MAX_ITER         = ITERS_PER_EPOCH * EPOCHS
WARMUP_ITERS     = ITERS_PER_EPOCH * 5

model.backbone.net.pretrained = PRETRAINED_VIM
model.roi_heads.num_classes   = 1

train = dict(
    output_dir      = "./outputs/{run_name}",
    init_checkpoint = PRETRAINED_VIM,
    max_iter        = MAX_ITER,
    amp             = dict(enabled=True),
    ddp             = dict(broadcast_buffers=False, find_unused_parameters=False, fp16_compression=True),
    checkpointer    = dict(period=ITERS_PER_EPOCH, max_to_keep=3),
    eval_period     = ITERS_PER_EPOCH,
    log_period      = 20,
    device          = "cuda",
    freeze_backbone = True,
)

dataloader.train.total_batch_size = BATCH_SIZE

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values     =[1.0, 0.1, 0.01],
        milestones  =[int(MAX_ITER * 0.889), int(MAX_ITER * 0.963)],
        num_updates =MAX_ITER,
    ),
    warmup_length = WARMUP_ITERS / MAX_ITER,
    warmup_factor = 0.001,
)

{optimizer_block}
'''

# ── Optimizer blocks ──────────────────────────────────────────────────────────
def make_optimizer_block(optimizer_name: str, lr: float) -> str:
    if optimizer_name == "adamw":
        return f"""optimizer = L(torch.optim.AdamW)(
    params=L(get_default_optimizer_params)(
        base_lr="{{"..lr"}}",
        weight_decay_norm=0.0,
        lr_factor_func=partial(get_vim_lr_decay_rate, num_layers=DEPTH, lr_decay_rate=0.0),
        overrides={{"pos_embed": {{"weight_decay": 0.0}}}},
    ),
    lr          ={lr},
    betas       =(0.9, 0.999),
    weight_decay=1e-4,
)"""

    elif optimizer_name == "sgd":
        return f"""optimizer = L(torch.optim.SGD)(
    params=L(get_default_optimizer_params)(
        base_lr="{{"..lr"}}",
        weight_decay_norm=0.0,
        lr_factor_func=partial(get_vim_lr_decay_rate, num_layers=DEPTH, lr_decay_rate=0.0),
    ),
    lr          ={lr},
    momentum    =0.9,
    weight_decay=1e-4,
    nesterov    =True,
)"""

    elif optimizer_name == "rmsprop":
        return f"""optimizer = L(torch.optim.RMSprop)(
    params=L(get_default_optimizer_params)(
        base_lr="{{"..lr"}}",
        weight_decay_norm=0.0,
        lr_factor_func=partial(get_vim_lr_decay_rate, num_layers=DEPTH, lr_decay_rate=0.0),
    ),
    lr          ={lr},
    momentum    =0.9,
    weight_decay=1e-4,
    alpha       =0.99,
    eps         =1e-8,
)"""


# ── Generate configs ───────────────────────────────────────────────────────────
def slugify(v):
    return str(v).replace(".", "_").replace("-", "_")


def generate_configs():
    os.makedirs(os.path.dirname(__file__), exist_ok=True)

    seen = set()
    configs_generated = []

    # Baseline + semua variasi OFAT
    all_runs = []

    # Tambahkan baseline
    all_runs.append((dict(BASELINE), "baseline"))

    # Tambahkan variasi per faktor
    for factor, values in OFAT_FACTORS.items():
        for val in values:
            cfg = dict(BASELINE)
            cfg[factor] = val
            key = frozenset(cfg.items())
            if key in seen:
                continue
            seen.add(key)
            run_name = f"ofat_{factor}_{slugify(val)}"
            all_runs.append((cfg, run_name))

    for cfg, run_name in all_runs:
        opt_block = make_optimizer_block(cfg["optimizer"], cfg["lr"])
        content = TEMPLATE.format(
            run_name       =run_name,
            lr             =cfg["lr"],
            batch_size     =cfg["batch_size"],
            optimizer      =cfg["optimizer"],
            epochs         =cfg["epochs"],
            optimizer_block=opt_block,
        )
        out_path = os.path.join(os.path.dirname(__file__), f"{run_name}.py")
        with open(out_path, "w") as f:
            f.write(content)
        configs_generated.append((run_name, out_path))
        print(f"  Generated: {out_path}")

    print(f"\nTotal configs: {len(configs_generated)}")
    return configs_generated


if __name__ == "__main__":
    print("Generating OFAT configs...\n")
    generate_configs()
