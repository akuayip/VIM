import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "libs" / "detectron2_vim"))

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import AMPTrainer, SimpleTrainer, default_argument_parser, default_setup, default_writers, hooks, launch
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.evaluation import COCOEvaluator

logger = logging.getLogger("detectron2")

def freeze_vim_backbone(model):
    frozen = trainable = 0
    for name, param in model.named_parameters():
        if name.startswith("backbone.net."):
            param.requires_grad = False
            frozen += param.numel()
        else:
            param.requires_grad = True
            trainable += param.numel()
    logger.info(f"[Freeze] Backbone frozen    : {frozen:,} params\n[Freeze] Neck+Head trainable: {trainable:,} params")
    return model

def build_evaluator(cfg, dataset_name, output_folder=None):
    if output_folder is None:
        output_folder = os.path.join(cfg.train.output_dir, "inference")
    return COCOEvaluator(dataset_name, output_dir=output_folder)

class BestCheckpointerHook(hooks.HookBase):
    def __init__(self, checkpointer, output_dir, eval_period):
        self._ckpt        = checkpointer
        self._output_dir  = Path(output_dir)
        self._eval_period = eval_period
        self._best_map    = -1.0
        self._best_iter   = -1

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final  = (next_iter == self.trainer.max_iter)
        if (next_iter % self._eval_period == 0 or is_final) and comm.is_main_process():
            self._ckpt.save("last")
            logger.info(f"[Checkpoint] last.pth disimpan (iter {self.trainer.iter})")
        if (next_iter % self._eval_period == 0 or is_final) and comm.is_main_process():
            try:
                latest  = self.trainer.storage.latest()
                map_val = latest.get("bbox/AP") or latest.get("AP")
                if map_val is not None:
                    val = float(map_val[0]) if isinstance(map_val, tuple) else float(map_val)
                    if val > self._best_map:
                        self._best_map  = val
                        self._best_iter = self.trainer.iter
                        self._ckpt.save("best")
                        logger.info(f"[Checkpoint] ★ best.pth diperbarui! mAP@50:95={val:.3f} (iter {self.trainer.iter})")
            except Exception:
                pass

    def after_train(self):
        if comm.is_main_process():
            logger.info(f"\n{'='*50}\n  Training selesai!\n  Best mAP@50:95 : {self._best_map:.3f} (iter {self._best_iter})\n  best.pth → {self._output_dir}/best.pth\n  last.pth → {self._output_dir}/last.pth\n{'='*50}")

def do_train(args, cfg):
    model = instantiate(cfg.model)
    if cfg.train.get("freeze_backbone", True):
        model = freeze_vim_backbone(model)
    model.to(cfg.train.device)
    cfg.optimizer.params.model = model
    optim        = instantiate(cfg.optimizer)
    train_loader = instantiate(cfg.dataloader.train)
    trainer      = AMPTrainer(model, train_loader, optim) if cfg.train.amp.enabled else SimpleTrainer(model, train_loader, optim)
    checkpointer = DetectionCheckpointer(model, cfg.train.output_dir, trainer=trainer)
    eval_period  = cfg.train.eval_period
    trainer.register_hooks([
        hooks.IterationTimer(),
        hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
        hooks.EvalHook(eval_period, lambda: do_test(cfg, model)),
        BestCheckpointerHook(checkpointer, cfg.train.output_dir, eval_period) if comm.is_main_process() else None,
        hooks.PeriodicWriter(default_writers(cfg.train.output_dir, cfg.train.max_iter), period=cfg.train.log_period) if comm.is_main_process() else None,
    ])
    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    start_iter = 0 if not args.resume else trainer.iter + 1
    logger.info(f"Training: iter {start_iter} → {cfg.train.max_iter}")
    trainer.train(start_iter, cfg.train.max_iter)

def do_test(cfg, model):
    if "test" not in cfg.dataloader:
        return {}
    metrics = inference_on_dataset(model, instantiate(cfg.dataloader.test), build_evaluator(cfg, "head_val"))
    bbox    = metrics.get("bbox", {})
    logger.info(f"\n{'─'*40}\n  mAP@50    : {round(bbox.get('AP50',0),3):>7.3f}\n  mAP@50:95 : {round(bbox.get('AP',0),3):>7.3f}\n{'─'*40}")
    return metrics

def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)
    if args.eval_only:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
        print_csv_format(do_test(cfg, model))
        return
    do_train(args, cfg)

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(main, args.num_gpus, num_machines=args.num_machines, machine_rank=args.machine_rank, dist_url=args.dist_url, args=(args,))
