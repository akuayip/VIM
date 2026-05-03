"""
tools/eval_effectiveness.py

Menghitung metrik efektivitas model untuk perbandingan:
  - mAP@50       (COCO AP IoU=0.50)
  - mAP@50:95    (COCO AP IoU=0.50:0.95, main metric)
  - Precision    (pada threshold score tertentu)
  - Recall       (pada threshold score tertentu)

Cara pakai:
  # Eval Vim-Det (Detectron2)
  python tools/eval_effectiveness.py \
      --config-file  configs/head_detection_baseline.py \
      --checkpoint   outputs/baseline/model_best.pth \
      --model-name   "Vim-Det (Tiny)" \
      --model-type   detectron2 \
      --dataset-name head_val

  # Eval Mamba-YOLO (YOLO-style)
  python tools/eval_effectiveness.py \
      --model-name   "Mamba-YOLO" \
      --model-type   yolo \
      --yolo-weights path/to/mamba_yolo.pt \
      --val-json     data/val/annotations.json \
      --val-images   data/val/images

  # Bandingkan semua model dari JSON yang sudah ada
  python tools/eval_effectiveness.py --compare
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

import torch
import numpy as np
from collections import defaultdict

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "Vim" / "det"))

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Detectron2 COCO Evaluator (mAP50, mAP50:95)
# ─────────────────────────────────────────────────────────────────────────────
def eval_detectron2(model, cfg, dataset_name: str) -> dict:
    """
    Jalankan COCO evaluasi menggunakan Detectron2 COCOEvaluator.

    Returns dict:
      AP      : mAP@50:95  ← metrik utama, konsisten dengan Mamba-YOLO
      AP50    : mAP@50     ← metrik utama, konsisten dengan Mamba-YOLO
    """
    from detectron2.config import instantiate
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset

    print(f"\n[Detectron2] Evaluating on '{dataset_name}'...")
    model.eval()

    test_loader = instantiate(cfg.dataloader.test)
    evaluator   = COCOEvaluator(
        dataset_name,
        output_dir=os.path.join(cfg.train.output_dir, "inference"),
        tasks=["bbox"],            # hanya bbox, tidak segmentasi
    )

    results  = inference_on_dataset(model, test_loader, evaluator)
    bbox_res = results.get("bbox", {})

    # Hanya ambil AP (50:95) dan AP50 — konsisten dengan Mamba-YOLO
    return {
        "AP"   : round(bbox_res.get("AP",   0), 3),   # mAP@50:95
        "AP50" : round(bbox_res.get("AP50", 0), 3),   # mAP@50
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. COCO-style P/R dari predictions JSON
# ─────────────────────────────────────────────────────────────────────────────
def compute_precision_recall_from_coco(
    predictions: list,
    gt_annotations: list,
    iou_threshold: float = 0.50,
    score_threshold: float = 0.50,
) -> dict:
    """
    Hitung Precision dan Recall dari predictions dan ground-truth COCO format.

    Args:
        predictions   : list of dicts {"image_id", "bbox":[x,y,w,h], "score", "category_id"}
        gt_annotations: list of COCO annotation dicts (dari annotations.json)
        iou_threshold : IoU minimum untuk TP (default 0.50)
        score_threshold: score minimum untuk menganggap prediksi valid

    Returns:
        {"precision": float, "recall": float, "f1": float,
         "TP": int, "FP": int, "FN": int, "score_threshold": float}
    """
    # Filter prediksi di bawah score threshold
    preds = [p for p in predictions if p.get("score", 0) >= score_threshold]

    # Group GT per image
    gt_by_image = defaultdict(list)
    for ann in gt_annotations:
        if ann.get("iscrowd", 0):
            continue
        gt_by_image[ann["image_id"]].append(ann)

    # Group prediksi per image, sorted by score descending
    pred_by_image = defaultdict(list)
    for p in preds:
        pred_by_image[p["image_id"]].append(p)

    TP = FP = FN = 0

    all_image_ids = set(gt_by_image.keys()) | set(pred_by_image.keys())
    for img_id in all_image_ids:
        gts   = gt_by_image[img_id]
        dets  = sorted(pred_by_image[img_id], key=lambda x: -x.get("score", 0))
        matched_gt = [False] * len(gts)

        for det in dets:
            best_iou = iou_threshold - 1e-5
            best_idx = -1

            for j, gt in enumerate(gts):
                if matched_gt[j]:
                    continue
                iou = _box_iou_xywh(det["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j

            if best_idx >= 0:
                TP += 1
                matched_gt[best_idx] = True
            else:
                FP += 1

        FN += matched_gt.count(False)

    precision = TP / (TP + FP + 1e-9)
    recall    = TP / (TP + FN + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)

    return {
        "precision"      : round(precision, 4),
        "recall"         : round(recall, 4),
        "f1"             : round(f1, 4),
        "TP"             : TP,
        "FP"             : FP,
        "FN"             : FN,
        "score_threshold": score_threshold,
        "iou_threshold"  : iou_threshold,
    }


def _box_iou_xywh(box1, box2) -> float:
    """IoU antara dua box format [x, y, w, h]."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)
    inter = max(0, xb - xa) * max(0, yb - ya)
    union = w1 * h1 + w2 * h2 - inter
    return inter / (union + 1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Baca predictions Detectron2 dari inference output
# ─────────────────────────────────────────────────────────────────────────────
def load_detectron2_predictions(inference_dir: str) -> list:
    """Load coco_instances_results.json dari Detectron2 inference output."""
    pred_path = Path(inference_dir) / "coco_instances_results.json"
    if not pred_path.exists():
        logger.warning(f"Prediction file tidak ditemukan: {pred_path}")
        return []
    with open(pred_path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Full effectiveness report untuk satu model
# ─────────────────────────────────────────────────────────────────────────────
def eval_model_effectiveness(
    model_name: str,
    map_results: dict,
    pr_results: dict,
) -> dict:
    """
    Gabungkan mAP dan P/R ke satu dict hasil.
    Metrik konsisten dengan Mamba-YOLO: mAP@50 dan mAP@50:95.
    """
    return {
        "model_name" : model_name,
        "mAP_50"     : map_results.get("AP50", 0),   # mAP@50
        "mAP_50_95"  : map_results.get("AP",   0),   # mAP@50:95
        "precision"  : pr_results.get("precision", 0),
        "recall"     : pr_results.get("recall",    0),
        "f1"         : pr_results.get("f1",        0),
        "details"    : {
            "map" : map_results,
            "pr"  : pr_results,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Pretty print & save
# ─────────────────────────────────────────────────────────────────────────────
def print_effectiveness_report(results: list):
    """Print tabel perbandingan efektivitas semua model."""
    print("\n" + "=" * 75)
    print(f"  {'Model':<25} {'mAP@50':>8} {'mAP@50:95':>11} {'Precision':>11} {'Recall':>9} {'F1':>7}")
    print("-" * 75)
    for r in results:
        print(
            f"  {r['model_name']:<25}"
            f"  {r['mAP_50']:>8.2f}"
            f"  {r['mAP_50_95']:>11.2f}"
            f"  {r['precision']:>11.4f}"
            f"  {r['recall']:>9.4f}"
            f"  {r['f1']:>7.4f}"
        )
    print("=" * 75)


def save_effectiveness_results(results: list, output_path: str):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Effectiveness] Hasil disimpan: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Evaluasi Efektivitas Model")
    p.add_argument("--config-file",   default="")
    p.add_argument("--checkpoint",    default="")
    p.add_argument("--model-name",    default="Model")
    p.add_argument("--model-type",    default="detectron2",
                   choices=["detectron2", "yolo"])
    p.add_argument("--dataset-name",  default="head_val",
                   help="Nama dataset terdaftar di Detectron2 (untuk d2 eval)")
    p.add_argument("--val-json",      default="",
                   help="Path COCO val annotations.json (untuk P/R manual)")
    p.add_argument("--val-images",    default="",
                   help="Path direktori val images")
    p.add_argument("--yolo-weights",  default="")
    p.add_argument("--score-thresh",  default=0.50, type=float,
                   help="Score threshold untuk P/R")
    p.add_argument("--iou-thresh",    default=0.50, type=float,
                   help="IoU threshold untuk TP")
    p.add_argument("--output",
                   default="outputs/effectiveness_results.json")
    p.add_argument("--compare",       action="store_true")
    p.add_argument("--compare-file",  default="outputs/effectiveness_results.json")
    return p.parse_args()


def main():
    args = parse_args()

    if args.compare:
        with open(args.compare_file) as f:
            results = json.load(f)
        print_effectiveness_report(results)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    map_results = {}
    pr_results  = {}

    # ── Detectron2 ──────────────────────────────────────────────────────────
    if args.model_type == "detectron2" and args.config_file:
        from detectron2.config import LazyConfig, instantiate
        from detectron2.checkpoint import DetectionCheckpointer

        # Register dataset dulu
        from configs.data.head_coco_loader import (  # noqa: F401 — side effect
            register_coco_instances,
            TRAIN_JSON, TRAIN_IMG, VAL_JSON, VAL_IMG,
        )

        cfg   = LazyConfig.load(args.config_file)
        model = instantiate(cfg.model)
        model.to(device)

        if args.checkpoint:
            DetectionCheckpointer(model).load(args.checkpoint)

        # mAP via COCOEvaluator
        map_results = eval_detectron2(model, cfg, args.dataset_name)

        # Precision & Recall dari predictions file
        inference_dir = Path(cfg.train.output_dir) / "inference"
        preds         = load_detectron2_predictions(str(inference_dir))

        if preds and args.val_json:
            with open(args.val_json) as f:
                coco_gt = json.load(f)
            pr_results = compute_precision_recall_from_coco(
                preds,
                coco_gt["annotations"],
                iou_threshold  =args.iou_thresh,
                score_threshold=args.score_thresh,
            )
        elif preds:
            logger.warning("--val-json tidak diberikan, P/R tidak dapat dihitung.")

    # ── YOLO ────────────────────────────────────────────────────────────────
    elif args.model_type == "yolo" and args.yolo_weights:
        try:
            # Coba load sebagai ultralytics YOLO
            from ultralytics import YOLO
            yolo = YOLO(args.yolo_weights)
            val_res = yolo.val(data=args.val_json or "dataset.yaml", verbose=True)

            map_results = {
                "AP"   : round(float(val_res.box.map),    3),
                "AP50" : round(float(val_res.box.map50),  3),
                "AP75" : round(float(val_res.box.map75),  3),
            }
            pr_results = {
                "precision": round(float(val_res.box.mp), 4),
                "recall"   : round(float(val_res.box.mr), 4),
                "f1"       : round(2 * float(val_res.box.mp) * float(val_res.box.mr)
                                   / (float(val_res.box.mp) + float(val_res.box.mr) + 1e-9), 4),
            }
        except ImportError:
            logger.warning("ultralytics tidak tersedia. Install: pip install ultralytics")
            return

    else:
        print("[ERROR] Tentukan --config-file (detectron2) atau --yolo-weights (yolo)")
        return

    # ── Gabungkan & simpan ──────────────────────────────────────────────────
    result = eval_model_effectiveness(args.model_name, map_results, pr_results)

    # Append ke file yang ada
    out_path = Path(args.output)
    existing = []
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
        existing = [r for r in existing if r["model_name"] != args.model_name]

    existing.append(result)
    save_effectiveness_results(existing, args.output)
    print_effectiveness_report(existing)


if __name__ == "__main__":
    main()
