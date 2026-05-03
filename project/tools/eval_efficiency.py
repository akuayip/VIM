"""
tools/eval_efficiency.py

Menghitung metrik efisiensi model untuk perbandingan:
  - Params (jumlah parameter trainable & total)
  - FLOPs  (Floating Point Operations per forward pass)
  - Latency (ms/image, throughput FPS) pada CPU dan GPU

Cara pakai:
  # Eval model Vim-Det (Detectron2)
  python tools/eval_efficiency.py \
      --config-file configs/head_detection_baseline.py \
      --checkpoint  outputs/baseline/model_best.pth \
      --model-name  "Vim-Det (Tiny)" \
      --input-size  1024 \
      --batch-size  1 \
      --warmup      50 \
      --runs        200

  # Eval model Mamba-YOLO (PyTorch biasa / YOLO-style)
  python tools/eval_efficiency.py \
      --model-name  "Mamba-YOLO" \
      --model-type  yolo \
      --yolo-weights path/to/mamba_yolo.pt \
      --input-size  640 \
      --batch-size  1 \
      --warmup      50 \
      --runs        200

  # Bandingkan semua model sekaligus
  python tools/eval_efficiency.py --compare
"""

import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn
import numpy as np

# Tambahkan root ke path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "Vim" / "det"))

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: dummy input
# ─────────────────────────────────────────────────────────────────────────────
def make_dummy_input(input_size: int, batch_size: int = 1,
                     device: str = "cuda") -> torch.Tensor:
    return torch.zeros(batch_size, 3, input_size, input_size).to(device)


def make_detectron2_batched_input(input_size: int, batch_size: int = 1,
                                   device: str = "cuda") -> list:
    """Detectron2 model expects list of dicts."""
    return [
        {
            "image": torch.zeros(3, input_size, input_size).to(device),
            "height": input_size,
            "width":  input_size,
        }
        for _ in range(batch_size)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Params
# ─────────────────────────────────────────────────────────────────────────────
def count_params(model: nn.Module) -> dict:
    """
    Hitung jumlah parameter model.

    Returns:
        dict dengan keys:
          total_params     : semua parameter
          trainable_params : parameter yang memiliki requires_grad=True
          frozen_params    : parameter yang di-freeze
          total_M          : total dalam juta (M)
          trainable_M      : trainable dalam juta (M)
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable

    return {
        "total_params"     : total,
        "trainable_params" : trainable,
        "frozen_params"    : frozen,
        "total_M"          : round(total / 1e6, 2),
        "trainable_M"      : round(trainable / 1e6, 2),
        "frozen_M"         : round(frozen / 1e6, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. FLOPs
# ─────────────────────────────────────────────────────────────────────────────
def count_flops(model: nn.Module, input_size: int,
                model_type: str = "detectron2",
                device: str = "cuda") -> dict:
    """
    Hitung FLOPs menggunakan fvcore.

    Args:
        model      : model PyTorch
        input_size : resolusi input (e.g. 1024 → [3,1024,1024])
        model_type : 'detectron2' | 'yolo' | 'torch'
        device     : 'cuda' | 'cpu'

    Returns:
        dict dengan keys:
          flops       : total FLOPs (int)
          gflops      : dalam GigaFLOPs
          params      : params dari fvcore (cross-check)
    """
    try:
        from fvcore.nn import FlopCountAnalysis, parameter_count
    except ImportError:
        logger.warning("fvcore tidak tersedia. Install: pip install fvcore")
        return {"flops": 0, "gflops": 0.0, "params": 0}

    model.eval()

    if model_type == "detectron2":
        dummy = make_detectron2_batched_input(input_size, 1, device)
        try:
            flops_counter = FlopCountAnalysis(model, dummy)
            flops_counter.unsupported_ops_warnings(False)
            flops_counter.uncalled_modules_warnings(False)
            total_flops = flops_counter.total()
        except Exception as e:
            logger.warning(f"FLOPs count gagal untuk Detectron2 model: {e}")
            total_flops = 0
    else:
        # YOLO atau PyTorch biasa
        dummy = make_dummy_input(input_size, 1, device)
        try:
            flops_counter = FlopCountAnalysis(model, dummy)
            flops_counter.unsupported_ops_warnings(False)
            flops_counter.uncalled_modules_warnings(False)
            total_flops = flops_counter.total()
        except Exception as e:
            logger.warning(f"FLOPs count gagal: {e}")
            total_flops = 0

    params_fvcore = sum(parameter_count(model).values())

    return {
        "flops"  : int(total_flops),
        "gflops" : round(total_flops / 1e9, 2),
        "params" : int(params_fvcore),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Latency & Throughput
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def measure_latency(model: nn.Module, input_size: int,
                    batch_size: int = 1,
                    model_type: str = "detectron2",
                    device: str = "cuda",
                    warmup: int = 50,
                    runs: int = 200) -> dict:
    """
    Ukur latency dan throughput model.

    Args:
        model      : model PyTorch
        input_size : resolusi input
        batch_size : ukuran batch
        model_type : 'detectron2' | 'yolo' | 'torch'
        device     : 'cuda' | 'cpu'
        warmup     : iterasi warmup sebelum pengukuran
        runs       : iterasi pengukuran

    Returns:
        dict dengan:
          latency_mean_ms  : rata-rata latency (ms) per batch
          latency_std_ms   : std latency (ms)
          latency_p50_ms   : median latency (ms)
          latency_p95_ms   : P95 latency (ms)
          latency_p99_ms   : P99 latency (ms)
          throughput_fps   : gambar per detik (batch_size / latency)
          per_image_ms     : latency per gambar (ms)
          device           : device yang digunakan
          batch_size       : batch size yang digunakan
    """
    model.eval()
    use_cuda = (device == "cuda" and torch.cuda.is_available())

    # Buat input dummy
    if model_type == "detectron2":
        dummy = make_detectron2_batched_input(input_size, batch_size, device)
    else:
        dummy = make_dummy_input(input_size, batch_size, device)

    # Warmup
    print(f"  Warmup ({warmup} iters)...")
    for _ in range(warmup):
        if model_type == "detectron2":
            _ = model(dummy)
        else:
            _ = model(dummy)

    if use_cuda:
        torch.cuda.synchronize()

    # Pengukuran
    print(f"  Measuring ({runs} iters)...")
    times = []
    for _ in range(runs):
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        if model_type == "detectron2":
            _ = model(dummy)
        else:
            _ = model(dummy)

        if use_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)   # ms

    times = np.array(times)
    latency_mean = float(np.mean(times))

    return {
        "latency_mean_ms" : round(latency_mean, 3),
        "latency_std_ms"  : round(float(np.std(times)), 3),
        "latency_p50_ms"  : round(float(np.percentile(times, 50)), 3),
        "latency_p95_ms"  : round(float(np.percentile(times, 95)), 3),
        "latency_p99_ms"  : round(float(np.percentile(times, 99)), 3),
        "throughput_fps"  : round(batch_size / (latency_mean / 1000), 2),
        "per_image_ms"    : round(latency_mean / batch_size, 3),
        "device"          : device,
        "batch_size"      : batch_size,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Full efficiency report untuk satu model
# ─────────────────────────────────────────────────────────────────────────────
def eval_model_efficiency(
    model: nn.Module,
    model_name: str,
    input_size: int,
    model_type: str = "detectron2",
    batch_size: int = 1,
    device: str = "cuda",
    warmup: int = 50,
    runs: int = 200,
) -> dict:
    """
    Jalankan semua evaluasi efisiensi untuk satu model.

    Returns dict:
      model_name, params, flops, latency (GPU + CPU)
    """
    print(f"\n{'='*60}")
    print(f"  Efficiency Eval: {model_name}")
    print(f"  Input: {batch_size} x 3 x {input_size} x {input_size}")
    print(f"{'='*60}")

    result = {"model_name": model_name, "input_size": input_size}

    # 1. Params
    print("\n[1/3] Counting parameters...")
    result["params"] = count_params(model)

    # 2. FLOPs
    print("\n[2/3] Counting FLOPs...")
    result["flops"] = count_flops(model, input_size, model_type, device)

    # 3. Latency GPU
    if torch.cuda.is_available() and device == "cuda":
        print("\n[3a/3] Measuring GPU latency...")
        result["latency_gpu"] = measure_latency(
            model, input_size, batch_size, model_type, "cuda", warmup, runs
        )

    # 3b. Latency CPU
    print("\n[3b/3] Measuring CPU latency (fewer runs)...")
    try:
        model_cpu = model.cpu()
        result["latency_cpu"] = measure_latency(
            model_cpu, input_size, batch_size, model_type, "cpu",
            warmup=min(warmup, 10), runs=min(runs, 50)
        )
        if torch.cuda.is_available():
            model.to(device)
    except Exception as e:
        logger.warning(f"CPU latency gagal: {e}")
        result["latency_cpu"] = {}

    print(f"\n  ✓ {model_name} selesai.")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 5. Pretty print & save
# ─────────────────────────────────────────────────────────────────────────────
def print_efficiency_report(results: list):
    """Print tabel perbandingan efisiensi semua model."""
    print("\n" + "=" * 90)
    print(f"  {'Model':<25} {'Params(M)':>10} {'GFLOPs':>8} "
          f"{'GPU ms':>8} {'FPS':>8} {'CPU ms':>9}")
    print("-" * 90)

    for r in results:
        name       = r["model_name"]
        params_m   = r.get("params", {}).get("total_M", 0)
        gflops     = r.get("flops",  {}).get("gflops", 0)
        gpu_ms     = r.get("latency_gpu", {}).get("per_image_ms", 0)
        fps        = r.get("latency_gpu", {}).get("throughput_fps", 0)
        cpu_ms     = r.get("latency_cpu", {}).get("per_image_ms", 0)
        print(f"  {name:<25} {params_m:>10.2f} {gflops:>8.1f} "
              f"{gpu_ms:>8.1f} {fps:>8.1f} {cpu_ms:>9.1f}")

    print("=" * 90)


def save_efficiency_results(results: list, output_path: str):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Efficiency] Hasil disimpan: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Evaluasi Efisiensi Model")
    p.add_argument("--config-file",  default="",  help="Detectron2 LazyConfig path")
    p.add_argument("--checkpoint",   default="",  help="Path checkpoint .pth")
    p.add_argument("--model-name",   default="Model", help="Nama model untuk laporan")
    p.add_argument("--model-type",   default="detectron2",
                   choices=["detectron2", "yolo", "torch"],
                   help="Tipe model")
    p.add_argument("--yolo-weights", default="",  help="Path weights Mamba-YOLO")
    p.add_argument("--input-size",   default=1024, type=int)
    p.add_argument("--batch-size",   default=1,    type=int)
    p.add_argument("--device",       default="cuda")
    p.add_argument("--warmup",       default=50,   type=int)
    p.add_argument("--runs",         default=200,  type=int)
    p.add_argument("--output",       default="outputs/efficiency_results.json")
    p.add_argument("--compare",      action="store_true",
                   help="Mode compare: load hasil JSON semua model dan print tabel")
    p.add_argument("--compare-file", default="outputs/efficiency_results.json")
    return p.parse_args()


def main():
    args = parse_args()

    # Mode compare: hanya print tabel dari JSON yang sudah ada
    if args.compare:
        with open(args.compare_file) as f:
            results = json.load(f)
        print_efficiency_report(results)
        return

    results = []
    device  = args.device if torch.cuda.is_available() else "cpu"

    # ── Load Detectron2 model ────────────────────────────────────────────────
    if args.model_type == "detectron2" and args.config_file:
        from detectron2.config import LazyConfig, instantiate
        from detectron2.checkpoint import DetectionCheckpointer

        cfg   = LazyConfig.load(args.config_file)
        model = instantiate(cfg.model)

        if args.checkpoint:
            DetectionCheckpointer(model).load(args.checkpoint)

        model.eval()
        model.to(device)

    # ── Load Mamba-YOLO model ─────────────────────────────────────────────────
    elif args.model_type == "yolo" and args.yolo_weights:
        try:
            # YOLO-style load (ultralytics / custom)
            import torch
            model = torch.load(args.yolo_weights, map_location=device)
            if hasattr(model, "model"):       # ultralytics format
                model = model.model
            model.eval()
            model.to(device)
        except Exception as e:
            print(f"[ERROR] Gagal load YOLO weights: {e}")
            return

    else:
        print("[ERROR] Tentukan --config-file (detectron2) atau --yolo-weights (yolo)")
        return

    # ── Eval ─────────────────────────────────────────────────────────────────
    result = eval_model_efficiency(
        model      =model,
        model_name =args.model_name,
        input_size =args.input_size,
        model_type =args.model_type,
        batch_size =args.batch_size,
        device     =device,
        warmup     =args.warmup,
        runs       =args.runs,
    )
    results.append(result)

    # ── Append ke file hasil yang sudah ada ───────────────────────────────────
    out_path = Path(args.output)
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
        # Replace entry yang sama namanya, atau append
        existing = [r for r in existing if r["model_name"] != args.model_name]
        existing.append(result)
        results = existing

    save_efficiency_results(results, args.output)
    print_efficiency_report(results)


if __name__ == "__main__":
    main()
