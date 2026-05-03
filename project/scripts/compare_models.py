"""
scripts/compare_models.py

Script utama untuk membandingkan Vim-Det vs Mamba-YOLO secara lengkap.
Menggabungkan efisiensi (Params, FLOPs, Latency) dan 
efektivitas (mAP50, mAP50:95, Precision, Recall) dalam satu laporan.

Cara pakai:

  # Step 1: Eval efisiensi Vim-Det
  python tools/eval_efficiency.py \
      --config-file  configs/head_detection_baseline.py \
      --checkpoint   outputs/baseline/model_best.pth \
      --model-name   "Vim-Det Tiny" \
      --model-type   detectron2 \
      --input-size   1024

  # Step 2: Eval efisiensi Mamba-YOLO
  python tools/eval_efficiency.py \
      --model-name   "Mamba-YOLO" \
      --model-type   yolo \
      --yolo-weights path/to/mamba_yolo.pt \
      --input-size   640

  # Step 3: Eval efektivitas Vim-Det
  python tools/eval_effectiveness.py \
      --config-file  configs/head_detection_baseline.py \
      --checkpoint   outputs/baseline/model_best.pth \
      --model-name   "Vim-Det Tiny" \
      --val-json     data/val/annotations.json

  # Step 4: Eval efektivitas Mamba-YOLO
  python tools/eval_effectiveness.py \
      --model-name   "Mamba-YOLO" \
      --model-type   yolo \
      --yolo-weights path/to/mamba_yolo.pt \
      --val-json     data/val/annotations.json

  # Step 5: Gabungkan dan print laporan final
  python scripts/compare_models.py
"""

import json
import argparse
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Merge efficiency + effectiveness per model
# ─────────────────────────────────────────────────────────────────────────────
def merge_results(
    efficiency_file: str = "outputs/efficiency_results.json",
    effectiveness_file: str = "outputs/effectiveness_results.json",
) -> list:
    eff_map  = {}
    efft_map = {}

    if Path(efficiency_file).exists():
        with open(efficiency_file) as f:
            for r in json.load(f):
                eff_map[r["model_name"]] = r

    if Path(effectiveness_file).exists():
        with open(effectiveness_file) as f:
            for r in json.load(f):
                efft_map[r["model_name"]] = r

    all_names = set(eff_map.keys()) | set(efft_map.keys())
    merged = []
    for name in sorted(all_names):
        row = {"model_name": name}
        row.update(eff_map.get(name, {}))
        row.update(efft_map.get(name, {}))
        merged.append(row)

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Print full comparison table
# ─────────────────────────────────────────────────────────────────────────────
def print_comparison_table(results: list):
    SEP = "=" * 105

    print("\n" + SEP)
    print("  MODEL COMPARISON: Vim-Det vs Mamba-YOLO")
    print("  Task: Human Head Detection")
    print(SEP)

    # ── Efisiensi ─────────────────────────────────────────────────────────────
    print("\n  [ EFISIENSI ]\n")
    header = (f"  {'Model':<22} {'Params(M)':>10} {'FrozenP(M)':>11} "
              f"{'GFLOPs':>8} {'GPU ms/img':>11} {'FPS':>7} {'CPU ms/img':>11}")
    print(header)
    print("  " + "-" * 83)

    for r in results:
        name    = r.get("model_name", "?")
        params  = r.get("params", {})
        flops   = r.get("flops",  {})
        lat_gpu = r.get("latency_gpu", {})
        lat_cpu = r.get("latency_cpu", {})

        total_m   = params.get("total_M",    0)
        frozen_m  = params.get("frozen_M",   0)
        gflops    = flops.get("gflops",      0)
        gpu_ms    = lat_gpu.get("per_image_ms", 0)
        fps       = lat_gpu.get("throughput_fps", 0)
        cpu_ms    = lat_cpu.get("per_image_ms", 0)

        print(f"  {name:<22} {total_m:>10.2f} {frozen_m:>11.2f} "
              f"{gflops:>8.1f} {gpu_ms:>11.2f} {fps:>7.1f} {cpu_ms:>11.2f}")

    print()

    # ── Efektivitas ───────────────────────────────────────────────────────────
    print("  [ EFEKTIVITAS ]\n")
    header2 = (f"  {'Model':<22} {'mAP@50':>8} {'mAP@50:95':>11} "
               f"{'Precision':>11} {'Recall':>9} {'F1':>7}")
    print(header2)
    print("  " + "-" * 71)

    for r in results:
        name      = r.get("model_name", "?")
        map50     = r.get("mAP_50",    0)
        map5095   = r.get("mAP_50_95", 0)
        precision = r.get("precision", 0)
        recall    = r.get("recall",    0)
        f1        = r.get("f1",         0)

        print(f"  {name:<22} {map50:>8.2f} {map5095:>11.2f} "
              f"{precision:>11.4f} {recall:>9.4f} {f1:>7.4f}")

    print()

    # ── Trade-off analysis ────────────────────────────────────────────────────
    print("  [ TRADE-OFF ANALYSIS ]\n")

    if len(results) >= 2:
        r0 = results[0]
        r1 = results[1]

        def _delta(key, r_a, r_b, higher_is_better=True):
            va = r_a.get(key, 0) or 0
            vb = r_b.get(key, 0) or 0
            diff = vb - va
            sign = "+" if diff > 0 else ""
            arrow = "↑" if (diff > 0) == higher_is_better else "↓"
            return f"{sign}{diff:.2f} {arrow}"

        def _eff_delta(key, sub, r_a, r_b, higher_is_better=False):
            va = (r_a.get(sub, {}) or {}).get(key, 0) or 0
            vb = (r_b.get(sub, {}) or {}).get(key, 0) or 0
            diff = vb - va
            sign = "+" if diff > 0 else ""
            arrow = "↑" if (diff > 0) == higher_is_better else "↓"
            return f"{sign}{diff:.2f} {arrow}"

        n0 = r0.get("model_name", "Model A")
        n1 = r1.get("model_name", "Model B")

        print(f"  {n1} vs {n0}:\n")
        print(f"    Params (M)     : {_eff_delta('total_M',      'params', r0, r1, higher_is_better=False)}")
        print(f"    GFLOPs         : {_eff_delta('gflops',       'flops',  r0, r1, higher_is_better=False)}")
        print(f"    GPU latency ms : {_eff_delta('per_image_ms', 'latency_gpu', r0, r1, higher_is_better=False)}")
        print(f"    mAP@50         : {_delta('mAP_50',    r0, r1, higher_is_better=True)}")
        print(f"    mAP@50:95      : {_delta('mAP_50_95', r0, r1, higher_is_better=True)}")
        print(f"    Precision      : {_delta('precision', r0, r1, higher_is_better=True)}")
        print(f"    Recall         : {_delta('recall',    r0, r1, higher_is_better=True)}")

    print("\n" + SEP + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Save merged report
# ─────────────────────────────────────────────────────────────────────────────
def save_comparison(results: list, output: str = "outputs/comparison_report.json"):
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Compare] Laporan disimpan: {output}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Compare Vim-Det vs Mamba-YOLO")
    p.add_argument("--efficiency-file",
                   default="outputs/efficiency_results.json")
    p.add_argument("--effectiveness-file",
                   default="outputs/effectiveness_results.json")
    p.add_argument("--output",
                   default="outputs/comparison_report.json")
    return p.parse_args()


def main():
    args    = parse_args()
    results = merge_results(args.efficiency_file, args.effectiveness_file)

    if not results:
        print("[Compare] Tidak ada hasil ditemukan. "
              "Jalankan eval_efficiency.py dan eval_effectiveness.py terlebih dahulu.")
        return

    print_comparison_table(results)
    save_comparison(results, args.output)


if __name__ == "__main__":
    main()
