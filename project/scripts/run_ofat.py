"""
scripts/run_ofat.py

Runner OFAT: generate semua config lalu jalankan satu per satu.

Cara pakai:
  # Dry run (lihat config saja, tidak training)
  python scripts/run_ofat.py --dry_run

  # Jalankan semua eksperimen OFAT
  python scripts/run_ofat.py --num_gpus 1

  # Jalankan hanya satu faktor
  python scripts/run_ofat.py --factor lr --num_gpus 1

  # Resume eksperimen yang gagal
  python scripts/run_ofat.py --num_gpus 1 --resume
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Tambahkan root ke path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from configs.ofat.generate_ofat_configs import generate_configs, BASELINE, OFAT_FACTORS


# ── Semua kombinasi OFAT ──────────────────────────────────────────────────────
def get_all_runs():
    """
    Returns list of (run_name, config_path) untuk semua eksperimen OFAT.
    """
    # Generate config files terlebih dahulu
    configs = generate_configs()
    return configs


# ── Baca hasil training dari output_dir ──────────────────────────────────────
def read_best_ap(output_dir: str) -> float:
    """Baca best AP50 dari metrics.json di output dir."""
    metrics_path = Path(output_dir) / "metrics.json"
    if not metrics_path.exists():
        return float("inf")

    best_ap = 0.0
    with open(metrics_path) as f:
        for line in f:
            try:
                m = json.loads(line.strip())
                ap = m.get("bbox/AP50", m.get("AP50", 0.0))
                if ap > best_ap:
                    best_ap = ap
            except Exception:
                continue
    return best_ap


# ── Print summary table ───────────────────────────────────────────────────────
def print_summary(results: list):
    print("\n" + "=" * 80)
    print("  OFAT EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"  {'Run Name':<45} {'Best AP50':>10} {'Status':>10}")
    print("-" * 80)

    sorted_r = sorted(results, key=lambda r: -r.get("ap50", 0))
    for r in sorted_r:
        ap   = r.get("ap50", 0)
        stat = r.get("status", "?")
        print(f"  {r['run_name']:<45} {ap:>10.2f} {stat:>10}")

    print("=" * 80)
    if sorted_r:
        best = sorted_r[0]
        print(f"\n  ★ BEST: {best['run_name']}  AP50={best.get('ap50', 0):.2f}")
    print("=" * 80 + "\n")


# ── Jalankan satu eksperimen ──────────────────────────────────────────────────
def run_one(run_name: str, config_path: str, num_gpus: int,
            resume: bool, dry_run: bool) -> dict:

    output_dir = str(ROOT / "outputs" / run_name)

    cmd = [
        sys.executable, str(ROOT / "tools" / "train_net.py"),
        "--config-file", config_path,
        "--num-gpus",    str(num_gpus),
    ]
    if resume:
        cmd.append("train.resume=True")

    print(f"\n{'='*70}")
    print(f"  RUN : {run_name}")
    print(f"  CMD : {' '.join(cmd)}")
    print(f"{'='*70}")

    if dry_run:
        return {"run_name": run_name, "status": "dry_run", "ap50": 0}

    result = {"run_name": run_name, "config": config_path, "output_dir": output_dir}

    try:
        proc = subprocess.run(cmd, check=True)
        result["status"] = "done"
        result["ap50"]   = read_best_ap(output_dir)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Run '{run_name}' failed: {e}")
        result["status"] = "failed"
        result["ap50"]   = 0.0
    except KeyboardInterrupt:
        print(f"\n[INTERRUPTED] Run '{run_name}' interrupted.")
        result["status"] = "interrupted"
        result["ap50"]   = read_best_ap(output_dir)

    return result


# ── Main ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="OFAT Runner untuk Head Detection")
    p.add_argument("--num_gpus", default=1,   type=int,
                   help="Jumlah GPU per eksperimen")
    p.add_argument("--factor",   default=None,
                   choices=["lr", "batch_size", "optimizer", "epochs"],
                   help="Jalankan hanya satu faktor OFAT")
    p.add_argument("--dry_run",  action="store_true",
                   help="Print config saja, tidak training")
    p.add_argument("--resume",   action="store_true",
                   help="Resume training yang belum selesai")
    return p.parse_args()


def main():
    args = parse_args()

    # Generate semua config
    all_runs = get_all_runs()

    # Filter faktor kalau diminta
    if args.factor:
        all_runs = [
            (name, path) for name, path in all_runs
            if "baseline" in name or args.factor in name
        ]

    print(f"\n[OFAT] Total eksperimen: {len(all_runs)}")
    for i, (name, path) in enumerate(all_runs):
        print(f"  [{i+1:2d}] {name}")

    if args.dry_run:
        print("\n[OFAT] Dry run — tidak ada training.")
        return

    # Jalankan semua
    results = []
    for run_name, config_path in all_runs:
        res = run_one(run_name, config_path, args.num_gpus, args.resume, args.dry_run)
        results.append(res)

    # Simpan hasil
    out_path = ROOT / "outputs" / "ofat_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OFAT] Hasil disimpan di: {out_path}")

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
