"""
scripts/analyze_ofat.py

Analisis hasil OFAT dari ofat_results.json.

Cara pakai:
  python scripts/analyze_ofat.py --results outputs/ofat_results.json
"""

import json
import argparse


FACTORS = ["lr", "batch_size", "optimizer", "epochs"]

BASELINE = dict(lr=0.001, batch_size=16, optimizer="adamw", epochs=100)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="outputs/ofat_results.json")
    return p.parse_args()


def group_by_factor(results):
    groups = {f: [] for f in FACTORS}
    groups["baseline"] = []
    for r in results:
        name = r["run_name"]
        if name == "baseline":
            groups["baseline"].append(r)
        else:
            for f in FACTORS:
                if f"ofat_{f}_" in name:
                    groups[f].append(r)
                    break
    return groups


def main():
    args = parse_args()
    with open(args.results) as f:
        results = json.load(f)

    print(f"\n[Analyzer] {len(results)} eksperimen dimuat.\n")

    groups  = group_by_factor(results)
    best_per_factor = {}

    # Baseline
    if groups["baseline"]:
        bl = groups["baseline"][0]
        print(f"  Baseline AP50: {bl.get('ap50', 0):.2f}")

    # Per faktor
    for factor in FACTORS:
        group = groups[factor]
        if not group:
            continue

        print(f"\n  Faktor: {factor.upper()}")
        print(f"  {'Nilai':<20} {'AP50':>8}")
        print(f"  {'-'*30}")

        sorted_g = sorted(group, key=lambda r: -r.get("ap50", 0))
        for r in sorted_g:
            # Ekstrak nilai faktor dari run_name
            val = r["run_name"].replace(f"ofat_{factor}_", "").replace("_", ".")
            marker = " ← best" if r is sorted_g[0] else ""
            print(f"  {val:<20} {r.get('ap50', 0):>8.2f}{marker}")

        if sorted_g:
            best_val = sorted_g[0]["run_name"].replace(f"ofat_{factor}_", "").replace("_", ".")
            best_per_factor[factor] = best_val

    # Recommended config
    print("\n" + "="*55)
    print("  REKOMENDASI CONFIG OPTIMAL (per faktor)")
    print("="*55)
    for factor, val in best_per_factor.items():
        print(f"  {factor:<15}: {val}")
    print("="*55)

    # Overall best
    best = max(results, key=lambda r: r.get("ap50", 0))
    print(f"\n  Overall best : {best['run_name']}")
    print(f"  Best AP50    : {best.get('ap50', 0):.2f}\n")


if __name__ == "__main__":
    main()
