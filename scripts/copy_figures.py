"""
Copy generated figures from result/ folders into manuscript/figures/.
Run this after re-running any plotting script to sync themed figures.
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).parents[1]
DEST = ROOT / "manuscript" / "figures"
DEST.mkdir(parents=True, exist_ok=True)

COPIES = [
    # (source, dest_filename)
    (ROOT / "result" / "c1_contrastive" / "tsne_provider.png",  "tsne_provider.png"),
    (ROOT / "result" / "c1_contrastive" / "tsne_quality.png",   "tsne_quality.png"),
    (ROOT / "result" / "c2_validation"  / "cp_barchart.png",    "cp_barchart.png"),
    (ROOT / "result" / "analysis"       / "ablation.png",       "ablation.png"),
    (ROOT / "result" / "analysis"       / "ks_comparison.png",  "ks_comparison.png"),
    (ROOT / "result" / "c3_affinity"    / "affinity_heatmap.png","affinity_heatmap.png"),
]

for src, fname in COPIES:
    dst = DEST / fname
    if src.exists():
        shutil.copy2(src, dst)
        print(f"  copied  {src.name}  ->  manuscript/figures/{fname}")
    else:
        print(f"  MISSING {src}  (re-run the relevant script first)")

print(f"\nDone. {DEST}")
