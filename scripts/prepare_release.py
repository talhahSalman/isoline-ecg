#!/usr/bin/env python3
"""Copy what the training notebook produced (weights, metrics, example pictures) into this repo.

Run it once in Colab, after the training cells have finished:

    python scripts/prepare_release.py --drive /content/drive/MyDrive/isoline --images /content/isoline_imgs

It never invents numbers: calibration values are read from the metric files the notebook saved,
and the example pictures are drawn at random (fixed seed) from the held-out test split.
"""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
SEED = 42                                  # the seed the training notebook used when it drew the pictures
PHOTO_MIX = [0.40, 0.35, 0.25]             # share of good / fair / bad photos, as in the notebook
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
WEIGHTS = ["isoline_image_net.pt", "isoline_signal_net_layout.pt", "isoline_signal_net.pt", "signal_norm.npz"]
RESULT_FILES = ["signal_metrics.json", "image_metrics.json", "signal_layout_metrics.json", "digitization_metrics.json",
                "reliability_metrics.json", "final_policy_metrics.json", "signal_results.png", "image_results.png",
                "digitization_results.png", "reliability_results.png", "policy_tradeoff.png", "image_gallery.png",
                "image_studio_preview.png"]


def rebuild_picture(row, signals, dest: Path) -> None:
    """Draw one picture again, exactly the way the notebook did (same code, same per-record seed).

    Used when the rendered JPEGs are gone, which happens whenever Colab restarts and wipes /content.
    """
    import cv2
    from isoline.paper import degrade, render_ecg

    i = int(row["idx"])
    rng = np.random.default_rng(SEED * 100003 + i)
    clean = render_ecg(np.asarray(signals[i], np.float32), rng)
    severity = int(rng.choice(3, p=PHOTO_MIX))
    photo, params = degrade(clean, severity, rng)
    image, quality = (clean, 95) if row["kind"] == "clean" else (photo, int(params["jpeg"]))
    cv2.imwrite(str(dest), image, [cv2.IMWRITE_JPEG_QUALITY, quality])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", required=True, help="the isoline folder on Google Drive")
    ap.add_argument("--images", required=True, help="folder with the rendered JPEGs (/content/isoline_imgs)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    drive, images = Path(args.drive), Path(args.images)

    # 1 · weights
    (REPO / "weights").mkdir(exist_ok=True)
    for name in WEIGHTS:
        src = drive / "weights" / name
        if src.exists():
            shutil.copy2(src, REPO / "weights" / name)
            print(f"  weights/{name}  ({src.stat().st_size / 1e6:.1f} MB)")
        elif name != "isoline_signal_net.pt":
            raise SystemExit(f"Missing {src}. Did the training cells finish?")

    # 2 · results (metrics + figures)
    (REPO / "results").mkdir(exist_ok=True)
    for name in RESULT_FILES:
        src = drive / "results" / name
        if src.exists():
            shutil.copy2(src, REPO / "results" / name)

    # 3 · calibration: the few numbers the pipeline needs at prediction time
    image = json.loads((drive / "results" / "image_metrics.json").read_text())
    digit = json.loads((drive / "results" / "digitization_metrics.json").read_text())
    calibration = {"image_temperature": image["temperature"], "image_thresholds": image["thresholds"],
                   "alignment_threshold": digit["tau"]}
    (REPO / "weights" / "calibration.json").write_text(json.dumps(calibration, indent=2))
    print("  weights/calibration.json")

    # 4 · example pictures from the test split
    manifest = list(csv.DictReader((drive / "data" / "image_manifest.csv").open()))
    labels = np.load(drive / "data" / "Y.npy")

    clean = [r for r in manifest if r["kind"] == "clean"]
    stats = {"records": len(clean), "images": len(manifest),
             "per_split": {s: sum(r["split"] == s for r in clean) for s in ("train", "val", "test")},
             "class_counts": {c: int(labels[:, k].sum()) for k, c in enumerate(CLASSES)},
             "photo_mix": {q: sum(r["kind"] == "photo" and r["quality"] == q for r in manifest) for q in ("good", "fair", "bad")}}
    (REPO / "results" / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"  results/dataset_stats.json  ({stats['records']} records, {stats['images']} pictures)")
    rng = np.random.default_rng(args.seed)
    test = [r for r in manifest if r["split"] == "test"]
    chosen = []
    for k, cls in enumerate(CLASSES):                      # one clean print per class (single-label records)
        pool = [r for r in test if r["kind"] == "clean" and labels[int(r["idx"])].sum() == 1 and labels[int(r["idx"]), k] == 1]
        chosen.append(pool[int(rng.integers(len(pool)))])
    for quality, count in [("good", 2), ("fair", 2), ("bad", 2)]:   # phone photos at each quality level
        pool = [r for r in test if r["kind"] == "photo" and r["quality"] == quality]
        for i in rng.choice(len(pool), count, replace=False):
            chosen.append(pool[int(i)])

    signals = None
    (REPO / "samples").mkdir(exist_ok=True)
    for old in (REPO / "samples").glob("ecg_*.jpg"):
        old.unlink()
    with (REPO / "samples" / "labels.csv").open("w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["file", "labels", "kind"])
        for n, r in enumerate(chosen, 1):
            name = f"ecg_{n:02d}.jpg"
            if (images / r["file"]).exists():
                shutil.copy2(images / r["file"], REPO / "samples" / name)
            else:
                if signals is None:
                    print("  rendered pictures not found, drawing the examples again from the saved signals")
                    signals = np.load(drive / "data" / "X.npy", mmap_mode="r")
                rebuild_picture(r, signals, REPO / "samples" / name)
            truth = "|".join(c for c, v in zip(CLASSES, labels[int(r["idx"])]) if v)
            kind = "clean print" if r["kind"] == "clean" else f"{r['quality']} photo"
            out.writerow([name, truth, kind])
    print(f"  samples/  {len(chosen)} pictures + labels.csv (random draw from the test split, seed {args.seed})")
    print("Done. Next: python evaluate.py")


if __name__ == "__main__":
    main()
