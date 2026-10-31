#!/usr/bin/env python3
"""Run Isoline over the example pictures in samples/ and compare with the known answers.

    python evaluate.py

Needs samples/labels.csv (columns: file, labels, kind). Writes samples/predictions.json,
which the website uses for its example gallery.
"""
import csv
import json
import sys
from pathlib import Path

import cv2

from isoline.pipeline import Isoline, to_plain
from isoline.report import build_report

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "samples"


def main() -> int:
    labels_file = SAMPLES / "labels.csv"
    if not labels_file.exists():
        print("samples/labels.csv not found. See the README for how the samples are set up.", file=sys.stderr)
        return 2
    rows = list(csv.DictReader(labels_file.open(encoding="utf-8")))
    try:
        engine = Isoline()
    except FileNotFoundError as err:
        print(err, file=sys.stderr)
        return 2

    out, answered, correct = [], 0, 0
    (SAMPLES / "reports").mkdir(exist_ok=True)
    print(f"\n{'picture':<22}{'kind':<14}{'true':<12}{'verdict':<12}{'answer':<8}")
    for row in rows:
        image = cv2.imread(str(SAMPLES / row["file"]))
        if image is None:
            print(f"  skipped {row['file']} (unreadable)")
            continue
        res = engine.analyze(image)
        truth = [t for t in row["labels"].split("|") if t]
        ok = res["top"] in truth
        if res["outcome"] != 3:
            answered += 1
            correct += ok
        shown = "-" if res["outcome"] == 3 else f"{res['top']}{'  ✓' if ok else '  ✗'}"
        print(f"{row['file']:<22}{row['kind']:<14}{'|'.join(truth):<12}{res['outcome_name']:<12}{shown:<8}")
        (SAMPLES / "reports" / (Path(row["file"]).stem + ".html")).write_text(build_report(res), encoding="utf-8")
        out.append({"file": row["file"], "kind": row["kind"], "truth": truth, "correct": bool(ok), **to_plain(res)})

    (SAMPLES / "predictions.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n{len(out)} pictures, {answered} answered, {correct} of those correct on the top pick.")
    print("Wrote samples/predictions.json and one report per picture in samples/reports/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
