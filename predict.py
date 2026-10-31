#!/usr/bin/env python3
"""Run Isoline on one ECG picture.

    python predict.py --image samples/ecg_01.jpg
    python predict.py --image samples/ecg_01.jpg --report report.html
    python predict.py --image samples/ecg_01.jpg --json
"""
import argparse
import json
import sys

import cv2

from isoline.pipeline import CLASS_NAMES, Isoline, to_plain
from isoline.report import build_report


def main() -> int:
    ap = argparse.ArgumentParser(description="Isoline: reads the ECG, and knows when not to.")
    ap.add_argument("--image", required=True, help="path to a photo or scan of a 12-lead ECG")
    ap.add_argument("--report", help="also write a one-page HTML report to this path")
    ap.add_argument("--json", action="store_true", help="print the result as JSON instead of text")
    ap.add_argument("--no-heatmap", action="store_true", help="skip the explanation heatmap (a bit faster)")
    ap.add_argument("--weights", help="folder with the model weights (default: ./weights)")
    ap.add_argument("--device", default="cpu", help="cpu (default) or cuda")
    args = ap.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        print(f"Could not read an image from '{args.image}'.", file=sys.stderr)
        return 2
    try:
        engine = Isoline(args.weights, device=args.device)
        result = engine.analyze(image, explain=not args.no_heatmap)
    except FileNotFoundError as err:
        print(err, file=sys.stderr)
        return 2

    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(build_report(result))

    if args.json:
        print(json.dumps(to_plain(result), indent=2))
        return 0

    print("\nIsoline · reads the ECG, and knows when not to\n")
    print(f"  Picture       {args.image}")
    print(f"  Photo quality {result['quality']} (alignment {result['alignment']:.2f})")
    print(f"  Verdict       {result['outcome_name']}")
    print(f"  {result['message']}\n")
    if result["outcome"] != 3:
        print("  Image model:")
        for name, p in result["image_probs"].items():
            mark = "◀" if name in result["found"] else " "
            print(f"    {name:<5} {p * 100:5.1f}%  {mark}  {CLASS_NAMES[name]}")
        if result["signal_probs"]:
            print(f"\n  Rebuilt waveform says: {result['signal_top']} ({CLASS_NAMES[result['signal_top']]})")
    print(f"\n  Next step: {result['advice']}")
    print(f"\n  {result['disclaimer']}")
    if args.report:
        print(f"  Report saved to {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
