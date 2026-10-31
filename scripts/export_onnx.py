#!/usr/bin/env python3
"""Export the image model to ONNX, so the website can run it inside the visitor's browser.

    pip install onnx onnxruntime
    python scripts/export_onnx.py

Writes weights/isoline_image_net.onnx (about 16 MB). Then it checks the exported file against the
PyTorch model on every example picture, and stops with an error if they disagree.
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from isoline.digitize import rectify  # noqa: E402
from isoline.models import IsolineImageNet  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def prepare(path: Path) -> np.ndarray:
    """The picture as the model sees it (the same steps as Isoline._image_tensor)."""
    page, _ = rectify(cv2.imread(str(path)))
    small = cv2.resize(page, (384, 192), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.ascontiguousarray(((rgb - MEAN) / STD).transpose(2, 0, 1)[None])


def export(model, example, out: Path) -> None:
    """Try the exporters one after another; which one works depends on the installed PyTorch version."""
    attempts = [("classic exporter", dict(dynamo=False)), ("default exporter", dict()), ("dynamo exporter", dict(dynamo=True))]
    last = None
    for name, extra in attempts:
        try:
            torch.onnx.export(model, example, str(out), input_names=["image"], output_names=["diagnosis", "quality"],
                              opset_version=17, **extra)
            print(f"  exported with the {name}")
            return
        except Exception as err:  # noqa: BLE001 - try the next exporter
            last = err
            print(f"  {name} failed: {type(err).__name__}: {str(err)[:160]}")
    raise SystemExit(f"Could not export the model: {last}\nTry: pip install onnxscript")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(REPO / "weights"))
    args = ap.parse_args()
    weights = Path(args.weights)
    out = weights / "isoline_image_net.onnx"

    model = IsolineImageNet(pretrained=False)
    model.load_state_dict(torch.load(weights / "isoline_image_net.pt", map_location="cpu"))
    model.eval()
    example = torch.zeros(1, 3, 192, 384)
    print("Exporting the image model ...")
    export(model, example, out)
    print(f"  {out.name}: {out.stat().st_size / 1e6:.1f} MB")

    import onnxruntime as ort
    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    rows = list(csv.DictReader((REPO / "samples" / "labels.csv").open(encoding="utf-8")))
    worst, same = 0.0, 0
    for row in rows:
        x = prepare(REPO / "samples" / row["file"])
        with torch.no_grad():
            d_ref, q_ref = (t.numpy() for t in model(torch.from_numpy(x)))
        d_out, q_out = session.run(["diagnosis", "quality"], {"image": x})
        worst = max(worst, float(np.abs(d_ref - d_out).max()), float(np.abs(q_ref - q_out).max()))
        same += int(d_ref.argmax() == d_out.argmax() and q_ref.argmax() == q_out.argmax())
    print(f"Checked {len(rows)} example pictures: largest difference {worst:.2e}, same answers on {same} of {len(rows)}.")
    if worst > 1e-3 or same != len(rows):
        print("The exported model does not match the original. Do not publish it.", file=sys.stderr)
        out.unlink()
        return 1
    print("The ONNX model matches the PyTorch model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
