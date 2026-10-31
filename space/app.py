"""Isoline live demo for Hugging Face Spaces.

The Space only holds this small file. On start-up it downloads the Isoline code and the trained
weights from the GitHub repository, so GitHub stays the single source of truth.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_URL = os.environ.get("ISOLINE_REPO") or (HERE / "repo_url.txt").read_text().strip()
CODE = HERE / "isoline-ecg"
if not CODE.exists():
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(CODE)], check=True)
sys.path.insert(0, str(CODE))

import cv2          # noqa: E402
import gradio as gr  # noqa: E402

from isoline.pipeline import Isoline  # noqa: E402
from isoline.report import build_report  # noqa: E402

engine = Isoline(CODE / "weights")
VERDICT = {
    "reliable": "✅ Reliable",
    "flagged": "⚠️ Uncertain: the two checks disagree",
    "image-only": "ℹ️ Image check only",
    "rejected": "🔁 Please retake the photo",
}


def run(picture):
    if picture is None:
        return "Upload a picture of a 12-lead ECG first.", None, None, None
    result = engine.analyze(cv2.cvtColor(picture, cv2.COLOR_RGB2BGR))
    q = result["quality_probs"]
    text = (f"### {VERDICT[result['outcome_name']]}\n\n{result['message']}\n\n"
            f"**Photo quality:** {result['quality']} (good {q['good']:.0%}, fair {q['fair']:.0%}, bad {q['bad']:.0%})  \n"
            f"**Alignment with the ECG template:** {result['alignment']:.2f}\n\n"
            f"**Next step:** {result['advice']}\n\n_{result['disclaimer']}_")
    heat = None if result["heatmap"] is None else cv2.cvtColor(result["heatmap"], cv2.COLOR_BGR2RGB)
    scores = None if result["outcome"] == 3 else result["image_probs"]
    report_path = Path(tempfile.mkdtemp()) / "isoline_report.html"
    report_path.write_text(build_report(result), encoding="utf-8")
    return text, heat, scores, str(report_path)


with gr.Blocks(title="Isoline · Reads the ECG. Knows when not to.") as demo:
    gr.Markdown("# Isoline\n**Reads the ECG. Knows when not to.**  \n"
                "Upload a scan or photo of a 12-lead ECG. This is a learning project, not a medical device.")
    with gr.Row():
        with gr.Column():
            picture = gr.Image(type="numpy", label="ECG picture")
            go = gr.Button("Analyse", variant="primary")
            samples = sorted(str(p) for p in (CODE / "samples").glob("ecg_*.jpg"))
            if samples:
                gr.Examples(examples=samples, inputs=picture, label="Example pictures (random draw from the test split)")
        with gr.Column():
            summary = gr.Markdown()
            scores = gr.Label(num_top_classes=5, label="Image model scores")
            heat = gr.Image(label="Where the image model looked")
            report = gr.File(label="One-page report (open in a browser)")
    go.click(run, inputs=picture, outputs=[summary, heat, scores, report])

if __name__ == "__main__":
    demo.launch()
