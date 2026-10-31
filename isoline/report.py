"""Builds the one-page ECG report as a single, self-contained HTML file (pictures included).
Open it in any browser; use "Print" to save it as a PDF."""
from __future__ import annotations

import base64
import datetime
import html

import cv2
import numpy as np

from .paper import FS, LAYOUT
from .pipeline import CLASS_NAMES, CLASSES, DISCLAIMER

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
BANNER = {
    0: ("#1e8e5a", "Reliable", "The picture and the rebuilt waveform agree."),
    1: ("#d9822b", "Uncertain", "The two checks disagree, so please treat this as a rough hint only."),
    2: ("#2f7bd1", "Image check only", "The waveform could not be rebuilt from this photo, so only the image model was used."),
    3: ("#c0392b", "Please retake the photo", "The image was not clear enough to read safely."),
}


def _jpeg64(img_bgr, width=720) -> str:
    h, w = img_bgr.shape[:2]
    img = cv2.resize(img_bgr, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def _path(values, w, h, lo=-1.5, hi=1.5) -> str:
    """SVG path for one waveform; gaps (NaN) break the line."""
    cmds, pen_down = [], False
    step = w / max(len(values) - 1, 1)
    for i, v in enumerate(values):
        if np.isnan(v):
            pen_down = False
            continue
        y = h - (min(max(v, lo), hi) - lo) / (hi - lo) * h
        cmds.append(f"{'L' if pen_down else 'M'}{i * step:.1f},{y:.1f}")
        pen_down = True
    return " ".join(cmds)


def _waveforms_svg(sig) -> str:
    cells = []
    for r in range(3):
        for c in range(4):
            li = LAYOUT[r][c]
            x0, y0 = c * 200, r * 90
            window = sig[c * 250:(c + 1) * 250, li]
            cells.append(f'<g transform="translate({x0},{y0})"><text x="4" y="12" class="lead">{LEAD_NAMES[li]}</text>'
                         f'<path d="{_path(window, 190, 76)}" class="wave" transform="translate(0,10)"/></g>')
    rhythm = f'<g transform="translate(0,280)"><text x="4" y="12" class="lead">II · 10 s rhythm</text><path d="{_path(sig[:, 1], 800, 70)}" class="wave" transform="translate(0,12)"/></g>'
    return f'<svg viewBox="0 0 800 365" class="waves">{"".join(cells)}{rhythm}</svg>'


def _bars(probs: dict, found: list) -> str:
    rows = []
    for c in CLASSES:
        p = probs[c]
        rows.append(f'<div class="row"><span class="name">{c}<small>{CLASS_NAMES[c]}</small></span>'
                    f'<span class="bar"><i style="width:{p * 100:.0f}%" class="{"hit" if c in found else ""}"></i></span>'
                    f'<span class="pct">{p * 100:.0f}%</span></div>')
    return "".join(rows)


CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:860px;margin:32px auto;padding:0 20px;color:#1b1f27;line-height:1.5}
h1{margin:0;font-size:26px}h1 b{color:#d63a2f}h2{font-size:15px;letter-spacing:.08em;text-transform:uppercase;color:#6b7280;margin:28px 0 8px}
.sub{color:#6b7280;font-size:14px}.banner{border-radius:10px;padding:14px 18px;color:#fff;margin:18px 0}.banner b{font-size:20px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}img{width:100%;border-radius:8px;border:1px solid #e5e7eb}
.card{background:#f6f7f9;border-radius:10px;padding:14px 16px}.row{display:flex;align-items:center;gap:10px;margin:7px 0}
.name{width:200px;font-weight:600}.name small{display:block;font-weight:400;color:#6b7280;font-size:12px}
.bar{flex:1;background:#e5e7eb;border-radius:6px;height:12px;overflow:hidden}.bar i{display:block;height:100%;background:#9aa3af}
.bar i.hit{background:#d63a2f}.pct{width:44px;text-align:right}.waves{width:100%;background:#fff;border:1px solid #e5e7eb;border-radius:8px}
.wave{fill:none;stroke:#1b1f27;stroke-width:1.1}.lead{font-size:11px;fill:#6b7280}.note{font-size:13px;color:#6b7280}
.foot{margin-top:30px;padding-top:14px;border-top:1px solid #e5e7eb;font-size:12px;color:#6b7280}
@media print{body{margin:0}}
"""


def build_report(result: dict, title: str = "ECG screening report") -> str:
    colour, headline, sub = BANNER[result["outcome"]]
    rejected = result["outcome"] == 3
    q = result["quality_probs"]
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>Isoline · {html.escape(title)}</title><style>{CSS}</style></head><body>",
             f"<h1>Iso<b>line</b> · {html.escape(title)}</h1>",
             f"<div class='sub'>Reads the ECG. Knows when not to. · {datetime.datetime.now():%d %b %Y, %H:%M}</div>",
             f"<div class='banner' style='background:{colour}'><b>{headline}</b><br>{sub}</div>",
             f"<p>{html.escape(result['message'])}</p>"]

    parts.append("<h2>The picture</h2><div class='grid'>")
    parts.append(f"<div><img src='{_jpeg64(result['page'])}' alt='uploaded ECG'><div class='note'>What you uploaded</div></div>")
    if result["heatmap"] is not None:
        parts.append(f"<div><img src='{_jpeg64(result['heatmap'])}' alt='heatmap'><div class='note'>Where the image model looked (red = most influence)</div></div>")
    parts.append("</div>")

    parts.append("<h2>Photo quality</h2><div class='card'>"
                 f"<b>{result['quality'].capitalize()}</b> · good {q['good'] * 100:.0f}% · fair {q['fair'] * 100:.0f}% · bad {q['bad'] * 100:.0f}%"
                 f"<div class='note'>Alignment of the page with the ECG template: {result['alignment']:.2f} "
                 f"({'good enough to rebuild the waveform' if result['aligned'] else 'too low to rebuild the waveform'}).</div></div>")

    if not rejected:
        parts.append("<h2>What the image model found</h2><div class='card'>" + _bars(result["image_probs"], result["found"]) + "</div>")
        if result["signal_probs"]:
            parts.append("<h2>What the rebuilt waveform says</h2><div class='card'>" + _bars(result["signal_probs"], [result["signal_top"]]) + "</div>")
        if result["waveforms"] is not None:
            parts.append("<h2>Waveforms rebuilt from the picture</h2>" + _waveforms_svg(result["waveforms"])
                         + f"<div class='note'>Each lead shows only the 2.5 seconds printed on the page. Sample rate {FS} Hz.</div>")
    parts.append(f"<h2>What to do next</h2><div class='card'>{html.escape(result['advice'])}</div>")
    parts.append(f"<div class='foot'>{html.escape(DISCLAIMER)} Trained on PTB-XL (rendered as ECG printouts); "
                 "it has not been validated on real clinic scans.</div></body></html>")
    return "".join(parts)
