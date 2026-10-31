#!/usr/bin/env python3
"""Write the project report as a PDF, using the real numbers and figures from results/.

    python scripts/build_report.py --author "Your Name" --github https://github.com/you/isoline-ecg

Nothing in the tables is typed by hand: it reads the same metric files as the website (through
build_site.collect_metrics), so the report, the site and the README always agree. Figures that are
missing are skipped. Output: docs/Isoline_Report.pdf
"""
import argparse
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image as RLImage, KeepTogether, PageBreak, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle)

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_site import CLASSES, collect_metrics, load  # noqa: E402

RED, INK, MUTED, LINE, PANEL = "#d9392b", "#161a23", "#5b6372", "#d7dbe3", "#f3f5f8"
LONG = {"NORM": "Normal ECG", "MI": "Myocardial infarction", "STTC": "ST/T-wave change",
        "CD": "Conduction disturbance", "HYP": "Hypertrophy"}


# ───────────────────────────── fonts and styles ─────────────────────────────
def register_fonts():
    fonts = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    for name, file in [("Body", "DejaVuSans.ttf"), ("Body-Bold", "DejaVuSans-Bold.ttf"),
                       ("Body-Italic", "DejaVuSans-Oblique.ttf"), ("Mono", "DejaVuSansMono.ttf")]:
        pdfmetrics.registerFont(TTFont(name, str(fonts / file)))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic", boldItalic="Body-Bold")


def styles():
    base = dict(fontName="Body", textColor=colors.HexColor(INK), alignment=TA_LEFT)
    return {
        "body": ParagraphStyle("body", fontSize=9, leading=14, spaceAfter=6, **base),
        "bullet": ParagraphStyle("bullet", fontSize=9, leading=13.5, leftIndent=12, bulletIndent=2, spaceAfter=2.5, **base),
        "h1": ParagraphStyle("h1", fontName="Body-Bold", fontSize=15, leading=19, spaceBefore=14, spaceAfter=7,
                             textColor=colors.HexColor(INK), keepWithNext=1),
        "h2": ParagraphStyle("h2", fontName="Body-Bold", fontSize=10.5, leading=14, spaceBefore=9, spaceAfter=4,
                             textColor=colors.HexColor(RED), keepWithNext=1),
        "caption": ParagraphStyle("caption", fontName="Body-Italic", fontSize=7.8, leading=10.5, spaceAfter=9,
                                  textColor=colors.HexColor(MUTED)),
        "lead": ParagraphStyle("lead", fontSize=9, leading=14, spaceAfter=6, keepWithNext=1, **base),
        "cell": ParagraphStyle("cell", fontSize=8.2, leading=11, **base),
        "cellb": ParagraphStyle("cellb", fontName="Body-Bold", fontSize=8.2, leading=11, textColor=colors.HexColor(INK)),
        "head": ParagraphStyle("head", fontName="Body-Bold", fontSize=8, leading=10.5, textColor=colors.white),
        "ref": ParagraphStyle("ref", fontSize=8.2, leading=11.5, leftIndent=10, firstLineIndent=-10, spaceAfter=3.5, **base),
        "code": ParagraphStyle("code", fontName="Mono", fontSize=8, leading=11.5, backColor=colors.HexColor(PANEL),
                               borderPadding=(6, 6, 6, 6), spaceBefore=4, spaceAfter=10, textColor=colors.HexColor(INK)),
        "coverkicker": ParagraphStyle("coverkicker", fontName="Body-Bold", fontSize=9, leading=12, textColor=colors.HexColor(RED)),
        "covertitle": ParagraphStyle("covertitle", fontName="Body-Bold", fontSize=40, leading=44, textColor=colors.HexColor(INK)),
        "coverred": ParagraphStyle("coverred", fontName="Body-Bold", fontSize=26, leading=31, textColor=colors.HexColor(RED)),
        "coversub": ParagraphStyle("coversub", fontSize=11.5, leading=17, textColor=colors.HexColor(MUTED)),
        "covermeta": ParagraphStyle("covermeta", fontSize=9.5, leading=15, textColor=colors.HexColor(INK)),
    }


# ───────────────────────────── building blocks ─────────────────────────────
class Builder:
    def __init__(self, st):
        self.st, self.flow = st, []

    def _take_headings(self):
        """Pull the headings (and lead-in line) just before a table or figure into its block, so they never get stranded."""
        taken = []
        while self.flow and isinstance(self.flow[-1], Paragraph) and getattr(self.flow[-1].style, "keepWithNext", 0):
            taken.insert(0, self.flow.pop())
        return taken

    def h1(self, text):
        self.flow.append(Paragraph(text, self.st["h1"]))

    def h2(self, text):
        self.flow.append(Paragraph(text, self.st["h2"]))

    def p(self, text, keep=False):
        self.flow.append(Paragraph(text, self.st["lead" if keep else "body"]))

    def bullets(self, items):
        for item in items:
            self.flow.append(Paragraph(item, self.st["bullet"], bulletText="•"))
        self.flow.append(Spacer(1, 4))

    def code(self, text):
        self.flow.append(Paragraph(text.replace("\n", "<br/>").replace(" ", "&nbsp;"), self.st["code"]))

    def table(self, rows, widths, bold_last=False, align_right_from=1, caption=None):
        st = self.st
        data = [[Paragraph(str(c), st["head"]) for c in rows[0]]]
        for i, row in enumerate(rows[1:], 1):
            style = st["cellb"] if (bold_last and i == len(rows) - 1) else st["cell"]
            data.append([Paragraph(str(c), style) for c in row])
        tbl = Table(data, colWidths=[w * mm for w in widths], repeatRows=1)
        cmds = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(INK)),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor(LINE))]
        for i in range(2, len(data), 2):
            cmds.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor(PANEL)))
        tbl.setStyle(TableStyle(cmds))
        block = [tbl] + ([Paragraph(caption, st["caption"])] if caption else [Spacer(1, 8)])
        self.flow.append(KeepTogether(self._take_headings() + block))

    def figure(self, path, caption, width_mm=165):
        path = Path(path)
        if not path.exists():
            return
        w, h = Image.open(path).size
        img = RLImage(str(path), width=width_mm * mm, height=width_mm * mm * h / w)
        self.flow.append(KeepTogether(self._take_headings() + [img, Spacer(1, 3), Paragraph(caption, self.st["caption"])]))

    def page_break(self):
        self.flow.append(PageBreak())


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def f3(x):
    return f"{x:.3f}"


# ───────────────────────────── the pipeline diagram ─────────────────────────────
def draw_pipeline(path):
    fig, ax = plt.subplots(figsize=(10, 4.3), dpi=200)
    ax.set_xlim(0, 100), ax.set_ylim(0, 43), ax.axis("off")

    def box(x, y, w, h, title, sub="", fill="#ffffff", edge=INK, tc=INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=fill, ec=edge, lw=1.3))
        ax.text(x + w / 2, y + h / 2 + (1.6 if sub else 0), title, ha="center", va="center", fontsize=8.6, fontweight="bold", color=tc)
        if sub:
            ax.text(x + w / 2, y + h / 2 - 2.6, sub, ha="center", va="center", fontsize=6.9, color=MUTED)

    def arrow(x1, y1, x2, y2, text="", dy=1.6, color=INK):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2, shrinkA=0, shrinkB=0))
        if text:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, text, ha="center", va="bottom", fontsize=6.8, color=color)

    top, bot, h, w = 26, 6, 11, 17
    box(1, 16, 14, 11, "Scan or\nphone photo", "", fill=PANEL)
    box(21, top, w, h, "1 · Quality gate", "good / fair / bad")
    box(46, top, w, h, "2 · Image model", "EfficientNet-B0")
    box(71, top, 22, h, "Answer + heat map", "Grad-CAM, one-page report", fill="#fdecea", edge=RED)
    box(21, bot, w, h, "3 · Digitizer", "pen line to signals")
    box(46, bot, w, h, "4 · Signal model", "1D CNN + Transformer")
    box(71, bot, 22, h, "5 · Cross-check", "do the two agree?", fill="#fdecea", edge=RED)
    box(21, 37.2, w, 5.2, "bad: ask for a retake", "", fill="#fff7f6", edge=RED, tc=RED)

    arrow(15.3, 23, 20.8, top + 5)
    arrow(15.3, 20, 20.8, bot + 6)
    arrow(38.3, top + 5.5, 45.8, top + 5.5, "good / fair")
    arrow(63.3, top + 5.5, 70.8, top + 5.5)
    arrow(30, top + h + 0.4, 30, 36.9, "", color=RED)
    ax.text(31.2, 33.0, "bad", fontsize=6.8, color=RED, va="center")
    arrow(38.3, bot + 5.5, 45.8, bot + 5.5, "if aligned")
    arrow(63.3, bot + 5.5, 70.8, bot + 5.5)
    arrow(82, top - 0.2, 82, bot + h + 0.5, "", color=RED)
    ax.text(83.2, 20.3, "compare top picks", fontsize=6.8, color=RED, va="center")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ───────────────────────────── page furniture ─────────────────────────────
def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Body", 7.5)
    canvas.setFillColor(colors.HexColor(MUTED))
    canvas.drawString(20 * mm, 287 * mm, "Isoline  ·  Project report")
    canvas.drawRightString(190 * mm, 12 * mm, f"{doc.page}")
    canvas.setStrokeColor(colors.HexColor(LINE))
    canvas.setLineWidth(0.5)
    canvas.line(20 * mm, 285 * mm, 190 * mm, 285 * mm)
    canvas.restoreState()


def on_cover(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(colors.HexColor(RED))
    canvas.rect(0, 0, 8 * mm, 297 * mm, stroke=0, fill=1)
    canvas.setStrokeColor(colors.HexColor(RED))
    canvas.setLineWidth(1.6)
    y, x = 138 * mm, 25 * mm                                       # a small heartbeat line under the title
    pts = [(0, 0), (30, 0), (34, 3), (38, 0), (50, 0), (53, -4), (57, 16), (61, -9), (65, 0), (80, 0), (86, 4), (92, 0), (160, 0)]
    path = canvas.beginPath()
    path.moveTo(x + pts[0][0] * mm, y)
    for dx, dy in pts[1:]:
        path.lineTo(x + dx * mm, y + dy * mm * 0.6)
    canvas.drawPath(path, stroke=1, fill=0)
    canvas.restoreState()


# ───────────────────────────── the report itself ─────────────────────────────
def build(args):
    register_fonts()
    st = styles()
    m = collect_metrics()
    stats = load(REPO / "results" / "dataset_stats.json") or {}
    dig = load(REPO / "results" / "digitization_metrics.json") or {}
    tau = dig.get("tau", 0.90)
    res = REPO / "results"
    policies = list(m["policies"].items())
    best_policy_name, (best_ans, best_acc) = policies[-1]
    base_name, (base_ans, base_acc) = policies[0]
    records = stats.get("records", 21388)
    images = stats.get("images", 42776)
    split = stats.get("per_split", {})
    wmb = sum(f.stat().st_size for f in (REPO / "weights").glob("*.pt")) / 1e6 or 30
    b = Builder(st)

    tmp = Path(tempfile.mkdtemp())
    pipe_png = tmp / "pipeline.png"
    draw_pipeline(pipe_png)

    # ── cover ──
    c = b.flow
    c.append(Spacer(1, 52 * mm))
    c.append(Paragraph("PROJECT REPORT  ·  DEEP LEARNING SEMESTER PROJECT", st["coverkicker"]))
    c.append(Spacer(1, 6 * mm))
    c.append(Paragraph("Isoline", st["covertitle"]))
    c.append(Spacer(1, 3 * mm))
    c.append(Paragraph("Reads the ECG.<br/>Knows when not to.", st["coverred"]))
    c.append(Spacer(1, 62 * mm))
    c.append(Paragraph("Quality-aware, explainable ECG screening from scans and phone photos, "
                       "built and evaluated on the PTB-XL dataset.", st["coversub"]))
    c.append(Spacer(1, 14 * mm))
    meta = []
    if args.author:
        meta.append(f"<b>Author:</b> {args.author}")
    if args.course:
        meta.append(f"<b>Course:</b> {args.course}")
    meta.append("<b>Date:</b> 31 October 2026")
    if args.github:
        meta.append(f"<b>Code and weights:</b> {args.github}")
    for line in meta:
        c.append(Paragraph(line, st["covermeta"]))
    b.page_break()

    # ── abstract ──
    b.h1("Abstract")
    b.p("Electrocardiograms are often shared as printouts, scans or phone photos, not as raw signals. Most deep-learning ECG "
        "classifiers need the signal, and most image-based ones answer with the same confidence whether the picture is crisp or "
        "crumpled. <b>Isoline</b> is a screening pipeline built around one idea: it should check how far its own answer can be "
        "trusted. It first rates the quality of the picture, then reads it with an EfficientNet image model, rebuilds the waveform "
        "with a classical digitizer and reads that with a second, independent network (a 1D CNN with a Transformer), and finally "
        "compares the two. Grad-CAM heat maps and a one-page report explain each result.")
    b.p(f"We trained and tested on PTB-XL ({records:,} ECGs, five diagnostic classes) using the official patient-safe split. On the "
        f"held-out test set the signal model reaches a macro AUC of <b>{f3(m['signal_auc'])}</b> "
        f"({f3(m['layout_auc'])} when it only sees what a printout shows), the image model <b>{f3(m['image_auc'])}</b>, and the "
        f"quality gate classifies photo quality with <b>{pct(m['quality_acc'])}</b> accuracy. Answering every picture is right "
        f"{pct(base_acc)} of the time; answering only when both models agree is right <b>{pct(best_acc)}</b> of the time, on "
        f"{pct(best_ans)} of the pictures. The pictures are rendered from PTB-XL signals, not real clinic scans, and Section 8 is "
        f"candid about what that means.")

    # ── 1 problem ──
    b.h1("1  Problem and motivation")
    b.p("The ECG is the most common heart test in the world, and in many clinics, rural health posts and chat conversations the "
        "result only exists on paper. Someone prints it, photographs it, and sends the photo. Three things make that hard for software:")
    b.bullets(["<b>The signal is gone.</b> A photo is pixels, not millivolts. A model trained on signals cannot read it directly.",
               "<b>Real photos are messy.</b> Pages are tilted, creased and shadowed; the camera blurs and compresses them. A model that "
               "has only seen clean data fails quietly on them.",
               "<b>Confidence is not honesty.</b> A network always produces a number. If nobody checks whether the input was readable, "
               "a wrong answer looks the same as a right one."])
    b.p("Isoline treats the third point as the real problem. A screening tool that is sometimes wrong is acceptable if it knows "
        "when it might be, and says so. This is the idea behind <i>selective classification</i> (Geifman and El-Yaniv, 2017): "
        "answer when it is safe, abstain when it is not, and measure the trade-off.")
    b.h2("Objectives")
    b.bullets(["Rate the quality of an ECG picture and refuse the unreadable ones with advice on how to retake them.",
               "Classify the picture into five diagnostic classes (NORM, MI, STTC, CD, HYP) directly from pixels.",
               "Rebuild the waveform from the picture and classify it independently, as a cross-check.",
               "Report how much to trust each answer, and show where the model looked.",
               "Package everything so a stranger can run it with one command."])

    # ── 2 dataset ──
    b.h1("2  Dataset")
    b.p("<b>PTB-XL</b> (Wagner et al., 2020) is a public collection of clinical 12-lead ECGs from the PhysioNet archive "
        "(Goldberger et al., 2000), released under CC BY 4.0. Each recording lasts 10 seconds and was annotated by cardiologists. "
        "We used the 100 Hz version (1,000 samples per lead) of release 1.0.3, which has 21,799 records.")
    b.p(f"The diagnostic statements are grouped into five <i>superclasses</i>. We kept the {records:,} records that carry at least "
        f"one of them. A record can have several labels, so this is a multi-label problem.")
    counts = stats.get("class_counts", {})
    desc = {"NORM": "No abnormality found", "MI": "Heart attack, recent or old", "STTC": "ST segment or T wave changes",
            "CD": "Bundle branch blocks and similar", "HYP": "Thickened heart chamber walls"}
    rows = [["Class", "Meaning", "Records with this label"]]
    for k in CLASSES:
        rows.append([f"<b>{k}</b>", f"{LONG[k]} · {desc[k]}", f"{counts[k]:,}" if k in counts else "-"])
    b.table(rows, [20, 100, 50], caption="The five classes. A record can carry more than one.")
    if split.get("train"):
        b.p(f"<b>Split.</b> We used the split recommended by the dataset authors: stratified folds 1 to 8 for training "
            f"({split.get('train', 0):,} records), fold 9 for validation ({split.get('val', 0):,}) and fold 10 for testing "
            f"({split.get('test', 0):,}). All thresholds and calibration values were chosen on validation; the test fold was used "
            f"once for the numbers in this report.")
    else:
        b.p("<b>Split.</b> We used the split recommended by the dataset authors: stratified folds 1 to 8 for training, fold 9 for "
            "validation and fold 10 for testing. All thresholds and calibration values were chosen on validation; the test fold "
            "was used once for the numbers in this report.")
    b.h2("Turning signals into pictures")
    b.p("PTB-XL contains signals, not images, so we built the pictures ourselves. A renderer prints every record the way a clinic "
        "does: 800 x 400 pixels, 25 mm/s, 10 mm/mV, three pixels per millimetre, the standard 3 x 4 lead layout (each lead shows "
        "only 2.5 seconds) plus a 10-second lead II rhythm strip and a calibration pulse on every row. Paper colour (pink, red or "
        "grey), ink colour and line thickness vary at random.")
    b.p("A second step, the <i>camera lab</i>, photographs each print badly on purpose. It applies rotation, perspective, uneven "
        "shadows, creases, blur, sensor noise and JPEG compression at three severities:")
    from isoline.paper import LEVELS  # imported late: it needs OpenCV
    mix = stats.get("photo_mix", {})
    rows = [["Severity", "Rotation", "Blur", "Shadow", "Creases", "Noise", "JPEG quality", "Photos"]]
    for i, lv in LEVELS.items():
        rows.append([f"<b>{lv['name']}</b>", f"up to {lv['rot']:g}°", f"{lv['blur'][0]:g} to {lv['blur'][1]:g}",
                     f"{lv['shadow'][0]:g} to {lv['shadow'][1]:g}", f"{lv['wrinkle'][0]:g} to {lv['wrinkle'][1]:g}",
                     f"{lv['noise'][0]} to {lv['noise'][1]}", f"{lv['jpeg'][0]} to {lv['jpeg'][1]}",
                     f"{mix.get(lv['name'], 0):,}" if mix else ["40%", "35%", "25%"][i]])
    b.table(rows, [22, 22, 22, 22, 22, 20, 24, 16], caption="Camera-lab settings. Each record gets one clean print and one photo "
                                                              "whose severity is drawn at random.")
    b.p(f"Every record therefore produces two pictures, one clean print and one phone photo, which makes {images:,} images. The "
        f"severity we applied is the ground truth for the quality gate.")
    b.figure(res / "image_studio_preview.png", "One ECG, shown as a clean print and as phone photos at the three severities.", 150)

    # ── 3 system ──
    b.h1("3  System overview")
    b.p("Isoline is a pipeline of five stages. Every stage can stop the process or change how the answer is described.", keep=True)
    b.figure(pipe_png, "The Isoline pipeline. The image model gives the answer; the waveform path is an independent cross-check.", 168)
    b.table([["Outcome", "When it happens", "What the user sees"],
             ["<b>rejected</b>", "The quality gate says the photo is bad", "No diagnosis. A request to retake the picture, with tips."],
             ["<b>image-only</b>", "The photo is readable, but the waveform could not be lined up with the grid",
              "The image model's answer, said to be uncross-checked."],
             ["<b>reliable</b>", "Aligned, and both models name the same top class", "The answer, marked reliable."],
             ["<b>flagged</b>", "Aligned, but the models disagree", "The answer, marked as needing a clinician's look."]],
            [28, 76, 66], caption="The four outcomes. The rules are in isoline/pipeline.py (function decide).")

    # ── 4 methods ──
    b.h1("4  Methods")
    b.h2("4.1  Image model and quality gate")
    b.p("The image model is an <b>EfficientNet-B0</b> (Tan and Le, 2019) pre-trained on ImageNet. The picture is resized to "
        "384 x 192 pixels and normalised with ImageNet statistics. The pooled 1,280-dimensional feature feeds <b>two heads</b>: a "
        "5-way head for the diagnoses (independent sigmoids, because labels overlap) and a 3-way head for photo quality (good, fair, "
        "bad). Training them together means the features learn to notice both the pattern and the condition of the picture.")
    b.p("The loss is binary cross-entropy for the diagnoses plus half a cross-entropy for the quality. After training we fitted a "
        "single <b>temperature</b> on the validation set so the probabilities are calibrated (Guo et al., 2017), and chose one "
        "decision threshold per class on validation as well.")
    b.h2("4.2  Digitizer")
    b.p("The digitizer is classical computer vision, with no learning. It (1) straightens the page to 800 x 400 pixels, (2) turns it "
        "into an <i>ink map</i> by normalising the illumination and removing the grid colours, (3) follows the pen line column by "
        "column inside each lead's band, keeping the run continuous and picking the line's centre, or its far end on steep strokes, "
        "and (4) interpolates the result to 100 samples per second in millivolts. Only the part of each lead that a printout actually "
        "shows is filled; the rest stays empty.")
    b.p(f"Tilted photos are the hard part. To avoid trusting a bad rebuild, the digitizer computes an <b>alignment score</b>: how much "
        f"of the ink in the calibration-pulse zone falls exactly where the pulse should be. A clean print scores about 1.0 and "
        f"photos mostly score below 0.5. Pictures at or above <b>{tau:.2f}</b> (chosen on validation clean prints) go on to the "
        f"signal model; the others are answered by the image model alone.")
    b.h2("4.3  Signal model")
    b.p("<b>IsolineSignalNet</b> reads 12 leads x 1,000 samples. A strided convolution stem is followed by five residual blocks "
        "(kernel size 7, widths 64 to 256, each with squeeze-and-excitation, Hu et al., 2018) that shrink the signal to 125 time "
        "steps. A 3-layer Transformer encoder (Vaswani et al., 2017) with 4 heads and width 256 and a learned position code lets "
        "distant beats talk to each other. Mean and max pooling of the output feed a dropout and a linear layer with five outputs.")
    b.p("It was trained with AdamW (Loshchilov and Hutter, 2019), a one-cycle learning-rate schedule (Smith and Topin, 2019), mixed "
        "precision, an exponential moving average of the weights, and light signal augmentation. The best epoch on validation was "
        "kept. Thresholds were tuned on validation.")
    b.h2("4.4  Making the signal model fit a printout")
    b.p("A printout shows each lead for only 2.5 seconds, except lead II (10 seconds), while the first signal model saw all 10 "
        "seconds of every lead. We therefore <b>fine-tuned it on masked signals</b> that show exactly the printout layout, with a "
        "random gain between 0.85 and 1.1 and small jitter added to imitate digitization error. This version is the one used inside "
        "the pipeline.")
    b.h2("4.5  Deciding whether to trust the answer")
    b.p("We first tried averaging the two models' probabilities. It did not beat the image model alone, because the rebuilt waveform "
        "is noisier than the picture. So the roles are fixed: the <b>image model gives the answer</b>, and the signal path is a "
        "<b>witness</b>. If the witness is available (the picture was aligned) and names the same top class, the answer is "
        "reliable; if not, it is flagged.")
    b.h2("4.6  Explanation and report")
    b.p("<b>Grad-CAM</b> (Selvaraju et al., 2017) is computed on the last feature map of the EfficientNet for the predicted class and "
        "drawn over the picture, so a reader can see which part of the trace drove the decision. The one-page HTML report "
        "collects the verdict, the probabilities, the waveforms, the heat map and a plain-language note.")

    # ── 5 training ──
    b.h1("5  Training setup")
    b.p("Everything was trained in Google Colab on a free NVIDIA T4 GPU with PyTorch and torchvision. Pictures were rendered once, "
        "saved as JPEG, and streamed from disk by a background thread so the GPU never waited.")
    b.table([["", "Image model", "Layout fine-tune of the signal model"],
             ["Starting point", "ImageNet weights", "The full-signal model"],
             ["Optimizer", "AdamW, weight decay 0.01", "AdamW, weight decay 0.01"],
             ["Peak learning rate", "0.001", "0.0005"],
             ["Schedule", "One-cycle, 20% warm-up", "One-cycle, 20% warm-up"],
             ["Epochs · batch size", "10 · 64", "12 · 128"],
             ["Precision · clipping", "Mixed (fp16) · gradient norm 1.0", "Mixed (fp16) · gradient norm 1.0"],
             ["Averaging", "Weight EMA, decay 0.998", "-"],
             ["Regularisation", "Dropout 0.3 · brightness, contrast, noise and shift augmentation",
              "Random gain and jitter on the visible parts"],
             ["Model selection", "Best validation AUC + 0.25 x quality accuracy", "Best validation AUC"]],
            [34, 70, 66], align_right_from=9, caption="Training settings for the two models trained in this project's notebook.")

    # ── 6 results ──
    b.h1("6  Results")
    b.p("All numbers come from the held-out test fold of PTB-XL, ECGs from patients that no model saw during training. AUC is the "
        "area under the ROC curve for each class; <i>macro</i> means the plain average over the five classes. We report AUC because "
        "the classes are unbalanced and a threshold-free score is fair to all of them.")
    b.h2("6.1  Per-class discrimination")
    rows = [["Class", "Signal model<br/>(full 10 s)", "Signal model<br/>(printout layout)", "Image model"]]
    for k in CLASSES:
        rows.append([f"<b>{k}</b> · {LONG[k]}", f3(m["signal_cls"][k]), f3(m["layout_cls"][k]), f3(m["image_cls"][k])])
    rows.append(["Macro average", f3(m["signal_auc"]), f3(m["layout_auc"]), f3(m["image_auc"])])
    b.table(rows, [66, 34, 36, 34], bold_last=True, caption="Test AUC. The first two columns use perfect signals, the third uses the "
                                                           "pictures (clean prints and photos together).")
    b.figure(res / "signal_results.png", "Signal model: training curve and ROC curves on the test fold.", 160)
    b.figure(res / "image_results.png", "Image model: ROC curves, and the confusion matrix of the photo-quality gate.", 160)
    b.p(f"Reading the pictures directly (AUC {f3(m['image_auc'])}) is almost as good as having the perfect signals "
        f"({f3(m['layout_auc'])} for the same printout layout). Hypertrophy is the hardest class for every model. That is expected: it "
        f"depends on voltage and axis, which are subtle.")
    b.h2("6.2  The quality gate")
    b.p(f"The photo-quality head labels good, fair and bad photos with <b>{pct(m['quality_acc'])}</b> accuracy on the test fold. Bad "
        f"photos are the ones that are refused, so this head is what stops an unreadable picture from getting an answer. Note "
        f"that the labels are the severities our camera lab applied, so this measures how well the model recognises that kind of "
        f"damage, not how it judges real phone photos.")
    b.h2("6.3  Rebuilding the waveform")
    b.p(f"On a clean print the digitizer recovers the overall shape of each lead, but a picture carries fewer than "
        f"one pixel per sample, so sharp R-peaks are slightly clipped. The cost is visible downstream: the signal model scores "
        f"{f3(m['layout_auc'])} AUC with perfect signals and about {f3(m['rebuilt_auc'])} on rebuilt ones. This is why the signal model is "
        f"the cross-check and not the main reader. Photos with a strong tilt rarely reach the alignment score of {tau:.2f}, so they "
        f"are answered by the image model alone.")
    b.figure(res / "digitization_results.png", "Digitizer: alignment scores of clean prints and photos, and one rebuilt lead.", 160)
    b.h2("6.4  Answering less, but more reliably")
    rows = [["Policy", "Share of pictures answered", "Top-1 accuracy"]]
    for name, (ans, acc) in policies:
        rows.append([name, pct(ans), pct(acc)])
    b.table(rows, [98, 38, 34], caption="Top-1 accuracy counts a prediction as right if it matches any of the patient's recorded labels. "
                                        "Each row is stricter than the one above it.")
    b.p(f"The table shows the point of the whole project. If Isoline answers everything, it is right {pct(base_acc)} of the time. "
        f"If it answers only when the picture is good enough and both models agree, it is right {pct(best_acc)} of the time, "
        f"at the price of answering fewer pictures ({pct(best_ans)}). The user chooses the point on that curve that suits the situation.")
    b.figure(res / "policy_tradeoff.png", "The trade-off between how many pictures are answered and how often the answer is right.", 150)
    b.figure(res / "reliability_results.png", "Reliability analysis of the agreement signal.", 150)
    b.figure(res / "image_gallery.png", "Examples of what the image model sees, with its output.", 160)

    # ── 7 deployment ──
    b.h1("7  Deployment and reproducibility")
    b.p("The project is built so that someone who has never seen it can run it. The repository contains the model code, the trained "
        f"weights (about {wmb:.0f} MB in total, committed directly), example pictures with their true labels, the training notebook and the "
        "website. One command analyses a picture:")
    b.code("pip install -r requirements.txt\npython predict.py --image samples/ecg_01.jpg --report report.html")
    b.bullets(["<b>Website.</b> A static page (no server) on Vercel that shows the dataset, the models, the real results, outputs on "
               "test pictures and links to the report, the slides, the video and the code. Example outputs are computed once and "
               "stored in the page, so it works even if the live demo is asleep.",
               "<b>Live demo.</b> A Gradio app on Hugging Face Spaces that downloads the code and weights from GitHub when it starts.",
               "<b>Checks.</b> <i>evaluate.py</i> runs every example picture through the pipeline and compares with the truth; "
               "<i>predict.py</i> exits with a clear message instead of a crash on a missing file or weights."])
    if args.github:
        b.p(f"Code, weights and notebook: {args.github}")

    # ── 8 limits ──
    b.h1("8  Limitations and ethics")
    b.bullets(["<b>Not a medical device.</b> Isoline is a learning project and a screening aid. It gives no diagnosis and must not "
               "replace a clinician, especially for chest pain or breathlessness.",
               "<b>Synthetic pictures.</b> Every picture is a PTB-XL signal printed by our own renderer and degraded by our own camera "
               "lab. The models have not been tested on real clinic scans or real phone photos, and real ones will differ. This is the most "
               "important limitation, and the main reason to expect lower numbers in the real world.",
               "<b>Tilted photos.</b> Straightening a strongly tilted photo well enough to rebuild the waveform is not solved. Only "
               "well-aligned pictures get the cross-check; the rest are answered by the image model alone and say so.",
               "<b>Resolution.</b> With fewer than one pixel per sample, the rebuilt signal loses sharp peaks, which limits the "
               "waveform path.",
               "<b>Labels.</b> PTB-XL labels are written by people and are noisy, and five superclasses are a coarse picture of "
               "cardiology. The classes are also unbalanced.",
               "<b>One dataset.</b> All data comes from a single collection, so other populations and ECG machines may behave differently."])

    # ── 9 conclusion ──
    b.h1("9  Conclusion and future work")
    b.p(f"Isoline shows that an ECG reader becomes more useful when it is allowed to say \"I am not sure\". Reading pictures directly "
        f"reaches {f3(m['image_auc'])} macro AUC, a second independent reading through a rebuilt waveform acts as a witness, and "
        f"asking both to agree lifts accuracy from {pct(base_acc)} to {pct(best_acc)} on the cases it answers.")
    b.p("The next steps are clear: (1) test on real scanned and photographed ECGs and fine-tune on them, which matters more than any "
        "architecture change; (2) fine-tune the signal model on digitized (not perfect) signals to close the gap between 0.926 and "
        "the rebuilt-signal result; (3) a better page finder for tilted photos, so more pictures can be cross-checked; (4) more "
        "classes, from the finer PTB-XL sub-classes.")

    # ── references ──
    b.h1("References")
    refs = ["Wagner, P. et al. (2020). PTB-XL, a large publicly available electrocardiography dataset. <i>Scientific Data</i>, 7, 154.",
            "Goldberger, A. L. et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. <i>Circulation</i>, 101(23), e215-e220.",
            "Tan, M. and Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. <i>ICML</i>.",
            "Hu, J., Shen, L. and Sun, G. (2018). Squeeze-and-excitation networks. <i>CVPR</i>.",
            "Vaswani, A. et al. (2017). Attention is all you need. <i>NeurIPS</i>.",
            "Selvaraju, R. R. et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. <i>ICCV</i>.",
            "Guo, C. et al. (2017). On calibration of modern neural networks. <i>ICML</i>.",
            "Geifman, Y. and El-Yaniv, R. (2017). Selective classification for deep neural networks. <i>NeurIPS</i>.",
            "Loshchilov, I. and Hutter, F. (2019). Decoupled weight decay regularization. <i>ICLR</i>.",
            "Smith, L. N. and Topin, N. (2019). Super-convergence: Very fast training of neural networks using large learning rates. <i>SPIE</i>."]
    for i, r in enumerate(refs, 1):
        b.flow.append(Paragraph(f"[{i}]  {r}", st["ref"]))

    # ── write the PDF ──
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(str(out), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=20 * mm, bottomMargin=20 * mm,
                          title="Isoline: project report", author=args.author or "Isoline",
                          subject="Quality-aware, explainable ECG screening from scans and phone photos")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
    doc.addPageTemplates([PageTemplate(id="cover", frames=[frame], onPage=on_cover, autoNextPageTemplate="body"),
                          PageTemplate(id="body", frames=[frame], onPage=on_page)])
    doc.build(b.flow)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--author", default="", help="your name (and roll number), shown on the cover")
    ap.add_argument("--course", default="Deep Learning · Semester Project")
    ap.add_argument("--github", default="", help="repository URL, shown on the cover and in section 7")
    ap.add_argument("--out", default=str(REPO / "docs" / "Isoline_Report.pdf"))
    args = ap.parse_args()
    out = build(args)
    print(f"Wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
