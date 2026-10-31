#!/usr/bin/env python3
"""Build the presentation: a PowerPoint file (for recording the video) and a PDF (for the website).

    python scripts/build_slides.py --author "Your Name" --github https://github.com/you/isoline-ecg

Both files are drawn from the same slide list, with the real numbers from results/ (the same source as
the report and the website). Each slide carries its voice-over text as speaker notes, and the whole
script is also written to docs/Isoline_Video_Script.md so you can read it while you record.
Outputs: docs/Isoline_Slides.pptx, docs/Isoline_Slides.pdf, docs/Isoline_Video_Script.md
"""
import argparse
import sys
import tempfile
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.platypus import Paragraph

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_report import draw_pipeline  # noqa: E402
from build_site import collect_metrics, load  # noqa: E402

SW, SH = 13.333, 7.5                      # slide size in inches
MX = 0.75                                 # side margin
BG, TEXT, MUTED, RED, CARD, LINE = "0B0D12", "E9ECF2", "9AA3B2", "E84A3C", "141821", "2A3140"


def rgb(hex_):
    return RGBColor.from_string(hex_)


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


# ───────────────────────────── the slides ─────────────────────────────
def make_slides(m, stats, tmp, author, github):
    res = REPO / "results"
    policies = list(m["policies"].items())
    base_ans, base_acc = policies[0][1]
    best_ans, best_acc = policies[-1][1]
    records = stats.get("records", 21388)
    images = stats.get("images", 42776)
    tau = (load(res / "digitization_metrics.json") or {}).get("tau", 0.90)
    pipe = tmp / "pipeline.png"
    draw_pipeline(pipe)

    def img(name):
        p = res / name
        return str(p) if p.exists() else None

    S = []
    S.append(dict(kind="title", title="Isoline", subtitle="Reads the ECG. Knows when not to.",
                  small=f"Quality-aware, explainable ECG screening from scans and phone photos\n"
                        f"{author + '  ·  ' if author else ''}Deep Learning Semester Project",
                  notes=f"Hello{', I am ' + author if author else ''}. This is Isoline, my deep learning semester project. "
                        f"The tagline is the whole idea: it reads the ECG, and it knows when not to."))
    S.append(dict(kind="bullets", title="The problem",
                  bullets=["ECGs often travel as printouts and phone photos, not as clean signals",
                           "Real photos are tilted, creased, shadowed and blurry",
                           "A network always gives an answer, sure or not",
                           "A wrong answer that looks confident is the dangerous case"],
                  notes="ECGs often travel as printouts and phone photos. Most deep learning models need the clean signal, and image "
                        "models usually answer with the same confidence whether the photo is crisp or crumpled. A wrong answer that looks "
                        "sure of itself is the dangerous case. So my question was not only what does this ECG show, but can I trust this "
                        "picture, and can I trust my own answer."))
    S.append(dict(kind="bullets", title="The idea: five steps",
                  bullets=["1  Check the picture: good, fair or bad",
                           "2  Read the image with an EfficientNet",
                           "3  Rebuild the waveform with a digitizer",
                           "4  Cross-check with a signal model (CNN + Transformer)",
                           "5  Explain with a heat map and a one-page report"],
                  notes="Isoline works in five steps. First it checks the picture and refuses the bad ones. Then an image model reads "
                        "the picture directly. A digitizer rebuilds the waveform, and a second, independent network reads that. If the two "
                        "agree, the answer is marked reliable. Finally a heat map and a one-page report explain the result."))
    S.append(dict(kind="stats", title="Dataset: PTB-XL",
                  stats=[(f"{records:,}", "ECG records with a diagnosis"), ("5", "classes: NORM, MI, STTC, CD, HYP"),
                         ("12", "leads, 10 seconds, 100 Hz"), (f"{images:,}", "pictures made from them")],
                  bullets=["Public, cardiologist-annotated, CC BY 4.0 (PhysioNet)",
                           "Official split: folds 1 to 8 train, 9 validation, 10 test"],
                  notes=f"I used PTB-XL, a public collection of twelve-lead ECGs annotated by cardiologists. I kept the {records:,} records "
                        f"that have a diagnosis, grouped into five classes: normal, heart attack, ST T change, conduction disturbance and "
                        f"hypertrophy. I used the official split, so test patients were never seen in training."))
    pic = img("image_studio_preview.png")
    S.append(dict(kind="image" if pic else "bullets", title="Making the pictures", image=pic,
                  caption="One ECG: a clean print, then phone photos at three severities",
                  bullets=["A renderer prints each signal like a clinic does",
                           "A camera lab ruins it on purpose: tilt, shadows, creases, blur, noise, JPEG",
                           "Every record gives one clean print and one phone photo"],
                  notes="PTB-XL has signals, not pictures, so I made the pictures. A renderer prints each record the way a clinic does, with "
                        "grid, layout and calibration pulses. A camera lab then photographs it badly on purpose, with tilt, shadows, creases, "
                        f"blur, noise and compression, at three severities. That gives {images:,} pictures."))
    S.append(dict(kind="image", title="The pipeline", image=str(pipe),
                  caption="The image model gives the answer; the waveform path is an independent cross-check",
                  notes="Here is the full pipeline. The quality gate can stop a bad photo right away and ask for a retake. Otherwise the "
                        "image model reads it. In parallel, the digitizer rebuilds the waveform, and if it lined up with the grid, the signal "
                        "model reads that. The cross-check compares the two answers and decides how far to trust the result."))
    S.append(dict(kind="two", title="The two models",
                  left=("Image model", ["EfficientNet-B0, pre-trained on ImageNet", "Two heads: 5 diagnoses + photo quality",
                                        "Temperature-calibrated, thresholds tuned on validation", "Grad-CAM heat maps"]),
                  right=("Signal model", ["1D residual CNN with squeeze-excitation", "3-layer Transformer on top",
                                          "Fine-tuned on what a printout really shows (2.5 s per lead)",
                                          "Used as the cross-check"]),
                  notes="The image model is an EfficientNet-B0 with two heads, one for the five diagnoses and one for photo quality. The signal "
                        "model is a one dimensional residual network with squeeze and excitation blocks, followed by a three layer Transformer. "
                        "I fine-tuned it on masked signals that show only what a printout actually contains."))
    pic = img("digitization_results.png")
    S.append(dict(kind="image" if pic else "bullets", title="Rebuilding the waveform", image=pic,
                  caption="Alignment scores of clean prints and photos, and one rebuilt lead",
                  bullets=["Classical computer vision: ink map, then trace each lead",
                           f"Alignment score: does the calibration pulse land where it should? (threshold {tau:.2f})",
                           "Tilted photos rarely align, so they are answered by the image model alone"],
                  notes="The digitizer is classical computer vision. It turns the page into an ink map, traces each lead and converts it to "
                        f"millivolts. To avoid trusting a bad rebuild it computes an alignment score from the calibration pulse. Clean prints "
                        f"score about one. Photos mostly score lower, and below {tau:.2f} the image model answers alone."))
    S.append(dict(kind="stats", title="Results on the held-out test set",
                  stats=[(f"{m['signal_auc']:.3f}", "macro AUC, signal model"), (f"{m['image_auc']:.3f}", "macro AUC, image model"),
                         (pct(m["quality_acc"]), "photo-quality gate accuracy"), (f"{m['layout_auc']:.3f}", "signal model, printout layout")],
                  bullets=["AUC averaged over five classes; hypertrophy is the hardest",
                           f"Rebuilding the signal from a picture costs accuracy (about {m['rebuilt_auc']:.3f} AUC), so the image model answers"],
                  notes=f"On the held-out test set the signal model reaches a macro AUC of {m['signal_auc']:.3f}, and the image model "
                        f"{m['image_auc']:.3f}, reading only pixels. The quality gate classifies photo quality with {pct(m['quality_acc'])} "
                        f"accuracy. Rebuilding the signal from a picture loses some information, which is why the image model gives the answer "
                        f"and the signal model is the cross-check."))
    rows = [["When Isoline answers", "Answered", "Top-1 accuracy"]]
    for name, (a, c) in policies:
        rows.append([name, pct(a), pct(c)])
    S.append(dict(kind="table", title="Answering less, but more reliably", rows=rows, widths=[7.4, 2.2, 2.3],
                  footnote="Top-1 counts a prediction as right if it matches any recorded label of the patient.",
                  notes=f"This is the point of the project. If Isoline answers every picture, it is right {pct(base_acc)} of the time. If it "
                        f"answers only when the photo is good enough and both models agree, it is right {pct(best_acc)} of the time, on "
                        f"{pct(best_ans)} of the pictures. The user chooses where to sit on that trade-off."))
    pic = img("image_gallery.png")
    S.append(dict(kind="image" if pic else "bullets", title="Explaining each answer", image=pic,
                  caption="What the image model sees, with its output",
                  bullets=["Grad-CAM shows where the model looked", "A one-page report: verdict, probabilities, waveforms, plain advice",
                           "Four outcomes: reliable, flagged, image-only, rejected"],
                  notes="Every answer comes with a Grad-CAM heat map that shows where the model looked, and a one page report with the verdict, "
                        "the probabilities, the rebuilt waveforms and plain advice. There are four outcomes: reliable, flagged, image only, and "
                        "rejected, when the photo is too poor to read."))
    S.append(dict(kind="bullets", title="Built to be run by anyone",
                  bullets=["GitHub: code, trained weights, notebook, example pictures",
                           "One command: python predict.py --image samples/ecg_01.jpg",
                           "Website on Vercel: dataset, results, real example outputs, report, slides, video",
                           "Live demo on Hugging Face Spaces"],
                  notes="The repository holds the code, the trained weights, the notebook and example pictures, and one command analyses a "
                        "picture. The website on Vercel shows the dataset, the results and real example outputs, and links to the report, "
                        "these slides and the video. There is also a live demo on Hugging Face."))
    S.append(dict(kind="bullets", title="Limits, said plainly",
                  bullets=["Not a medical device: a learning project and a screening aid",
                           "Trained on rendered ECGs, not tested on real clinic photos",
                           "Strongly tilted photos cannot be cross-checked yet",
                           "Fewer than one pixel per sample: sharp peaks are slightly clipped"],
                  notes="A few honest limits. This is not a medical device. All pictures are rendered from PTB-XL signals, so I have not tested "
                        "on real clinic photos, and real ones will be harder. Strongly tilted photos cannot yet be cross-checked, and the "
                        "resolution clips sharp peaks. The next step would be real scans."))
    S.append(dict(kind="title", title="Thank you", subtitle="Reads the ECG. Knows when not to.",
                  small=(github + "\n" if github else "") + "Questions welcome",
                  notes="Thank you for watching. The code, the weights, the report and the live demo are all linked on the project website."))
    return S


# ───────────────────────────── PowerPoint ─────────────────────────────
def fit(path, max_w, max_h):
    w, h = Image.open(path).size
    k = min(max_w / w, max_h / h)
    return w * k, h * k


def pp_text(slide, x, y, w, h, text, size, color=TEXT, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    for i, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = line
        r.font.size, r.font.bold, r.font.name = Pt(size), bold, "Arial"
        r.font.color.rgb = rgb(color)
    return tb


def pp_bullets(slide, x, y, w, h, items, size=22):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(size * 0.55)
        r = p.add_run()
        r.text = item
        r.font.size, r.font.name = Pt(size), "Arial"
        r.font.color.rgb = rgb(TEXT)
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", str(int(Inches(0.38))))
        pPr.set("indent", str(-int(Inches(0.38))))
        clr = pPr.makeelement(qn("a:buClr"), {})
        srgb = clr.makeelement(qn("a:srgbClr"), {"val": RED})
        clr.append(srgb)
        pPr.append(clr)
        pPr.append(pPr.makeelement(qn("a:buChar"), {"char": "•"}))
    return tb


def pp_rect(slide, x, y, w, h, fill, line=None, round_=True):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if round_ else MSO_SHAPE.RECTANGLE,
                                 Inches(x), Inches(y), Inches(w), Inches(h))
    if round_:
        shp.adjustments[0] = 0.06
    shp.fill.solid()
    shp.fill.fore_color.rgb = rgb(fill)
    if line:
        shp.line.color.rgb = rgb(line)
        shp.line.width = Pt(1)
    else:
        shp.line.fill.background()
    shp.shadow.inherit = False
    return shp


def pp_frame(slide, s, n, total):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = rgb(BG)
    if s["kind"] == "title":
        return
    pp_rect(slide, MX, 0.55, 0.7, 0.07, RED, round_=False)
    pp_text(slide, MX, 0.72, SW - 2 * MX, 0.9, s["title"], 34, bold=True)
    pp_text(slide, MX, 7.0, 3, 0.3, "Isoline", 11, MUTED, bold=True)
    pp_text(slide, SW - MX - 2, 7.0, 2, 0.3, f"{n} / {total}", 11, MUTED, align=PP_ALIGN.RIGHT)


def build_pptx(slides, out):
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(SW), Inches(SH)
    total = len(slides)
    for n, s in enumerate(slides, 1):
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        pp_frame(slide, s, n, total)
        k = s["kind"]
        if k == "title":
            pp_rect(slide, 0, 0, 0.35, SH, RED, round_=False)
            pp_text(slide, 1.2, 1.9, 11, 1.4, s["title"], 80, bold=True)
            pp_text(slide, 1.2, 3.4, 11, 1.0, s["subtitle"], 40, RED, bold=True)
            pp_text(slide, 1.2, 5.2, 10.5, 1.2, s["small"], 18, MUTED)
        elif k == "bullets":
            pp_bullets(slide, MX, 2.0, SW - 2 * MX, 4.7, s["bullets"], 28)
        elif k == "stats":
            n_t = len(s["stats"])
            gap = 0.3
            w = (SW - 2 * MX - gap * (n_t - 1)) / n_t
            for i, (val, label) in enumerate(s["stats"]):
                x = MX + i * (w + gap)
                pp_rect(slide, x, 1.9, w, 2.1, CARD, LINE)
                pp_rect(slide, x, 1.9, w, 0.07, RED, round_=False)
                pp_text(slide, x + 0.25, 2.25, w - 0.5, 0.9, val, 44, bold=True)
                pp_text(slide, x + 0.25, 3.2, w - 0.5, 0.7, label, 15, MUTED)
            if s.get("bullets"):
                pp_bullets(slide, MX, 4.5, SW - 2 * MX, 2.2, s["bullets"], 20)
        elif k == "image":
            iw, ih = fit(s["image"], SW - 2 * MX, 4.55)
            pp_rect(slide, (SW - iw) / 2 - 0.1, 1.75 - 0.1, iw + 0.2, ih + 0.2, "FFFFFF", round_=False)
            slide.shapes.add_picture(s["image"], Inches((SW - iw) / 2), Inches(1.75), Inches(iw), Inches(ih))
            pp_text(slide, MX, 6.5, SW - 2 * MX, 0.4, s["caption"], 14, MUTED, align=PP_ALIGN.CENTER)
        elif k == "two":
            gap, w = 0.5, (SW - 2 * MX - 0.5) / 2
            for i, (head, items) in enumerate([s["left"], s["right"]]):
                x = MX + i * (w + gap)
                pp_rect(slide, x, 1.9, w, 4.7, CARD, LINE)
                pp_text(slide, x + 0.35, 2.15, w - 0.7, 0.5, head, 24, RED, bold=True)
                pp_bullets(slide, x + 0.35, 2.9, w - 0.7, 3.5, items, 19)
        elif k == "table":
            rows, widths = s["rows"], s["widths"]
            shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(MX), Inches(1.9), Inches(sum(widths)), Inches(0.62 * len(rows)))
            tbl = shape.table
            for j, wd in enumerate(widths):
                tbl.columns[j].width = Inches(wd)
            for i, row in enumerate(rows):
                tbl.rows[i].height = Inches(0.62)
                for j, val in enumerate(row):
                    cell = tbl.cell(i, j)
                    cell.fill.solid()
                    cell.fill.fore_color.rgb = rgb(RED if i == 0 else (CARD if i % 2 else "0F121A"))
                    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
                    tf = cell.text_frame
                    tf.text = ""
                    r = tf.paragraphs[0].add_run()
                    r.text = val
                    r.font.size, r.font.name = Pt(18), "Arial"
                    r.font.bold = i == 0 or i == len(rows) - 1
                    r.font.color.rgb = rgb("FFFFFF" if i == 0 else TEXT)
                    tf.paragraphs[0].alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.RIGHT
            pp_text(slide, MX, 1.9 + 0.62 * len(rows) + 0.3, SW - 2 * MX, 0.5, s["footnote"], 14, MUTED)
        slide.notes_slide.notes_text_frame.text = s["notes"]
    prs.save(out)


# ───────────────────────────── PDF ─────────────────────────────
def build_pdf(slides, out):
    P = 72.0
    c = pdfcanvas.Canvas(str(out), pagesize=(SW * P, SH * P))
    c.setTitle("Isoline: presentation")
    total = len(slides)
    hexc = lambda h: colors.HexColor("#" + h)  # noqa: E731

    def Y(y, h=0):                      # top-left inches -> bottom-left points
        return (SH - y - h) * P

    def rect(x, y, w, h, fill, line=None, r=0):
        c.setFillColor(hexc(fill))
        if line:
            c.setStrokeColor(hexc(line))
            c.setLineWidth(1)
        if r:
            c.roundRect(x * P, Y(y, h), w * P, h * P, r * P, stroke=1 if line else 0, fill=1)
        else:
            c.rect(x * P, Y(y, h), w * P, h * P, stroke=1 if line else 0, fill=1)

    def text(x, y, w, h, txt, size, color=TEXT, bold=False, align=0, lead=1.22):
        st = ParagraphStyle("t", fontName="Helvetica-Bold" if bold else "Helvetica", fontSize=size, leading=size * lead,
                            textColor=hexc(color), alignment=align)
        para = Paragraph(txt.replace("&", "&amp;").replace("\n", "<br/>"), st)
        pw, ph = para.wrapOn(c, w * P, h * P)
        para.drawOn(c, x * P, Y(y, 0) - ph)

    def bullets(x, y, w, items, size):
        yy = y
        st = ParagraphStyle("b", fontName="Helvetica", fontSize=size, leading=size * 1.22, textColor=hexc(TEXT),
                            leftIndent=0.38 * P, bulletIndent=0, bulletColor=hexc(RED), bulletFontSize=size)
        for it in items:
            para = Paragraph(it.replace("&", "&amp;"), st, bulletText="•")
            pw, ph = para.wrapOn(c, w * P, 5 * P)
            para.drawOn(c, x * P, Y(yy, 0) - ph)
            yy += ph / P + size * 0.55 / 72

    for n, s in enumerate(slides, 1):
        rect(0, 0, SW, SH, BG)
        k = s["kind"]
        if k != "title":
            rect(MX, 0.55, 0.7, 0.07, RED)
            text(MX, 0.72, SW - 2 * MX, 0.9, s["title"], 34, bold=True)
            text(MX, 7.0, 3, 0.3, "Isoline", 11, MUTED, bold=True)
            text(SW - MX - 2, 7.0, 2, 0.3, f"{n} / {total}", 11, MUTED, align=2)
        if k == "title":
            rect(0, 0, 0.35, SH, RED)
            text(1.2, 1.9, 11, 1.4, s["title"], 80, bold=True)
            text(1.2, 3.4, 11, 1.0, s["subtitle"], 40, RED, bold=True)
            text(1.2, 5.2, 10.5, 1.2, s["small"], 18, MUTED)
        elif k == "bullets":
            bullets(MX, 2.0, SW - 2 * MX, s["bullets"], 28)
        elif k == "stats":
            n_t, gap = len(s["stats"]), 0.3
            w = (SW - 2 * MX - gap * (n_t - 1)) / n_t
            for i, (val, label) in enumerate(s["stats"]):
                x = MX + i * (w + gap)
                rect(x, 1.9, w, 2.1, CARD, LINE, r=0.12)
                rect(x, 1.9, w, 0.07, RED)
                text(x + 0.25, 2.25, w - 0.5, 0.9, val, 44, bold=True)
                text(x + 0.25, 3.2, w - 0.5, 0.7, label, 15, MUTED)
            if s.get("bullets"):
                bullets(MX, 4.5, SW - 2 * MX, s["bullets"], 20)
        elif k == "image":
            iw, ih = fit(s["image"], SW - 2 * MX, 4.55)
            rect((SW - iw) / 2 - 0.1, 1.65, iw + 0.2, ih + 0.2, "FFFFFF")
            c.drawImage(ImageReader(s["image"]), (SW - iw) / 2 * P, Y(1.75, ih), iw * P, ih * P)
            text(MX, 6.5, SW - 2 * MX, 0.4, s["caption"], 14, MUTED, align=1)
        elif k == "two":
            gap, w = 0.5, (SW - 2 * MX - 0.5) / 2
            for i, (head, items) in enumerate([s["left"], s["right"]]):
                x = MX + i * (w + gap)
                rect(x, 1.9, w, 4.7, CARD, LINE, r=0.12)
                text(x + 0.35, 2.15, w - 0.7, 0.5, head, 24, RED, bold=True)
                bullets(x + 0.35, 2.9, w - 0.7, items, 19)
        elif k == "table":
            rows, widths = s["rows"], s["widths"]
            rh, y0 = 0.62, 1.9
            for i, row in enumerate(rows):
                rect(MX, y0 + i * rh, sum(widths), rh, RED if i == 0 else (CARD if i % 2 else "0F121A"))
                x = MX
                for j, val in enumerate(row):
                    bold = i == 0 or i == len(rows) - 1
                    text(x + 0.15, y0 + i * rh + 0.17, widths[j] - 0.3, rh, val, 18, "FFFFFF" if i == 0 else TEXT,
                         bold=bold, align=0 if j == 0 else 2)
                    x += widths[j]
            text(MX, y0 + rh * len(rows) + 0.3, SW - 2 * MX, 0.5, s["footnote"], 14, MUTED)
        c.showPage()
    c.save()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--author", default="")
    ap.add_argument("--github", default="")
    ap.add_argument("--outdir", default=str(REPO / "docs"))
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    m = collect_metrics()
    stats = load(REPO / "results" / "dataset_stats.json") or {}
    slides = make_slides(m, stats, Path(tempfile.mkdtemp()), args.author, args.github)
    build_pptx(slides, out / "Isoline_Slides.pptx")
    build_pdf(slides, out / "Isoline_Slides.pdf")
    words = 0
    lines = ["# Isoline: video voice-over script", "", "Read each part while its slide is on screen. About 5 minutes at a relaxed pace.", ""]
    for n, s in enumerate(slides, 1):
        lines += [f"## Slide {n}: {s['title']}", "", s["notes"], ""]
        words += len(s["notes"].split())
    (out / "Isoline_Video_Script.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(slides)} slides to {out}  (PPTX, PDF) and the voice-over script ({words} words, about {words / 140:.1f} min)")


if __name__ == "__main__":
    main()
