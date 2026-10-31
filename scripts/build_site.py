#!/usr/bin/env python3
"""Fill the website template with real numbers, example outputs and your links.

    python scripts/build_site.py --github https://github.com/you/isoline-ecg \
        --space https://you-isoline-ecg.hf.space --youtube https://youtu.be/... --linkedin https://www.linkedin.com/...

Reads results/*.json and samples/predictions.json (made by evaluate.py). If a file is missing it
falls back to the numbers from the training run, so the site always builds. It also writes README.md
from README.template.md (filling the <<TOKENS>>), so you can run it again whenever a link changes. Output: site/index.html (static, no JavaScript needed).
"""
import argparse
import html
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SITE = REPO / "site"
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

# Numbers from the training run, used only if the metric files are not there.
FALLBACK = {
    "signal_auc": 0.929, "layout_auc": 0.9256, "image_auc": 0.921, "quality_acc": 0.991, "rebuilt_auc": 0.814,
    "signal_cls": {"NORM": .951, "MI": .931, "STTC": .938, "CD": .925, "HYP": .902},
    "layout_cls": {"NORM": .947, "MI": .923, "STTC": .930, "CD": .923, "HYP": .904},
    "image_cls": {"NORM": .943, "MI": .914, "STTC": .934, "CD": .910, "HYP": .905},
    "policies": {"A · answer every image": (1.0, .824), "B · answer unless the quality gate rejects": (.876, .830),
                 "C · B, but withhold when the two models disagree": (.598, .849), "D · answer only when both models agree": (.228, .896)},
}
FIGURES = [("signal_results.png", "Signal model: training curve and ROC"), ("image_results.png", "Image model: ROC and quality-gate confusion"),
           ("digitization_results.png", "Digitizer: alignment scores and a rebuilt lead"), ("policy_tradeoff.png", "Answering less, but more reliably"),
           ("image_studio_preview.png", "One ECG: clean print and three phone-photo severities")]


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def pct(x, digits=1):
    return f"{x * 100:.{digits}f}%"


def collect_metrics():
    """Real numbers from results/*.json. Anything missing or shaped differently keeps the training-run value."""
    r = REPO / "results"
    m = dict(FALLBACK)

    def take(name, update):
        data = load(r / name)
        if data is None:
            return
        try:
            update(data)
        except (KeyError, TypeError, ValueError, IndexError):
            print(f"  note: could not read {name}, using the training-run numbers for it")

    def signal(s):
        macro = next((s[k] for k in ("macro_auc", "test_macro_auc", "macro_auc_test", "auc_macro") if k in s), None)
        if macro is None:
            macro = next((v for k, v in s.items() if "macro" in k and "auc" in k and isinstance(v, float)), None)
        if macro is None:
            raise KeyError("macro AUC")
        m["signal_auc"], m["signal_cls"] = float(macro), {c: float(s["per_class_auc"][c]) for c in CLASSES}

    def layout(s):
        m["layout_auc"], m["layout_cls"] = float(s["layout_test_macro_auc"]), {c: float(s["per_class_auc"][c]) for c in CLASSES}

    def image(s):
        m["image_auc"], m["quality_acc"] = float(s["macro_auc_all"]), float(s["quality_accuracy"])
        m["image_cls"] = {c: float(s["per_class_auc"][c]) for c in CLASSES}

    def policy(s):
        m["policies"] = {k: (float(v["answered"]), float(v["top1"])) for k, v in s["policies"].items()}

    take("signal_metrics.json", signal)
    take("signal_layout_metrics.json", layout)
    take("image_metrics.json", image)
    take("final_policy_metrics.json", policy)
    return m


def example_cards(link_reports: bool) -> str:
    preds = load(REPO / "samples" / "predictions.json")
    if not preds:
        return '<p class="note">Example outputs appear here once <code>python evaluate.py</code> has been run.</p>'
    (SITE / "assets" / "samples").mkdir(parents=True, exist_ok=True)
    (SITE / "assets" / "reports").mkdir(parents=True, exist_ok=True)
    cards = []
    for p in preds:
        shutil.copy2(REPO / "samples" / p["file"], SITE / "assets" / "samples" / p["file"])
        report = REPO / "samples" / "reports" / (Path(p["file"]).stem + ".html")
        if report.exists():
            shutil.copy2(report, SITE / "assets" / "reports" / report.name)
        truth = " + ".join(p["truth"])
        if p["outcome"] == 3:
            verdict = "asked for a better photo"
        else:
            verdict = f"top pick <b>{p['top']}</b> " + ("✓ matches" if p["correct"] else "✗ does not match")
        link = f' · <a href="assets/reports/{report.name}">full report</a>' if report.exists() and link_reports else ""
        cards.append(
            f'<figure><img src="assets/samples/{html.escape(p["file"])}" alt="ECG example, {html.escape(p["kind"])}" loading="lazy">'
            f'<figcaption><span class="badge b-{p["outcome_name"]}">{p["outcome_name"]}</span>{html.escape(p["kind"])}<br>'
            f'Recorded: <b>{html.escape(truth)}</b> · {verdict}<br>'
            f'<span class="note">{html.escape(p["message"])}</span>{link}</figcaption></figure>')
    return "\n".join(cards)


def build(args):
    m = collect_metrics()
    per_class = "\n".join(
        f'<tr><td>{c}</td><td class="num">{m["signal_cls"][c]:.3f}</td><td class="num">{m["layout_cls"][c]:.3f}</td><td class="num">{m["image_cls"][c]:.3f}</td></tr>'
        for c in CLASSES)
    policy_rows = "\n".join(
        f'<tr><td>{html.escape(k)}</td><td class="num">{pct(a)}</td><td class="num">{pct(t)}</td></tr>' for k, (a, t) in m["policies"].items())
    base, reliable = list(m["policies"].values())[0][1], list(m["policies"].values())[-1][1]

    figs = []
    (SITE / "assets").mkdir(parents=True, exist_ok=True)
    for name, caption in FIGURES:
        src = REPO / "results" / name
        if src.exists():
            shutil.copy2(src, SITE / "assets" / name)
            figs.append(f'<figure><img src="assets/{name}" alt="{html.escape(caption)}" loading="lazy"><figcaption class="note">{html.escape(caption)}</figcaption></figure>')

    if args.space and args.space != "#":
        demo_lead = "Upload a picture of a 12-lead ECG below, or open the demo in its own tab. The free demo server sleeps when idle and takes up to a minute to wake, so the real outputs further down always work."
        demo = (f'<iframe class="demo" src="{html.escape(args.space)}" title="Isoline live demo" loading="lazy"></iframe>'
                f'<p class="note"><a href="{html.escape(args.space)}">Open the live demo in a new tab</a></p>')
    else:
        demo_lead = "The live demo is being set up. Real outputs on test pictures are below."
        demo = ""

    tokens = {"GITHUB": args.github, "SPACE": args.space, "YOUTUBE": args.youtube, "LINKEDIN": args.linkedin,
              "REPORT": args.report, "SLIDES": args.slides,
              "M_SIGNAL_AUC": f"{m['signal_auc']:.3f}", "M_LAYOUT_AUC": f"{m['layout_auc']:.3f}", "M_IMAGE_AUC": f"{m['image_auc']:.3f}",
              "M_REBUILT_AUC": f"{m['rebuilt_auc']:.3f}", "M_QUALITY": pct(m["quality_acc"]), "M_RELIABLE": pct(reliable), "M_BASE": pct(base),
              "PER_CLASS_ROWS": per_class, "POLICY_ROWS": policy_rows, "FIGURES": "\n".join(figs),
              "DEMO_LEAD": demo_lead, "DEMO": demo, "EXAMPLES": example_cards(link_reports=True)}
    page = (SITE / "index.template.html").read_text(encoding="utf-8")
    for k, v in tokens.items():
        page = page.replace("{{" + k + "}}", v)
    (SITE / "index.html").write_text(page, encoding="utf-8")

    for pdf in (REPO / "docs").glob("*.pdf"):                    # the report and slides are served from the site as well
        shutil.copy2(pdf, SITE / "assets" / pdf.name)

    if args.github != "#":                                       # the live demo reads its repository address from here
        (REPO / "space" / "repo_url.txt").write_text(args.github + "\n")

    text = (REPO / "README.template.md").read_text(encoding="utf-8")
    if args.space == "#":                                        # no live demo: leave that row out instead of linking to nothing
        text = "\n".join(line for line in text.splitlines() if "<<SPACE>>" not in line) + "\n"
    for k in ("GITHUB", "SPACE", "YOUTUBE", "LINKEDIN", "SITE"):
        text = text.replace(f"<<{k}>>", args.site if k == "SITE" else tokens[k])
    (REPO / "README.md").write_text(text, encoding="utf-8")
    print(f"Wrote {SITE / 'index.html'}  ({len(page) // 1024} KB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for name, default in [("github", "#"), ("space", "#"), ("youtube", "#"), ("linkedin", "#"), ("site", "#"),
                          ("report", "assets/Isoline_Report.pdf"), ("slides", "assets/Isoline_Slides.pdf")]:
        ap.add_argument(f"--{name}", default=default)
    build(ap.parse_args())
