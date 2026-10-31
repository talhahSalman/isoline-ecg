#!/usr/bin/env python3
"""Check that everything the submission points at is alive, the way an automatic grader would.

    python scripts/check_links.py --site https://your-project.vercel.app \
        --github https://github.com/you/isoline-ecg --youtube https://youtu.be/XXXX \
        --linkedin https://www.linkedin.com/posts/... --space https://huggingface.co/spaces/you/isoline-ecg --try-clone

It opens the website and every link on it, looks for the key files in the GitHub repository (without
downloading the big ones), asks YouTube whether the video is public or unlisted, and, with --try-clone,
clones the repository into a fresh folder and runs predict.py on an example picture.
Exit code 0 means everything that can be checked automatically passed.
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"}
REPO_FILES = ["README.md", "LICENSE", "requirements.txt", "predict.py", "evaluate.py", "isoline/pipeline.py",
              "weights/isoline_image_net.pt", "weights/isoline_signal_net_layout.pt", "weights/signal_norm.npz",
              "weights/calibration.json", "samples/labels.csv", "samples/ecg_01.jpg", "docs/Isoline_Report.pdf",
              "docs/Isoline_Slides.pdf", "notebooks/isoline_training.ipynb", "site/index.html"]
results = []


def record(ok, label, detail=""):
    results.append(ok)
    print(f"  {'OK  ' if ok else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))


def get(url, ranged=False, limit=2_000_000):
    """Return (status, final_url, body) or (None, url, error text)."""
    headers = dict(UA)
    if ranged:
        headers["Range"] = "bytes=0-0"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as r:
            return r.status, r.geturl(), r.read(limit)
    except urllib.error.HTTPError as e:
        return e.code, url, b""
    except Exception as e:  # noqa: BLE001 - any network problem is a failed check
        return None, url, str(e).encode()


def check_site(site):
    print(f"\nWebsite  {site}")
    status, _, body = get(site)
    if status != 200:
        record(False, "home page opens", f"status {status}")
        return
    record(True, "home page opens")
    html = body.decode("utf-8", "replace")
    links = sorted(set(re.findall(r'(?:href|src)="([^"]+)"', html)))
    placeholders = [m for m in re.findall(r'href="(#)"', html)]
    record(not placeholders, "no empty '#' placeholder links", f"{len(placeholders)} left" if placeholders else "")
    for link in links:
        if link.startswith(("#", "mailto:", "javascript:")) or link == "":
            continue
        url = urllib.parse.urljoin(site.rstrip("/") + "/", link)
        external = urllib.parse.urlparse(url).netloc != urllib.parse.urlparse(site).netloc
        st, _, _ = get(url, ranged=True)
        ok = st in (200, 206) or (external and st in (401, 403, 405, 429, 999))     # big sites refuse robots; that is not our fault
        record(ok, ("external  " if external else "same site ") + link, "" if ok else f"status {st}")
    return html


def check_github(github, html):
    m = re.match(r"https?://github\.com/([^/]+)/([^/#?]+)", github)
    if not m:
        record(False, "GitHub URL looks right", github)
        return
    owner, repo = m.group(1), m.group(2).removesuffix(".git")
    print(f"\nGitHub  {owner}/{repo}")
    st, _, _ = get(f"https://github.com/{owner}/{repo}")
    record(st == 200, "repository page is public", f"status {st}" if st != 200 else "")
    for path in REPO_FILES:
        st, _, _ = get(f"https://raw.githubusercontent.com/{owner}/{repo}/main/{path}", ranged=True)
        record(st in (200, 206), path, f"status {st}" if st not in (200, 206) else "")
    if html is not None:
        record(github.rstrip("/") in html, "website links to the repository")


def check_youtube(url, html):
    print(f"\nYouTube  {url}")
    st, _, body = get("https://www.youtube.com/oembed?format=json&url=" + urllib.parse.quote(url, safe=""))
    if st == 200:
        title = json.loads(body.decode()).get("title", "")
        record(True, "video exists and is public or unlisted", f'"{title}"')
    else:
        record(False, "video exists and is public or unlisted", f"status {st} (private, deleted or a wrong link)")
    if html is not None:
        record(url in html, "website links to the video")


def check_linkedin(url, html):
    print(f"\nLinkedIn  {url}")
    st, _, _ = get(url, ranged=True)
    if st in (200, 206):
        record(True, "post opens without signing in")
    else:
        print(f"  NOTE  LinkedIn blocks automatic checks (status {st}). Open the link in a private window to confirm it is public.")
    if html is not None:
        record(url in html, "website links to the post")


def check_clone(github, keep=False):
    print("\nFresh clone test")
    tmp = Path(tempfile.mkdtemp())
    url = github if github.endswith(".git") else github + ".git"
    proc = subprocess.run(["git", "clone", "--depth", "1", url, str(tmp / "repo")], capture_output=True, text=True)
    record(proc.returncode == 0, "git clone works without a password", proc.stderr.strip()[-120:] if proc.returncode else "")
    if proc.returncode:
        return
    run = subprocess.run([sys.executable, "predict.py", "--image", "samples/ecg_01.jpg", "--json"], cwd=tmp / "repo",
                         capture_output=True, text=True)
    ok = run.returncode == 0
    detail = ""
    if ok:
        try:
            out = json.loads(run.stdout)
            detail = f"outcome: {out.get('outcome_name')}, top class: {out.get('top')}"
        except json.JSONDecodeError:
            ok, detail = False, "output was not JSON"
    else:
        detail = (run.stderr.strip().splitlines() or ["no message"])[-1][:150]
    record(ok, "python predict.py runs on a sample picture", detail)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--site", help="Vercel URL")
    ap.add_argument("--github", help="repository URL")
    ap.add_argument("--youtube")
    ap.add_argument("--linkedin")
    ap.add_argument("--space", help="Hugging Face Space URL (optional)")
    ap.add_argument("--try-clone", action="store_true", help="clone the repo and run predict.py (needs the project requirements installed)")
    args = ap.parse_args()
    html = check_site(args.site) if args.site else None
    if args.github:
        check_github(args.github, html)
    if args.youtube:
        check_youtube(args.youtube, html)
    if args.linkedin:
        check_linkedin(args.linkedin, html)
    if args.space:
        print(f"\nHugging Face Space  {args.space}")
        st, _, _ = get(args.space)
        record(st == 200, "Space page opens", f"status {st}" if st != 200 else "")
    if args.try_clone and args.github:
        check_clone(args.github)
    failed = results.count(False)
    print(f"\n{len(results) - failed} of {len(results)} checks passed." + ("" if not failed else f"  {failed} need fixing."))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
