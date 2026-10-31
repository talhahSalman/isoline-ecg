#!/usr/bin/env python3
"""Commit everything in this folder and push it to your GitHub repository.

    GITHUB_TOKEN=... python scripts/publish.py --user your-name --repo isoline-ecg --email you@example.com

The token (a GitHub personal access token with permission to write to the repo) is read from the
GITHUB_TOKEN environment variable and is never printed or written into the repository.
Safe to run again: if nothing changed, it just says so.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def git(*cmd, check=True, secret=None):
    shown = " ".join(cmd).replace(secret, "***") if secret else " ".join(cmd)
    proc = subprocess.run(["git", *cmd], cwd=REPO, capture_output=True, text=True)
    out = (proc.stdout + proc.stderr).strip()
    if secret:
        out = out.replace(secret, "***")
    if check and proc.returncode != 0:
        sys.exit(f"git {shown} failed:\n{out}")
    return proc.returncode, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True, help="your GitHub user name")
    ap.add_argument("--repo", default="isoline-ecg")
    ap.add_argument("--email", default="isoline@example.com")
    ap.add_argument("--message", default="Update Isoline")
    ap.add_argument("--remote", help="full remote URL (only for testing; normally built from --user and --repo)")
    args = ap.parse_args()

    token = os.environ.get("GITHUB_TOKEN", "")
    if not args.remote and not token:
        sys.exit("Set the GITHUB_TOKEN environment variable first (see the README, 'Publishing').")
    url = args.remote or f"https://{args.user}:{token}@github.com/{args.user}/{args.repo}.git"

    if not (REPO / ".git").exists():
        git("init", "-b", "main")
    git("config", "user.name", args.user)
    git("config", "user.email", args.email)
    git("add", "-A")
    code, _ = git("diff", "--cached", "--quiet", check=False)
    if code != 0:
        git("commit", "-m", args.message)
        print("Committed.")
    else:
        print("Nothing new to commit.")
    git("remote", "remove", "origin", check=False)
    git("remote", "add", "origin", url, secret=token or None)
    code, out = git("push", "-u", "origin", "main", check=False, secret=token or None)
    git("remote", "set-url", "origin", f"https://github.com/{args.user}/{args.repo}.git", check=False)   # keep the token out of .git/config
    if code != 0:
        sys.exit(f"Push failed:\n{out}\n\nCheck the user name, the repository name and the token. If GitHub says the remote "
                 f"has changes you don't have, clone the repo fresh and run this script from the clone.")
    print(f"Pushed to https://github.com/{args.user}/{args.repo}")


if __name__ == "__main__":
    main()
