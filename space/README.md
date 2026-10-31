---
title: Isoline ECG
emoji: 🫀
colorFrom: red
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
---

# Isoline · live demo

Reads the ECG. Knows when not to. Upload a scan or photo of a 12-lead ECG and Isoline will check
the picture, read it, and tell you how far to trust the answer.

This Space only contains the small `app.py`. The code and trained weights are downloaded from the
GitHub repository when it starts. To point it at your repository, set the Space variable
`ISOLINE_REPO`, or edit `repo_url.txt`.

A learning project. Not a medical device.
