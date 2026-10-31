"""Digitizer: turns a picture of a 12-lead ECG printout back into millivolt signals.

It follows the pen line column by column inside each lead's window, then reports an alignment
score: how well the calibration pulses sit where they should. A score near 1 means the page is
lined up with the template and the rebuilt signal can be trusted; low scores mean it can't.
"""
import cv2
import numpy as np

from .paper import H, LAYOUT, MV, PXS, W, X0, Y0, ROW_H

BAND, NS = 58, 250                                        # search half-height (px) · samples per 2.5 s window
ROW_YC   = [int(Y0 + (r + .5) * ROW_H) for r in range(4)]  # baselines of the 3 lead rows + the rhythm strip

def rectify(img):
    """Bring the picture to the canonical 800x400 landscape page. (Straightening tilted photos is not solved;
    the alignment score below tells us when a photo is too far off to trust.)"""
    if img.shape[0] > img.shape[1]:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return (cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA) if img.shape[:2] != (H, W) else img), 1.0

def ink_map(img):
    """1 where the pen ink is, 0 on paper/grid. Illumination-normalised so shadows do not matter."""
    mx = img.max(2).astype(np.float32)                         # grid lines are bright in at least one channel, ink is dark in all
    bg = cv2.GaussianBlur(cv2.dilate(mx, np.ones((11, 11), np.uint8)), (0, 0), 12)
    n  = mx / np.maximum(bg, 1.0)
    return (n < 0.62).astype(np.uint8)

def trace_window(ink, xs, yc, nsamp):
    """Follow the waveform column by column. Returns y (pixels) per sample and the fraction of columns found."""
    ncol = int(round(nsamp * PXS)); ys = np.full(ncol, np.nan); prev = float(yc)
    lo = max(0, int(yc - BAND)); hi = min(ink.shape[0], int(yc + BAND))
    for j in range(ncol):
        x = int(round(xs)) + j
        if x >= ink.shape[1]: break
        idx = np.flatnonzero(ink[lo:hi, x])
        if idx.size == 0: continue
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 2) + 1)
        def dist(r):
            a, b = r[0] + lo, r[-1] + lo
            return 0 if a <= prev <= b else min(abs(a - prev), abs(b - prev))
        r = min(runs, key=dist); top, bot = r[0] + lo, r[-1] + lo
        y = (top + bot) / 2 if bot - top <= 3 else (top if abs(top - prev) > abs(bot - prev) else bot)   # thin line → centre · steep stroke → far end (keeps peaks)
        ys[j] = y; prev = y
    ok = ~np.isnan(ys)
    if ok.sum() < 0.3 * ncol: return None, float(ok.mean())
    col = np.interp(np.arange(ncol), np.flatnonzero(ok), ys[ok])
    return np.interp(np.arange(nsamp) * PXS, np.arange(ncol), col), float(ok.mean())   # sample k sits at column k * PXS

def _cal_masks():
    m, z = np.zeros((H, W), np.uint8), np.zeros((H, W), np.uint8)
    for r in range(4):
        yc = ROW_YC[r]
        cv2.polylines(m, [np.array([[6, yc], [11, yc], [11, yc - MV], [26, yc - MV], [26, yc], [31, yc]], np.int32)], False, 1, 5)
        z[yc - MV - 8: yc + 8, 0:38] = 1
    return m.astype(bool), z.astype(bool)
CAL_MASK, CAL_ZONE = _cal_masks()

def digitize(photo):
    """photo (BGR) -> dict(sig=(1000,12) mV, NaN where the printout shows nothing; conf; align; page)."""
    page, reg = rectify(photo)
    ink = ink_map(page)
    align = float(ink[CAL_MASK].sum() / (ink[CAL_ZONE].sum() + 1e-6))                        # ~1 if the page sits exactly on the template, ~0 if shifted/tilted/blurred
    sig = np.full((1000, 12), np.nan, np.float32); covs = []
    for r in range(3):
        for c in range(4):
            li = LAYOUT[r][c]
            y, cov = trace_window(ink, X0 + c * NS * PXS, ROW_YC[r], NS); covs.append(cov)
            if y is not None: sig[c * NS:(c + 1) * NS, li] = (ROW_YC[r] - y) / MV
    y, cov = trace_window(ink, X0, ROW_YC[3], 1000); covs.append(cov)       # 10 s rhythm strip = full lead II
    if y is not None: sig[:, 1] = (ROW_YC[3] - y) / MV
    return dict(sig=sig, conf=float(np.mean(covs)), align=align, page=page)

def layout_mask(sig):
    """What a printed 12-lead ECG actually shows: each lead only inside its 2.5 s window, lead II for the full 10 s."""
    out = np.full_like(sig, np.nan)
    for r in range(3):
        for c in range(4):
            li = LAYOUT[r][c]; out[c * NS:(c + 1) * NS, li] = sig[c * NS:(c + 1) * NS, li]
    out[:, 1] = sig[:, 1]
    return out
