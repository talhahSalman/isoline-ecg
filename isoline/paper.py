"""ECG paper: a renderer that prints a 12-lead ECG the way a clinic does, plus a "camera lab" that
photographs it badly (tilt, shadows, creases, blur, noise, JPEG).

This is how Isoline's training images were made, and it also defines the page geometry the
digitizer relies on. Page: 800 x 400 px, 25 mm/s, 10 mm/mV, 3 px per mm, standard 3x4 layout
plus a 10-second lead II rhythm strip.
"""
import cv2
import numpy as np

cv2.setNumThreads(0)

# ---------- page geometry: 25 mm/s · 10 mm/mV · 3 px per mm ----------
FS = 100                                                  # samples per second in PTB-XL (100 Hz version)
PX, W, H, X0, Y0 = 3, 800, 400, 40, 28
ROW_H, MV, PXS = 30 * PX, 10 * PX, 25 * PX / FS          # row height · px per mV · px per sample
LAYOUT = [[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11]]    # standard 3x4 grid (I/II/III · aVR/aVL/aVF · V1-3 · V4-6)
NAMES  = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
FONT   = cv2.FONT_HERSHEY_SIMPLEX

# ---------- three kinds of ECG paper (BGR): paper, minor grid, major grid ----------
PAPERS = {}
for name, (paper, minor, major) in {'pink': ((250, 247, 255), (222, 214, 255), (170, 150, 245)),
                                    'red' : ((245, 245, 252), (200, 200, 250), (120, 120, 230)),
                                    'grey': ((250, 250, 250), (228, 228, 228), (185, 185, 185))}.items():
    pg = np.empty((H, W, 3), np.uint8); pg[:] = paper
    for x in range(X0 % PX, W, PX):
        pg[:, x] = major if ((x - X0) // PX) % 5 == 0 else minor
    for y in range(Y0 % PX, H, PX):
        pg[y, :] = major if ((y - Y0) // PX) % 5 == 0 else minor
    PAPERS[name] = pg
INKS = [(25, 25, 25), (60, 30, 20), (40, 40, 60)]


def render_ecg(sig, rng, paper=None, thick=None):
    """sig: (1000, 12) in mV  ->  800x400 BGR picture of a standard 12-lead printout."""
    paper = paper if paper is not None else str(rng.choice(list(PAPERS), p=[.6, .2, .2]))
    thick = thick if thick is not None else int(rng.choice([1, 2], p=[.6, .4]))
    ink   = INKS[int(rng.integers(len(INKS)))]
    img   = PAPERS[paper].copy()
    sig   = np.clip(sig - np.median(sig, 0), -2.5, 2.5)          # centre baseline, tame huge spikes

    def trace(seg, x_start, yc):
        xs  = x_start + np.arange(len(seg)) * PXS
        pts = np.stack([xs, yc - seg * MV], 1).round().astype(np.int32)
        cv2.polylines(img, [pts], False, ink, thick, cv2.LINE_AA)

    for r in range(3):                                            # 3 rows x 4 columns of 2.5 s
        yc = int(Y0 + (r + .5) * ROW_H)
        for c in range(4):
            li, xs = LAYOUT[r][c], X0 + c * 250 * PXS
            trace(sig[c * 250:(c + 1) * 250, li], xs, yc)
            cv2.putText(img, NAMES[li], (int(xs) + 4, int(yc - ROW_H / 2 + 11)), FONT, .38, ink, 1, cv2.LINE_AA)
            if c: cv2.line(img, (int(xs), yc - 6), (int(xs), yc + 6), ink, 1, cv2.LINE_AA)
    yc = int(Y0 + 3.5 * ROW_H)                                    # 10-second rhythm strip, lead II
    trace(sig[:, 1], X0, yc)
    cv2.putText(img, 'II', (X0 + 4, int(yc - ROW_H / 2 + 11)), FONT, .38, ink, 1, cv2.LINE_AA)
    for r in range(4):                                            # 1 mV calibration pulse on every row
        yc = int(Y0 + (r + .5) * ROW_H)
        cal = np.array([[6, yc], [11, yc], [11, yc - MV], [26, yc - MV], [26, yc], [31, yc]], np.int32)
        cv2.polylines(img, [cal], False, ink, thick, cv2.LINE_AA)
    cv2.putText(img, '25 mm/s    10 mm/mV    100 Hz', (X0, 16), FONT, .4, ink, 1, cv2.LINE_AA)
    return img


# ---------- the camera lab: three levels of "real-world" abuse ----------
LEVELS = {
    0: dict(name='good', rot=1.0,  persp=.008, blur=(0., .6),  shadow=(0., .12),  wrinkle=(0., .04),   noise=(0, 3),  jpeg=(85, 96), shift=.01, bright=(.92, 1.08)),
    1: dict(name='fair', rot=4.0,  persp=.03,  blur=(.9, 1.6), shadow=(.15, .4),  wrinkle=(.06, .16),  noise=(3, 8),  jpeg=(55, 80), shift=.04, bright=(.78, 1.18)),
    2: dict(name='bad',  rot=11.0, persp=.07,  blur=(2., 3.6), shadow=(.4, .7),   wrinkle=(.16, .32),  noise=(8, 18), jpeg=(25, 50), shift=.10, bright=(.55, 1.3)),
}

def smooth_field(rng, h, w, gh=4, gw=6):
    return cv2.resize(rng.normal(size=(gh, gw)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)

def degrade(img, sev, rng):
    """Turn a clean render into a 'phone photo'. Returns (uint8 image, parameter dict)."""
    L, p = LEVELS[sev], {}
    f = img.astype(np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    # 1 · crumpled paper: smooth geometric ripples
    wr = p['wrinkle'] = rng.uniform(*L['wrinkle'])
    if wr > .03:
        amp = wr * 14
        f = cv2.remap(f, xx + amp * smooth_field(rng, H, W, 5, 8), yy + amp * smooth_field(rng, H, W, 5, 8),
                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # 2 · camera pose: rotation + perspective + shift, page lying on a table
    ang = p['rot'] = rng.uniform(-L['rot'], L['rot'])
    sc  = rng.uniform(.9, .98) if sev else rng.uniform(.96, 1.0)
    c, a = np.array([W / 2, H / 2]), np.deg2rad(ang)
    R   = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    src = np.array([[0, 0], [W, 0], [W, H], [0, H]], np.float32)
    dst = (src - c) @ R.T * sc + c
    dst = dst + rng.normal(0, L['persp'], (4, 2)) * np.array([W, H])
    dst = dst + rng.uniform(-L['shift'], L['shift'], 2) * np.array([W, H])
    bg  = rng.integers(30, 235, 3) if rng.random() < .6 else np.full(3, rng.integers(215, 255))
    M   = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    f   = cv2.warpPerspective(f, M, (W, H), borderMode=cv2.BORDER_CONSTANT, borderValue=tuple(float(v) for v in bg))

    # 3 · lighting: soft shadow, optional hand/phone blob, creases, uneven exposure
    s = p['shadow'] = rng.uniform(*L['shadow'])
    th = rng.uniform(0, 2 * np.pi)
    g = np.cos(th) * (xx - W / 2) / W + np.sin(th) * (yy - H / 2) / H
    g = 1 / (1 + np.exp(-g * rng.uniform(4, 14) - rng.uniform(-1, 1)))
    shade = 1 - s * g
    if rng.random() < .5:
        blob = np.zeros((H, W), np.float32)
        cv2.ellipse(blob, (int(rng.uniform(0, W)), int(rng.uniform(0, H))),
                    (int(rng.uniform(60, 250)), int(rng.uniform(40, 150))), float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
        shade = shade * (1 - s * .8 * cv2.GaussianBlur(blob, (0, 0), 35))
    wl = np.zeros((H, W), np.float32)
    for _ in range(int(rng.integers(3, 9)) if wr > .03 else 0):
        p1 = rng.uniform((0, 0), (W, H)); a2 = rng.uniform(0, np.pi); ln = rng.uniform(150, 700)
        p2 = p1 + ln * np.array([np.cos(a2), np.sin(a2)])
        cv2.line(wl, tuple(int(v) for v in p1), tuple(int(v) for v in p2), float(rng.choice([-1, 1])), int(rng.integers(1, 3)))
    wl = cv2.GaussianBlur(wl, (0, 0), 2.5); wl = wl / (np.abs(wl).max() + 1e-6)
    shade = shade * (1 + wr * 1.6 * wl + wr * .6 * smooth_field(rng, H, W, 4, 6))
    f = f * np.clip(shade, .08, 1.4)[..., None]

    # 4 · focus: lens blur and (sometimes) hand-shake
    sg = p['blur'] = rng.uniform(*L['blur'])
    if sg > .05: f = cv2.GaussianBlur(f, (0, 0), sg)
    p['motion'] = 0
    if sev and rng.random() < .35:
        k = p['motion'] = int(rng.integers(5, 9 + 4 * sev)) | 1
        ker = np.zeros((k, k), np.float32); ker[k // 2] = 1
        ker = cv2.warpAffine(ker, cv2.getRotationMatrix2D(((k - 1) / 2, (k - 1) / 2), float(rng.uniform(0, 180)), 1.), (k, k))
        f = cv2.filter2D(f, -1, ker / ker.sum())

    # 5 · sensor: noise, exposure, contrast, colour cast
    ns = p['noise'] = rng.uniform(*L['noise'])
    f = f + rng.standard_normal(f.shape, dtype=np.float32) * ns
    ct = rng.uniform(.8, 1.15) if sev else rng.uniform(.95, 1.05)
    f = ((f - 128) * ct + 128) * rng.uniform(*L['bright']) * (1 + rng.normal(0, .02 + .03 * sev, 3))
    p['jpeg'] = int(rng.integers(*L['jpeg']))
    return np.clip(f, 0, 255).astype(np.uint8), p
