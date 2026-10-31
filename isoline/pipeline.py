"""The Isoline pipeline.

    photo -> quality check -> image model -> (rebuild waveforms -> signal model) -> decision

The image model gives the answer. The waveform path is a cross-check: when the photo is lined up
well enough to rebuild the signals and both models agree, we trust the answer more. When the
photo is too poor to read, we say so and ask for a better one instead of guessing.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import cv2
import numpy as np

from .digitize import digitize

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
CLASS_NAMES = {
    "NORM": "normal ECG",
    "MI": "myocardial infarction pattern",
    "STTC": "ST/T-wave change",
    "CD": "conduction disturbance",
    "HYP": "hypertrophy pattern",
}
ADVICE = {
    "NORM": "Nothing abnormal was flagged. A normal screening result does not rule out heart disease, so "
            "symptoms still matter.",
    "MI": "The pattern resembles a heart attack, old or new. If there is chest pain or breathlessness, seek "
          "urgent medical care. Otherwise, ask a clinician to review the ECG.",
    "STTC": "ST/T-wave changes were flagged. They have many causes, so please have a clinician review the ECG.",
    "CD": "A conduction pattern (for example a bundle branch block) was flagged. A clinician should confirm it.",
    "HYP": "Signs that look like an enlarged heart chamber were flagged. A clinician should confirm them, "
           "usually with an echocardiogram.",
}
QUALITY = ["good", "fair", "bad"]
OUTCOMES = ["reliable", "flagged", "image-only", "rejected"]
RETAKE_TIPS = "Lay the page flat, use even light without shadows, keep the whole printout in the frame, and hold the camera steady."
DISCLAIMER = "Isoline is a learning project and a screening aid. It is not a medical device and does not give a diagnosis."

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def load_lead_spread(path) -> np.ndarray:
    """Per-lead standard deviation saved at training time, as an array of shape (1, 12, 1)."""
    data = np.load(path)
    names = list(data.files)
    key = next((k for k in ("std", "sd", "sigma", "scale") if k in names), None)
    if key is None and len(names) == 2:                 # (mean, std) saved in that order under other names
        key = names[1]
    if key is None:
        raise ValueError(f"{path} holds the arrays {names}, but a per-lead 'std' was expected.")
    return np.asarray(data[key], np.float32).reshape(1, 12, 1)


def decide(top_image: int, top_signal: int | None, quality_idx: int, aligned: bool) -> int:
    """0 reliable, 1 flagged (the two models disagree), 2 image-only, 3 rejected (retake the photo)."""
    if quality_idx == 2:
        return 3
    if not aligned or top_signal is None:
        return 2
    return 0 if top_image == top_signal else 1


def message_for(outcome: int, top: str) -> str:
    name = CLASS_NAMES[top]
    if outcome == 0:
        return f"Both checks agree. The most likely finding is: {name}."
    if outcome == 1:
        return f"The picture and the rebuilt waveform point to different answers, so treat this as uncertain. The image model's best guess is: {name}."
    if outcome == 2:
        return (f"Only the image model could read this one, because the photo is not lined up well enough to rebuild "
                f"the waveform. Its best guess is: {name}.")
    return "This image is not clear enough to read safely. Please retake it. " + RETAKE_TIPS


class Isoline:
    """Load once, then call analyze() on as many pictures as you like."""

    def __init__(self, weights_dir: str | Path | None = None, device: str = "cpu"):
        self.dir = Path(weights_dir) if weights_dir else ROOT / "weights"
        self.device = device
        calib_path = self.dir / "calibration.json"
        if not calib_path.exists():
            raise FileNotFoundError(f"Missing {calib_path}. See the 'Weights' section of the README.")
        calib = json.loads(calib_path.read_text())
        self.temperature = float(calib["image_temperature"])
        self.thresholds = np.array([calib["image_thresholds"][c] for c in CLASSES], np.float32)
        self.tau = float(calib["alignment_threshold"])
        self._image_net = None
        self._signal_net = None
        self._std = None

    # ------------------------------------------------------------------ models
    def _load(self):
        if self._image_net is not None:
            return
        import torch
        from .models import IsolineImageNet, IsolineSignalNet

        for name in ("isoline_image_net.pt", "isoline_signal_net_layout.pt", "signal_norm.npz"):
            if not (self.dir / name).exists():
                raise FileNotFoundError(f"Missing weight file {self.dir / name}. See the 'Weights' section of the README.")
        self._torch = torch
        self._image_net = IsolineImageNet(pretrained=False)
        self._image_net.load_state_dict(torch.load(self.dir / "isoline_image_net.pt", map_location=self.device))
        self._signal_net = IsolineSignalNet()
        self._signal_net.load_state_dict(torch.load(self.dir / "isoline_signal_net_layout.pt", map_location=self.device))
        for net in (self._image_net, self._signal_net):
            net.to(self.device).eval()
        self._std = load_lead_spread(self.dir / "signal_norm.npz")                        # shape (1, 12, 1)

    def _image_tensor(self, page_bgr):
        small = cv2.resize(page_bgr, (384, 192), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        arr = ((rgb - _MEAN) / _STD).transpose(2, 0, 1)[None]
        return self._torch.from_numpy(np.ascontiguousarray(arr)).to(self.device)

    def _signal_tensor(self, sig):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            centred = sig - np.nanmedian(sig, axis=0, keepdims=True)
        arr = np.clip(np.nan_to_num(centred), -2.5, 2.5).T[None] / self._std
        return self._torch.from_numpy(arr.astype(np.float32)).to(self.device)

    def _forward_image(self, x):
        with self._torch.no_grad():
            diag, qual = self._image_net(x)
        return diag[0].cpu().numpy(), qual[0].cpu().numpy()

    def _forward_signal(self, x):
        with self._torch.no_grad():
            return self._signal_net(x)[0].cpu().numpy()

    def _gradcam(self, x, class_idx):
        """Grad-CAM: which parts of the picture pushed the model towards this class."""
        torch = self._torch
        store = {}

        def hook(_module, _inputs, output):
            output.retain_grad()
            store["a"] = output

        handle = self._image_net.features.register_forward_hook(hook)
        try:
            with torch.enable_grad():
                diag, _ = self._image_net(x.clone().requires_grad_(True))
                self._image_net.zero_grad()
                diag[0, class_idx].backward()
            act = store["a"]
            weights = act.grad.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * act).sum(1, keepdim=True)).detach()
            cam = torch.nn.functional.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
            cam = cam.cpu().numpy()
            cam = cam - cam.min()
            return cam / (cam.max() + 1e-8)
        finally:
            handle.remove()

    # ------------------------------------------------------------------ main entry point
    def analyze(self, image_bgr: np.ndarray, explain: bool = True) -> dict:
        self._load()
        dig = digitize(image_bgr)                       # also straightens portrait pictures and resizes to the 800x400 page
        page = dig["page"]
        x = self._image_tensor(page)

        diag_logits, qual_logits = self._forward_image(x)
        p_img = _sigmoid(diag_logits / self.temperature)
        q_prob = _softmax(qual_logits)
        q_idx = int(q_prob.argmax())
        top_img = int(p_img.argmax())
        aligned = dig["align"] >= self.tau

        p_sig, top_sig = None, None
        if aligned and q_idx != 2:
            p_sig = _sigmoid(self._forward_signal(self._signal_tensor(dig["sig"])))
            top_sig = int(p_sig.argmax())

        outcome = decide(top_img, top_sig, q_idx, aligned)
        top = CLASSES[top_img]

        overlay = None
        if explain and outcome != 3:
            try:
                cam = self._gradcam(x, top_img)
                heat = cv2.applyColorMap((cv2.resize(cam, (page.shape[1], page.shape[0])) * 255).astype(np.uint8), cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(page, 0.55, heat, 0.45, 0)
            except Exception:                           # an explanation must never break a prediction
                overlay = None

        return {
            "outcome": outcome,
            "outcome_name": OUTCOMES[outcome],
            "message": message_for(outcome, top),
            "top": top,
            "advice": ADVICE[top] if outcome != 3 else RETAKE_TIPS,
            "found": [c for c, p, t in zip(CLASSES, p_img, self.thresholds) if p >= t],
            "image_probs": {c: float(p) for c, p in zip(CLASSES, p_img)},
            "quality": QUALITY[q_idx],
            "quality_probs": {q: float(p) for q, p in zip(QUALITY, q_prob)},
            "alignment": float(dig["align"]),
            "aligned": bool(aligned),
            "signal_probs": None if p_sig is None else {c: float(p) for c, p in zip(CLASSES, p_sig)},
            "signal_top": None if top_sig is None else CLASSES[top_sig],
            "waveforms": dig["sig"] if aligned else None,
            "page": page,
            "heatmap": overlay,
            "disclaimer": DISCLAIMER,
        }


def to_plain(result: dict) -> dict:
    """The same result without the pictures and arrays, ready for json.dumps."""
    return {k: v for k, v in result.items() if k not in ("waveforms", "page", "heatmap")}
