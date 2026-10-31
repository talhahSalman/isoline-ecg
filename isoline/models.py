"""The two neural networks behind Isoline.

IsolineImageNet   EfficientNet-B0 that looks at the ECG picture. It has two heads: the five
                  cardiac categories and the photo quality (good / fair / bad).
IsolineSignalNet  Residual 1D-CNN with a small Transformer on top. It reads waveforms that
                  were rebuilt from the picture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# ---------------------------------------------------------------- image model
class IsolineImageNet(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = torchvision.models.efficientnet_b0(weights=weights)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.3)
        self.diag = nn.Linear(1280, 5)   # NORM, MI, STTC, CD, HYP
        self.qual = nn.Linear(1280, 3)   # good, fair, bad

    def forward(self, x):
        h = self.drop(self.pool(self.features(x)).flatten(1))
        return self.diag(h), self.qual(h)


# ---------------------------------------------------------------- signal model
class SE(nn.Module):
    """Squeeze-and-excitation: learns which channels matter."""

    def __init__(self, c, r=8):
        super().__init__()
        self.f1, self.f2 = nn.Linear(c, c // r), nn.Linear(c // r, c)

    def forward(self, x):
        w = torch.sigmoid(self.f2(F.relu(self.f1(x.mean(-1)))))
        return x * w.unsqueeze(-1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1, self.b1 = nn.Conv1d(cin, cout, 7, stride, 3, bias=False), nn.BatchNorm1d(cout)
        self.c2, self.b2 = nn.Conv1d(cout, cout, 7, 1, 3, bias=False), nn.BatchNorm1d(cout)
        self.se, self.drop = SE(cout), nn.Dropout(0.1)
        if stride != 1 or cin != cout:
            self.sc = nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout))
        else:
            self.sc = nn.Identity()

    def forward(self, x):
        out = self.drop(F.relu(self.b1(self.c1(x))))
        out = self.se(self.b2(self.c2(out)))
        return F.relu(out + self.sc(x))


class IsolineSignalNet(nn.Module):
    def __init__(self, n_leads=12, n_cls=5, d=256, heads=4, layers=3, length=125):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(n_leads, 64, 15, 2, 7, bias=False), nn.BatchNorm1d(64), nn.ReLU())
        plan = [(64, 64, 1), (64, 128, 2), (128, 128, 1), (128, 256, 2), (256, 256, 1)]
        self.cnn = nn.Sequential(*[ResBlock(a, b, s) for a, b, s in plan])      # 1000 samples -> 125 tokens
        self.pos = nn.Parameter(torch.zeros(1, length, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=512, dropout=0.1,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.tf = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * d, n_cls))

    def forward(self, x):                                   # x: (batch, 12 leads, 1000 samples)
        h = self.cnn(self.stem(x)).transpose(1, 2)
        h = self.norm(self.tf(h + self.pos))
        return self.head(torch.cat([h.mean(1), h.amax(1)], 1))
