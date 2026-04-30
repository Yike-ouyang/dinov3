"""Standalone DINOv3 model-loading smoke test.

Imports the backbone factory directly (skipping torch.hub.load, which can
stall under nix when sourcing the local repo). Mirrors what `server.py` does
on setup.
"""

import torch
from dinov3.hub.backbones import dinov3_vits16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = dinov3_vits16(pretrained=False).to(device).eval()

x = torch.randn(1, 3, 224, 224, device=device)
with torch.no_grad():
    y = model(x)

print(y.shape)
