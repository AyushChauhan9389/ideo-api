"""BiRefNet background removal for the Ideogram 4 API server.

Two MIT-licensed model variants, lazy-loaded and cached on first use:
  birefnet-hr (default) — ZhengPeng7/BiRefNet_HR, 2048px processing
  birefnet              — ZhengPeng7/BiRefNet, 1024px processing (faster,
                          right fit for small images)

Output edge quality depends on foreground refinement: the alpha mask alone
leaves background color baked into semi-transparent edge pixels (white halo on
light backgrounds). ``refine_foreground`` below is a port of BiRefNet's own
image_proc.py two-pass blur-fusion estimator and removes that bleed.

Extra deps (beyond the base package): pip install ".[bg]"
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

BG_MODELS: dict[str, tuple[str, int]] = {
  "birefnet-hr": ("ZhengPeng7/BiRefNet_HR", 2048),
  "birefnet": ("ZhengPeng7/BiRefNet", 1024),
}
DEFAULT_BG_MODEL = "birefnet-hr"

_removers: dict[tuple[str, str], "BiRefNetRemover"] = {}


def _fb_blur_fusion(image, F, B, alpha, r=90):
  import cv2

  blurred_alpha = cv2.blur(alpha, (r, r))[:, :, None]
  blurred_FA = cv2.blur(F * alpha, (r, r))
  blurred_F = blurred_FA / (blurred_alpha + 1e-5)
  blurred_B1A = cv2.blur(B * (1 - alpha), (r, r))
  blurred_B = blurred_B1A / ((1 - blurred_alpha) + 1e-5)
  F = blurred_F + alpha * (image - alpha * blurred_F - (1 - alpha) * blurred_B)
  return np.clip(F, 0, 1), blurred_B


def refine_foreground(image: Image.Image, mask: Image.Image, r: int = 90) -> Image.Image:
  """Return ``image`` with background color bleed removed from edge pixels."""
  img = np.asarray(image, dtype=np.float64) / 255.0
  alpha = (np.asarray(mask, dtype=np.float64) / 255.0)[:, :, None]
  F, blur_B = _fb_blur_fusion(img, img, img, alpha, r=r)
  F, _ = _fb_blur_fusion(img, F, blur_B, alpha, r=6)
  return Image.fromarray((F * 255.0).astype(np.uint8))


class BiRefNetRemover:
  def __init__(self, model_key: str, device: str):
    try:
      from torchvision import transforms
      from transformers import AutoModelForImageSegmentation
    except ImportError as e:
      raise RuntimeError(
        "background removal needs extra deps; install with: pip install '.[bg]'"
      ) from e

    repo, resolution = BG_MODELS[model_key]
    self.device = device
    # Checkpoints are stored fp16; force fp32 off-CUDA so CPU/MPS still work.
    self.dtype = torch.float16 if device.startswith("cuda") else torch.float32
    self.model = AutoModelForImageSegmentation.from_pretrained(
      repo, trust_remote_code=True, torch_dtype=self.dtype
    )
    self.model.to(device).eval()
    self.transform = transforms.Compose([
      transforms.Resize((resolution, resolution)),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

  @torch.no_grad()
  def remove(self, image: Image.Image) -> Image.Image:
    """RGB PIL image in, RGBA PIL image (transparent background) out."""
    image = image.convert("RGB")
    x = self.transform(image).unsqueeze(0).to(self.device, self.dtype)
    preds = self.model(x)[-1].sigmoid().float().cpu()
    mask = Image.fromarray(
      (preds[0].squeeze().numpy() * 255).astype(np.uint8)
    ).resize(image.size, Image.LANCZOS)
    out = refine_foreground(image, mask)
    out.putalpha(mask)
    return out


def get_remover(model_key: str, device: str) -> BiRefNetRemover:
  """Cached accessor — both variants are small (~0.5-1 GB) and may coexist."""
  if model_key not in BG_MODELS:
    raise ValueError(f"unknown bg model {model_key!r}, expected one of {sorted(BG_MODELS)}")
  key = (model_key, device)
  if key not in _removers:
    _removers[key] = BiRefNetRemover(model_key, device)
  return _removers[key]


def remove_background(image: Image.Image, model_key: str, device: str) -> Image.Image:
  return get_remover(model_key, device).remove(image)
