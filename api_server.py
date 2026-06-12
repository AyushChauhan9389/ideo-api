"""FastAPI server exposing Ideogram 4 text-to-image generation.

Run:
  pip install -e ".[api]"
  python api_server.py --host 127.0.0.1 --port 8000

Endpoints:
  POST /generate            generate image(s); every pipeline/sampler/magic-prompt
                            parameter can be passed in the request body
  GET  /images              list generated images (newest first) with metadata
  GET  /images/{filename}   fetch a generated image
  GET  /presets             named sampler presets and their parameters
  GET  /magic-prompt-models available magic-prompt configurations
  GET  /health              server + loaded-model status

Generated images are saved under ./generated/ together with a JSON sidecar
holding the full request parameters (API keys excluded).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import threading
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

# Disable expandable segments unless the user configured the allocator themselves:
# it requires working NVML, which is blocked on fractional/virtualized GPUs
# (RunPod, vast.ai, WSL2), crashing the first forward pass with
# "NVML_SUCCESS == r INTERNAL ASSERT FAILED at CUDACachingAllocator.cpp".
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from html import escape as html_escape
from PIL import Image
from pydantic import BaseModel, Field, field_validator

from ideogram4 import (
  DEFAULT_MAGIC_PROMPT,
  MAGIC_PROMPTS,
  PRESETS,
  Ideogram4Pipeline,
  Ideogram4PipelineConfig,
  aspect_ratio_from_size,
  moderate_image,
  moderate_prompt,
)

QUANTIZATION_REPOS = {
  "nf4": "ideogram-ai/ideogram-4-nf4",
  "fp8": "ideogram-ai/ideogram-4-fp8",
}

DTYPES = {
  "bfloat16": torch.bfloat16,
  "float16": torch.float16,
  "float32": torch.float32,
}

GENERATED_DIR = Path(__file__).resolve().parent / "generated"
_FILENAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-]*\.png")


def _default_device() -> str:
  if torch.cuda.is_available():
    return "cuda"
  if torch.backends.mps.is_available():
    return "mps"
  return "cpu"


def _default_quantization() -> str:
  return "nf4" if torch.cuda.is_available() else "fp8"


# --------------------------------------------------------------------------- #
# Pipeline cache: one pipeline resident at a time, swapped when the requested
# (quantization, device, dtype) changes. Loading + generation are serialized
# behind a single lock since the model owns the device.
# --------------------------------------------------------------------------- #

_lock = threading.Lock()
_pipeline: Optional[Ideogram4Pipeline] = None
_pipeline_key: Optional[tuple[str, str, str]] = None

SERVER_DEFAULTS = {
  "quantization": _default_quantization(),
  "device": _default_device(),
  "dtype": "bfloat16",
}


def _get_pipeline(quantization: str, device: str, dtype: str) -> Ideogram4Pipeline:
  """Return the cached pipeline, (re)loading if the key changed. Caller holds _lock."""
  global _pipeline, _pipeline_key
  key = (quantization, device, dtype)
  if _pipeline is not None and _pipeline_key == key:
    return _pipeline

  if _pipeline is not None:
    _pipeline = None
    _pipeline_key = None
    gc.collect()
    if torch.cuda.is_available():
      torch.cuda.empty_cache()

  _pipeline = Ideogram4Pipeline.from_pretrained(
    config=Ideogram4PipelineConfig(weights_repo=QUANTIZATION_REPOS[quantization]),
    device=device,
    dtype=DTYPES[dtype],
  )
  _pipeline_key = key
  return _pipeline


# --------------------------------------------------------------------------- #
# Request / response models
# --------------------------------------------------------------------------- #


class GenerateRequest(BaseModel):
  # Prompting
  prompt: str | list[str] = Field(
    description="Plain prompt (or structured JSON caption string). A list generates one image per prompt."
  )
  magic_prompt: bool = Field(
    default=True,
    description="Expand the plain prompt into a structured JSON caption via a magic-prompt LLM.",
  )
  magic_prompt_model: str = Field(
    default=DEFAULT_MAGIC_PROMPT,
    description=f"Magic-prompt configuration. One of: {sorted(MAGIC_PROMPTS)}",
  )
  magic_prompt_key: Optional[str] = Field(
    default=None,
    description="API key for the magic-prompt backend. Falls back to MAGIC_PROMPT_API_KEY / IDEOGRAM_API_KEY env vars.",
  )
  strip_bboxes: bool = Field(
    default=True, description="Strip bbox layout hints from the expanded caption."
  )
  raise_on_caption_issues: bool = Field(
    default=True,
    description="Reject the request (400) when the caption verifier flags issues; false emits warnings instead.",
  )

  # Image / sampler
  height: int = 1024
  width: int = 1024
  sampler_preset: Optional[str] = Field(
    default="V4_QUALITY_48",
    description=f"Named sampler preset, one of: {sorted(PRESETS)}. Set null to configure manually.",
  )
  num_steps: Optional[int] = Field(
    default=None, description="Override the preset's step count."
  )
  guidance_scale: Optional[float] = Field(
    default=None,
    description="Constant CFG weight. Overrides the preset's guidance schedule.",
  )
  guidance_schedule: Optional[list[float]] = Field(
    default=None,
    description="Per-step CFG weights in loop-index order (index 0 = final polish step). Length must equal num_steps.",
  )
  mu: Optional[float] = Field(
    default=None, description="Logit-normal schedule mean override."
  )
  std: Optional[float] = Field(
    default=None, description="Logit-normal schedule std override."
  )
  seed: Optional[int] = Field(
    default=None, description="Random seed. Omit for a random seed (returned in the response)."
  )

  # Model selection (changing these reloads the pipeline)
  quantization: Optional[Literal["nf4", "fp8"]] = None
  device: Optional[str] = None
  dtype: Optional[Literal["bfloat16", "float16", "float32"]] = None

  # Safety screening (Hive)
  hive_text_key: Optional[str] = Field(
    default=None, description="Hive Text Moderation key. Falls back to HIVE_TEXT_MODERATION_KEY."
  )
  hive_visual_key: Optional[str] = Field(
    default=None, description="Hive Visual Moderation key. Falls back to HIVE_VISUAL_MODERATION_KEY."
  )
  moderate_input: bool = Field(
    default=True, description="Screen prompts with Hive text moderation when a key is available."
  )
  moderate_output: bool = Field(
    default=True, description="Screen generated images with Hive visual moderation when a key is available."
  )

  # Background removal (BiRefNet)
  remove_background: bool = Field(
    default=False,
    description="Also produce a transparent-background version of each image (saved as <name>_nobg.png).",
  )
  bg_model: Literal["birefnet-hr", "birefnet"] = Field(
    default="birefnet-hr",
    description="Background-removal model: 'birefnet-hr' (2048px, best quality) or 'birefnet' (1024px, faster).",
  )

  # Response shape
  response_format: Literal["json", "image"] = Field(
    default="json",
    description="'json' returns metadata + image URLs; 'image' streams the first image back directly.",
  )

  @field_validator("height", "width")
  @classmethod
  def _check_size(cls, v: int) -> int:
    if not 256 <= v <= 2048:
      raise ValueError(f"must be in [256, 2048], got {v}")
    if v % 16 != 0:
      raise ValueError(f"must be a multiple of 16, got {v}")
    return v

  @field_validator("sampler_preset")
  @classmethod
  def _check_preset(cls, v: Optional[str]) -> Optional[str]:
    if v is not None and v not in PRESETS:
      raise ValueError(f"unknown preset {v!r}, expected one of {sorted(PRESETS)}")
    return v

  @field_validator("magic_prompt_model")
  @classmethod
  def _check_magic_model(cls, v: str) -> str:
    if v not in MAGIC_PROMPTS:
      raise ValueError(f"unknown magic-prompt model {v!r}, expected one of {sorted(MAGIC_PROMPTS)}")
    return v


def _resolve_sampler(req: GenerateRequest) -> dict[str, Any]:
  """Merge the named preset with explicit per-request overrides."""
  preset = PRESETS[req.sampler_preset] if req.sampler_preset else None

  num_steps = req.num_steps or (preset.num_steps if preset else 48)
  mu = req.mu if req.mu is not None else (preset.mu if preset else 0.5)
  std = req.std if req.std is not None else (preset.std if preset else 1.0)

  if req.guidance_schedule is not None:
    if len(req.guidance_schedule) != num_steps:
      raise HTTPException(
        status_code=400,
        detail=f"guidance_schedule has length {len(req.guidance_schedule)}, expected num_steps={num_steps}",
      )
    schedule: Optional[tuple[float, ...]] = tuple(req.guidance_schedule)
  elif req.guidance_scale is not None:
    schedule = None  # constant guidance_scale takes over
  elif preset and num_steps == preset.num_steps:
    schedule = preset.guidance_schedule
  else:
    schedule = None  # custom step count invalidates the preset's schedule

  return {
    "num_steps": num_steps,
    "guidance_scale": req.guidance_scale if req.guidance_scale is not None else 7.0,
    "guidance_schedule": schedule,
    "mu": mu,
    "std": std,
  }


def _expand_prompts(req: GenerateRequest, prompts: list[str]) -> list[str]:
  key = (
    req.magic_prompt_key
    or os.environ.get("MAGIC_PROMPT_API_KEY")
    or os.environ.get("IDEOGRAM_API_KEY")
  )
  if not key:
    raise HTTPException(
      status_code=400,
      detail=(
        "magic_prompt is enabled but no API key was found. Pass magic_prompt_key, "
        "set MAGIC_PROMPT_API_KEY / IDEOGRAM_API_KEY, or disable with magic_prompt=false."
      ),
    )
  aspect_ratio = aspect_ratio_from_size(req.width, req.height)
  magic = MAGIC_PROMPTS[req.magic_prompt_model](  # type: ignore[call-arg]
    api_key=key, strip_bboxes=req.strip_bboxes
  )
  try:
    return [magic.expand(p, aspect_ratio=aspect_ratio) for p in prompts]
  except Exception as e:
    raise HTTPException(status_code=502, detail=f"magic prompt expansion failed: {e}")


def _public_params(req: GenerateRequest) -> dict[str, Any]:
  """Request params safe to persist in the metadata sidecar (no API keys)."""
  return req.model_dump(
    exclude={"magic_prompt_key", "hive_text_key", "hive_visual_key"}
  )


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #

app = FastAPI(title="Ideogram 4 API", version="0.1.0")

# Allow all cross-origin requests so browser frontends on any host can call the API.
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=False,
  allow_methods=["*"],
  allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
  return {
    "status": "ok",
    "loaded_pipeline": dict(zip(("quantization", "device", "dtype"), _pipeline_key))
    if _pipeline_key
    else None,
    "server_defaults": SERVER_DEFAULTS,
    "cuda_available": torch.cuda.is_available(),
  }


@app.get("/presets")
def presets() -> dict[str, Any]:
  return {name: asdict(p) for name, p in PRESETS.items()}


@app.get("/magic-prompt-models")
def magic_prompt_models() -> dict[str, Any]:
  return {
    "models": sorted(MAGIC_PROMPTS),
    "default": DEFAULT_MAGIC_PROMPT,
  }


@app.post("/generate")
def generate(req: GenerateRequest):
  t_start = time.perf_counter()
  prompts = [req.prompt] if isinstance(req.prompt, str) else list(req.prompt)
  if not prompts:
    raise HTTPException(status_code=400, detail="prompt list is empty")

  quantization = req.quantization or SERVER_DEFAULTS["quantization"]
  device = req.device or SERVER_DEFAULTS["device"]
  dtype = req.dtype or SERVER_DEFAULTS["dtype"]
  if quantization == "nf4" and not device.startswith("cuda"):
    raise HTTPException(
      status_code=400, detail="nf4 weights require a CUDA device; use quantization='fp8'"
    )

  # 1. Prompt safety screening (pre-expansion, matching run_inference.py).
  text_key = req.hive_text_key or os.environ.get("HIVE_TEXT_MODERATION_KEY")
  if text_key and req.moderate_input:
    for i, p in enumerate(prompts):
      flags = moderate_prompt(p, text_key)
      if flags:
        raise HTTPException(
          status_code=400,
          detail={"error": "prompt rejected by Hive text moderation", "prompt_index": i, "flags": flags},
        )

  # 2. Magic-prompt expansion.
  original_prompts = list(prompts)
  expanded = None
  if req.magic_prompt:
    prompts = _expand_prompts(req, prompts)
    expanded = prompts

  # 3. Sampler resolution.
  sampler = _resolve_sampler(req)
  seed = req.seed if req.seed is not None else int.from_bytes(os.urandom(4), "big")

  # 4. Generate (serialized: the model owns the device).
  with _lock:
    try:
      pipe = _get_pipeline(quantization, device, dtype)
    except Exception as e:
      raise HTTPException(status_code=500, detail=f"failed to load pipeline: {e}")
    t_loaded = time.perf_counter()
    try:
      images = pipe(
        prompts,
        height=req.height,
        width=req.width,
        num_steps=sampler["num_steps"],
        guidance_scale=sampler["guidance_scale"],
        guidance_schedule=sampler["guidance_schedule"],
        mu=sampler["mu"],
        std=sampler["std"],
        seed=seed,
        raise_on_caption_issues=req.raise_on_caption_issues,
      )
    except ValueError as e:
      raise HTTPException(status_code=400, detail=str(e))
  t_generated = time.perf_counter()

  # 5. Output safety screening; flagged images are dropped, not saved.
  visual_key = req.hive_visual_key or os.environ.get("HIVE_VISUAL_MODERATION_KEY")
  rejected: list[dict[str, Any]] = []
  kept: list[tuple[int, Any]] = []
  for i, img in enumerate(images):
    if visual_key and req.moderate_output:
      flags = moderate_image(img, visual_key)
      if flags:
        rejected.append({"image_index": i, "flags": flags})
        continue
    kept.append((i, img))

  # 6. Save to ./generated/ with a JSON metadata sidecar.
  GENERATED_DIR.mkdir(exist_ok=True)
  generation_id = uuid.uuid4().hex[:12]
  created_at = datetime.now(timezone.utc).isoformat()
  saved: list[dict[str, Any]] = []
  for i, img in kept:
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{generation_id}_{i}.png"
    img.save(GENERATED_DIR / filename)
    saved.append({"filename": filename, "url": f"/images/{filename}", "prompt_index": i})

  # 6b. Optional background removal — one transparent PNG per kept image.
  if req.remove_background and kept:
    from bg_removal import remove_background as _remove_bg

    with _lock:  # BiRefNet shares the device with the pipeline
      for entry, (_, img) in zip(saved, kept):
        try:
          rgba = _remove_bg(img, req.bg_model, device)
        except Exception as e:
          entry["nobg_error"] = str(e)
          continue
        nobg_name = Path(entry["filename"]).stem + "_nobg.png"
        rgba.save(GENERATED_DIR / nobg_name)
        entry["nobg_filename"] = nobg_name
        entry["nobg_url"] = f"/images/{nobg_name}"

  metadata = {
    "id": generation_id,
    "created_at": created_at,
    "seed": seed,
    "prompts": original_prompts,
    "expanded_captions": expanded,
    "sampler": {k: list(v) if isinstance(v, tuple) else v for k, v in sampler.items()},
    "pipeline": {"quantization": quantization, "device": device, "dtype": dtype},
    "request": _public_params(req),
    "images": saved,
    "rejected_by_moderation": rejected,
    "timing_s": {
      "model_load": round(t_loaded - t_start, 2),
      "generation": round(t_generated - t_loaded, 2),
    },
  }
  for entry in saved:
    sidecar = GENERATED_DIR / (Path(entry["filename"]).stem + ".json")
    sidecar.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

  if not saved:
    raise HTTPException(
      status_code=422,
      detail={"error": "all generated images were rejected by Hive visual moderation", "rejected": rejected},
    )

  if req.response_format == "image":
    return FileResponse(GENERATED_DIR / saved[0]["filename"], media_type="image/png")
  return metadata


@app.get("/images")
def list_images(limit: int = 50, offset: int = 0) -> dict[str, Any]:
  """List generated images, newest first, with their sidecar metadata."""
  GENERATED_DIR.mkdir(exist_ok=True)
  files = sorted(
    GENERATED_DIR.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True
  )
  page = files[offset : offset + limit]
  items = []
  for f in page:
    sidecar = f.with_suffix(".json")
    meta = None
    if sidecar.exists():
      try:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
      except json.JSONDecodeError:
        meta = None
    items.append(
      {
        "filename": f.name,
        "url": f"/images/{f.name}",
        "size_bytes": f.stat().st_size,
        "modified_at": datetime.fromtimestamp(
          f.stat().st_mtime, tz=timezone.utc
        ).isoformat(),
        "metadata": meta,
      }
    )
  return {"total": len(files), "limit": limit, "offset": offset, "images": items}


def _resolve_generated(filename: str) -> Path:
  """Validate a filename and resolve it inside GENERATED_DIR (no traversal)."""
  if not _FILENAME_RE.fullmatch(filename):
    raise HTTPException(status_code=400, detail="invalid filename")
  path = (GENERATED_DIR / filename).resolve()
  if path.parent != GENERATED_DIR or not path.exists():
    raise HTTPException(status_code=404, detail="image not found")
  return path


@app.get("/images/{filename}")
def get_image(filename: str):
  """Fetch a generated image by filename."""
  return FileResponse(_resolve_generated(filename), media_type="image/png")


@app.get("/images/{filename}/metadata")
def get_image_metadata(filename: str) -> dict[str, Any]:
  """Fetch the JSON metadata sidecar for a generated image."""
  path = _resolve_generated(filename)
  sidecar = path.with_suffix(".json")
  if not sidecar.exists():
    raise HTTPException(status_code=404, detail="no metadata for this image")
  return json.loads(sidecar.read_text(encoding="utf-8"))


@app.delete("/images/{filename}")
def delete_image(filename: str) -> dict[str, Any]:
  """Delete a generated image and its metadata sidecar."""
  path = _resolve_generated(filename)
  path.unlink()
  sidecar = path.with_suffix(".json")
  if sidecar.exists():
    sidecar.unlink()
  nobg = GENERATED_DIR / (path.stem + "_nobg.png")
  if nobg.exists():
    nobg.unlink()
  return {"deleted": filename}


@app.post("/images/{filename}/remove-background")
def remove_image_background(
  filename: str, model: Literal["birefnet-hr", "birefnet"] = "birefnet-hr"
) -> dict[str, Any]:
  """Create (or overwrite) a transparent-background version of an existing image."""
  path = _resolve_generated(filename)
  if path.stem.endswith("_nobg"):
    raise HTTPException(status_code=400, detail="image is already a background-removed output")

  from bg_removal import remove_background as _remove_bg

  device = _pipeline_key[1] if _pipeline_key else SERVER_DEFAULTS["device"]
  with _lock:
    try:
      rgba = _remove_bg(Image.open(path), model, device)
    except RuntimeError as e:
      raise HTTPException(status_code=500, detail=str(e))
  nobg_name = path.stem + "_nobg.png"
  rgba.save(GENERATED_DIR / nobg_name)
  return {
    "filename": filename,
    "nobg_filename": nobg_name,
    "nobg_url": f"/images/{nobg_name}",
    "bg_model": model,
  }


_CHECKER_CSS = (
  "background-image:linear-gradient(45deg,#ccc 25%,transparent 25%),"
  "linear-gradient(-45deg,#ccc 25%,transparent 25%),"
  "linear-gradient(45deg,transparent 75%,#ccc 75%),"
  "linear-gradient(-45deg,transparent 75%,#ccc 75%);"
  "background-size:20px 20px;background-position:0 0,0 10px,10px -10px,-10px 0;"
  "background-color:#fff;"
)


@app.get("/gallery", response_class=HTMLResponse)
def gallery() -> str:
  """Static page: every generated image next to its background-removed version."""
  GENERATED_DIR.mkdir(exist_ok=True)
  originals = sorted(
    (p for p in GENERATED_DIR.glob("*.png") if not p.stem.endswith("_nobg")),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
  )
  cards = []
  for p in originals:
    sidecar = p.with_suffix(".json")
    prompt = seed = ""
    if sidecar.exists():
      try:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        prompts = meta.get("prompts") or []
        prompt = (prompts[0] if prompts else "")[:160]
        seed = meta.get("seed", "")
      except json.JSONDecodeError:
        pass
    nobg = GENERATED_DIR / (p.stem + "_nobg.png")
    if nobg.exists():
      nobg_html = (
        f'<div class="checker"><img src="/images/{nobg.name}" loading="lazy"></div>'
        f'<div class="cap">{nobg.name}</div>'
      )
    else:
      nobg_html = (
        f'<div class="missing">no transparent version yet<br>'
        f'<button onclick="removeBg(this, \'{p.name}\')">remove background</button></div>'
      )
    cards.append(f"""
    <div class="card">
      <div class="pair">
        <div><img src="/images/{p.name}" loading="lazy"><div class="cap">{p.name}</div></div>
        <div>{nobg_html}</div>
      </div>
      <div class="meta">{html_escape(prompt)}{f" &middot; seed {seed}" if seed != "" else ""}</div>
    </div>""")

  body = "".join(cards) or "<p>No generated images yet. POST /generate to create some.</p>"
  return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Ideogram 4 — gallery</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 24px; background: #f5f5f5; }}
  h1 {{ font-size: 20px; }}
  .toolbar {{ margin-bottom: 16px; color: #444; font-size: 14px; }}
  .card {{ background: #fff; border-radius: 10px; padding: 14px; margin-bottom: 18px;
           box-shadow: 0 1px 4px rgba(0,0,0,.08); display: inline-block; margin-right: 14px;
           vertical-align: top; }}
  .pair {{ display: flex; gap: 12px; }}
  .pair img {{ max-width: 320px; max-height: 320px; display: block; }}
  .checker {{ {_CHECKER_CSS} }}
  .cap {{ font-size: 11px; color: #777; margin-top: 4px; max-width: 320px; word-break: break-all; }}
  .meta {{ font-size: 12px; color: #444; margin-top: 8px; max-width: 660px; }}
  .missing {{ display: flex; flex-direction: column; gap: 8px; align-items: center;
              justify-content: center; min-width: 200px; min-height: 200px;
              color: #999; font-size: 13px; text-align: center; }}
  button {{ cursor: pointer; padding: 6px 10px; }}
</style></head><body>
<h1>Ideogram 4 — generated images</h1>
<div class="toolbar">bg model for on-demand removal:
  <select id="bgmodel"><option value="birefnet-hr" selected>birefnet-hr (quality)</option>
  <option value="birefnet">birefnet (fast)</option></select>
</div>
{body}
<script>
async function removeBg(btn, filename) {{
  btn.disabled = true; btn.textContent = "processing...";
  const model = document.getElementById("bgmodel").value;
  try {{
    const r = await fetch(`/images/${{filename}}/remove-background?model=${{model}}`, {{ method: "POST" }});
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    location.reload();
  }} catch (e) {{ btn.disabled = false; btn.textContent = "failed: " + e.message; }}
}}
</script>
</body></html>"""


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #


def main() -> None:
  parser = argparse.ArgumentParser(description="Ideogram 4 API server")
  parser.add_argument("--host", default="0.0.0.0")
  parser.add_argument("--port", type=int, default=8000)
  parser.add_argument(
    "--quantization",
    choices=sorted(QUANTIZATION_REPOS),
    default=_default_quantization(),
    help="Default weight quantization (per-request override via 'quantization').",
  )
  parser.add_argument("--device", default=_default_device())
  parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
  parser.add_argument(
    "--preload",
    action="store_true",
    help="Load the pipeline at startup instead of on the first /generate request.",
  )
  args = parser.parse_args()

  SERVER_DEFAULTS["quantization"] = args.quantization
  SERVER_DEFAULTS["device"] = args.device
  SERVER_DEFAULTS["dtype"] = args.dtype

  if args.preload:
    with _lock:
      _get_pipeline(args.quantization, args.device, args.dtype)

  uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
  main()
