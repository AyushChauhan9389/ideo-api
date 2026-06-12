"""Offline model resolution for Ideogram 4.

All model loading goes through this module so weights are read from a local
``offline/`` folder instead of being fetched from the Hugging Face Hub on every
run. Download the repos once with ``python download_models.py``; afterwards the
pipeline loads entirely from disk.

Layout (each Hugging Face repo is snapshotted into a folder named after the
repo's last path component)::

    offline/
      ideogram-4-nf4/      <- ideogram-ai/ideogram-4-nf4
      ideogram-4-fp8/      <- ideogram-ai/ideogram-4-fp8
      BiRefNet_HR/         <- ZhengPeng7/BiRefNet_HR
      BiRefNet/            <- ZhengPeng7/BiRefNet

The offline directory defaults to ``<project root>/offline`` and can be
overridden with the ``IDEOGRAM_OFFLINE_DIR`` environment variable.

``resolve_repo`` / ``resolve_file`` gracefully fall back to the Hub when a repo
has not been downloaded yet, so existing online behaviour still works.
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

# Every repo the project may load, so the downloader can fetch them all.
IDEOGRAM_REPOS: dict[str, str] = {
  "nf4": "ideogram-ai/ideogram-4-nf4",
  "fp8": "ideogram-ai/ideogram-4-fp8",
}
BG_REPOS: dict[str, str] = {
  "birefnet-hr": "ZhengPeng7/BiRefNet_HR",
  "birefnet": "ZhengPeng7/BiRefNet",
}
ALL_REPOS: tuple[str, ...] = tuple(IDEOGRAM_REPOS.values()) + tuple(BG_REPOS.values())

# Project root = two levels up from this file (.../src/ideogram4/offline.py).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def offline_dir() -> Path:
  """Return the directory that holds downloaded model snapshots."""
  override = os.environ.get("IDEOGRAM_OFFLINE_DIR")
  if override:
    return Path(override).expanduser().resolve()
  return _PROJECT_ROOT / "offline"


def repo_dirname(repo_id: str) -> str:
  """Local folder name for a repo, e.g. ``ideogram-ai/ideogram-4-nf4`` -> ``ideogram-4-nf4``."""
  return repo_id.rstrip("/").split("/")[-1]


def repo_local_dir(repo_id: str) -> Path:
  """Absolute path where ``repo_id`` is (or would be) stored offline."""
  return offline_dir() / repo_dirname(repo_id)


def resolve_repo(repo_id: str) -> str:
  """Return a local snapshot directory for ``repo_id`` if present, else ``repo_id``.

  The result is suitable to pass to ``from_pretrained`` (which accepts either a
  Hub repo id or a local directory).
  """
  if os.path.isdir(repo_id):
    return repo_id
  local = repo_local_dir(repo_id)
  if local.is_dir():
    return str(local)
  return repo_id


def resolve_file(repo_id: str, filename: str, **kwargs) -> str:
  """Return a path to ``filename`` within ``repo_id``.

  If ``repo_id`` resolves to a local offline snapshot the file is read from
  disk (raising :class:`EntryNotFoundError` when missing, so callers' index/
  single-file fallback logic still works). Otherwise it is downloaded from the
  Hub via ``hf_hub_download``.
  """
  local_dir: str | None = None
  if os.path.isdir(repo_id):
    local_dir = repo_id
  else:
    candidate = repo_local_dir(repo_id)
    if candidate.is_dir():
      local_dir = str(candidate)

  if local_dir is None:
    return hf_hub_download(repo_id=repo_id, filename=filename, **kwargs)

  path = os.path.join(local_dir, *filename.split("/"))
  if os.path.exists(path):
    return path
  raise EntryNotFoundError(f"{filename!r} not found in offline snapshot {local_dir!r}")
