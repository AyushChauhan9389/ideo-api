"""Download all Ideogram 4 models into the local ``offline/`` folder.

Run once before going offline:

    python download_models.py

It prompts for a Hugging Face access token (the Ideogram 4 weights are gated --
accept the licence at https://huggingface.co/ideogram-ai/ideogram-4-nf4 first
and create a token at https://huggingface.co/settings/tokens), logs in, then
snapshots every repo into ``offline/<repo-name>/``. After that the pipeline and
background-removal code load entirely from disk with no network access.

Examples:
    python download_models.py                       # nf4 + fp8 + both BiRefNet
    python download_models.py --quantization nf4     # just the nf4 weights
    python download_models.py --quantization fp8 --no-bg
    python download_models.py --token hf_xxx         # non-interactive
"""

from __future__ import annotations

import argparse
import os
import sys
from getpass import getpass
from pathlib import Path

# Allow running straight from a checkout even without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from huggingface_hub import login, snapshot_download  # noqa: E402

from ideogram4.offline import (  # noqa: E402
  BG_REPOS,
  IDEOGRAM_REPOS,
  offline_dir,
  repo_dirname,
)


def _prompt_token(cli_token: str | None) -> str:
  token = cli_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
  if token:
    return token
  print(
    "The Ideogram 4 weights are gated. Accept the licence at\n"
    "  https://huggingface.co/ideogram-ai/ideogram-4-nf4\n"
    "and paste a token from https://huggingface.co/settings/tokens below.\n"
  )
  token = getpass("Hugging Face token (input hidden): ").strip()
  if not token:
    print("ERROR: no token provided.", file=sys.stderr)
    sys.exit(2)
  return token


def _select_repos(quantization: str, include_bg: bool) -> list[str]:
  repos: list[str] = []
  if quantization == "both":
    repos.extend(IDEOGRAM_REPOS.values())
  else:
    repos.append(IDEOGRAM_REPOS[quantization])
  if include_bg:
    repos.extend(BG_REPOS.values())
  return repos


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument(
    "--quantization",
    choices=["nf4", "fp8", "both"],
    default="both",
    help="Which Ideogram 4 weights to download (default: both).",
  )
  parser.add_argument(
    "--bg",
    dest="bg",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Also download the BiRefNet background-removal models (default: yes).",
  )
  parser.add_argument(
    "--token",
    default=None,
    help="Hugging Face token. If omitted, reads HF_TOKEN or prompts interactively.",
  )
  args = parser.parse_args()

  token = _prompt_token(args.token)
  login(token=token)

  target = offline_dir()
  target.mkdir(parents=True, exist_ok=True)
  repos = _select_repos(args.quantization, args.bg)

  print(f"\nDownloading {len(repos)} repo(s) into {target}\n")
  for i, repo_id in enumerate(repos, 1):
    dest = target / repo_dirname(repo_id)
    print(f"[{i}/{len(repos)}] {repo_id} -> {dest}")
    snapshot_download(
      repo_id=repo_id,
      local_dir=str(dest),
      token=token,
    )

  print("\nDone. Models are available offline at:")
  for repo_id in repos:
    print(f"  {target / repo_dirname(repo_id)}")
  print(
    "\nThe pipeline now loads from this folder automatically. To use a custom "
    "location, set IDEOGRAM_OFFLINE_DIR to the same path when running inference."
  )


if __name__ == "__main__":
  main()
