"""Publish the trained CRF models to the Hugging Face Hub.

Uploads every `train/out/kx_<lang>.pkl` plus `train/MODEL_CARD.md` (as the repo
README) to a model repo under the TigreGotico org. Needs an HF token with write
access (`huggingface-cli login` / `HF_TOKEN`). `--dry-run` lists without uploading.

Run::

    python train/push_model_to_hub.py --dry-run
    HF_TOKEN=... python train/push_model_to_hub.py
"""
from __future__ import annotations

import argparse
import glob
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
CARD = os.path.join(HERE, "MODEL_CARD.md")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="TigreGotico/crf-query-xtract")
    ap.add_argument("--models-dir", default=OUT)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--public", action="store_true", help="create the repo public")
    args = ap.parse_args()

    pkls = sorted(glob.glob(os.path.join(args.models_dir, "kx_*.pkl")))
    if not pkls:
        raise SystemExit(f"no kx_*.pkl in {args.models_dir} — run train_from_dataset.py first")
    print(f"{len(pkls)} models from {args.models_dir}:",
          ", ".join(os.path.basename(p)[3:-4] for p in pkls))
    if args.dry_run:
        print("[dry-run] nothing uploaded.")
        return

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=not args.public, exist_ok=True)
    for p in pkls:
        api.upload_file(path_or_fileobj=p, path_in_repo=os.path.basename(p),
                        repo_id=args.repo, repo_type="model")
    if os.path.exists(CARD):
        api.upload_file(path_or_fileobj=CARD, path_in_repo="README.md",
                        repo_id=args.repo, repo_type="model")
    print(f"published -> https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
