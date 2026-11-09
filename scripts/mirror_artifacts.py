"""Copy the pinned RAG artifacts to a Hugging Face dataset you own, so installs no longer depend on a teammate's Space.

    pip install huggingface_hub
    huggingface-cli login                                 # a token with write access to your account
    python scripts/mirror_artifacts.py your-name/steam-review-rag-artifacts

It downloads (or reuses) the pinned files, verifies every size and checksum, uploads them under outputs/rag/ in
the dataset, and prints the STEAM_ARTIFACT_MIRROR value to set. Because the loader checks the same pins, a mirror
can never serve a different file.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from steam_review_rag.artifacts import FILES, ensure_artifacts, valid_artifact  # noqa: E402


def verified_files(project_dir: Path) -> list[Path]:
    ensure_artifacts(project_dir)
    paths = [project_dir / "outputs" / "rag" / name for name in FILES]
    for path, (size, checksum) in zip(paths, FILES.values()):
        if not valid_artifact(path, size, checksum):
            raise SystemExit(f"{path.name} does not match its pinned checksum; refusing to mirror it")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repo_id", help="the dataset to create or update, e.g. your-name/steam-review-rag-artifacts")
    parser.add_argument("--private", action="store_true", help="create the dataset as private")
    args = parser.parse_args()
    from huggingface_hub import HfApi

    paths = verified_files(ROOT)
    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    for path in paths:
        api.upload_file(path_or_fileobj=str(path), path_in_repo=f"outputs/rag/{path.name}", repo_id=args.repo_id, repo_type="dataset")
        print(f"uploaded {path.name}")
    print(f"\nSet STEAM_ARTIFACT_MIRROR=https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
