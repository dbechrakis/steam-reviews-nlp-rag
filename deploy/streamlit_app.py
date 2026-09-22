"""Community Cloud entry point with notebook dependencies kept isolated."""

import os
from pathlib import Path
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "90")
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))

from steam_review_rag.app import main  # noqa: E402


main(root)
