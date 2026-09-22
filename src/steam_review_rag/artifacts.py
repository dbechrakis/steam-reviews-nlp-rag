"""Download and integrity-check the original team's public RAG artifacts."""

import hashlib
import os
from pathlib import Path
import tempfile
import time
from urllib.request import urlopen


REVISION = "c9a4d43b9422edde3fe85b20cc8f4645e258ac02"
SOURCE = "https://huggingface.co/spaces/Nebuchedeser/steam-game-review-explorer"
FILES = {
    "faiss_index.bin": (
        63_237_165,
        "a50c5a1abaa3697d2ed07628e2d8f64c046583ad6d483a498be37f1217180836",
    ),
    "rag_corpus.csv": (
        46_764_404,
        "0cd69805eac314790744c1baa611d54fe879f6e78db4ea221e9cfe5db55395b4",
    ),
    "rag_config.json": (900, "ddfec331f0a5573faa2558a542539b9e841c8dec"),
}


def valid_artifact(path: Path, size: int, expected_checksum: str) -> bool:
    """Return whether an artifact matches its pinned size and checksum."""
    if not path.exists() or path.stat().st_size != size:
        return False

    digest = hashlib.sha1() if len(expected_checksum) == 40 else hashlib.sha256()
    if len(expected_checksum) == 40:
        digest.update(f"blob {size}\0".encode())
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == expected_checksum


def ensure_artifacts(project_dir: Path) -> None:
    """Ensure the pinned matching corpus, configuration, and index are present."""
    target = project_dir / "outputs" / "rag"
    target.mkdir(parents=True, exist_ok=True)

    for name, (size, checksum) in FILES.items():
        destination = target / name
        if valid_artifact(destination, size, checksum):
            continue

        for attempt in range(3):
            temporary: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(dir=target, delete=False) as output:
                    temporary = Path(output.name)
                    url = f"{SOURCE}/resolve/{REVISION}/outputs/rag/{name}"
                    with urlopen(url, timeout=90) as response:
                        downloaded = 0
                        while chunk := response.read(1024 * 1024):
                            downloaded += len(chunk)
                            if downloaded > size:
                                raise ValueError("Artifact exceeds expected size")
                            output.write(chunk)

                if not valid_artifact(temporary, size, checksum):
                    raise ValueError(f"Integrity check failed for {name}")
                os.replace(temporary, destination)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)


def main() -> None:
    """Verify/download artifacts for the repository containing the package."""
    project_dir = Path(__file__).resolve().parents[2]
    ensure_artifacts(project_dir)
    print("All three original RAG artifacts verified.")


if __name__ == "__main__":
    main()
