import hashlib
from pathlib import Path
import tempfile
import unittest

from steam_review_rag.artifacts import valid_artifact


class ArtifactIntegrityTests(unittest.TestCase):
    def test_valid_artifact_accepts_matching_sha256(self):
        content = b"matching retrieval artifact"
        checksum = hashlib.sha256(content).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.bin"
            path.write_bytes(content)
            self.assertTrue(valid_artifact(path, len(content), checksum))

    def test_valid_artifact_rejects_wrong_size_or_checksum(self):
        content = b"retrieval artifact"
        checksum = hashlib.sha256(content).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.bin"
            path.write_bytes(content)
            self.assertFalse(valid_artifact(path, len(content) + 1, checksum))
            self.assertFalse(valid_artifact(path, len(content), "0" * 64))


if __name__ == "__main__":
    unittest.main()
