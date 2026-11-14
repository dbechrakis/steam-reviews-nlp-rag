import hashlib
import io
import os
import tempfile
import unittest
from pathlib import Path

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


class MirrorTests(unittest.TestCase):
    def test_without_a_mirror_only_the_original_space_is_used(self):
        from unittest import mock

        from steam_review_rag import artifacts
        with mock.patch.dict(os.environ, {"STEAM_ARTIFACT_MIRROR": ""}):
            self.assertEqual(artifacts.sources(), [(artifacts.SOURCE, artifacts.REVISION)])

    def test_a_mirror_is_tried_first_and_a_bad_copy_falls_back_to_the_original(self):
        from unittest import mock

        from steam_review_rag import artifacts
        good = b"x" * 10
        pins = {"rag_config.json": (10, hashlib.sha256(good).hexdigest())}
        asked = []

        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def fake_urlopen(url, timeout):
            asked.append(url)
            return Response(b"y" * 10 if "mirror.example" in url else good)

        env = {"STEAM_ARTIFACT_MIRROR": "https://mirror.example/datasets/me/copy/", "STEAM_ARTIFACT_MIRROR_REVISION": ""}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, env), \
                mock.patch.object(artifacts, "FILES", pins), mock.patch.object(artifacts, "urlopen", fake_urlopen), \
                mock.patch.object(artifacts.time, "sleep"):
            artifacts.ensure_artifacts(Path(tmp))
            self.assertEqual((Path(tmp) / "outputs" / "rag" / "rag_config.json").read_bytes(), good)
        self.assertEqual(asked[0], "https://mirror.example/datasets/me/copy/resolve/main/outputs/rag/rag_config.json")
        self.assertEqual(sum("mirror.example" in u for u in asked), 3)
        self.assertIn(artifacts.SOURCE, asked[-1])
