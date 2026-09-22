import contextlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from scripts import download_sa1b_1m_subset as download


class Response(io.BytesIO):
    def __init__(self, data):
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data))}


def archive(stem, include_json=True):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w") as tar:
        for extension in (["jpg", "json"] if include_json else ["jpg"]):
            contents = b"test-image" if extension == "jpg" else b'{"annotations": []}'
            member = tarfile.TarInfo(f"nested/{stem}.{extension}")
            member.size = len(contents)
            tar.addfile(member, io.BytesIO(contents))
    return data.getvalue()


class DownloadTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "sa1b"
        self.links = self.root / "links.tsv"
        # Input order must not determine shard order.
        self.links.write_text("file_name\tcdn_link\nsa_000001.tar\thttps://test/1\nsa_000000.tar\thttps://test/0\n")
        self.calls = []

    def open_url(self, request, **kwargs):
        self.calls.append(request.full_url)
        index = request.full_url.rsplit("/", 1)[1]
        return Response(archive(f"sa_{index}"))

    def run_download(self, count):
        argv = ["download", "--num-tars", str(count), "--links-file", str(self.links),
                "--output-dir", str(self.output), "--retries", "0"]
        with patch("sys.argv", argv), patch.object(download, "urlopen", side_effect=self.open_url), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            download.main()

    def test_order_complete_shards_resume_and_extend(self):
        self.run_download(1)
        self.assertEqual(self.calls, ["https://test/0"])
        self.run_download(1)
        self.assertEqual(len(self.calls), 1)
        self.run_download(2)
        self.assertEqual(self.calls, ["https://test/0", "https://test/1"])
        self.assertEqual(len(list(self.output.glob("*.jpg"))), 2)
        self.assertEqual(len(list(self.output.glob("*.json"))), 3)  # includes state
        state = json.loads((self.output / download.STATE_NAME).read_text())
        self.assertEqual(state["completed"], ["sa_000000.tar", "sa_000001.tar"])
        self.assertIsNone(state["in_progress"])
        with self.assertRaisesRegex(ValueError, "beyond"):
            self.run_download(1)

    def test_partial_shard_resume_preserves_saved_file(self):
        self.output.mkdir()
        saved = self.output / "sa_0.jpg"
        saved.write_bytes(b"test-image")
        before = saved.stat().st_mtime_ns
        download.save_state(self.output / download.STATE_NAME,
                            {"version": 2, "selection": "first-tars", "completed": [], "in_progress": "sa_000000.tar"})
        self.run_download(1)
        self.assertEqual(saved.stat().st_mtime_ns, before)
        self.assertTrue((self.output / "sa_0.json").exists())

    def test_old_shuffled_state_rejected_without_network(self):
        self.output.mkdir()
        old = {"seed": 42, "completed": [], "shards": ["sa_000900.tar"]}
        download.save_state(self.output / download.STATE_NAME, old)
        with self.assertRaisesRegex(ValueError, "old shuffled"):
            self.run_download(1)
        self.assertEqual(self.calls, [])
        self.assertEqual(json.loads((self.output / download.STATE_NAME).read_text()), old)

    def test_missing_shard_fails_before_downloading(self):
        with self.assertRaisesRegex(ValueError, "sa_000002.tar"):
            self.run_download(3)
        self.assertEqual(self.calls, [])

    def test_incomplete_archive_is_not_marked_complete(self):
        self.open_url = lambda *args, **kwargs: Response(archive("sa_0", include_json=False))
        with self.assertRaisesRegex(RuntimeError, "lack a JPEG or JSON"):
            self.run_download(1)
        state = json.loads((self.output / download.STATE_NAME).read_text())
        self.assertEqual(state["completed"], [])
        self.assertEqual(state["in_progress"], "sa_000000.tar")


if __name__ == "__main__":
    unittest.main()
