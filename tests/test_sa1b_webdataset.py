import contextlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from pycocotools import mask as masks
from botocore.exceptions import ClientError

from scripts import prepare_sa1b_webdataset as prep


class FakeS3:
    def __init__(self, fail=False):
        self.objects = {}
        self.fail = fail
        self.threads = []

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        return self.objects[Key]

    def upload_file(self, filename, bucket, key, ExtraArgs, Config, Callback=None):
        self.threads.append(threading.current_thread().name)
        if self.fail:
            raise OSError("network unavailable")
        if Callback:
            Callback(Path(filename).stat().st_size)
        self.objects[key] = {"ContentLength": Path(filename).stat().st_size,
                             "Metadata": ExtraArgs["Metadata"]}


class PreparationTests(unittest.TestCase):
    def test_service_error_reports_status_without_credentials_or_urls(self):
        error = ClientError({
            "Error": {"Code": "AccessDenied", "Message": "Denied secret-value https://example.test/?token=private"},
            "ResponseMetadata": {"HTTPStatusCode": 403},
        }, "HeadBucket")
        with patch.dict(prep.os.environ, {"B2_APPLICATION_KEY": "secret-value"}, clear=True):
            message = prep.client_error_details(error)
        self.assertIn("HeadBucket: HTTP 403, AccessDenied", message)
        self.assertNotIn("secret-value", message)
        self.assertNotIn("token=private", message)

    def test_b2_credentials_override_aws_pair(self):
        with patch.dict(prep.os.environ, {
            "B2_APPLICATION_ID": "b2-id", "B2_APPLICATION_KEY": "b2-key",
            "AWS_ACCESS_KEY_ID": "aws-id", "AWS_SECRET_ACCESS_KEY": "aws-key",
        }, clear=True):
            self.assertEqual(prep.b2_credentials(), {
                "aws_access_key_id": "b2-id", "aws_secret_access_key": "b2-key",
            })

    def test_partial_b2_credentials_fail_instead_of_using_another_account(self):
        for name in ("B2_APPLICATION_ID", "B2_APPLICATION_KEY"):
            with self.subTest(name=name), patch.dict(prep.os.environ, {name: "placeholder"}, clear=True):
                with self.assertRaisesRegex(ValueError, "Set both"):
                    prep.b2_credentials()

    def test_no_b2_credentials_preserves_default_lookup(self):
        with patch.dict(prep.os.environ, {}, clear=True):
            self.assertEqual(prep.b2_credentials(), {})

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def make_pair(self, stem="sa_1", shape=(12, 24)):
        mask = np.zeros(shape, dtype=np.uint8)
        mask[2:10, 4:20] = 1
        encoded = masks.encode(np.asfortranarray(mask))
        area = int(masks.area(encoded))
        bbox = masks.toBbox(encoded).tolist()
        encoded["counts"] = encoded["counts"].decode()
        image = self.root / f"{stem}.jpg"
        Image.fromarray(np.repeat((mask * 255)[:, :, None], 3, axis=2)).save(image)
        annotation = image.with_suffix(".json")
        annotation.write_text(json.dumps({"image": {"height": shape[0], "width": shape[1]},
                                         "annotations": [{"id": 7, "segmentation": encoded,
                                                           "area": area, "bbox": bbox}]}))
        return image, annotation, mask

    def args(self, **kwargs):
        defaults = dict(work_dir=self.root, size=896, jpeg_quality=95,
                        samples_per_shard=2, bucket="test",
                        endpoint="https://s3.us-west-004.backblazeb2.com", prefix="data",
                        final_shard="retain", upload_workers=2, max_pending_uploads=2,
                        retries=0, keep_local=False, timeout=10)
        return SimpleNamespace(**(defaults | kwargs))

    def test_image_resizes_but_masks_pass_through_at_original_resolution(self):
        image, annotation, original = self.make_pair()
        source = json.loads(annotation.read_text())
        key, jpg, metadata, timing = prep.encode_pair(image.stem, image.read_bytes(), annotation.read_bytes(), size=6)
        self.assertEqual(timing["num_annotations"], 1)
        self.assertGreaterEqual(timing["image_seconds"], 0)
        self.assertGreaterEqual(timing["mask_seconds"], 0)
        self.assertEqual(key, "sa_1")
        self.assertEqual(Image.open(io.BytesIO(jpg)).size, (12, 6))  # image is resized
        result = json.loads(metadata)
        annotation = result["annotations"][0]
        source_annotation = source["annotations"][0]
        # Masks are passed through unresized: training resizes them itself.
        self.assertEqual(annotation["segmentation"], source_annotation["segmentation"])
        self.assertEqual(annotation["area"], source_annotation["area"])
        self.assertEqual(annotation["bbox"], source_annotation["bbox"])
        np.testing.assert_array_equal(masks.decode(annotation["segmentation"]), original)
        self.assertEqual(result["original_size"], [12, 24])

    def test_small_images_are_not_enlarged(self):
        image, annotation, _ = self.make_pair()
        _, jpg, metadata, _ = prep.encode_pair(image.stem, image.read_bytes(), annotation.read_bytes())
        self.assertEqual(Image.open(io.BytesIO(jpg)).size, (24, 12))
        self.assertEqual(json.loads(metadata)["annotations"][0]["segmentation"]["size"], [12, 24])

    def fake_samples(self, args, links, cursor, pool=None):
        timing = {"image_seconds": 0.0, "mask_seconds": 0.0, "num_annotations": 0}
        for source in range(cursor[0], len(links)):
            for offset in range(cursor[1] if source == cursor[0] else 0, 5):
                sample = (f"sa_{source * 5 + offset}", b"jpg", b'{"annotations":[]}')
                yield (sample, timing), [source, offset + 1]

    def test_equal_shards_remainder_resume_and_extend(self):
        args = self.args()
        links = [("sa_000000.tar", "unused")]
        client = FakeS3()
        with patch.object(prep, "iter_samples", self.fake_samples), contextlib.redirect_stdout(io.StringIO()):
            prep.run(args, links, client)
            state = json.loads((self.root / "state.json").read_text())
            self.assertEqual([x["samples"] for x in state["shards"]], [2, 2])
            self.assertEqual(len(client.objects), 2)
            self.assertFalse(list((self.root / "shards").glob("*.tar")))
            with tarfile.open(self.root / "shards/remainder.tar.pending") as archive:
                self.assertEqual(archive.getnames(), ["sa_4.jpg", "sa_4.json"])
            with patch.object(prep, "iter_samples", side_effect=AssertionError("must not redownload")):
                prep.run(args, links, client)
            prep.run(args, links + [("sa_000001.tar", "unused")], client)
        state = json.loads((self.root / "state.json").read_text())
        self.assertEqual([x["samples"] for x in state["shards"]], [2] * 5)
        self.assertFalse((self.root / "shards/remainder.tar.pending").exists())
        self.assertTrue(all(name != "MainThread" for name in client.threads))

    def test_upload_failure_retains_tar_and_resume_uploads_it(self):
        args = self.args(final_shard="keep")
        links = [("sa_000000.tar", "unused")]
        with patch.object(prep, "iter_samples", self.fake_samples), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "Upload failed"):
                prep.run(args, links, FakeS3(fail=True))
            self.assertTrue(list((self.root / "shards").glob("*.tar")))
            client = FakeS3()
            prep.run(args, links, client)
        state = json.loads((self.root / "state.json").read_text())
        self.assertEqual([x["samples"] for x in state["shards"]], [2, 2, 1])
        self.assertEqual(len(client.objects), 3)

    def test_remote_collision_does_not_overwrite_or_delete_local(self):
        path = self.root / "test.tar"
        path.write_bytes(b"payload")
        client = FakeS3()
        client.objects["data/test.tar"] = {"ContentLength": 7, "Metadata": {"sha256": "different"}}
        uploads = prep.BackgroundUploads(self.args(), client)
        uploads.submit(path)
        with self.assertRaises(FileExistsError):
            uploads.close()
        self.assertTrue(path.exists())
        self.assertEqual(client.threads, [])

    def test_interrupted_producer_replays_only_uncommitted_samples(self):
        args = self.args(final_shard="keep")
        links = [("sa_000000.tar", "unused")]
        def interrupted(args, links, cursor, pool=None):
            for index, sample in enumerate(self.fake_samples(args, links, cursor)):
                if index == 3:
                    raise OSError("interrupted source")
                yield sample
        with contextlib.redirect_stdout(io.StringIO()):
            with patch.object(prep, "iter_samples", interrupted), self.assertRaises(OSError):
                prep.run(args, links, None)
            with patch.object(prep, "iter_samples", self.fake_samples):
                prep.run(args, links, None)
        keys = []
        for path in sorted((self.root / "shards").glob("*.tar")):
            with tarfile.open(path) as archive:
                keys.extend(name for name in archive.getnames() if name.endswith(".jpg"))
        self.assertEqual(keys, [f"sa_{i}.jpg" for i in range(5)])

    def archive_response(self, stems):
        data = io.BytesIO()
        with tarfile.open(fileobj=data, mode="w") as archive:
            for stem in stems:
                image, annotation, _ = self.make_pair(stem)
                archive.add(annotation, arcname=f"nested/{annotation.name}")
                archive.add(image, arcname=f"nested/{image.name}")
        class Response(io.BytesIO):
            headers = {}
        return lambda *args, **kwargs: Response(data.getvalue())

    def test_upload_starts_before_source_download_finishes(self):
        response = self.archive_response(["sa_3", "sa_1", "sa_2"])
        upload_started = threading.Event()
        download_continued = threading.Event()
        original = prep.stream_pairs

        def stream(*args, **kwargs):
            for item in original(*args, **kwargs):
                yield item
                if item[0][0] == "sa_3":
                    self.assertTrue(upload_started.wait(5), "upload waited for source EOF")
                    download_continued.set()

        class OverlappingS3(FakeS3):
            def upload_file(inner, *args, **kwargs):
                upload_started.set()
                if not download_continued.wait(5):
                    raise AssertionError("download did not overlap upload")
                super().upload_file(*args, **kwargs)

        client = OverlappingS3()
        with patch.object(prep.download, "urlopen", side_effect=response), \
                patch.object(prep, "stream_pairs", side_effect=stream), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            prep.run(self.args(samples_per_shard=1), [("sa_000000.tar", "https://test")], client)
        self.assertEqual(len(client.objects), 3)
        self.assertTrue(download_continued.is_set())
        self.assertFalse((self.root / "source").exists())

    def test_stream_retry_keeps_completion_order_without_duplicates(self):
        response = self.archive_response(["sa_3", "sa_1", "sa_2"])
        original = prep.stream_pairs
        failed = False

        def stream(*args, **kwargs):
            nonlocal failed
            for item in original(*args, **kwargs):
                yield item
                if not failed:
                    failed = True
                    raise OSError("connection lost after first pair")

        with patch.object(prep.download, "urlopen", side_effect=response), \
                patch.object(prep, "stream_pairs", side_effect=stream), \
                contextlib.redirect_stderr(io.StringIO()):
            samples = list(prep.iter_samples(self.args(retries=1), [("sa_000000.tar", "https://test")], [0, 0]))
        self.assertEqual([sample[0][0][0] for sample in samples], ["sa_3", "sa_1", "sa_2"])
        self.assertEqual([sample[1] for sample in samples], [[0, 1], [0, 2], [0, 3]])

    def test_stream_resume_replays_archive_without_staging(self):
        response = self.archive_response(["sa_3", "sa_1", "sa_2"])
        args = self.args(max_pending_pairs=1)
        links = [("sa_000000.tar", "https://test")]
        with patch.object(prep.download, "urlopen", side_effect=response), \
                contextlib.redirect_stderr(io.StringIO()):
            samples = prep.iter_samples(args, links, [0, 0])
            self.assertEqual(next(samples)[0][0][0], "sa_3")
            samples.close()  # Also checks cancellation with a bounded/full queue.
            resumed = list(prep.iter_samples(args, links, [0, 1]))
        self.assertEqual([sample[0][0][0] for sample in resumed], ["sa_1", "sa_2"])
        self.assertFalse(any(t.name == "sa1b-download" for t in threading.enumerate()))

    def test_legacy_checkpoint_finishes_current_source_then_streams(self):
        first = self.archive_response(["sa_1", "sa_3", "sa_2"])
        second = self.archive_response(["sa_6", "sa_4", "sa_5"])
        args = self.args(samples_per_shard=1, final_shard="keep")
        links = [("sa_000000.tar", "https://first")]
        with patch.object(prep.download, "urlopen", side_effect=first), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            prep.run(args, links, None)
        state_path = self.root / "state.json"
        state = json.loads(state_path.read_text())
        state.update(version=1, cursor=[0, 1], finished=False, shards=state["shards"][:1])
        prep.download.save_state(state_path, state)
        for path in sorted((self.root / "shards").glob("*.tar"))[1:]:
            path.unlink()

        def response(request, **kwargs):
            return (first if request.full_url == "https://first" else second)()

        with patch.object(prep.download, "urlopen", side_effect=response), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            prep.run(args, [("sa_000000.tar", "https://first"),
                            ("sa_000001.tar", "https://second")], None)
        keys = []
        for path in sorted((self.root / "shards").glob("*.tar")):
            with tarfile.open(path) as archive:
                keys.extend(name for name in archive.getnames() if name.endswith(".jpg"))
        self.assertEqual(keys, [f"sa_{i}.jpg" for i in [1, 3, 2, 6, 4, 5]])
        self.assertEqual(json.loads(state_path.read_text())["version"], 2)

    def test_stream_failure_propagates_without_staging(self):
        response = self.archive_response(["sa_3", "sa_1"])
        original = prep.stream_pairs

        def stream(*args, **kwargs):
            for item in original(*args, **kwargs):
                yield item
                raise OSError("connection lost")

        with patch.object(prep.download, "urlopen", side_effect=response), \
                patch.object(prep, "stream_pairs", side_effect=stream), \
                contextlib.redirect_stderr(io.StringIO()):
            samples = prep.iter_samples(self.args(), [("sa_000000.tar", "https://test")], [0, 0])
            self.assertEqual(next(samples)[0][0][0], "sa_3")
            with self.assertRaisesRegex(RuntimeError, "Download failed"):
                next(samples)
        self.assertFalse((self.root / "source").exists())

    def member_response(self, members):
        data = io.BytesIO()
        with tarfile.open(fileobj=data, mode="w") as archive:
            for name, payload in members:
                member = tarfile.TarInfo(name)
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
        class Response(io.BytesIO):
            headers = {}
        return lambda *args, **kwargs: Response(data.getvalue())

    def test_nonadjacent_pairs_and_replay_boundary(self):
        response = self.member_response([
            ("nested/sa_2.jpg", b"image2"), ("sa_1.json", b"json1"),
            ("sa_1.jpg", b"image1"), ("sa_2.json", b"json2"),
            ("sa_1.jpg", b"duplicate"), ("unrelated.txt", b"ignored"),
        ])
        with patch.object(prep.download, "urlopen", side_effect=response), \
                contextlib.redirect_stderr(io.StringIO()):
            pairs = list(prep.stream_pairs(self.args(), "https://test", "test", threading.Event(), skip=1))
        self.assertEqual(pairs, [(("sa_2", b"image2", b"json2"), 2)])
        self.assertFalse((self.root / "source").exists())

    def test_pair_buffer_limit_fails_without_disk_spill(self):
        response = self.member_response([("sa_1.jpg", b"x" * 20), ("sa_1.json", b"{}")])
        with patch.object(prep.download, "urlopen", side_effect=response), \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "max-buffer-mb"):
                list(prep.stream_pairs(self.args(max_buffer_mb=0.00001), "https://test", "test", threading.Event()))
        self.assertEqual(list(self.root.iterdir()), [])

    def test_missing_pair_member_fails_without_marking_finished(self):
        response = self.member_response([("sa_1.jpg", b"image")])
        with patch.object(prep.download, "urlopen", side_effect=response), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                prep.run(self.args(), [("sa_000000.tar", "https://test")], None)
        state = json.loads((self.root / "state.json").read_text())
        self.assertFalse(state["finished"])
        self.assertEqual(state["cursor"], [0, 0])
        self.assertFalse((self.root / "source").exists())

    def test_dashboard_reports_real_upload_queue_progress_and_completion(self):
        from rich.console import Console
        dashboard = prep.Dashboard("s3://test/data")
        dashboard.fill("sa1b-000002.tar", 375, 1000)
        started = threading.Event()
        release = threading.Event()
        test = self

        class BlockingS3(FakeS3):
            def upload_file(inner, filename, bucket, key, ExtraArgs, Config, Callback):
                if key.endswith("first.tar"):
                    Callback(4)
                    started.set()
                    test.assertTrue(release.wait(5))
                super().upload_file(filename, bucket, key, ExtraArgs, Config, Callback)

        first, second = self.root / "first.tar", self.root / "second.tar"
        first.write_bytes(b"12345678")
        second.write_bytes(b"12345678")
        uploads = prep.BackgroundUploads(self.args(dashboard=dashboard, upload_workers=1), BlockingS3())
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                uploads.submit(first)
                self.assertTrue(started.wait(5))
                uploads.submit(second)
                output = io.StringIO()
                Console(file=output, width=110).print(dashboard.render())
                rendered = output.getvalue()
                self.assertIn("1 active | 1 queued", rendered)
                self.assertIn("375/1,000 images (37.5%)", rendered)
                self.assertIn("50.0%", rendered)
                self.assertIn("s3://test/data", rendered)
                self.assertIn("first.tar", rendered)
                self.assertIn("second.tar", rendered)
            finally:
                release.set()
                uploads.close()
        self.assertEqual(dashboard.uploaded, 2)
        self.assertEqual(dashboard.uploads, {})
        self.assertEqual(list(dashboard.recent), ["first.tar", "second.tar"])

    def test_dashboard_marks_failed_upload_without_counting_it_uploaded(self):
        dashboard = prep.Dashboard("s3://test/data")
        path = self.root / "test.tar"
        path.write_bytes(b"payload")
        uploads = prep.BackgroundUploads(self.args(dashboard=dashboard), FakeS3(fail=True))
        uploads.submit(path)
        with self.assertRaisesRegex(RuntimeError, "Upload failed"):
            uploads.close()
        self.assertEqual(dashboard.uploads["test.tar"]["status"], "Failed")
        self.assertEqual(dashboard.uploaded, 0)
        self.assertTrue(path.exists())

    def test_live_dashboard_restores_terminal_on_failure(self):
        from rich.console import Console
        output = io.StringIO()
        console = Console(file=output, force_terminal=True, width=100)
        dashboard = prep.Dashboard("Local only", enabled=True, console=console)
        with self.assertRaisesRegex(RuntimeError, "example"):
            with dashboard:
                dashboard.fill("sa1b-000000.tar", 500, 1000)
                dashboard.live.refresh()
                raise RuntimeError("example")
        self.assertFalse(dashboard.live.is_started)
        self.assertEqual(dashboard.status, "Failed")
        self.assertIn("50.0%", output.getvalue())
        self.assertIn("Failed", output.getvalue())

    def test_download_convert_pipeline_with_synthetic_archive(self):
        image, annotation, _ = self.make_pair()
        data = io.BytesIO()
        with tarfile.open(fileobj=data, mode="w") as archive:
            archive.add(annotation, arcname="nested/sa_1.json")
            archive.add(image, arcname="nested/sa_1.jpg")
        class Response(io.BytesIO):
            headers = {}
        args = self.args(samples_per_shard=1000, final_shard="keep")
        with patch.object(prep.download, "urlopen", return_value=Response(data.getvalue())), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            prep.run(args, [("sa_000000.tar", "https://test")], None)
        with tarfile.open(self.root / "shards/sa1b-000000.tar") as archive:
            metadata = json.load(archive.extractfile("sa_1.json"))
            self.assertEqual(metadata["image"]["height"], 12)
            self.assertEqual(len(metadata["annotations"]), 1)


if __name__ == "__main__":
    unittest.main()
