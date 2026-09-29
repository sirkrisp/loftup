import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from botocore.exceptions import ClientError, ResponseStreamingError
import numpy as np
from PIL import Image
from pycocotools import mask as masks
import torch
from torchvision import transforms as T
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf

from datasets.sa1b_webdataset import (
    FiniteSA1BWebDataset, SA1BWebDataset, StreamingDataLoader, consumer_shards, decode_sample,
    iter_encoded_samples, open_shard, resolve_shards, sample_split, training_sample,
)
from datasets.loaders import create_training_loaders
from scripts.prepare_sa1b_webdataset import write_sample
import tarfile


class StreamingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.mask = np.zeros((8, 12), dtype=np.uint8)
        self.mask[2:6, 4:8] = 1
        rle = masks.encode(np.asfortranarray(self.mask))
        rle['counts'] = rle['counts'].decode()
        self.metadata = {'annotations': [{'id': 1, 'segmentation': rle}], 'original_size': [8, 12]}
        image = io.BytesIO()
        Image.fromarray(np.repeat(self.mask[:, :, None] * 255, 3, axis=2)).save(image, 'JPEG')
        self.jpg = image.getvalue()
        for shard in range(2):
            with tarfile.open(self.root / f'sa1b-{shard:06d}.tar', 'w') as archive:
                for index in range(shard * 100, (shard + 1) * 100):
                    write_sample(archive, (f'sa_{index}', self.jpg, json.dumps(self.metadata).encode()))

    def dataset(self, **kwargs):
        options = dict(shards=self.root, batch_size=2, batches_per_epoch=4,
                       max_masks=3, val_fraction=0.2, shuffle_buffer=0)
        return SA1BWebDataset(**(options | kwargs))

    def interrupted_body(self, fail_after):
        payload = (self.root / 'sa1b-000000.tar').read_bytes()

        class Body(io.BytesIO):
            def read(self, size=-1):
                if self.tell() >= fail_after:
                    # A failed network read may consume bytes without returning them.
                    super().read(127)
                    raise ResponseStreamingError(error=OSError('Connection broken'))
                return super().read(min(size, 512))

        return Body(payload)

    def s3_responses(self, bodies):
        responses = iter(bodies)

        def get_object(**kwargs):
            body = next(responses)
            offset = int(kwargs.get('Range', 'bytes=0-')[6:-1])
            length = len(body.getvalue())
            body.seek(offset)
            response = dict(Body=body, ContentLength=length - offset, ETag='"original"')
            if offset:
                response['ContentRange'] = f'bytes {offset}-{length - 1}/{length}'
            return response

        return get_object

    def test_s3_body_retries_preserve_all_samples_and_close_streams(self):
        path = self.root / 'sa1b-000000.tar'
        expected = list(iter_encoded_samples(path))
        # Fail inside members, including a failure immediately after reopening.
        bodies = [self.interrupted_body(25088), self.interrupted_body(12800),
                  io.BytesIO(path.read_bytes())]
        with patch('datasets.sa1b_webdataset.s3_client') as client, \
                patch('datasets.sa1b_webdataset.time.sleep') as sleep, \
                self.assertLogs('datasets.sa1b_webdataset', level='WARNING'):
            client.return_value.get_object.side_effect = self.s3_responses(bodies)
            actual = list(iter_encoded_samples('s3://bucket/data/one.tar'))
            requests = client.return_value.get_object.call_args_list
        self.assertNotIn('Range', requests[0].kwargs)
        for request in requests[1:]:
            self.assertEqual(request.kwargs['Range'], 'bytes=25088-')
            self.assertEqual(request.kwargs['IfMatch'], '"original"')
        self.assertEqual([s['__key__'] for s in actual], [s['__key__'] for s in expected])
        self.assertEqual([(s['jpg'], s['json']) for s in actual],
                         [(s['jpg'], s['json']) for s in expected])
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 2])
        self.assertTrue(all(body.closed for body in bodies))

    def test_s3_progress_resets_retry_limit_and_backoff(self):
        path = self.root / 'sa1b-000000.tar'
        expected = list(iter_encoded_samples(path))
        # Each pair fails twice at one offset; subsequent connections make progress.
        # More than 20 interruptions across the shard must still recover.
        bodies = [self.interrupted_body(offset)
                  for offset in range(512, 512 * 26, 512) for _ in range(2)]
        bodies.append(io.BytesIO(path.read_bytes()))
        with patch('datasets.sa1b_webdataset.s3_client') as client, \
                patch('datasets.sa1b_webdataset.time.sleep') as sleep, \
                self.assertLogs('datasets.sa1b_webdataset', level='WARNING'):
            client.return_value.get_object.side_effect = self.s3_responses(bodies)
            actual = list(iter_encoded_samples('s3://bucket/data/one.tar'))
        self.assertEqual([(s['__key__'], s['jpg'], s['json']) for s in actual],
                         [(s['__key__'], s['jpg'], s['json']) for s in expected])
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 2] * 25)
        self.assertTrue(all(body.closed for body in bodies))

    def test_s3_body_retry_exhaustion_raises_without_repeating_samples(self):
        bodies = [self.interrupted_body(25088) for _ in range(21)]
        actual = []
        with patch('datasets.sa1b_webdataset.s3_client') as client, \
                patch('datasets.sa1b_webdataset.time.sleep') as sleep, \
                self.assertLogs('datasets.sa1b_webdataset', level='WARNING'):
            client.return_value.get_object.side_effect = self.s3_responses(bodies)
            with self.assertRaises(ResponseStreamingError):
                for sample in iter_encoded_samples('s3://bucket/data/one.tar'):
                    actual.append(sample['__key__'])
        self.assertTrue(actual)
        self.assertEqual(len(actual), len(set(actual)))
        self.assertEqual(sleep.call_count, 20)
        self.assertEqual([call.args[0] for call in sleep.call_args_list],
                         [1, 2, 4, 8, 16] + [30] * 15)
        self.assertTrue(all(body.closed for body in bodies))

    def test_s3_access_errors_are_not_retried(self):
        with patch('datasets.sa1b_webdataset.s3_client') as client, \
                patch('datasets.sa1b_webdataset.time.sleep') as sleep:
            client.return_value.get_object.side_effect = ClientError(
                {'Error': {'Code': 'AccessDenied'}}, 'GetObject')
            with self.assertRaises(ClientError):
                list(iter_encoded_samples('s3://bucket/data/one.tar'))
        sleep.assert_not_called()

    def test_s3_resume_rejects_changed_objects_and_incorrect_ranges(self):
        for field, value in [('ETag', '"changed"'), ('ContentRange', 'bytes 0-9/10'),
                             ('ContentLength', 1)]:
            with self.subTest(field=field):
                bodies = [self.interrupted_body(25088),
                          io.BytesIO((self.root / 'sa1b-000000.tar').read_bytes())]
                get_object = self.s3_responses(bodies)

                def response(**kwargs):
                    result = get_object(**kwargs)
                    if 'Range' in kwargs:
                        result[field] = value
                    return result

                with patch('datasets.sa1b_webdataset.s3_client') as client, \
                        patch('datasets.sa1b_webdataset.time.sleep') as sleep, \
                        self.assertLogs('datasets.sa1b_webdataset', level='WARNING'):
                    client.return_value.get_object.side_effect = response
                    with self.assertRaises(ValueError):
                        list(iter_encoded_samples('s3://bucket/data/one.tar'))
                self.assertEqual(sleep.call_count, 1)
                self.assertTrue(all(body.closed for body in bodies))

    def test_s3_premature_eof_resumes_at_last_delivered_byte(self):
        payload = (self.root / 'sa1b-000000.tar').read_bytes()
        bodies = [io.BytesIO(payload[:25088]), io.BytesIO(payload)]
        get_object = self.s3_responses(bodies)

        def response(**kwargs):
            result = get_object(**kwargs)
            if 'Range' not in kwargs:
                result['ContentLength'] = len(payload)
            return result

        with patch('datasets.sa1b_webdataset.s3_client') as client, \
                patch('datasets.sa1b_webdataset.time.sleep'), \
                self.assertLogs('datasets.sa1b_webdataset', level='WARNING'):
            client.return_value.get_object.side_effect = response
            actual = list(iter_encoded_samples('s3://bucket/data/one.tar'))
            self.assertEqual(client.return_value.get_object.call_args.kwargs['Range'], 'bytes=25088-')
        self.assertEqual([sample['__key__'] for sample in actual], [f'sa_{i}' for i in range(100)])
        self.assertTrue(all(body.closed for body in bodies))

    def test_s3_early_close_closes_body(self):
        body = io.BytesIO((self.root / 'sa1b-000000.tar').read_bytes())
        with patch('datasets.sa1b_webdataset.s3_client') as client:
            client.return_value.get_object.side_effect = self.s3_responses([body])
            samples = iter_encoded_samples('s3://bucket/data/one.tar')
            next(samples)
            samples.close()
        self.assertTrue(body.closed)

    def test_decoding_transform_alignment_padding_and_empty_masks(self):
        sample = next(iter_encoded_samples(str(self.root / 'sa1b-000000.tar')))
        image, metadata = decode_sample(sample)
        self.assertEqual(image.size, (12, 8))
        img_transform = T.Compose([T.Resize(4), T.CenterCrop(4), T.ToTensor()])
        target_transform = T.Compose([T.PILToTensor(), T.Resize(4, T.InterpolationMode.NEAREST), T.CenterCrop(4)])
        result = training_sample(sample, img_transform, target_transform, 3)
        expected = target_transform(Image.fromarray(self.mask)).squeeze(0)
        torch.testing.assert_close(result['label'][0], expected.float())
        self.assertEqual(result['img'].shape, (3, 4, 4))
        self.assertTrue((result['label'][1:] == -1).all())
        sample['json'] = b'{"annotations": [], "original_size": [8, 12]}'
        self.assertTrue((training_sample(sample, img_transform, target_transform, 3)['label'] == -1).all())

    def test_split_is_stable_and_disjoint(self):
        train = {f'sa_{i}' for i in range(200) if sample_split(f'sa_{i}', .2) == 'train'}
        val = {f'sa_{i}' for i in range(200) if sample_split(f'sa_{i}', .2) == 'val'}
        self.assertTrue(train and val)
        self.assertFalse(train & val)
        for split in ('train', 'val'):
            dataset = self.dataset(split=split, batches_per_epoch=2)
            keys = [sample['__key__'] for sample in dataset]
            self.assertTrue(all(sample_split(key, .2) == split for key in keys))
            self.assertEqual(len(keys), len(dataset))

    def test_multiworker_batches_have_exact_epoch_length_without_duplicates(self):
        dataset = self.dataset(batches_per_epoch=5)
        loader = StreamingDataLoader(dataset, batch_size=2, num_workers=3)
        batches = list(loader)
        keys = [key for batch in batches for key in batch['__key__']]
        self.assertEqual(len(batches), len(loader))
        self.assertEqual(len(batches), 5)
        self.assertEqual(len(set(keys)), 10)

    def test_distributed_ranks_do_not_overlap(self):
        keys = []
        for rank in range(3):
            dataset = self.dataset()
            dataset.distributed_context = (rank, 3)
            keys.append({sample['__key__'] for sample in dataset})
        self.assertTrue(all(not keys[a] & keys[b] for a in range(3) for b in range(a)))
        self.assertTrue(all(len(part) == 8 for part in keys))

    def test_cycles_to_fill_epoch_and_validation_repeats_deterministically(self):
        dataset = self.dataset(split='val', batches_per_epoch=100)
        first = [sample['__key__'] for sample in dataset]
        self.assertEqual(len(first), 200)
        self.assertLess(len(set(first)), len(first))
        self.assertEqual(first, [sample['__key__'] for sample in dataset])

    def test_empty_split_fails_instead_of_hanging(self):
        with patch('datasets.sa1b_webdataset.sample_split', return_value='val'):
            with self.assertRaisesRegex(ValueError, 'No train samples'):
                list(self.dataset())

    def test_incomplete_pairs_rejected(self):
        path = self.root / 'broken.tar'
        with tarfile.open(path, 'w') as archive:
            member = tarfile.TarInfo('bad.jpg')
            member.size = len(self.jpg)
            archive.addfile(member, io.BytesIO(self.jpg))
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            list(iter_encoded_samples(path))

    def test_s3_stream_is_closed_without_local_cache(self):
        data = (self.root / 'sa1b-000000.tar').read_bytes()
        stream = io.BytesIO(data)
        with patch('datasets.sa1b_webdataset.s3_client') as client:
            client.return_value.get_object.side_effect = self.s3_responses([stream])
            samples = list(iter_encoded_samples('s3://bucket/data/one.tar', 'https://example.test'))
            client.return_value.get_object.assert_called_once_with(Bucket='bucket', Key='data/one.tar')
        self.assertTrue(stream.closed)
        self.assertEqual(len(samples), 100)

    def test_s3_listing_snapshots_completed_shards(self):
        with patch('datasets.sa1b_webdataset.s3_client') as client:
            client.return_value.get_paginator.return_value.paginate.return_value = [
                {'Contents': [{'Key': 'data/b.tar'}, {'Key': 'data/a.tar'}, {'Key': 'data/a.tar.part'}]}]
            self.assertEqual(resolve_shards('s3://bucket/data'), ['s3://bucket/data/a.tar', 's3://bucket/data/b.tar'])

    def test_finite_epoch_exact_unique_membership_across_ranks_and_workers(self):
        for world in (1, 2, 4):
            all_keys = []
            for rank in range(world):
                dataset = FiniteSA1BWebDataset(shards=self.root, sample_start=0,
                    sample_count=160, samples_per_shard=100, world_size=world,
                    batch_size=2, max_masks=1, shuffle_buffer=5)
                with patch.dict('os.environ', RANK=str(rank), WORLD_SIZE=str(world)):
                    loader = StreamingDataLoader(dataset, batch_size=2, num_workers=3)
                    batches = list(loader)
                self.assertEqual(len(batches), len(loader))
                all_keys.extend(key for batch in batches for key in batch['__key__'])
            self.assertEqual(len(all_keys), 160)
            self.assertEqual(set(all_keys), {f'sa_{i}' for i in range(160)})
        validation = FiniteSA1BWebDataset(shards=self.root, split='val', sample_start=160,
            sample_count=40, samples_per_shard=100, world_size=1, batch_size=1, max_masks=1)
        self.assertEqual({x['__key__'] for x in validation}, {f'sa_{i}' for i in range(160, 200)})

    def test_finite_rejects_insufficient_data_and_uneven_ddp_batches(self):
        for count, message in ((202, 'Not enough'), (159, 'divisible')):
            with self.assertRaisesRegex(ValueError, message):
                FiniteSA1BWebDataset(shards=self.root, sample_start=0, sample_count=count,
                    samples_per_shard=100, world_size=1, batch_size=2)

    def test_finite_short_shard_fails_without_repeating(self):
        dataset = FiniteSA1BWebDataset(shards=self.root, sample_start=0,
            sample_count=102, samples_per_shard=101, world_size=1, batch_size=2, max_masks=1)
        with self.assertRaisesRegex(ValueError, 'fewer than'):
            list(dataset)

    def test_finite_more_workers_than_batches(self):
        dataset = FiniteSA1BWebDataset(shards=self.root, sample_start=0,
            sample_count=2, samples_per_shard=100, world_size=1, batch_size=2, max_masks=1)
        batches = list(StreamingDataLoader(dataset, batch_size=2, num_workers=3))
        self.assertEqual(len(batches), 1)
        self.assertEqual(set(batches[0]['__key__']), {'sa_0', 'sa_1'})

    def test_finite_training_factory_ignores_legacy_batch_budgets(self):
        cfg = OmegaConf.create(dict(dataset='sa1b_webdataset', batch_size=2, num_workers=0,
            num_gpus=1, sa1b_val_fraction=.2, sa1b_split_seed=42,
            webdataset=dict(shards=str(self.root), endpoint=None, train_samples=160,
                            val_samples=40, samples_per_shard=100, train_batches=1,
                            val_batches=1, max_masks=1, shuffle_buffer=0)))
        train, val = create_training_loaders(cfg, T.ToTensor(), T.PILToTensor())
        self.assertEqual((len(train), len(val)), (80, 40))
        train_keys = {key for batch in train for key in batch['__key__']}
        val_keys = {key for batch in val for key in batch['__key__']}
        self.assertEqual((len(train_keys), len(val_keys)), (160, 40))
        self.assertFalse(train_keys & val_keys)

    def test_training_factory(self):
        cfg = OmegaConf.create(dict(dataset='sa1b_webdataset', batch_size=2, num_workers=0,
            sa1b_val_fraction=.2, sa1b_split_seed=42,
            webdataset=dict(shards=str(self.root), endpoint=None, train_batches=2, val_batches=1,
                            max_masks=3, shuffle_buffer=0)))
        train, val = create_training_loaders(cfg, T.ToTensor(), T.PILToTensor())
        self.assertEqual((len(train), len(val)), (2, 1))
        self.assertEqual(next(iter(train))['label'].shape, (2, 3, 8, 12))
        with patch('datasets.sa1b_webdataset.decode_masks', side_effect=AssertionError('validation decoded masks')):
            validation = next(iter(val))
        self.assertEqual(validation['img'].shape, (1, 3, 8, 12))
        self.assertNotIn('label', validation)
        self.assertEqual((train.timeout, val.timeout), (0, 0))
        cfg.num_workers = 2
        train, val = create_training_loaders(cfg, T.ToTensor(), T.PILToTensor())
        self.assertEqual((train.timeout, val.timeout), (600, 600))
        cfg.webdataset.worker_timeout = 120
        train, val = create_training_loaders(cfg, T.ToTensor(), T.PILToTensor())
        self.assertEqual((train.timeout, val.timeout), (120, 120))

    def test_corrupt_sample_error_identifies_shard_and_key(self):
        sample = {'__url__': 's3://bucket/bad.tar', '__key__': 'broken', 'jpg': b'bad', 'json': b'{}'}
        with self.assertRaises(Exception) as raised:
            training_sample(sample, None, None, 1, include_masks=False)
        self.assertIn('s3://bucket/bad.tar::broken', ' '.join(raised.exception.__notes__))

    def test_shard_checker_reads_all_images_and_detects_wrong_count(self):
        import contextlib
        from scripts.check_sa1b_shards import check_shard
        with contextlib.redirect_stdout(io.StringIO()):
            path = str(self.root / 'sa1b-000000.tar')
            self.assertEqual(check_shard(path, expected_samples=100, check_masks=True), 100)
            with self.assertRaisesRegex(ValueError, 'expected 101 samples, found 100'):
                check_shard(path, expected_samples=101)


if __name__ == '__main__':
    unittest.main()
