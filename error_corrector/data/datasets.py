import abc
import logging
import os 
import json
import torch
import tempfile
import warnings
from typing import Any, Dict, List, Optional, Sequence

import datasets as hf_datasets
from pathlib import Path
from datasets import load_from_disk
import jinja2
import numpy as np
import transformers
from torch.utils import data
from error_corrector.data.data_sample import CorrectorSample
from error_corrector.data import data_sample, types, text_proc



def _get_worker_info(length: int):
    """
    Calculate number of samples for this worker, accounting for max workers limit.
    Returns 0 if worker_id exceeds max allowed workers.
    """
    worker_id = 0
    num_workers = 1
    worker_info = data.get_worker_info()
    if worker_info is not None:
        worker_id = worker_info.id
        num_workers = worker_info.num_workers

    # Calculate samples for this worker
    worker_samples = length // num_workers
    extra_samples = length % num_workers

    # Workers with id < extra_samples get one extra sample
    if worker_id < extra_samples:
        worker_samples += 1

    return num_workers, worker_id, worker_samples


class SizedIterableDataset(abc.ABC, data.IterableDataset):
    """
    An abstract IterableDataset that provides a length method
    """
    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @abc.abstractmethod
    def __str__(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass


class EmptyDataset(SizedIterableDataset):
    def __init__(self, length: int = 1) -> None:
        self._length = length

    def __iter__(self):
        return iter([])

    def __len__(self):
        return self._length
    
    def __str__(self):
        return f"EmptyDataset(length={self._length})"

    @property
    def name(self):
        return "empty"


class InterleaveDataset(SizedIterableDataset):
    """Interleaves multiple SizedIterableDataset objects based on normalized weights."""

    def __init__(
        self,
        datasets: Sequence[SizedIterableDataset],
        weights: Optional[Sequence[float]] = None,
    ) -> None:
        """
        Args:
            datasets: A list of SizedIterableDataset objects.
            weights: An optional list of dataset weights, i.e., the number of times it should be repeated.
            seed: Optional seed for reproducibility.
        """
        self._datasets = datasets
        if weights is not None:
            assert len(weights) == len(datasets)
        else:
            weights = [1.0] * len(datasets)
        self._weights = weights
        self._weighted_samples = [int(w * len(d)) for w, d in zip(weights, datasets)]
        self._total_samples = sum(self._weighted_samples)

    def __iter__(self):
        ds_iters = [iter(ds) for ds in self._datasets]
        ds_pos = [0] * len(ds_iters)
        num_workers, worker_id, worker_samples = _get_worker_info(self._total_samples)
        # Find the iterator that is least far along and vend from it.
        for i in range(worker_samples):
            min_fraction = 1.0
            for j in range(len(ds_iters)):
                iter_fraction = ds_pos[j] / self._weighted_samples[j]
                if iter_fraction < min_fraction:
                    min_fraction = iter_fraction
                    iter_index = j
            try:
                yield next(ds_iters[iter_index])
            except StopIteration:
                ds_iters[iter_index] = iter(self._datasets[iter_index])
                try:
                    yield next(ds_iters[iter_index])
                except StopIteration:
                    warnings.warn(
                        f"Dataset {iter_index} is empty for worker {worker_id}/{num_workers}. num_workers is likely too high. Stopping iteration."
                    )
                    break
            ds_pos[iter_index] += 1

    def __len__(self):
        return self._total_samples

    def __str__(self):
        return "+".join([f"{d}:{w:.2f}" for w, d in zip(self._weights, self._datasets)])

    @property
    def name(self):
        return "+".join([ds.name for ds in self._datasets])


class Dataproc(SizedIterableDataset):
    """Base class to preprocess error correction data"""
    def __init__(self, dataset: SizedIterableDataset):
        self._dataset = dataset

    @abc.abstractmethod
    def _process(self, sample: data_sample.CorrectorSample):
        pass

    def __iter__(self):
        for sample in self._dataset:
            yield self._process(sample)

    def __len__(self):
        return len(self._dataset)
    
    def __str__(self):
        return f"Dataproc({str(self._dataset)})"

    @property
    def name(self):
        return self._dataset.name


class Range(SizedIterableDataset):
    """Limits the number of samples from another dataset."""

    def __init__(
        self,
        dataset: SizedIterableDataset,
        num_samples: Optional[int] = None,
    ) -> None:
        self._dataset = dataset
        self._length = num_samples or len(dataset)
        if self._length > len(dataset):
            warnings.warn(
                f"num_samples ({self._length}) exceeds dataset length ({len(dataset)}). Truncating to {len(dataset)}."
            )
            self._length = len(dataset)
        self._name = f"{dataset.name}.{self._length}"

    def __iter__(self):
        num_workers, worker_id, worker_samples = _get_worker_info(self._length)
        if worker_samples == 0:
            return iter([])
        yielded_samples = 0
        try:
            for sample in self._dataset:
                yielded_samples += 1
                yield sample
                if yielded_samples == worker_samples:
                    break
        except Exception as e:
            logging.error(
                f"Worker {worker_id}/{num_workers} failed after yielding {yielded_samples}/{worker_samples} samples, out of {self._length} total samples with error: {e}"
            )
            raise e
        if yielded_samples < worker_samples:
            logging.warn(
                f"Worker {worker_id}/{num_workers} only yielded {yielded_samples} (expected {worker_samples}) samples, out of {self._length} total samples"
            )

    def __str__(self):
        return f"Range({self._dataset}%{len(self)})"

    def __len__(self):
        return self._length

    @property
    def name(self):
        return self._name

    def get_config(self):
        if isinstance(self._dataset, GenericDataset):
            return self._dataset.get_config()
        else:
            raise ValueError("Cannot get config for non-GenericDataset")


class SliceDataset(SizedIterableDataset):
    """Returns a contiguous slice of another SizedIterableDataset.

    Useful for creating train/val splits without mutating the underlying dataset.
    """

    def __init__(
        self,
        dataset: SizedIterableDataset,
        start: int = 0,
        length: Optional[int] = None,
    ) -> None:
        self._dataset = dataset
        dataset_len = len(dataset)
        if start < 0:
            start = max(dataset_len + start, 0)
        if start > dataset_len:
            warnings.warn(
                f"SliceDataset start ({start}) exceeds dataset length ({dataset_len}). Truncating to dataset length."
            )
            start = dataset_len

        slice_len = dataset_len - start if length is None else length
        if slice_len < 0:
            slice_len = 0
        if start + slice_len > dataset_len:
            warnings.warn(
                f"SliceDataset slice [{start}:{start + slice_len}) exceeds dataset length ({dataset_len}). Truncating to {dataset_len - start}."
            )
            slice_len = dataset_len - start

        self._start = start
        self._length = slice_len
        self._name = f"{dataset.name}[{self._start}:{self._start + self._length}]"

    def __iter__(self):
        num_workers, worker_id, worker_samples = _get_worker_info(self._length)
        if worker_samples == 0:
            return iter([])

        # Compute worker-specific starting offset within the slice.
        # Distribute remainder samples to lower worker IDs, mirroring _get_worker_info.
        base = self._length // num_workers
        remainder = self._length % num_workers
        worker_offset = base * worker_id + min(worker_id, remainder)
        
        # The slice region in the global dataset
        global_start = self._start + worker_offset
        global_end = global_start + worker_samples

        # Iterate through the parent dataset and take only samples in [global_start, global_end)
        idx = 0
        yielded = 0
        try:
            for sample in self._dataset:
                if idx >= global_end:
                    break
                if idx >= global_start:
                    yield sample
                    yielded += 1
                idx += 1
        except Exception as e:
            logging.error(
                f"Worker {worker_id}/{num_workers} failed after yielding {yielded}/{worker_samples} samples from slice [{global_start}:{global_end}) with error: {e}"
            )
            raise e

        if yielded < worker_samples:
            logging.warn(
                f"Worker {worker_id}/{num_workers} only yielded {yielded} (expected {worker_samples}) samples from slice [{global_start}:{global_end})"
            )

    def __len__(self):
        return self._length

    def __str__(self):
        return f"Slice({self._dataset}[{self._start}:{self._start + self._length}])"

    @property
    def name(self):
        return self._name


class CoreCorrectorDataset(data.Dataset):
    def __init__(self, samples: List[data_sample.CorrectorSample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        item = {
            "k_best_candidates": s.k_best_candidates,          # List[str]
            "num_candidates": s.num_candidates,                # int
            "continuation_transcript": s.continuation_transcript,  # str
            "previous_transcript": s.previous_transcript,      # str
            "audio_encoder_path": s.audio_encoder_path,        # Optional[str]
            "extra_kwargs": s.extra_kwargs,                    # Optional[dict]
        }

        if s.audio_embed is not None:
            item["audio_embed"] = torch.from_numpy(
                s.audio_embed.astype(np.float32)
            )
        else:
            item["audio_embed"] = None

        return item



def load_corrector_samples(path: str) -> hf_datasets.Dataset:
    """
    Load corrector samples from a JSONL file and return a HuggingFace Dataset.
    Each row is a dict; conversion to CorrectorSample happens later.
    """
    dataset = hf_datasets.load_dataset(
        "json",
        data_files=path,
        split="train",
    )

    return dataset


class CorrectorDataset(SizedIterableDataset):
    def __init__(self, args: types.CorrectorDatasetArgs):
        super().__init__()
        self._args = args
        self._rng = np.random.default_rng(self._args.shuffle_seed)
        self._name = "[unset]"
        self._length = -1

    def _init_dataset(
        self,
        dataset: data.Dataset,
        name: str,
        num_samples: int,
    ):
        self._dataset = dataset
        self._name = name
        self._length = num_samples

    def __len__(self):
        return self._length

    @property
    def name(self):
        return self._name

    def _load_dataset(
        self,
        path: str,
        name: Optional[str] = None,
        *,
        split: Optional[str] = None,
        batch_size: int = 1,
    ):
        dataset = load_corrector_samples(path)
        if self._args.shuffle:
            dataset = dataset.shuffle(seed=self._args.shuffle_seed)
        return dataset

    def __iter__(self):
        num_workers, _, _ = _get_worker_info(self._length)
        if num_workers > 1:
            assert hasattr(
                self._dataset, "n_shards"
            ), f"{self._name} does not have n_shards attribute, which is required when num_workers ({num_workers}) > 1"
            assert (
                self._dataset.n_shards >= num_workers
            ), f"{self._name} has {self._dataset.n_shards} shards, which is less than the number of workers ({num_workers})."

        actual_length = 0
        skipped_samples = 0
        bad_samples = 0
        dataset_iter = iter(self._dataset)
        for row in dataset_iter:
            actual_length += 1
            sample = self._get_sample(row)
            if sample is None:
                print(f"Sample is None in dataset {self.name} for row {row}")
                bad_samples += 1
                continue

            if self._args.include_audio_embed:
                if sample.audio_embed is None:
                    print(f"Audio embed is None for sample {sample}")
                    bad_samples += 1
                    continue
                if sample.audio_embed.shape[-1] == 0:
                    print(f"Audio embed length is 0 for sample {sample}")
                    bad_samples += 1
                    continue
            yield sample

        logging.info(
            f"Extracted {actual_length} samples from {self.name} (total: {len(self)}), removed {bad_samples} bad samples."
        )

    def _get_sample(
        self, row: transformers.BatchFeature
    ) -> Optional[data_sample.CorrectorSample]:
        audio_embed = self._get_audio_embed(row)

        return self._make_sample(
            k_best_candidates=row["k_best_candidates"],
            num_candidates=row["num_candidates"],
            chunk_size=row["chunk_size"],
            previous_transcript=row["previous_transcript"],
            continuation_transcript=row.get("continuation_transcript"),
            audio_embed=audio_embed,
            extra_kwargs=row.get("extra_kwargs"),
        )

    def _get_audio_embed(
        self,
        row: transformers.BatchFeature,
        column_name: Optional[str] = "audio_embed_path",
    ):
        if not self._args.include_audio_embed:
            return None

        path = row.get(column_name)
        if path is None:
            return None

        path = os.path.join("error_corrector/data/sample_custom_data", path)
        return np.load(path)

    def _make_sample(
        self,
        k_best_candidates: List[str],
        num_candidates: int,
        chunk_size: int,
        previous_transcript: str,
        audio_embed: Optional[np.ndarray] = None,
        continuation_transcript: str = None,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if not self._args.include_audio_embed:
            return data_sample.CorrectorSample(
                k_best_candidates=k_best_candidates,
                num_candidates=num_candidates,
                chunk_size=chunk_size,
                audio_embed=None,
                previous_transcript=previous_transcript,
                continuation_transcript=continuation_transcript,
                extra_kwargs=extra_kwargs,
            )
        return data_sample.CorrectorSample(
            k_best_candidates=k_best_candidates,
            num_candidates=num_candidates,
            chunk_size=chunk_size,
            audio_embed=audio_embed,
            previous_transcript=previous_transcript,
            continuation_transcript=continuation_transcript,
            extra_kwargs=extra_kwargs,
        )


class GenericDataset(CorrectorDataset):
    def __init__(
        self,
        args: types.CorrectorDatasetArgs,
        config: types.DatasetConfig,
    ):
        assert config.splits is not None
        assert config.path is not None
        assert config.batch_size is not None
        super().__init__(args)
        self._config = config
        dsets = []
        total_samples = 0
        for split in config.splits:
            if split.split == self._args.split:
                ds = self._load_dataset(
                    config.path,
                    name=config.subset,
                    split=split.name,
                    batch_size=config.batch_size,
                )
                dsets.append(ds)
                total_samples += split.num_samples
        assert (
            len(dsets) > 0
        ), f"The {config.name} dataset has no {self._args.split} splits."
        dataset = ds if len(dsets) == 1 else hf_datasets.concatenate_datasets(dsets)

        dataset_name = f"{config.name}.{self._args.split.value}"

        super()._init_dataset(dataset, dataset_name, total_samples)

    def __str__(self):
        return f"GenericDataset{self._config}"

    # def _get_sample(self, row) -> Optional[data_sample.CorrectorSample]:
    #     # Setting up extra kwargs
    #     extra_kwargs = None
    #     if (
    #         self._config.eval_config is not None
    #         and self._config.eval_config.extra_kwargs_map is not None
    #     ):
    #         extra_kwargs = {
    #             key: row.get(value)
    #             for key, value in self._config.eval_config.extra_kwargs_map.items()
    #         }
        
    #     pass


class CorrectorDummyDataset(GenericDataset):
    def __init__(self, args: types.CorrectorDatasetArgs) -> None:
        CorrectorDataset.__init__(self, args)
        # This dataset doesn't support streaming.
        dataset = None
        # self._init_dataset(dataset, "dummy", 73)

    def __len__(self):
        return 1

    def __str__(self):
        return "CorrectorDummyDataset"

    @property
    def name(self):
        return "dummy"

    def get_config(self):
        return types.DatasetConfig(
            name="dummy",
            path="hf-internal-testing/librispeech_asr_dummy",
        )

    def _get_sample(
        self, row: transformers.BatchFeature
    ) -> Optional[data_sample.CorrectorSample]:
        text = text_proc.format_asr_text(row["text"])
        user_content = "Transcribe\n"
        user_content += (
            types.AUDIO_PLACEHOLDER if self._args.include_audio else f'"{text}"'
        )
        return self._make_sample(
            self._make_messages(user_content, text),
            # some of our test models that use this dataset can only handle up to 4 seconds of audio
            self._get_audio(row, "audio")[: 4 * data_sample.SAMPLE_RATE],
            audio_transcript=text,
        )