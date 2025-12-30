import dataclasses
import enum
import json
import logging
from typing import Any, Dict, List, Optional
from simple_parsing import helpers

AUDIO_PLACEHOLDER = "<|audio|>"

ERROR_CORRECTOR_TEMPLATE = (
    "You are an ASR error corrector. Given the audio, the previous transcription, and N candidate "
    "continuations, respond with ONLY the corrected appended text for the transcript.\n\n"
    "Previous transcription:\n{prev_display}\n\n"
    "Candidate continuations:\n{prompt_lines}"
    "\n\nCorrected appended text:"
)


class DatasetSplit(str, enum.Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclasses.dataclass
class DatasetOptions:
    name: str
    weight: float = 1.0


@dataclasses.dataclass
class CorrectorDatasetArgs:
    """Global arguments for train/val/test dataset creation."""

    split: DatasetSplit = DatasetSplit.TRAIN
    """Which split of the dataset to use."""
    include_audio_embed: bool = True
    """Whether to include audio embeddings in the data samples."""
    shuffle: bool = False
    """Whether to shuffle the dataset."""
    shuffle_seed: int = 21
    """Random seed for shuffling."""
    max_input_characters: Optional[int] = 2200
    """Used for direct messages input. Skips samples with input characters longer than this value."""
    max_samples: int = -1
    """Max number of samples to use per dataset."""

    def __post_init__(self):
        if isinstance(self.split, str):
            self.split = DatasetSplit(self.split.lower())

@dataclasses.dataclass
class TrainDatasetArgs(CorrectorDatasetArgs):
    split: DatasetSplit = DatasetSplit.TRAIN
    shuffle: bool = True


@dataclasses.dataclass
class ValDatasetArgs(CorrectorDatasetArgs):
    split: DatasetSplit = DatasetSplit.VALIDATION
    max_samples: int = 256


@dataclasses.dataclass
class EvalDatasetArgs(CorrectorDatasetArgs):
    split: DatasetSplit = DatasetSplit.TEST


@dataclasses.dataclass
class DatasetSplitConfig(helpers.Serializable):
    name: str
    """Name of the split."""
    num_samples: int
    """Number of samples in the split"""
    split: Optional[DatasetSplit] = None
    """Type of split, i.e., train, test, or validation."""

    def __post_init__(self):
        """Automatically set split type based on split name"""
        if self.split is None:
            try:
                self.split = DatasetSplit(self.name.lower())
            except ValueError:
                raise ValueError(
                    f"Could not automatically determine split type from split name '{self.name}'. Please explicitly specify split_type for splits that are not named 'train', 'validation', or 'test'."
                )


# Eval config for a single metric, added to the dataset config
@dataclasses.dataclass
class EvalConfig(helpers.Serializable):
    metric: str
    args: Dict[str, Any] = dataclasses.field(default_factory=dict)
    extra_kwargs_map: Dict[str, str] = dataclasses.field(default_factory=dict)
    """Mapping of field names to use as extra_kwargs for the sample.
    key is name in extra_kwargs, value is name in dataset row."""


@dataclasses.dataclass
class DatasetConfig(helpers.Serializable):
    name: str
    """Name of the dataset."""
    base: Optional[str] = None
    """Base dataset config to inherit from."""
    path: Optional[str] = None
    """Directory of the dataset, or huggingface dataset name; must be set for "generic" datasets. If not set, it is automatically inferred for predefined dataset types."""
    subset: Optional[str] = None
    """Name of the dataset, or huggingface dataset config/subset name."""
    splits: Optional[List[DatasetSplitConfig]] = None
    """List of splits to use, e.g. [{"name": "train", "num_samples": 1000}, {"name": "validation", "num_samples": 100}]."""
    audio_field: Optional[str] = None
    """Field in the dataset that contains the audio, use None if the dataset does not contain audio."""
    error_corrector_template: Optional[str] = None
    """Template for the error corrector model."""
    eval_config: Optional[EvalConfig] = None
    """Eval config for the dataset."""
    batch_size: Optional[int] = None
    """Batch size to use for this dataset."""

    def __post_init__(self):
        """Set defaults only if this is a root config, so that said defaults in a subclass don't act as overrides."""
        DEFAULTS = {
            "splits": [],
            "error_corrector_template": ERROR_CORRECTOR_TEMPLATE,
            "audio_field": "audio",
            "eval_config": None,
            "batch_size": 32,
        }
        if self.base is None:
            for attr, default_value in DEFAULTS.items():
                if getattr(self, attr) is None:
                    setattr(self, attr, default_value)

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
