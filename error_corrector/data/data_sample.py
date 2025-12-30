import base64
import dataclasses
import io
from typing import Any, Dict, List, Optional

import librosa
import numpy as np
import soundfile as sf
from numpy import typing as npt

@dataclasses.dataclass
class CorrectorSample:

    k_best_candidates : List[str]
    """List of k-best candidate continuations from ASR system."""
    num_candidates: int
    """Number of candidate continuations."""
    chunk_size: int
    """The chunk size of streaming mechanism."""
    continuation_transcript: str
    """The ground truth continuation transcript."""
    previous_transcript: str
    """The previous transcription before the continuation."""
    audio_embed: Optional[npt.NDArray[np.float32]] = None
    """Audio embedding array, if included."""
    audio_encoder_path: Optional[str] = None
    """Path to the audio encoder used."""
    extra_kwargs: Optional[Dict[str, Any]] = None
    """For evaluations, extra columns from the sample."""