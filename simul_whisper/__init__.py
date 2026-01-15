"""
SimulStreaming Whisper module.

Provides both OpenAI Whisper and HuggingFace Whisper model support.
"""

from .simul_whisper import PaddedAlignAttWhisper, create_whisper_model
from .config import AlignAttConfig

__all__ = [
    'PaddedAlignAttWhisper',
    'create_whisper_model',
    'AlignAttConfig',
]
