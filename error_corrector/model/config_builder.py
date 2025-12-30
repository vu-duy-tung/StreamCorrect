"""Helpers for building Corrector configs aligned with pretrained Ultravox weights."""

from __future__ import annotations

import copy
from functools import lru_cache
from typing import Optional

from transformers import AutoConfig, LlamaConfig, WhisperConfig

from error_corrector.model.corrector_config import CorrectorConfig

ULTRAVOX_MODEL_ID = "fixie-ai/ultravox-v0_5-llama-3_2-1b"


@lru_cache(maxsize=None)
def _load_ultravox_base_configs(pretrained_id: str):
    """Load the Ultravox composite config and return deep copies of the sub-configs."""
    base_cfg = AutoConfig.from_pretrained(pretrained_id, trust_remote_code=True)
    return base_cfg


def build_error_corrector_config(
    *,
    pretrained_id: str = ULTRAVOX_MODEL_ID,
    audio_ctx: Optional[int] = None,
    text_ctx: Optional[int] = None,
    stack_factor: Optional[int] = None,
) -> CorrectorConfig:
    """Return an CorrectorConfig aligned with the pretrained Ultravox architecture."""
    base_cfg = _load_ultravox_base_configs(pretrained_id)
    return CorrectorConfig(**base_cfg.to_dict())