import os
from transformers import AutoProcessor, AutoModel, AutoTokenizer
from peft import PeftModel
from whisper_streaming.base import OnlineProcessorInterface
from whisper_streaming.silero_vad_iterator import FixedVADIterator
from error_corrector.model.corrector_config import CorrectorConfig
from error_corrector.model.config_builder import (
    ULTRAVOX_MODEL_ID,
    build_error_corrector_config as _build_ultravox_error_corrector_config,
)
from error_corrector.model.corrector_model import CorrectorModel

import torch
import numpy as np
import logging
import sys

logger = logging.getLogger(__name__)


class VACOnlineASRProcessor(OnlineProcessorInterface):
    '''Wraps OnlineASRProcessor with VAC (Voice Activity Controller).

    It works the same way as OnlineASRProcessor: it receives chunks of audio (e.g. 0.04 seconds),
    it runs VAD and continuously detects whether there is speech or not.
    When it detects end of speech (non-voice for 500ms), it makes OnlineASRProcessor to end the utterance immediately.
    '''

    def __init__(
        self,
        online_chunk_size,
        online,
        min_buffered_length=1,
        use_error_corrector=False,
        error_corrector_ckpt=None,
        error_corrector_base_model=None,
        error_corrector_type="speechlm",  # "speechlm" or "lm"
    ):
        self.online_chunk_size = online_chunk_size
        self.online = online
        self.min_buffered_frames = int(min_buffered_length * self.SAMPLING_RATE)

        # VAC:
        model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad'
        )
        self.vac = FixedVADIterator(model)  # Default options: 500ms silence, 100ms padding, etc.

        self.init()

        # Error corrector initialization
        self._corrector_model = None
        self._corrector_processor = None
        self._corrector_type = error_corrector_type  # "speechlm" or "lm"

        if use_error_corrector:
            if error_corrector_type == "lm":
                # LM corrector (text-only, Llama with LoRA)
                from transformers import AutoModelForCausalLM
                
                base_model_id = error_corrector_base_model or "meta-llama/Llama-3.2-1B"
                if error_corrector_ckpt:
                    checkpoint_path = error_corrector_ckpt
                else:
                    checkpoint_path = os.path.join(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "LMCorrector/runs/llama_lora_finetuned"
                    )
                logger.info(f"Loading LM corrector from {checkpoint_path}")
                
                # Load tokenizer
                self._corrector_processor = AutoTokenizer.from_pretrained(
                    base_model_id,
                    trust_remote_code=True
                )
                if self._corrector_processor.pad_token is None:
                    self._corrector_processor.pad_token = self._corrector_processor.eos_token
                    self._corrector_processor.pad_token_id = self._corrector_processor.eos_token_id
                
                # Load base model
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_id,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    device_map="cuda",
                )
                
                # Load LoRA adapter
                self._corrector_model = PeftModel.from_pretrained(
                    base_model,
                    checkpoint_path,
                    is_trainable=False,
                )
                self._corrector_model.eval()
                logger.info("LM corrector loaded successfully")
            else:
                # SpeechLM corrector (audio + text, Ultravox with LoRA)
                base_model_id = error_corrector_base_model or "fixie-ai/ultravox-v0_5-llama-3_2-1b"
                if error_corrector_ckpt:
                    checkpoint_path = error_corrector_ckpt
                else:
                    checkpoint_path = os.path.join(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "SpeechLMCorrector/ultravox_lora_continued_more_erroneous_5/checkpoint-2895"
                    )
                logger.info(f"Loading SpeechLM corrector from {checkpoint_path}")

                # Load processor
                self._corrector_processor = AutoProcessor.from_pretrained(
                    base_model_id,
                    trust_remote_code=True
                )

                # Load base model
                base_model = AutoModel.from_pretrained(
                    base_model_id,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    device_map="cuda",
                )

                # Load LoRA adapter
                self._corrector_model = PeftModel.from_pretrained(
                    base_model,
                    checkpoint_path,
                    is_trainable=False,
                )
                self._corrector_model.eval()
                logger.info("SpeechLM corrector loaded successfully")
        else:
            logger.info("Error corrector disabled")

    @property
    def first_token_latency(self):
        """Delegate first_token_latency to the wrapped online processor."""
        return getattr(self.online, 'first_token_latency', None)

    def init(self):
        self.online.init()
        self.vac.reset_states()
        self.current_online_chunk_buffer_size = 0
        self.is_currently_final = False
        self.status = None  # or "voice" or "nonvoice"
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_offset = 0  # in frames

    def clear_buffer(self):
        self.audio_buffer = np.array([], dtype=np.float32)

    def insert_audio_chunk(self, audio):
        res = self.vac(audio)
        logger.info(f"VAD result: {res}")
        self.audio_buffer = np.append(self.audio_buffer, audio)

        if res is not None:
            frame = list(res.values())[0] - self.buffer_offset
            frame = max(0, frame)

            if 'start' in res and 'end' not in res:
                self.status = 'voice'
                send_audio = self.audio_buffer[frame:]
                self.online.init(offset=(frame + self.buffer_offset) / self.SAMPLING_RATE)
                self.online.insert_audio_chunk(send_audio)
                self.current_online_chunk_buffer_size += len(send_audio)
                self.buffer_offset += len(self.audio_buffer)
                self.clear_buffer()

            elif 'end' in res and 'start' not in res:
                self.status = 'nonvoice'
                if frame > 0:
                    send_audio = self.audio_buffer[:frame]
                    self.online.insert_audio_chunk(send_audio)
                    self.current_online_chunk_buffer_size += len(send_audio)
                self.is_currently_final = True
                keep_frames = min(len(self.audio_buffer) - frame, self.min_buffered_frames)
                self.buffer_offset += len(self.audio_buffer) - keep_frames
                self.audio_buffer = self.audio_buffer[-keep_frames:]

            else:
                beg = max(0, res["start"] - self.buffer_offset)
                end = max(0, res["end"] - self.buffer_offset)
                self.status = 'nonvoice'
                if beg < end:
                    send_audio = self.audio_buffer[beg:end]
                    self.online.init(offset=((beg + self.buffer_offset) / self.SAMPLING_RATE))
                    self.online.insert_audio_chunk(send_audio)
                    self.current_online_chunk_buffer_size += len(send_audio)
                self.is_currently_final = True
                keep_frames = min(len(self.audio_buffer) - end, self.min_buffered_frames)
                self.buffer_offset += len(self.audio_buffer) - keep_frames
                self.audio_buffer = self.audio_buffer[-keep_frames:]

        else:
            if self.status == 'voice':
                self.online.insert_audio_chunk(self.audio_buffer)
                self.current_online_chunk_buffer_size += len(self.audio_buffer)
                self.buffer_offset += len(self.audio_buffer)
                self.clear_buffer()
            else:
                self.buffer_offset += max(0, len(self.audio_buffer) - self.min_buffered_frames)
                self.audio_buffer = self.audio_buffer[-self.min_buffered_frames:]

        logger.info(f"Current online chunk buffer size: {self.current_online_chunk_buffer_size}")

    def process_iter(self, start_time=None):
        if self.is_currently_final:
            return self.finish(start_time=start_time)
        elif self.current_online_chunk_buffer_size > self.SAMPLING_RATE * self.online_chunk_size:
            self.current_online_chunk_buffer_size = 0
            ret = self.online.process_iter(
                start_time=start_time,
                corrector_model=self._corrector_model,
                corrector_processor=self._corrector_processor,
                corrector_type=self._corrector_type,
            )
            return ret
        else:
            logger.info(f"No online update, only VAD. {self.status}")
            return {"first_token_latency": self.first_token_latency}

    def finish(self, start_time=None):
        ret = self.online.finish(
            start_time=start_time,
            corrector_model=self._corrector_model,
            corrector_processor=self._corrector_processor,
            corrector_type=self._corrector_type,
        )
        self.init()
        return ret