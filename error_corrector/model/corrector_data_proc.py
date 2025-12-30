from typing import Any, Dict, Optional

import math
import torch
import numpy as np

from error_corrector import data as datasets
from error_corrector.model import corrector_config
from error_corrector.model import corrector_processing
from error_corrector.data.types import ERROR_CORRECTOR_TEMPLATE, AUDIO_PLACEHOLDER


class CorrectorDataproc(datasets.Dataproc):
    def __init__(
        self,
        dataset: datasets.SizedIterableDataset,
        processor: corrector_processing.CorrectorProcessor,
        loss_mask_type: corrector_config.LossMaskType,
        error_corrector_template: Optional[str],
        inference_mode: bool = False,
        include_alt_fields: bool = False,
        max_response_tokens: Optional[int] = None,
    ) -> None:
        """
        Pre-processing for the Error Corrector model: applies tokenization and audio processing using the CorrectorProcessor
        and prepares the shape of the data for being fed into the model.

        Args:
            dataset: The dataset to wrap/preprocess.
            processor: The processor.
            inference_mode: If True, only the input message is included in input_ids and labels, and the assistant
                message is removed from the sample. This is used for inference (e.g. testing) since the model should
                generate the assistant message. For training and validation, this should be False.
            include_alt_fields: If True, the alt_input_ids, alt_attention_mask, and alt_labels are included in the output,
                computed with <|audio|> replaced by the audio transcript.
        """
        super().__init__(dataset)
        self.processor = processor
        self.inference_mode = inference_mode
        self.include_alt_fields = include_alt_fields
        self.max_response_tokens = max_response_tokens
        self.error_corrector_template = error_corrector_template
        self.loss_mask_type = loss_mask_type

    def _compute_loss_mask_len(
        self, sample: datasets.CorrectorSample, audio: Optional[np.ndarray]
    ) -> int:
        # TODO: this might be slow due to calling audio_processor twice. We can compute modified input_text_len directly too.
        # Revisit when using WhisperProcessor.
        # Computing the length of the mask.
        if self.loss_mask_type == corrector_config.LossMaskType.AFTER_AUDIO:
            user_text = user_text.split("<|audio|>")[0] + "<|audio|>"
            loss_mask_len = self.processor(
                text=user_text,
                audios=audio,
                sampling_rate=sample.sample_rate,
            )["input_ids"].shape[-1]

        elif self.loss_mask_type == corrector_config.LossMaskType.LAST_ASSISTANT:
            loss_mask_len = self.processor(
                text=user_text,
                audios=audio,
                sampling_rate=sample.sample_rate,
            )["input_ids"].shape[-1]

        elif self.loss_mask_type == corrector_config.LossMaskType.ALL:
            # This does not work with KL loss.
            loss_mask_len = 0
        return loss_mask_len

    def _process(self, sample: datasets.CorrectorSample) -> Dict[str, Any]:
        # ------------------------------------------------------------
        # 1. Build prompt text
        # ------------------------------------------------------------
        prompt_text = ERROR_CORRECTOR_TEMPLATE.format(
            prev_display=sample.previous_transcript,
            prompt_lines="\n".join(sample.k_best_candidates),
        )

        # ------------------------------------------------------------
        # 2. Tokenize prompt and continuation
        # ------------------------------------------------------------
        prompt_inputs = self.processor.tokenizer(
            prompt_text,
            return_tensors="pt",
        )
        prompt_ids = prompt_inputs["input_ids"].squeeze(0)

        continuation_ids = self.processor.tokenizer(
            sample.continuation_transcript,
            return_tensors="pt",
            add_special_tokens=False,  # Don't add BOS token to continuation
        )["input_ids"].squeeze(0)

        # Append EOS token to continuation so model learns when to stop
        eos_token_id = self.processor.tokenizer.eos_token_id
        if eos_token_id is not None:
            eos_token = torch.tensor([eos_token_id], dtype=continuation_ids.dtype)
            continuation_ids = torch.cat([continuation_ids, eos_token], dim=0)

        # ------------------------------------------------------------
        # 3. Audio token length calculation (unchanged)
        # ------------------------------------------------------------
        stack_factor = 8
        frames = sample.audio_embed.shape[1]
        audio_token_len_value = max(1, math.ceil(frames / stack_factor))

        batch_size = 1
        audio_batch_size = torch.ones(batch_size, dtype=torch.long)
        total_audio_segments = int(audio_batch_size.sum().item())

        # ------------------------------------------------------------
        # 4. Build sequence: prompt → audio PAD → continuation
        # ------------------------------------------------------------
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.processor.tokenizer.eos_token_id

        audio_padding = torch.full(
            (audio_token_len_value,),
            pad_token_id,
            dtype=prompt_ids.dtype,
            device=prompt_ids.device,
        )

        # audio starts immediately after prompt
        audio_token_start_idx = torch.full(
            (total_audio_segments,),
            prompt_ids.shape[0],
            dtype=torch.long,
        )

        audio_token_len = torch.full(
            (total_audio_segments,),
            audio_token_len_value,
            dtype=torch.long,
        )

        # concatenate in desired order
        input_ids = torch.cat(
            [prompt_ids, audio_padding, continuation_ids],
            dim=0,
        )

        attention_mask = torch.ones_like(input_ids)

        # ------------------------------------------------------------
        # 5. Labels: only continuation contributes to loss
        # ------------------------------------------------------------
        labels = input_ids.clone()

        # mask prompt
        labels[: prompt_ids.shape[0]] = -100

        # mask audio padding
        labels[
            prompt_ids.shape[0] : prompt_ids.shape[0] + audio_token_len_value
        ] = -100

        # continuation labels are kept as-is

        # ------------------------------------------------------------
        # 6. Assemble input dict
        # ------------------------------------------------------------
        inputs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "audio_values": sample.audio_embed,
            "audio_batch_size": audio_batch_size,
            "audio_token_start_idx": audio_token_start_idx,
            "audio_token_len": audio_token_len,
        }

        # Optional alternative fields
        if self.include_alt_fields:
            inputs["alt_input_ids"] = input_ids.clone()
            inputs["alt_attention_mask"] = attention_mask.clone()
            inputs["alt_labels"] = labels.clone().tolist()

        assert (
            inputs["input_ids"].shape
            == inputs["attention_mask"].shape
            == inputs["labels"].shape
        ), "input_ids, attention_mask, and labels must have the same shape"

        # HF Trainer prefers Python lists
        inputs["labels"] = inputs["labels"].tolist()

        return inputs