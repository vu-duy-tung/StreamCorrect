"""
Streaming ASR using Qwen3-ASR-1.7B as the backbone,
with optional error correction using the existing Qwen2-Audio/Ultravox corrector.

Replaces Whisper with Qwen3-ASR for producing top-k beam search candidates,
while keeping the error corrector unchanged.

Usage:
    python qwen3asr_streaming.py audio.wav --lan yue --beams 4 --use-error-corrector
"""

import os
import sys
import time
import logging
import argparse

import numpy as np
import torch

from whisper_streaming.base import OnlineProcessorInterface, ASRBase

logger = logging.getLogger(__name__)

LANG_CODE_TO_NAME = {
    'yue': 'Cantonese', 'zh': 'Chinese', 'en': 'English',
    'ja': 'Japanese', 'ko': 'Korean', 'de': 'German',
    'fr': 'French', 'es': 'Spanish', 'pt': 'Portuguese',
    'ru': 'Russian', 'ar': 'Arabic', 'hi': 'Hindi',
    'id': 'Indonesian', 'it': 'Italian', 'th': 'Thai',
    'vi': 'Vietnamese', 'tr': 'Turkish', 'ms': 'Malay',
    'auto': None,
}


def qwen3asr_args(parser):
    group = parser.add_argument_group('Qwen3-ASR arguments')
    group.add_argument(
        '--model_path', type=str, default='Qwen/Qwen3-ASR-1.7B',
        help='Qwen3-ASR model path or HuggingFace repo id.',
    )
    group.add_argument(
        '--beams', '-b', type=int, default=4,
        help='Number of beams for beam search to produce top-k candidates.',
    )

    group = parser.add_argument_group('Error corrector')
    group.add_argument('--use-error-corrector', action='store_true', default=False)
    group.add_argument(
        '--error-corrector-type', type=str, choices=['speechlm', 'lm'], default='speechlm',
    )
    group.add_argument('--error-corrector-ckpt', type=str, default=None)
    group.add_argument('--error-corrector-base-model', type=str, default=None)

    group = parser.add_argument_group('Prompt and context')
    group.add_argument('--init_prompt', type=str, default=None)
    group.add_argument('--static_init_prompt', type=str, default=None)
    group.add_argument('--max_context_tokens', type=int, default=None)


def qwen3_asr_factory(args):
    logger.setLevel(args.log_level)
    asr = Qwen3ASRBackendASR(
        language=args.lan,
        model_path=args.model_path,
        beams=args.beams,
        logdir=getattr(args, 'logdir', None),
    )
    return asr, Qwen3ASROnline(asr)


class Qwen3ASRBackendASR(ASRBase):

    sep = ''

    def __init__(self, language, model_path, beams, logdir):
        self.language = language
        self.beams = beams
        self.logdir = logdir
        self.force_language = LANG_CODE_TO_NAME.get(language)

        from qwen_asr import Qwen3ASRModel
        logger.info(f'Loading Qwen3-ASR model from {model_path}')
        self.qwen3 = Qwen3ASRModel.from_pretrained(
            model_path,
            dtype=torch.float32,
            device_map='cuda:0',
            max_new_tokens=512,
        )
        logger.info(f'Language: {language} -> {self.force_language}')

    def transcribe_topk(self, audio_np, num_beams=None):
        """Transcribe audio and return top-k candidate texts."""
        if num_beams is None:
            num_beams = self.beams

        try:
            candidates = self._beam_search(audio_np, num_beams)
            if candidates and len(candidates) > 1:
                return candidates
        except Exception as e:
            logger.warning(f'Beam search failed ({e}), falling back to greedy')

        results = self.qwen3.transcribe(
            audio=(audio_np, 16000),
            language=self.force_language,
        )
        text = results[0].text.strip()
        logger.info(f'[Qwen3-ASR greedy] {text}')
        return [text]

    def _beam_search(self, audio_np, num_beams):
        """Try beam search via the underlying HuggingFace model."""
        hf_model = self.qwen3.model
        processor = self.qwen3.processor

        prompt = self.qwen3._build_text_prompt(
            context='', force_language=self.force_language,
        )
        inputs = processor(
            text=[prompt], audio=[audio_np],
            return_tensors='pt', padding=True,
        )
        inputs = inputs.to(hf_model.device)
        prompt_len = inputs['input_ids'].shape[1]

        with torch.no_grad():
            output = hf_model.generate(
                **inputs,
                max_new_tokens=512,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                do_sample=False,
            )

        seqs = output.sequences if hasattr(output, 'sequences') else output
        if seqs.shape[0] < 2:
            raise RuntimeError('Model returned only 1 sequence')

        from qwen_asr.inference.utils import parse_asr_output
        candidates = []
        for seq in seqs:
            raw = processor.decode(
                seq[prompt_len:], skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            _, text = parse_asr_output(raw, user_language=self.force_language)
            candidates.append(text.strip())

        logger.info(f'[Qwen3-ASR beam] {len(candidates)} candidates:')
        for i, c in enumerate(candidates):
            logger.info(f'  [{i}] {c}')
        return candidates

    def warmup(self, audio, init_prompt=''):
        logger.info('Warming up Qwen3-ASR...')
        results = self.qwen3.transcribe(
            audio=(audio, 16000), language=self.force_language,
        )
        logger.info(f'Warmup result: {results[0].text}')

    def transcribe(self, audio, init_prompt=''):
        raise NotImplementedError('Use Qwen3ASROnline.process_iter()')

    def use_vad(self):
        pass

    def set_translate_task(self):
        pass


class Qwen3ASROnline(OnlineProcessorInterface):

    def __init__(self, asr):
        self.asr = asr
        self.init()

    def init(self, offset=None):
        self.audio_buffer = []
        self.offset = offset if offset is not None else 0.0
        self.is_last = False
        self.committed_text = ''
        self.first_token_latency = None
        self._first_token_generated = False
        self.frame_delay = False
        self.end = self.offset
        self.prev_full_text = ''

    def insert_audio_chunk(self, audio):
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()
        self.audio_buffer.append(audio)

    def process_iter(
        self, start_time=None, *,
        corrector_model=None, corrector_processor=None, corrector_type='speechlm',
    ):
        if not self.audio_buffer:
            return {'first_token_latency': self.first_token_latency}

        all_audio = np.concatenate(self.audio_buffer, axis=0)
        if all_audio.shape[0] < 1600:
            return {'first_token_latency': self.first_token_latency}

        self.end = self.offset + all_audio.shape[0] / self.SAMPLING_RATE

        # Qwen3-ASR is not strictly incremental, but we can fake streaming
        # by transcribing the growing buffer and returning any new prefix matches.
        self.frame_delay = False

        candidates = self.asr.transcribe_topk(all_audio)
        if not candidates or all(c.strip() == '' for c in candidates):
            return {'first_token_latency': self.first_token_latency}

        top1_text = candidates[0].strip()

        if corrector_model is not None:
            corrected_suffix = _run_error_corrector(
                audio_np=all_audio,
                candidates=candidates,
                previous_text=self.committed_text,
                corrector_model=corrector_model,
                corrector_processor=corrector_processor,
                corrector_type=corrector_type,
            )
            if corrected_suffix is not None:
                full_text = self.committed_text + corrected_suffix
                delta = corrected_suffix
            else:
                full_text = top1_text
        else:
            full_text = top1_text

        # Robust pseudo-streaming deduplication via Confirmed Prefix Matching:
        # We compare the current full_text with the previous chunk's full_text.
        # The longest common prefix (LCP) form the "confirmed" text that the model
        # is no longer hallucinating/changing. We only emit this confirmed text.
        # If this is the final chunk (is_last), we emit everything.
        lcp_len = 0
        min_len = min(len(self.prev_full_text), len(full_text))
        while lcp_len < min_len and self.prev_full_text[lcp_len] == full_text[lcp_len]:
            lcp_len += 1
            
        confirmed_text = self.prev_full_text[:lcp_len]
        if self.is_last:
            confirmed_text = full_text
            
        self.prev_full_text = full_text

        delta = confirmed_text[len(self.committed_text):] if confirmed_text.startswith(self.committed_text) else ''
        if not delta and confirmed_text and not confirmed_text.startswith(self.committed_text):
            # Fallback if the model drastically re-writes the absolute beginning of the string 
            # after we already committed it. We just append what we can.
            import difflib
            sm = difflib.SequenceMatcher(None, self.committed_text, confirmed_text)
            match = sm.find_longest_match(0, len(self.committed_text), 0, len(confirmed_text))
            if match.size > 0:
                delta = confirmed_text[match.b + match.size:]
            else:
                delta = confirmed_text

        if not self._first_token_generated and start_time is not None and delta:
            self.first_token_latency = time.time() - start_time
            self._first_token_generated = True

        self.committed_text += delta

        if not delta:
            return {'first_token_latency': self.first_token_latency}

        return {
            'start': self.offset,
            'end': self.end,
            'text': delta,
            'tokens': [],
            'words': [{
                'start': self.offset, 'end': self.end,
                'text': delta, 'tokens': [],
            }],
            'first_token_latency': self.first_token_latency,
        }

    def finish(
        self, start_time=None, *,
        corrector_model=None, corrector_processor=None, corrector_type='speechlm',
    ):
        self.is_last = True
        o = self.process_iter(
            start_time=start_time,
            corrector_model=corrector_model,
            corrector_processor=corrector_processor,
            corrector_type=corrector_type,
        )
        self.is_last = False
        self.init()
        return o


# ---------------------------------------------------------------------------
# Error corrector -- matches the exact format in simul_whisper.py
# ---------------------------------------------------------------------------

def _run_error_corrector(
    audio_np, candidates, previous_text,
    corrector_model, corrector_processor, corrector_type,
):
    """Run error corrector on top-k candidates.

    This replicates the format used in simul_whisper.py
    (_run_SpeechLM_error_corrector / _run_LM_error_corrector) so that the
    fine-tuned corrector receives inputs identical to what it was trained on.
    """
    prev_display = previous_text
    while prev_display.endswith('\ufffd'):
        prev_display = prev_display[:-1]

    cleaned = []
    for text in candidates:
        while text.endswith('\ufffd'):
            text = text[:-1]
        if text.strip():
            cleaned.append(text)
    if not cleaned:
        return None

    # ---- LM (text-only) corrector ----
    if corrector_type == 'lm':
        from LMCorrector.training import format_instruction_for_correction
        instruction = format_instruction_for_correction(
            k_best_candidates=cleaned,
            previous_transcript=prev_display,
        )
        bos_token = corrector_processor.bos_token or ''
        full_text = f'{bos_token}{instruction}\n'
        inputs = corrector_processor(
            full_text, return_tensors='pt', truncation=True, max_length=512,
        )
        model_device = next(corrector_model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        with torch.no_grad():
            gen = corrector_model.generate(
                **inputs, max_new_tokens=8, do_sample=False,
                pad_token_id=corrector_processor.pad_token_id,
                eos_token_id=corrector_processor.eos_token_id,
            )
        response = corrector_processor.decode(
            gen[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True,
        ).strip()

        print('============ LM CORRECTOR =============')
        print(f'Previous: {prev_display}')
        print(f'Candidates: {cleaned}')
        print(f'Corrected suffix: {response}')
        print('=======================================')
        return response

    # ---- SpeechLM corrector (audio + text) ----
    from SpeechLMCorrector.training_qwen2audio import format_instruction_for_correction
    instruction = format_instruction_for_correction(
        k_best_candidates=cleaned,
        previous_transcript=prev_display,
    )

    audio_array = np.asarray(audio_np, dtype=np.float32)
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=0) if audio_array.shape[0] <= 2 else audio_array[0]

    model_config = getattr(corrector_model, 'config', None)
    if model_config is None and hasattr(corrector_model, 'base_model'):
        model_config = getattr(corrector_model.base_model, 'config', None)
    model_type = getattr(model_config, 'model_type', 'ultravox')

    if model_type == 'qwen2_audio':
        conversation = [
            {'role': 'system', 'content': 'You are a helpful assistant specialized in ASR error correction.'},
            {'role': 'user', 'content': [
                {'type': 'audio', 'audio_url': 'placeholder'},
                {'type': 'text', 'text': instruction},
            ]},
        ]
        full_text = corrector_processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False,
        )
        inputs = corrector_processor(
            text=full_text, audios=[audio_array],
            return_tensors='pt', sampling_rate=16000, padding=True,
        )
    else:
        # Ultravox -- exact format from simul_whisper.py
        bos_token = corrector_processor.tokenizer.bos_token or ''
        full_text = f'{bos_token}<|audio|>\n{instruction}\n'
        inputs = corrector_processor(
            audio=audio_array, text=full_text,
            return_tensors='pt', sampling_rate=16000,
        )

    model_device = next(corrector_model.parameters()).device
    inputs = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}

    with torch.no_grad():
        gen = corrector_model.generate(
            **inputs, max_new_tokens=8, do_sample=False,
            pad_token_id=corrector_processor.tokenizer.pad_token_id,
            eos_token_id=corrector_processor.tokenizer.eos_token_id,
        )

    input_length = inputs['input_ids'].shape[1]
    new_tokens = gen[:, input_length:]
    response = corrector_processor.tokenizer.decode(
        new_tokens[0], skip_special_tokens=True,
    ).strip()

    print('======== SPEECHLM CORRECTOR ============')
    print(f'Previous: {prev_display}')
    print(f'Candidates: {cleaned}')
    print(f'Corrected suffix: {response}')
    print('========================================')
    return response


if __name__ == '__main__':
    from whisper_streaming.whisper_online_main import main_simulation_from_file
    main_simulation_from_file(qwen3_asr_factory, add_args=qwen3asr_args)
