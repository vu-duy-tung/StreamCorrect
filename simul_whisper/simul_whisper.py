# This code was originally in simul_whisper/transcriber/simul_whisper.py.
# It is adapted a lot for SimulStreaming.

import os
import sys
import time
import wave
import logging
import math

import numpy as np
import torch
import torch.nn.functional as F

from .whisper import load_model, DecodingOptions, tokenizer
from .config import AlignAttConfig
from .whisper.audio import (
    log_mel_spectrogram,
    TOKENS_PER_SECOND,
    pad_or_trim,
    N_SAMPLES,
    N_FRAMES,
)
from .whisper.timing import median_filter
from .whisper.decoding import GreedyDecoder, BeamSearchDecoder, SuppressTokens
from .beam import BeamPyTorchInference
from .eow_detection import fire_at_boundary, load_cif
from .generation_progress import *

from token_buffer import TokenBuffer
from transformers import AutoModelForCausalLM, AutoTokenizer

env = os.environ.copy()

DEC_PAD = 50257
logger = logging.getLogger(__name__)


def create_whisper_model(cfg: AlignAttConfig):
    logger.info(f"Using OpenAI Whisper format: {cfg.model_path}")
    return PaddedAlignAttWhisper(cfg)


# New features added to the original version of Simul-Whisper: 
# - large-v3 model support
# - translation support
# - beam search
# - prompt -- static vs. non-static
# - context
class PaddedAlignAttWhisper:
    def __init__(self, cfg: AlignAttConfig) -> None:
        self.logdir_i = 0
        self.log_segments = 0
        if cfg.logdir is not None and not os.path.exists(cfg.logdir):
            os.makedirs(cfg.logdir)
        
        # Load the model - keep it simple for OpenAI Whisper format
        model_name = os.path.basename(cfg.model_path).replace(".pt", "")
        model_path = os.path.dirname(os.path.abspath(cfg.model_path))
        self.model = load_model(name=model_name, download_root=model_path)

        logger.info(f"Model dimensions: {self.model.dims}")

        self.decode_options = DecodingOptions(
            language=cfg.language,
            without_timestamps=True,
            task=cfg.task
        )
        self.tokenizer_is_multilingual = not model_name.endswith(".en")
        self.create_tokenizer(cfg.language if cfg.language != "auto" else None)
        self.detected_language = cfg.language if cfg.language != "auto" else None
        
        self.max_text_len = self.model.dims.n_text_ctx
        self.num_decoder_layers = len(self.model.decoder.blocks)
        self.cfg = cfg
        
        # First Token Latency tracking
        self.first_token_generated = False
        self.first_token_latency = None
        self.is_warmup = False

        # model to detect end-of-word boundary at the end of the segment
        self.CIFLinear, self.always_fire, self.never_fire = load_cif(cfg,
                                                                     n_audio_state=self.model.dims.n_audio_state,
                                                                     device=self.model.device)

        # install hooks to access encoder-decoder attention
        self.dec_attns = []
        def layer_hook(module, net_input, net_output):
            # net_output[1]: B*num_head*token_len*audio_len
            t = F.softmax(net_output[1], dim=-1)
            self.dec_attns.append(t.squeeze(0))
        for b in self.model.decoder.blocks:
            b.cross_attn.register_forward_hook(layer_hook)
        
        self.kv_cache = {}
        def kv_hook(module: torch.nn.Linear, _, net_output: torch.Tensor):
            if module.cache_id not in self.kv_cache or net_output.shape[1] > self.max_text_len:
                # save as-is, for the first token or cross attention
                self.kv_cache[module.cache_id] = net_output
            else:
                x = self.kv_cache[module.cache_id]
                self.kv_cache[module.cache_id] = torch.cat([x, net_output], dim=1).detach()
            return self.kv_cache[module.cache_id]

        for i, b in enumerate(self.model.decoder.blocks):
            b.attn.key.register_forward_hook(kv_hook)
            b.attn.value.register_forward_hook(kv_hook)
            b.cross_attn.key.register_forward_hook(kv_hook)
            b.cross_attn.value.register_forward_hook(kv_hook)

        self.align_source = {}
        self.num_align_heads = 0
        for layer_rank, head_id in self.model.alignment_heads.indices().T:
            layer_rank = layer_rank.item()
            heads = self.align_source.get(layer_rank, [])
            heads.append((self.num_align_heads, head_id.item()))
            self.align_source[layer_rank] = heads
            self.num_align_heads += 1


        # tokens to be suppressed from decoding, to prevent hallucinations
        suppress_tokens = [
            self.tokenizer.transcribe,
            self.tokenizer.translate,
            self.tokenizer.sot,
            self.tokenizer.sot_prev,
            self.tokenizer.sot_lm,
            self.tokenizer.no_timestamps,
        ] + list(self.tokenizer.all_language_tokens)
        if self.tokenizer.no_speech is not None:
            suppress_tokens.append(self.tokenizer.no_speech)
        suppress_tokens = tuple(sorted(set(suppress_tokens)))
        logger.debug(f"Suppress tokens: {suppress_tokens}")
        sup_tokens = SuppressTokens(suppress_tokens)
        self.suppress_tokens = lambda logits: sup_tokens.apply(logits, None)
        # blank tokens are suppresed for new segments near the line 334

        # it's going to be regenerated after lang id
        self.segments = []
        self.init_tokens()
        
        self.last_attend_frame = -self.cfg.rewind_threshold

        if self.cfg.max_context_tokens is None:
            self.max_context_tokens = self.max_text_len
        else:
            self.max_context_tokens = self.cfg.max_context_tokens
        self.init_context()

        # decoder type: greedy or beam
        if cfg.decoder_type == "greedy":
            logger.info("Using greedy decoder")
            self.token_decoder = GreedyDecoder(0.0, self.tokenizer.eot)
            self.decoder_type = "greedy"

        elif cfg.decoder_type == "beam":
            self.decoder_type = "beam"
            self.inference = BeamPyTorchInference(self.model, self.initial_token_length)
            self.inference.kv_cache = self.kv_cache

            self.token_decoder = BeamSearchDecoder(inference=self.inference, eot=self.tokenizer.eot, beam_size=cfg.beam_size)

    def create_tokenizer(self, language=None):
        self.tokenizer = tokenizer.get_tokenizer(
            multilingual=self.tokenizer_is_multilingual,
            language=language,
            num_languages=self.model.num_languages,
            task=self.decode_options.task
        )

    def init_context(self):
        kw = {
            'tokenizer': self.tokenizer,
            'device': self.model.device,
            'prefix_token_ids': [self.tokenizer.sot_prev]
        }
        self.context = TokenBuffer.empty(**kw)
        if self.cfg.static_init_prompt is not None:
            self.context = TokenBuffer.from_text(self.cfg.static_init_prompt, **kw)
        if self.cfg.init_prompt is not None:
            self.context.text += self.cfg.init_prompt

    def init_tokens(self):
        logger.debug(f"init tokens, {len(self.segments)}")
        # init tokens (mandatory prompt)
        self.initial_tokens = torch.tensor(
            self.tokenizer.sot_sequence_including_notimestamps, 
            dtype=torch.long, 
            device=self.model.device).unsqueeze(0)
        self.initial_token_length = self.initial_tokens.shape[1]
        self.sot_index = self.tokenizer.sot_sequence.index(self.tokenizer.sot)
#        self.segments = []
        logger.debug(f"init tokens after, {len(self.segments)}")
        self.tokens = [self.initial_tokens]

    def trim_context(self):
        logger.info("Trimming context")
        c = len(self.context.as_token_ids()) - len(self.context.prefix_token_ids)
        logger.info(f"Context text: {self.context.as_text()}")
        total_len = sum(t.shape[1] for t in self.tokens) + c
        if self.cfg.static_init_prompt is None:
            after = 0
        else:
            after = len(self.cfg.static_init_prompt)
        while c > self.max_context_tokens or total_len > self.max_text_len - 20:
            t = self.context.trim_words(after=after)
            total_len -= t
            c -= t
            logger.debug(f"len {total_len}, c {c}, max_context_tokens {self.max_context_tokens}")
            if t == 0:
                break
        logger.info(f"Context after trim: {self.context.text} (len: {total_len})")


    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor) -> torch.Tensor:
        if self.cfg.decoder_type == "greedy":
            logit = self.model.decoder(tokens, audio_features, kv_cache=self.kv_cache)
        else:
            logger.debug(f"Logits shape: {tokens.shape}")
            logit = self.inference.logits(tokens, audio_features)
        return logit

    def refresh_segment(self, complete=False):
        logger.debug("Refreshing segment:")
        self.init_tokens()
        self.last_attend_frame = -self.cfg.rewind_threshold
        self.detected_language = None
        self.init_context()
        
        # Reset FTL tracking for new segment
        if complete:
            self.first_token_generated = False
            self.first_token_latency = None
            self.is_warmup = False
        
        logger.debug(f"Context: {self.context}")
        logger.debug("removing all segments.")
        self.segments = []
        self.log_segments += 1

    def fire_at_boundary(self, chunked_encoder_feature: torch.Tensor):
        if self.always_fire:
            return True
        if self.never_fire:
            return False
        return fire_at_boundary(chunked_encoder_feature, self.CIFLinear)


    def _current_tokens(self):
        toks = self.tokens
        # Very first infer: duplicate start of seq to beam_size
        if toks[0].shape[0] == 1:
            toks[0] = toks[0].repeat_interleave(self.cfg.beam_size, dim=0)

        if not self.context.is_empty():
            context_toks = self.context.as_tensor_beam(self.cfg.beam_size, device=self.model.device)
            toks = [context_toks] + toks

        # Make it one tensor
        if len(toks) > 1:
            current_tokens = torch.cat(toks, dim=1)
        else:
            current_tokens = toks[0]

        logger.debug("debug print current_tokens:")
        if logger.isEnabledFor(logging.DEBUG):
            self.debug_print_tokens(current_tokens)

        return current_tokens


    def debug_print_tokens(self, tokens):
        for i in range(self.cfg.beam_size):
            logger.debug(self.tokenizer.decode_with_timestamps(tokens[i].tolist()))

    # ==================== Audio Buffer ====================

    def segments_len(self):
        segments_len = sum(s.shape[0] for s in self.segments) / 16000
        return segments_len

    def _apply_minseglen(self):
        segments_len = self.segments_len()
        # Wait for long enough audio to start
        if segments_len < self.cfg.audio_min_len:
            logger.debug("waiting for next segment")
            return False
        return True

    def insert_audio(self, segment=None):
        if segment is not None:
            self.segments.append(segment)

        removed_len = 0
        # Length of audio is bigger than buffer_len. Going to remove the first segment
        segments_len = self.segments_len()
        while len(self.segments) > 1 and segments_len > self.cfg.audio_max_len:
            removed_len = self.segments[0].shape[0] / 16000
            segments_len -= removed_len
            self.last_attend_frame -= int(TOKENS_PER_SECOND * removed_len)
            self.segments = self.segments[1:]
            logger.debug(f"remove segments: {len(self.segments)} {len(self.tokens)}")
            if len(self.tokens) > 1:
                self.context.append_token_ids(self.tokens[1][0, :])
                self.tokens = [self.initial_tokens] + self.tokens[2:]
        return removed_len

    def _clean_cache(self):
        """Clean the cache that stores the attention matrices and kv_cache.
        
        It must be called every time after generation with the model.
        """
        self.dec_attns = []
        self.kv_cache = {}
        if self.decoder_type == "beam":
            self.inference.kv_cache = self.kv_cache
            self.token_decoder.reset()

    @torch.no_grad()
    def lang_id(self, encoder_features):
        """Language detection from encoder features.
        
        This code is trimmed and copy-pasted from whisper.decoding.detect_language.
        """
        # Forward pass using a single token, startoftranscript
        n_audio = encoder_features.shape[0]
        x = torch.tensor([[self.tokenizer.sot]] * n_audio).to(self.model.device)
        logits = self.model.logits(x, encoder_features)[:, 0]

        # Collect detected languages; suppress all non-language tokens
        mask = torch.ones(logits.shape[-1], dtype=torch.bool)
        mask[list(self.tokenizer.all_language_tokens)] = False
        logits[:, mask] = -np.inf
        language_tokens = logits.argmax(dim=-1)
        language_token_probs = logits.softmax(dim=-1).cpu()
        language_probs = [
            {
                c: language_token_probs[i, j].item()
                for j, c in zip(self.tokenizer.all_language_tokens, self.tokenizer.all_language_codes)
            }
            for i in range(n_audio)
        ]

        single = encoder_features.ndim == 2
        if single:
            language_tokens = language_tokens[0]
            language_probs = language_probs[0]

        self._clean_cache()
        return language_tokens, language_probs
    
    def _run_qwen_text_only_error_corrector(
        self,
        *,
        current_tokens: torch.Tensor,
        token_len_before_decoding: int,
    ):
        """Run a text-only error corrector using Qwen3-30B-A3B-Instruct-2507 model.
        
        This leverages the language model's capacity to choose the best appended transcript
        from beam search candidates without using audio features.
        """
        
        # Lazy load the Qwen model (cache it as class attribute)
        if not hasattr(self, '_qwen_text_model'):
            self._qwen_text_tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen3-30B-A3B-Instruct-2507",
                trust_remote_code=True
            )
            self._qwen_text_model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen3-30B-A3B-Instruct-2507",
                torch_dtype=torch.float32,
                device_map="auto",
                trust_remote_code=True
            )
        
        previous_tokens = current_tokens[0, 4:token_len_before_decoding].detach().cpu().tolist()
        previous_text = self.tokenizer.decode([t for t in previous_tokens if t >= 0]).strip()

        top_k = min(current_tokens.shape[0], getattr(self.cfg, "beam_size", 1))
        candidate_texts = []
        for i in range(top_k):
            toks = current_tokens[i, 4:].detach().cpu().tolist()
            text = self.tokenizer.decode([t for t in toks if t >= 0]).strip()
            if text:
                candidate_texts.append(text)
        if not candidate_texts:
            return None

        # Clean up prev_display (remove replacement characters at end)
        prev_display = previous_text
        while prev_display.endswith('\uFFFD'):
            prev_display = prev_display[:-1]

        # Clean up candidates (remove replacement characters)
        cleaned_candidates = []
        for text in candidate_texts:
            while text.endswith('\uFFFD'):
                text = text[:-1]
            cleaned_candidates.append(text)

        qwen_prompt_template = (
            "You are an ASR error corrector for streaming transcription.\n\n"
            "Task description:\n"
            "- Each candidate transcript is a FULL transcript.\n"
            "- Each candidate STRICTLY contains the previous transcription as a prefix.\n"
            "- Your job is to choose or minimally correct ONE candidate,\n"
            "  then output ONLY the newly appended suffix after removing the previous transcription.\n\n"
            "Strict rules:\n"
            "- Output ONLY the appended suffix (do NOT repeat the previous transcription).\n"
            "- Output length is typically 1 to 3 Chinese characters.\n"
            "- Output MUST be a valid spoken Chinese continuation.\n"
            "- Output MUST NOT be '...', '…', or any placeholder.\n"
            "- Do NOT add explanations, labels, or English words.\n\n"
            "Examples:\n"
            "Previous transcription:\n"
            "逐渐\n"
            "Candidate transcripts:\n"
            "逐渐的变\n"
            "逐渐地变\n"
            "逐渐的遍\n"
            "Final output:\n"
            "的变\n\n"
            "Previous transcription:\n"
            "在这种情况下我们\n"
            "Candidate transcripts:\n"
            "在这种情况下我们需要\n"
            "在这种情况下我们必须\n"
            "在这种情况下我们应当\n"
            "Final output:\n"
            "需要\n\n"
            "Previous transcription:\n"
            "{prev_display}\n"
            "Candidate transcripts:\n"
            "{prompt_lines}\n"
            "Final output:\n"
        )

        prompt_text = qwen_prompt_template.format(
            prev_display=prev_display,
            prompt_lines="\n".join(cleaned_candidates),
        )

        messages = [
            {"role": "user", "content": prompt_text}
        ]
        
        text = self._qwen_text_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        
        inputs = self._qwen_text_tokenizer([text], return_tensors="pt").to(self._qwen_text_model.device)
        
        generated_ids = self._qwen_text_model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            temperature=0,
            top_p=0.1,
        )
        
        output_ids = generated_ids[0, inputs.input_ids.shape[1]:].tolist()
        response = self._qwen_text_tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        
        return {
            "prompt": prompt_text,
            "previous_transcription": previous_text,
            "candidates": candidate_texts,
            "corrected_appended_text": response,
        }


    def _run_LM_error_corrector(
        self,
        *,
        current_tokens: torch.Tensor,
        token_len_before_decoding: int,
        corrector_model=None,
        corrector_tokenizer=None,
    ):
        """Run a text-only error corrector using fine-tuned Llama model (LMCorrector).
        
        This uses the same format as LMCorrector training - text-only input without audio.
        """
        previous_tokens = current_tokens[0, 4:token_len_before_decoding].detach().cpu().tolist()
        previous_text = self.tokenizer.decode([t for t in previous_tokens if t >= 0]).strip()

        top_k = min(current_tokens.shape[0], getattr(self.cfg, "beam_size", 1))
        candidate_texts = []
        for i in range(top_k):
            toks = current_tokens[i, 4:].detach().cpu().tolist()
            text = self.tokenizer.decode([t for t in toks if t >= 0]).strip()
            if text:
                candidate_texts.append(text)
        if not candidate_texts:
            return None

        # Clean up prev_display (remove replacement characters at end)
        prev_display = previous_text
        while prev_display.endswith('\uFFFD'):
            prev_display = prev_display[:-1]

        # Clean up candidates (remove replacement characters)
        cleaned_candidates = []
        for text in candidate_texts:
            while text.endswith('\uFFFD'):
                text = text[:-1]
            cleaned_candidates.append(text)

        # Lazy load the LM model (Llama with LoRA)
        if not hasattr(self, '_lm_corrector_model'):
            if corrector_model is not None and corrector_tokenizer is not None:
                self._lm_corrector_model = corrector_model
                self._lm_corrector_tokenizer = corrector_tokenizer
                logger.info("Using provided LM corrector model and tokenizer")
            else:
                raise ValueError("LM corrector model and tokenizer must be provided for loading")
        
        # Import and use the same format function as LMCorrector training
        from LMCorrector.training import format_instruction_for_correction
        
        instruction = format_instruction_for_correction(
            k_best_candidates=cleaned_candidates,
            previous_transcript=prev_display,
        )
        
        # Build full text (matching LMCorrector training format)
        # Training uses: f"{bos}{instruction}\n{response}{eos}"
        # For inference, we omit response and EOS so model generates them
        bos_token = self._lm_corrector_tokenizer.bos_token or ""
        full_text = f"{bos_token}{instruction}\n"
        
        # Tokenize input
        inputs = self._lm_corrector_tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        
        # Move inputs to model device
        model_device = next(self._lm_corrector_model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            generated_ids = self._lm_corrector_model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=self._lm_corrector_tokenizer.pad_token_id,
                eos_token_id=self._lm_corrector_tokenizer.eos_token_id,
            )
        
        # Decode only the new tokens
        input_length = inputs["input_ids"].shape[1]
        new_tokens = generated_ids[:, input_length:]
        response = self._lm_corrector_tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()
        
        print("============ LM CORRECTOR =============")
        print("=======================================")
        print(f"Previous: {previous_text}")
        print(f"Candidates: {cleaned_candidates}")
        print(f"Corrected: {response}")
        print("=======================================")
        
        return {
            "prompt": full_text,
            "previous_transcription": previous_text,
            "candidates": candidate_texts,
            "corrected_appended_text": response,
        }


    def _run_SpeechLM_error_corrector(
        self,
        *,
        input_audio: np.ndarray,
        current_tokens: torch.Tensor,
        token_len_before_decoding: int,
        corrector_model=None,
        corrector_processor=None,
    ):
        previous_tokens = current_tokens[0, 4:token_len_before_decoding].detach().cpu().tolist()
        previous_text = self.tokenizer.decode([t for t in previous_tokens if t >= 0]).strip()

        top_k = min(current_tokens.shape[0], getattr(self.cfg, "beam_size", 1))
        candidate_texts = []
        for i in range(top_k):
            toks = current_tokens[i, 4:].detach().cpu().tolist()
            text = self.tokenizer.decode([t for t in toks if t >= 0]).strip()
            if text:
                candidate_texts.append(text)
        if not candidate_texts:
            return None

        # Use candidates directly without numbering to match training format
        # Training data has plain candidates like ['对人', '队', ...] joined by newlines
        # NOTE: Training uses empty string directly, NOT '<empty>' placeholder
        prev_display = previous_text

        # Clean up prev_display (remove replacement characters at end)
        while prev_display.endswith('\uFFFD'):
            prev_display = prev_display[:-1]

        # Clean up candidates (remove replacement characters)
        cleaned_candidates = []
        for text in candidate_texts:
            while text.endswith('\uFFFD'):
                text = text[:-1]
            cleaned_candidates.append(text)

        # Lazy load the SpeechLM model (Ultravox with LoRA)
        if not hasattr(self, '_speechlm_model'):
            if corrector_model is not None and corrector_processor is not None:
                self._speechlm_model = corrector_model
                self._speechlm_processor = corrector_processor
                logger.info("Using provided SpeechLM corrector model and tokenizer")
            else:
                raise ValueError("SpeechLM corrector model and tokenizer must be provided for loading")
        
        # Import and use the same format function as training to ensure consistency
        from SpeechLMCorrector.training import format_instruction_for_correction
        
        instruction = format_instruction_for_correction(
            k_best_candidates=cleaned_candidates,
            previous_transcript=prev_display,
        )
        
        # Build full text with audio placeholder (matching training format)
        # Training uses: f"{bos}<|audio|>\n{instruction}\n{response}{eos}"
        # For inference, we omit response and EOS so model generates them
        bos_token = self._speechlm_processor.tokenizer.bos_token or ""
        full_text = f"{bos_token}<|audio|>\n{instruction}\n"
        
        # Ensure audio is float32 numpy array
        if isinstance(input_audio, torch.Tensor):
            audio_array = input_audio.cpu().numpy().astype(np.float32)
        else:
            audio_array = np.array(input_audio, dtype=np.float32)
        
        # Ensure 1D
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=0) if audio_array.shape[0] <= 2 else audio_array[0]
        
        # Process with Ultravox processor
        inputs = self._speechlm_processor(
            audio=audio_array,
            text=full_text,
            return_tensors="pt",
            sampling_rate=16000,
        )
        
        # Move inputs to model device
        model_device = next(self._speechlm_model.parameters()).device
        inputs = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            generated_ids = self._speechlm_model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=self._speechlm_processor.tokenizer.pad_token_id,
                eos_token_id=self._speechlm_processor.tokenizer.eos_token_id,
            )
        
        # Decode only the new tokens
        input_length = inputs["input_ids"].shape[1]
        new_tokens = generated_ids[:, input_length:]
        response = self._speechlm_processor.tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()
        
        # print("========= SPEECHLM CORRECTOR ==========")
        # print("=======================================")
        # print(f"Previous: {previous_text}")
        # print(f"Candidates: {cleaned_candidates}")
        # print(f"Corrected: {response}")
        # print("=======================================")
        
        return {
            "prompt": full_text,
            "previous_transcription": previous_text,
            "candidates": candidate_texts,
            "corrected_appended_text": response,
        }


    def _run_qwen_audio_error_corrector(
        self,
        *,
        input_audio: torch.Tensor,
        current_tokens: torch.Tensor,
        token_len_before_decoding: int,
    ):
        previous_tokens = current_tokens[0, 4:token_len_before_decoding].detach().cpu().tolist()
        previous_text = self.tokenizer.decode([t for t in previous_tokens if t >= 0]).strip()

        top_k = min(current_tokens.shape[0], getattr(self.cfg, "beam_size", 1))
        candidate_texts = []
        for i in range(top_k):
            toks = current_tokens[i, 4:].detach().cpu().tolist()
            text = self.tokenizer.decode([t for t in toks if t >= 0]).strip()
            if text:
                candidate_texts.append(text)
        if not candidate_texts:
            return None

        # Use candidates directly without numbering to match training format
        # Training data has plain candidates like ['对人', '队', ...] joined by newlines
        # NOTE: Training uses empty string directly, NOT '<empty>' placeholder
        prev_display = previous_text

        # Clean up prev_display (remove replacement characters at end)
        while prev_display.endswith('\uFFFD'):
            prev_display = prev_display[:-1]

        # Clean up candidates (remove replacement characters)
        cleaned_candidates = []
        for text in candidate_texts:
            while text.endswith('\uFFFD'):
                text = text[:-1]
            cleaned_candidates.append(text)
        
        # Lazy load the Qwen Audio model (cache it as instance attribute)
        if not hasattr(self, '_qwen_audio_model'):
            from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
            self._qwen_audio_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
            self._qwen_audio_model = Qwen2AudioForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-Audio-7B-Instruct", 
                device_map="auto"
            )
        
        qwen_prompt_template = (
            "You are an ASR error corrector for streaming transcription.\n\n"

            "Task description:\n"
            "- Each candidate transcript is a FULL transcript.\n"
            "- Each candidate STRICTLY contains the previous transcription as a prefix.\n"
            "- Your job is to choose or minimally correct ONE candidate,\n"
            "  then output ONLY the newly appended suffix after removing the previous transcription.\n\n"

            "Strict rules:\n"
            "- Output ONLY the appended suffix (do NOT repeat the previous transcription).\n"
            "- Output length is typically 1 to 3 Chinese characters.\n"
            "- Output MUST be a valid spoken Chinese continuation.\n"
            "- Output MUST NOT be empty.\n"
            "- Output MUST NOT be '...', '…', or any placeholder.\n"
            "- Do NOT add explanations, labels, or English words.\n\n"

            "Examples:\n"
            "Previous transcription:\n"
            "逐渐\n"
            "Candidate transcripts:\n"
            "逐渐的变\n"
            "逐渐地变\n"
            "逐渐的遍\n"
            "Final output:\n"
            "的变\n\n"

            "Previous transcription:\n"
            "{prev_display}\n"
            "Candidate transcripts:\n"
            "{prompt_lines}\n\n"
            "Final output:\n"
        )

        conversation = [
            {"role": "system", "content": "You are a helpful assistant that corrects streaming ASR errors."},
            {"role": "user", "content": [
                {"type": "audio", "audio": input_audio},
                {"type": "text", "text": qwen_prompt_template.format(
                    prev_display=prev_display,
                    prompt_lines="\n".join(cleaned_candidates),
                )}
            ]},
        ]
        text = self._qwen_audio_processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = self._qwen_audio_processor(text=text, audios=[input_audio], return_tensors="pt", padding=True)
        inputs = inputs.to(self._qwen_audio_model.device)
        generate_ids = self._qwen_audio_model.generate(**inputs, max_new_tokens=8)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]
        response = self._qwen_audio_processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return {
            "prompt": text,
            "previous_transcription": previous_text,
            "candidates": candidate_texts,
            "corrected_appended_text": response,
        }

    # ==================== Transcription / Translation ====================

    @torch.no_grad()
    def infer(
        self,
        is_last: bool = False,
        start_time=None,
        *,
        corrector_model=None,
        corrector_processor=None,
        corrector_type: str = "speechlm",  # "speechlm" or "lm"
    ):
        if (not self.is_warmup and not self.first_token_generated and start_time is not None):
            print(f"Entering infer at {time.time() - start_time:.3f} seconds")

        # start_time should be the time when audio processing started (before load_audio_chunk)
        if start_time is None and not self.is_warmup:
            raise ValueError("start_time must be provided for First Token Latency tracking.")
        infer_start_time = start_time
        
        new_segment = True
        if len(self.segments) == 0:
            logger.debug("No segments, nothing to do")
            # self.logdir_save([], [], {})
            return [], {}
        if not self._apply_minseglen():
            logger.debug(f"applied minseglen {self.cfg.audio_min_len} > {self.segments_len()}.")
            input_segments = torch.cat(self.segments, dim=0)
            # self.logdir_save(input_segments, [], {})
            return [], {}

        # input_segments is concatenation of audio, it's one array
        if len(self.segments) > 1:
            input_segments = torch.cat(self.segments, dim=0)
        else:
            input_segments = self.segments[0]
        
        # mel + padding to 30s
        mel_padded = log_mel_spectrogram(
            input_segments,
            n_mels=self.model.dims.n_mels,
            padding=N_SAMPLES,
            device=self.model.device
        ).unsqueeze(0)
        # Trim to 3000
        mel = pad_or_trim(mel_padded, N_FRAMES)

        # The length of actual audio
        content_mel_len = int((mel_padded.shape[2] - mel.shape[2]) / 2)

        # Encode
        encoder_feature = self.model.encoder(mel)

        if self.cfg.language == "auto" and self.detected_language is None:
            language_tokens, language_probs = self.lang_id(encoder_feature) 
            logger.debug(f"Language tokens: {language_tokens}, probs: {language_probs}")
            top_lan, p = max(language_probs[0].items(), key=lambda x: x[1])
            logger.info(f"Detected language: {top_lan} with p={p:.4f}")
            #self.tokenizer.language = top_lan
            #self.tokenizer.__post_init__()
            self.create_tokenizer(top_lan)
            self.detected_language = top_lan
            self.init_tokens()
            logger.info(f"Tokenizer language: {self.tokenizer.language}, {self.tokenizer.sot_sequence_including_notimestamps}")

        self.trim_context()
        current_tokens = self._current_tokens()

        fire_detected = self.fire_at_boundary(encoder_feature[:, :content_mel_len, :])

        # ==================== Decoding Loop ====================
        if (not self.is_warmup and not self.first_token_generated and start_time is not None):
            print(f"Entering decoding loop at {time.time() - start_time:.3f} seconds")
        logger.info("Decoding loop starts")
        t_before_decoding = time.time()

        sum_logprobs = torch.zeros(self.cfg.beam_size, device=mel.device)
        completed = False

        attn_of_alignment_heads = None
        most_attended_frame = None

        token_len_before_decoding = current_tokens.shape[1]
        
        generation_progress = []
        generation = {
            "starting_tokens": BeamTokens(current_tokens[0,:].clone(), self.cfg.beam_size),
            "token_len_before_decoding": token_len_before_decoding,
            #"fire_detected": fire_detected,
            "frames_len": content_mel_len,
            "frames_threshold": 4 if is_last else self.cfg.frame_threshold,

            # to be filled later
            "logits_starting": None,

            # to be filled later
            "no_speech_prob": None,
            "no_speech": False,

            # to be filled in the loop
            "progress": generation_progress,
            
            # First Token Latency (will be set in the loop if first token is generated)
            "first_token_latency": None,

            # If a token is delayed from generating due to attention reaching the end of audio
            "frame_delay" : False
        }

        logger.info(f"Previous confirmed transcript: {self.tokenizer.decode(current_tokens[0,4:])}")

        decode_iter = 0
        while not completed and current_tokens.shape[1] < self.max_text_len:
            decode_iter += 1
            cur_decode_iter_time = time.time()
            generation_progress_loop = []

            if new_segment:
                tokens_for_logits = current_tokens
            else:
                # only need to use the last token except in the first forward pass
                tokens_for_logits = current_tokens[:,-1:]

            logits = self.logits(tokens_for_logits, encoder_feature) # B, len(tokens), token dict size
            if new_segment:
                generation["logits_starting"] = Logits(logits[:,:,:])

            if new_segment and self.tokenizer.no_speech is not None:
                probs_at_sot = logits[:, self.sot_index, :].float().softmax(dim=-1)
                no_speech_probs = probs_at_sot[:, self.tokenizer.no_speech].tolist()
                generation["no_speech_prob"] = no_speech_probs[0]
                if no_speech_probs[0] > self.cfg.nonspeech_prob:
                    generation["no_speech"] = True
                    logger.info("no speech, stop")
                    break

            logits = logits[:, -1, :] # logits for the last token
            generation_progress_loop.append(("logits_before_suppress",Logits(logits)))

            # supress blank tokens only at the beginning of the segment
            if new_segment:
                logits[:, self.tokenizer.encode(" ") + [self.tokenizer.eot]] = -np.inf
            new_segment = False
            self.suppress_tokens(logits)
            generation_progress_loop.append(("logits_after_suppress", Logits(logits)))

            current_tokens, completed = self.token_decoder.update(current_tokens, logits, sum_logprobs)
            generation_progress_loop.append(("beam_tokens", Tokens(current_tokens[:, -1].clone())))
            generation_progress_loop.append(("sum_logprobs", sum_logprobs.tolist()))
            generation_progress_loop.append(("completed", completed))

            logger.debug(f"Decoding completed: {completed}, sum_logprobs: {sum_logprobs.tolist()}, tokens: ")

            self.debug_print_tokens(current_tokens)

            attn_of_alignment_heads = [[] for _ in range(self.num_align_heads)]
            for i, attn_mat in enumerate(self.dec_attns):
                layer_rank = int(i % len(self.model.decoder.blocks))
                align_heads_in_layer = self.align_source.get(layer_rank, [])
                if len(align_heads_in_layer) == 0:
                    continue
                for align_head_rank, head_id in align_heads_in_layer:
                    if self.cfg.beam_size == 1:
                        a = attn_mat[head_id, :, :]
                        a = a.unsqueeze(0)
                    else:
                        a = attn_mat[:, head_id, :, :]
                    attn_of_alignment_heads[align_head_rank].append(a)
            tmp = []
            for mat in attn_of_alignment_heads:
                t = torch.cat(mat, dim=1)
                tmp.append(t)
            attn_of_alignment_heads = torch.stack(tmp, dim=1)
            std, mean = torch.std_mean(attn_of_alignment_heads, dim=-2, keepdim=True, unbiased=False)
            attn_of_alignment_heads = (attn_of_alignment_heads - mean) / std
            attn_of_alignment_heads = median_filter(attn_of_alignment_heads, 7) # from whisper.timing
            attn_of_alignment_heads = attn_of_alignment_heads.mean(dim=1)
            attn_of_alignment_heads = attn_of_alignment_heads[:,:, :content_mel_len]

            # For each beam, the most attended frame is:
            most_attended_frames = torch.argmax(attn_of_alignment_heads[:, -1, :], dim=-1)
            generation_progress_loop.append(("most_attended_frames", most_attended_frames.clone().tolist()))
            logger.debug(str(most_attended_frames.tolist()) + " most att frames")

            most_attended_frame = most_attended_frames[0].item()

            generation_progress.append(dict(generation_progress_loop))
            logger.debug("current tokens " + str(current_tokens.shape))
            if completed:
                # Stripping the last token, the eot
                current_tokens = current_tokens[:, :-1]
                break
            
            # For some rare cases where the attention fails
            if not is_last and self.last_attend_frame - most_attended_frame > self.cfg.rewind_threshold:
                if current_tokens.shape[1] > 1 and current_tokens[0, -2] >= DEC_PAD:
                    logger.debug("omit rewinding from special tokens")
                    self.last_attend_frame = most_attended_frame
                else:
                    logger.debug(
                        f"[rewind detected] current attention pos: {most_attended_frame}, "
                        f"last attention pos: {self.last_attend_frame}; omit this segment"
                    )
                    self.last_attend_frame = -self.cfg.rewind_threshold
                    current_tokens = torch.cat(self.tokens, dim=1) if len(self.tokens) > 0 else self.tokens[0]
                    break
            else:
                self.last_attend_frame = most_attended_frame

            if content_mel_len - most_attended_frame <= (-1 if is_last else self.cfg.frame_threshold):
                logger.debug(f"attention reaches the end: {most_attended_frame}/{content_mel_len}")
                # Stripping the last token, the one that is attended too close to the end
                current_tokens = current_tokens[:, :-1]
                generation["frame_delay"] = True
                break

            # Only update FTL in decoding loop if error corrector is NOT being used
            # When error corrector is used, FTL will be updated after corrector generates result

            if (not self.is_warmup and not self.first_token_generated and
                start_time is not None and current_tokens.shape[1] > token_len_before_decoding and
                corrector_model is None):
                self.first_token_latency = time.time() - infer_start_time
                self.first_token_generated = True
                generation["first_token_latency"] = self.first_token_latency

            # Debug print
            for i in range(self.cfg.beam_size):
                logger.debug(
                    "attn: {}, current pos: {}, current token: {}({})".format(
                        attn_of_alignment_heads.shape if attn_of_alignment_heads is not None else None,
                        most_attended_frames[i],
                        current_tokens[i, -1].item(),
                        self.tokenizer.decode([current_tokens[i, -1].item()])
                    )
                )

        # Error Corrector
        use_error_corrector = False
        # For SpeechLM corrector, we need corrector_processor (which is the Ultravox processor)
        # For LM corrector, corrector_processor is actually the tokenizer
        if token_len_before_decoding > 0 and not self.is_warmup and corrector_model is not None:
            use_error_corrector = True
            error_corrector_start = time.time()

            
            estimated_run_time_start = time.time()

            if corrector_type == "lm":
                # LM corrector (text-only, Llama with LoRA)
                corrector_result = self._run_LM_error_corrector(
                    current_tokens=current_tokens,
                    token_len_before_decoding=token_len_before_decoding,
                    corrector_model=corrector_model,
                    corrector_tokenizer=corrector_processor,  # For LM, processor is actually the tokenizer
                )
            else:
                # SpeechLM corrector (audio + text, Ultravox with LoRA)
                corrector_result = self._run_SpeechLM_error_corrector(
                    input_audio=input_segments,
                    current_tokens=current_tokens,
                    token_len_before_decoding=token_len_before_decoding,
                    corrector_model=corrector_model,
                    corrector_processor=corrector_processor,
                )
            
            estimated_run_time_end = time.time()

            if corrector_result is not None:
                tmp_len = len(corrector_result['corrected_appended_text']) + 1
                estimated_run_time = (estimated_run_time_end - estimated_run_time_start) / tmp_len * (tmp_len - 1)

            # Update first token latency right after error corrector generates result
            if (not self.is_warmup and not self.first_token_generated and
                start_time is not None and corrector_result is not None):
                self.first_token_latency = time.time() - infer_start_time - estimated_run_time
                self.first_token_generated = True
                generation["first_token_latency"] = self.first_token_latency


            if corrector_result is not None:
                corrector_result['corrected_appended_token_ids'] = self.tokenizer.encode(
                    corrector_result['corrected_appended_text']
                )

            if corrector_result is not None:
                generation["error_corrector"] = corrector_result
            error_corrector_end = time.time()
            # print(f"Error corrector time: {1000 * (error_corrector_end - error_corrector_start):.3f} ms")

            logger.info("End of decoding loop")
            t_after_decoding = time.time()
            logger.info(f"Decoding time: {1000 * (t_after_decoding - t_before_decoding):.3f} ms")

        # Let's now operate only with the top beam hypothesis
        tokens_to_split = current_tokens[0, token_len_before_decoding:]

        if use_error_corrector and "error_corrector" in generation:
            new_hypothesis = generation["error_corrector"]["corrected_appended_token_ids"]
        elif fire_detected or is_last:
            new_hypothesis = tokens_to_split.flatten().tolist()
        else:
            # Going to truncate the tokens after the last space
            split_words, split_tokens = self.tokenizer.split_to_word_tokens(tokens_to_split.tolist())
            generation["result"] = {"split_words": split_words[:-1], "split_tokens": split_tokens[:-1]}
            generation["result_truncated"] = {"split_words": split_words[-1:], "split_tokens": split_tokens[-1:]}

            if len(split_words) > 1:
                new_hypothesis = [i for sublist in split_tokens[:-1] for i in sublist]
            else:
                new_hypothesis = []

        # New hypothesis
        logger.debug(f"new_hypothesis: {new_hypothesis}")
        new_tokens = torch.tensor(
            [new_hypothesis],
            dtype=torch.long
        ).repeat_interleave(self.cfg.beam_size, dim=0).to(device=self.model.device)
        self.tokens.append(new_tokens)

        logger.info(f"Output: {self.tokenizer.decode(new_hypothesis)}")
        self._clean_cache()

        return new_hypothesis, generation

    def logdir_save(self, input_segments, new_hypothesis, generation):
        """The audio and result from each iteration is saved to the logdir for debugging purposes."""
        # Only when the logdir arg is set
        if self.cfg.logdir is None:
            return

        self.logdir_i += 1

        # Every VAD segment is in a separate directory
        log_dir = os.path.join(self.cfg.logdir, f"seg_{self.log_segments:05d}")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        logger.debug(f"Saving to {log_dir}, iteration {self.logdir_i:05d}")

        # Saving wav
        wav_path = os.path.join(log_dir, f"iter_{self.logdir_i:05d}_audio.wav")
        audio_np = np.array(input_segments)
        # Ensure audio is float32 in range [-1, 1], convert to int16 for wav
        if audio_np.dtype != np.int16:
            audio_int16 = np.clip(audio_np * 32767, -32768, 32767).astype(np.int16)
        else:
            audio_int16 = audio_np

        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 2 bytes for int16
            wf.setframerate(16000)
            wf.writeframes(audio_int16.tobytes())

        # Saving readable text: context + hypothesis
        text = self.tokenizer.decode(new_hypothesis)
        with open(os.path.join(log_dir, f"iter_{self.logdir_i:05d}_hypothesis.txt"), "w") as f:
            if generation:
                context = generation["starting_tokens"].as_text(self.tokenizer)
            else:
                context = ""
            print("CONTEXT+FORCED:", context, sep="\t", file=f)
            print("HYPOTHESIS:", text, sep="\t", file=f)