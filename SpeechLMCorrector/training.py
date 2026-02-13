"""
Fine-tuning script for Ultravox model using LoRA.
This script uses only HuggingFace libraries (transformers, peft, datasets).

Supported Models:
- fixie-ai/ultravox-v0_5-llama-3_2-1b (Llama 3.2 1B, ~4GB VRAM)
- fixie-ai/ultravox-v0_5-llama-3_1-8b (Llama 3.1 8B, ~20GB VRAM)

Architecture:
- Language Model: Llama 3.2 1B or Llama 3.1 8B
- Audio Encoder: Whisper (frozen by default)
- Multi-modal Projector: Linear layers adapting speech to text space

Usage:
    python training.py --config training_config.yaml
    
For multi-GPU:
    torchrun --nproc_per_node=4 training.py --config training_config.yaml
"""

import os
import yaml
import argparse
import torch
import torchaudio
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Union
import logging

from transformers import (
    AutoModel,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)
from datasets import Dataset, Audio, load_dataset
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def is_main_process() -> bool:
    """Check if this is the main process in distributed training."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return local_rank == 0


def log_info(message: str):
    """Log info only on main process."""
    if is_main_process():
        logger.info(message)


def log_warning(message: str):
    """Log warning only on main process."""
    if is_main_process():
        logger.warning(message)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class UltravoxLoraConfig:
    """Configuration for LoRA fine-tuning of Ultravox model."""
    
    # Model
    model_id: str = "fixie-ai/ultravox-v0_5-llama-3_2-1b"
    
    # LoRA parameters for language model
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # Target modules for LoRA in the language model
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    
    # Whether to also fine-tune the multi-modal projector
    train_projector: bool = True
    
    # Whether to fine-tune the audio encoder (usually kept frozen)
    train_audio_encoder: bool = False
    
    # Training parameters
    output_dir: str = "./ultravox_lora_finetuned"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    
    # Data
    train_data_path: Optional[str] = None
    eval_data_path: Optional[str] = None
    max_audio_length_seconds: float = 30.0
    max_text_length: int = 512
    sample_rate: int = 16000
    
    # Train/validation split
    train_val_split: float = 0.1  # Fraction of data for validation (0.0 to disable)
    shuffle_data: bool = True     # Shuffle data before splitting
    
    # Distributed training (DDP)
    ddp_find_unused_parameters: bool = False
    ddp_backend: str = "nccl"
    local_rank: int = -1
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    
    # Misc
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None
    
    # WandB settings
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    wandb_project: Optional[str] = "ultravox-lora-finetune"
    wandb_run_name: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    
    # Learning rate scheduler
    lr_scheduler_type: str = "cosine"  # Options: linear, cosine, cosine_with_restarts, polynomial, constant, constant_with_warmup
    warmup_steps: int = 0  # If > 0, overrides warmup_ratio
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "UltravoxLoraConfig":
        """Load configuration from a YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, yaml_path: str):
        """Save configuration to a YAML file."""
        with open(yaml_path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)


# =============================================================================
# Data Processing
# =============================================================================

class UltravoxDataCollator:
    """
    Data collator for Ultravox model.
    Processes each sample individually then collates into batch.
    The Ultravox processor doesn't support batch mode, so we process one at a time.
    Supports lazy loading: if 'audio_path' and 'timestamp' are provided instead of 'audio',
    the audio will be loaded on-the-fly.
    """
    
    def __init__(
        self,
        processor,
        max_audio_length_seconds: float = 30.0,
        max_text_length: int = 512,
        sample_rate: int = 16000,
    ):
        self.processor = processor
        self.max_audio_length = int(max_audio_length_seconds * sample_rate)
        self.max_text_length = max_text_length
        self.sample_rate = sample_rate
        
        # Get pad token id
        self.pad_token_id = getattr(processor.tokenizer, 'pad_token_id', 0) or 0
    
    def _load_audio_lazy(self, audio_path: str, timestamp: float) -> np.ndarray:
        """Load audio segment lazily from file path."""
        audio_data = load_audio_segment(
            audio_path=audio_path,
            start_time=0.0,
            end_time=timestamp,
            target_sample_rate=self.sample_rate,
        )
        return np.array(audio_data['array'], dtype=np.float32)
    
    def _process_audio(self, audio: Union[Dict, np.ndarray, torch.Tensor]) -> np.ndarray:
        """Process a single audio sample."""
        if isinstance(audio, dict):
            audio_array = np.array(audio['array'], dtype=np.float32)
            sr = audio.get('sampling_rate', self.sample_rate)
        elif isinstance(audio, torch.Tensor):
            audio_array = audio.numpy()
            sr = self.sample_rate
        else:
            audio_array = np.array(audio, dtype=np.float32)
            sr = self.sample_rate
        
        # Ensure 1D
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=0) if audio_array.shape[0] <= 2 else audio_array[0]
        
        # Resample if needed
        if sr != self.sample_rate:
            audio_tensor = torch.tensor(audio_array).unsqueeze(0)
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, self.sample_rate)
            audio_array = audio_tensor.squeeze(0).numpy()
        
        # Truncate if too long
        if len(audio_array) > self.max_audio_length:
            audio_array = audio_array[:self.max_audio_length]
        
        return audio_array
    
    def _process_single_sample(self, feature: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Process a single sample with the Ultravox processor."""
        # Load audio - support both lazy loading (audio_path) and pre-loaded (audio)
        if 'audio_path' in feature and 'timestamp' in feature:
            # Lazy loading: load audio from file path
            audio_array = self._load_audio_lazy(feature['audio_path'], feature['timestamp'])
            # Truncate if too long
            if len(audio_array) > self.max_audio_length:
                audio_array = audio_array[:self.max_audio_length]
        elif 'audio' in feature:
            # Pre-loaded audio
            audio_array = self._process_audio(feature['audio'])
        else:
            raise ValueError("Feature must contain either 'audio' or ('audio_path' and 'timestamp')")
        
        # Format text
        instruction = feature.get('instruction', '')
        response = feature.get('response', '')
        
        # Get special tokens from tokenizer
        # BOS is needed at the start (the processor uses add_special_tokens=False)
        # EOS is needed at the end so model learns when to stop generating
        bos_token = self.processor.tokenizer.bos_token or ""
        eos_token = self.processor.tokenizer.eos_token or ""
        
        # Build full text with BOS, audio placeholder, and EOS token
        # Format: <bos><|audio|>\n{instruction}\n{response}<eos>
        full_text = f"{bos_token}<|audio|>\n{instruction}\n{response}{eos_token}"
        
        # Process with Ultravox processor (single sample)
        inputs = self.processor(
            audio=audio_array,
            text=full_text,
            return_tensors="pt",
            sampling_rate=self.sample_rate,
        )
        
        # Compute loss_mask_len by processing text WITHOUT the response
        # This is more accurate than tokenizing response separately because
        # tokenization can differ based on context (e.g., "\n" + response vs just response)
        # Following the approach in ultravox_data_proc.py
        #
        # Token structure after processing:
        #   [text_before_audio] + [audio_replacement_tokens] + [text_after_audio_placeholder]
        # For full_text = "<bos><|audio|>\n{instruction}\n{response}<eos>":
        #   [<bos>] + [audio_tokens] + ["\n{instruction}\n{response}<eos>" tokens]
        # For prompt_only = "<bos><|audio|>\n{instruction}\n":
        #   [<bos>] + [audio_tokens] + ["\n{instruction}\n" tokens]
        # So response tokens (including EOS) are at the END of the sequence.
        prompt_only_text = f"{bos_token}<|audio|>\n{instruction}\n"
        prompt_inputs = self.processor(
            audio=audio_array,
            text=prompt_only_text,
            return_tensors="pt",
            sampling_rate=self.sample_rate,
        )
        loss_mask_len = prompt_inputs["input_ids"].shape[-1]
        
        # Sanity check: loss_mask_len should be less than total input length
        total_len = inputs["input_ids"].shape[-1]
        if loss_mask_len >= total_len:
            logger.warning(
                f"loss_mask_len ({loss_mask_len}) >= total_len ({total_len}). "
                f"Response may be empty or tokenization issue. Response: '{response}'"
            )
        
        return inputs, loss_mask_len
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of features.
        
        Each feature should have:
        - 'audio': audio waveform (numpy array, dict with 'array' key, or path)
        - 'instruction': the task instruction
        - 'response': the expected output
        """
        batch_size = len(features)
        
        # Process each sample individually
        all_inputs = []
        loss_mask_lens = []  # Number of tokens to mask (everything before response)
        
        for feature in features:
            try:
                inputs, loss_mask_len = self._process_single_sample(feature)
                all_inputs.append(inputs)
                loss_mask_lens.append(loss_mask_len)
            except Exception as e:
                logger.error(f"Error processing sample: {e}")
                raise
        
        # Separate audio-related fields that need special handling
        # These are per-audio-segment, not per-sample
        audio_values_list = []
        audio_lens_list = []
        audio_token_len_list = []
        audio_token_start_idx_list = []
        audio_batch_size_list = []
        
        for inp in all_inputs:
            if "audio_values" in inp:
                av = inp["audio_values"]
                if av.dim() == 2:
                    # Shape: (num_segments, audio_len)
                    for i in range(av.shape[0]):
                        audio_values_list.append(av[i])
                else:
                    audio_values_list.append(av.squeeze(0) if av.dim() > 1 else av)
            
            if "audio_lens" in inp:
                al = inp["audio_lens"]
                if al.dim() > 0:
                    audio_lens_list.extend(al.tolist() if al.numel() > 1 else [al.item()])
                else:
                    audio_lens_list.append(al.item())
            
            if "audio_token_len" in inp:
                atl = inp["audio_token_len"]
                if atl.dim() > 0:
                    audio_token_len_list.extend(atl.tolist() if atl.numel() > 1 else [atl.item()])
                else:
                    audio_token_len_list.append(atl.item())
            
            if "audio_token_start_idx" in inp:
                atsi = inp["audio_token_start_idx"]
                if atsi.dim() > 0:
                    audio_token_start_idx_list.extend(atsi.tolist() if atsi.numel() > 1 else [atsi.item()])
                else:
                    audio_token_start_idx_list.append(atsi.item())
            
            if "audio_batch_size" in inp:
                abs_val = inp["audio_batch_size"]
                audio_batch_size_list.append(abs_val.item() if abs_val.numel() == 1 else abs_val[0].item())
        
        # Collate input_ids and attention_mask with RIGHT padding
        # (Ultravox uses left padding for inference, but right padding is fine for training)
        input_ids_list = [inp["input_ids"].squeeze(0) for inp in all_inputs]
        attention_mask_list = [inp["attention_mask"].squeeze(0) for inp in all_inputs]
        original_lengths = [ids.shape[-1] for ids in input_ids_list]
        
        max_len = max(t.shape[-1] for t in input_ids_list)
        
        padded_input_ids = []
        padded_attention_mask = []
        for ids, mask in zip(input_ids_list, attention_mask_list):
            pad_len = max_len - ids.shape[-1]
            if pad_len > 0:
                ids = torch.nn.functional.pad(ids, (0, pad_len), value=self.pad_token_id)
                mask = torch.nn.functional.pad(mask, (0, pad_len), value=0)
            padded_input_ids.append(ids)
            padded_attention_mask.append(mask)
        
        collated = {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
        }
        
        # Handle audio fields
        if audio_values_list:
            # Pad audio values to same length
            max_audio_len = max(av.shape[-1] for av in audio_values_list)
            padded_audio = []
            for av in audio_values_list:
                if av.dim() == 1:
                    av = av.unsqueeze(0)
                pad_len = max_audio_len - av.shape[-1]
                if pad_len > 0:
                    av = torch.nn.functional.pad(av, (0, pad_len), value=0)
                padded_audio.append(av)
            collated["audio_values"] = torch.stack(padded_audio).squeeze(1) if padded_audio[0].shape[0] == 1 else torch.stack(padded_audio)
        
        if audio_lens_list:
            collated["audio_lens"] = torch.tensor(audio_lens_list, dtype=torch.long)
        
        if audio_token_len_list:
            collated["audio_token_len"] = torch.tensor(audio_token_len_list, dtype=torch.long)
        
        if audio_token_start_idx_list:
            # audio_token_start_idx doesn't need adjustment for RIGHT padding
            # (only LEFT padding would shift the positions)
            collated["audio_token_start_idx"] = torch.tensor(audio_token_start_idx_list, dtype=torch.long)
        
        if audio_batch_size_list:
            collated["audio_batch_size"] = torch.tensor(audio_batch_size_list, dtype=torch.long)
        
        # Create labels with proper masking
        # Following ultravox_data_proc.py approach: mask everything up to loss_mask_len
        labels = collated["input_ids"].clone()
        
        for i in range(batch_size):
            loss_mask_len = loss_mask_lens[i]
            
            # Mask all tokens up to loss_mask_len (prompt + instruction, not response)
            labels[i, :loss_mask_len] = -100
            
            # Mask padding tokens (where attention_mask is 0)
            padding_mask = collated["attention_mask"][i] == 0
            labels[i][padding_mask] = -100
        
        collated["labels"] = labels
        
        return collated


class UltravoxTrainer(Trainer):
    """
    Custom Trainer for Ultravox model that properly computes eval_loss.
    
    The standard Trainer may not compute eval_loss correctly for some model types.
    This ensures loss is computed during evaluation for best model selection.
    """
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss for training and evaluation.
        
        The key is to pass labels to the model so it computes the loss internally.
        """
        # Pass all inputs including labels to the model
        outputs = model(**inputs)
        
        # Get loss from outputs
        if hasattr(outputs, "loss") and outputs.loss is not None:
            loss = outputs.loss
        else:
            # If model didn't compute loss, compute it manually
            labels = inputs.get("labels", None)
            if labels is not None:
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                
                # Shift for causal LM
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                # Flatten and compute cross entropy
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
            else:
                raise ValueError("No loss returned by model and no labels provided")
        
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Override prediction_step to ensure loss is computed during evaluation.
        """
        inputs = self._prepare_inputs(inputs)
        
        with torch.no_grad():
            # Always compute loss
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.mean().detach()
        
        # Return loss, logits=None, labels=None (we don't need predictions for eval_loss)
        return (loss, None, None)


def _is_custom_error_correction_format(data_path: str) -> bool:
    """
    Check if the JSONL file is in the custom ASR error correction format.
    
    Custom format has: k_best_candidates, timestamp, audio_path, continuation_transcript
    """
    import json
    
    if not data_path.endswith('.jsonl'):
        return False
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                # Check for custom format signature fields
                if all(key in sample for key in ['k_best_candidates', 'timestamp', 'audio_path', 'continuation_transcript']):
                    return True
                break
    except Exception:
        pass
    
    return False


def load_training_data(
    data_path: str,
    audio_column: str = "audio",
    instruction_column: str = "instruction",
    response_column: str = "response",
    sample_rate: int = 16000,
) -> Dataset:
    """
    Load training data from various formats.
    
    Supports:
    - JSON, JSONL, CSV, or HuggingFace dataset path with standard columns
    - Custom JSONL format for ASR error correction (auto-detected)
    
    Custom JSONL format (auto-detected):
    {
        "k_best_candidates": [...],
        "timestamp": 2.0,
        "audio_path": "/path/to/audio.wav",
        "previous_transcript": "prefix",
        "continuation_transcript": "continuation"
    }
    """
    # Check if this is the custom error correction format
    if _is_custom_error_correction_format(data_path):
        log_info("Detected custom ASR error correction format")
        return create_dataset_from_jsonl(data_path, sample_rate=sample_rate)
    
    # Standard format handling
    if data_path.endswith('.json') or data_path.endswith('.jsonl'):
        dataset = load_dataset('json', data_files=data_path, split='train')
    elif data_path.endswith('.csv'):
        dataset = load_dataset('csv', data_files=data_path, split='train')
    else:
        dataset = load_dataset(data_path, split='train')
    
    # Rename columns if needed
    if audio_column != "audio" and audio_column in dataset.column_names:
        dataset = dataset.rename_column(audio_column, "audio")
    if instruction_column != "instruction" and instruction_column in dataset.column_names:
        dataset = dataset.rename_column(instruction_column, "instruction")
    if response_column != "response" and response_column in dataset.column_names:
        dataset = dataset.rename_column(response_column, "response")
    
    # Cast audio column if it contains file paths
    if "audio" in dataset.column_names:
        first_audio = dataset[0]["audio"]
        if isinstance(first_audio, str):
            dataset = dataset.cast_column("audio", Audio(sampling_rate=sample_rate))
    
    return dataset


def create_dataset_from_samples(
    audio_paths: List[str],
    instructions: List[str],
    responses: List[str],
) -> Dataset:
    """Create a HuggingFace Dataset from lists."""
    assert len(audio_paths) == len(instructions) == len(responses), \
        "All input lists must have the same length"
    
    data = {
        'audio': audio_paths,
        'instruction': instructions,
        'response': responses,
    }
    dataset = Dataset.from_dict(data)
    dataset = dataset.cast_column('audio', Audio(sampling_rate=16000))
    return dataset


def load_audio_segment(
    audio_path: str,
    start_time: float = 0.0,
    end_time: float = None,
    target_sample_rate: int = 16000,
) -> Dict[str, Any]:
    """
    Load a segment of audio from a file.
    
    Args:
        audio_path: Path to the audio file
        start_time: Start time in seconds
        end_time: End time in seconds (None for full audio)
        target_sample_rate: Target sample rate for resampling
    
    Returns:
        Dictionary with 'array' and 'sampling_rate' keys
    """
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Calculate sample indices
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate) if end_time is not None else waveform.shape[1]
    
    # Clip to valid range
    start_sample = max(0, start_sample)
    end_sample = min(waveform.shape[1], end_sample)
    
    # Extract segment
    waveform = waveform[:, start_sample:end_sample]
    
    # Resample if needed
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
    
    return {
        'array': waveform.squeeze(0).numpy(),
        'sampling_rate': target_sample_rate,
    }


def format_instruction_for_correction(
    k_best_candidates: List[str],
    previous_transcript: str,
    num_candidates: int = None,
) -> str:
    """
    Format the instruction prompt for the error correction task.
    
    Args:
        k_best_candidates: List of k-best transcription candidates
        previous_transcript: The confirmed prefix transcript
        num_candidates: Number of candidates (optional, inferred from list)
    
    Returns:
        Formatted instruction string
    """
    if num_candidates is None:
        num_candidates = len(k_best_candidates)
    
    # Format candidates as a numbered list
    candidates_str = "\n".join([
        f"  {i+1}. {candidate}" 
        for i, candidate in enumerate(k_best_candidates[:num_candidates])
    ])
    
    instruction = f"""Given the audio and the following ASR transcription candidates, predict the correct continuation.

Previous confirmed transcript: "{previous_transcript}"

K-best candidates for the continuation:
{candidates_str}

Based on the audio, what is the correct text that should be appended after "{previous_transcript}"?"""
    
    return instruction


def create_dataset_from_jsonl(
    jsonl_path: str,
    sample_rate: int = 16000,
) -> Dataset:
    """
    Create a HuggingFace Dataset from the custom JSONL format for ASR error correction.
    Uses LAZY LOADING - audio is loaded on-the-fly to save memory.
    
    Expected JSONL format:
    {
        "k_best_candidates": ["candidate1", "candidate2", ...],
        "num_candidates": 4,
        "chunk_size": 500,
        "previous_transcript": "prefix text",
        "continuation_transcript": "continuation",
        "audio_path": "/path/to/audio.wav",
        "timestamp": 2.0
    }
    
    The audio is cut from 0s to timestamp seconds.
    The instruction includes k-best candidates and previous transcript.
    The response is the continuation_transcript.
    
    Args:
        jsonl_path: Path to the JSONL file
        sample_rate: Target sample rate for audio
    
    Returns:
        HuggingFace Dataset with lazy audio loading
    """
    import json
    from tqdm import tqdm
    
    # Only load metadata, not audio (lazy loading)
    metadata = []
    
    log_info(f"Loading metadata from {jsonl_path} (lazy loading - audio loaded on demand)")
    
    # Count total lines first for progress bar
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        # Only show progress bar on main process
        iterator = tqdm(f, total=total_lines, desc="Loading metadata", disable=not is_main_process())
        
        for line_num, line in enumerate(iterator, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                sample = json.loads(line)
                
                # Extract fields - store metadata only, don't load audio yet
                audio_path = sample['audio_path']
                timestamp = sample['timestamp']
                k_best_candidates = sample['k_best_candidates']
                previous_transcript = sample.get('previous_transcript', '')
                continuation_transcript = sample['continuation_transcript']
                num_candidates = sample.get('num_candidates', len(k_best_candidates))
                
                # Format instruction (text only, no audio loading)
                instruction = format_instruction_for_correction(
                    k_best_candidates=k_best_candidates,
                    previous_transcript=previous_transcript,
                    num_candidates=num_candidates,
                )
                
                metadata.append({
                    'audio_path': audio_path,
                    'timestamp': timestamp,
                    'instruction': instruction,
                    'response': continuation_transcript,
                })
                
            except Exception as e:
                if is_main_process():
                    log_warning(f"Error processing line {line_num}: {e}")
                continue
    
    log_info(f"Loaded metadata for {len(metadata)} samples from {jsonl_path}")
    
    # Create dataset with metadata only (audio paths, not loaded audio)
    dataset = Dataset.from_dict({
        'audio_path': [m['audio_path'] for m in metadata],
        'timestamp': [m['timestamp'] for m in metadata],
        'instruction': [m['instruction'] for m in metadata],
        'response': [m['response'] for m in metadata],
    })
    
    return dataset


# =============================================================================
# Model Setup
# =============================================================================

def load_model_from_checkpoint(checkpoint_path: str, config: UltravoxLoraConfig):
    """
    Load a model from a saved checkpoint for continued fine-tuning.
    
    This handles two cases:
    1. Trainer checkpoint (contains optimizer state, scheduler, etc.)
    2. Saved LoRA adapter (just the adapter weights)
    
    Args:
        checkpoint_path: Path to checkpoint directory
        config: Training configuration
        
    Returns:
        Tuple of (model, processor)
    """
    import os
    from peft import PeftModel
    
    log_info(f"Loading model from checkpoint: {checkpoint_path}")
    
    # Determine dtype
    if config.bf16:
        dtype = torch.bfloat16
    elif config.fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32
    
    is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    
    # Check if this is a LoRA adapter checkpoint or a full trainer checkpoint
    adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
    is_lora_checkpoint = os.path.exists(adapter_config_path)
    
    if is_lora_checkpoint:
        log_info("Detected LoRA adapter checkpoint")
        
        # Load base model
        base_model = AutoModel.from_pretrained(
            config.model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map=None if is_distributed else "auto",
        )
        
        # Configure audio encoder and projector
        if hasattr(base_model, 'audio_tower'):
            for param in base_model.audio_tower.parameters():
                param.requires_grad = config.train_audio_encoder
            log_info(f"Audio encoder trainable: {config.train_audio_encoder}")
        
        if hasattr(base_model, 'multi_modal_projector'):
            for param in base_model.multi_modal_projector.parameters():
                param.requires_grad = config.train_projector
            log_info(f"Projector trainable: {config.train_projector}")
        
        # Load LoRA adapter FIRST
        log_info("Loading LoRA adapter weights...")
        model = PeftModel.from_pretrained(
            base_model,
            checkpoint_path,
            is_trainable=True,  # Important: keep adapter trainable for continued training
        )
        
        # NOTE: Gradient checkpointing is enabled by Trainer via TrainingArguments
        # with gradient_checkpointing_kwargs={"use_reentrant": False} for LoRA compatibility
        # Do NOT enable it here to avoid double-enabling issues
        
        model.print_trainable_parameters()
        
    else:
        log_info("Checkpoint appears to be a Trainer checkpoint, will load base model and let Trainer resume")
        # For trainer checkpoints, we load the base model with fresh LoRA
        # The trainer will handle loading optimizer/scheduler state
        model = setup_model_for_lora_training_base(config)
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        config.model_id,
        trust_remote_code=True,
    )
    
    return model, processor


def setup_model_for_lora_training_base(config: UltravoxLoraConfig):
    """Base model setup without processor (used internally)."""
    
    # Determine dtype
    if config.bf16:
        dtype = torch.bfloat16
    elif config.fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32
    
    is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    
    # Load the model
    model = AutoModel.from_pretrained(
        config.model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=None if is_distributed else "auto",
    )
    
    # 1. Audio encoder - usually frozen
    if hasattr(model, 'audio_tower'):
        for param in model.audio_tower.parameters():
            param.requires_grad = config.train_audio_encoder
        log_info(f"Audio encoder trainable: {config.train_audio_encoder}")
    
    # 2. Multi-modal projector
    if hasattr(model, 'multi_modal_projector'):
        for param in model.multi_modal_projector.parameters():
            param.requires_grad = config.train_projector
        log_info(f"Projector trainable: {config.train_projector}")
    
    # 3. Configure LoRA
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    log_info("Applying LoRA to language model...")
    
    # Apply LoRA FIRST
    model = get_peft_model(model, lora_config)
    
    # NOTE: Gradient checkpointing is enabled by Trainer via TrainingArguments
    # with gradient_checkpointing_kwargs={"use_reentrant": False} for LoRA compatibility
    # Do NOT enable it here to avoid double-enabling issues
    
    model.print_trainable_parameters()
    
    return model


def setup_model_for_lora_training(config: UltravoxLoraConfig):
    """Load the Ultravox model and apply LoRA for fine-tuning."""
    log_info(f"Loading model: {config.model_id}")
    
    # Use the base function to setup model with LoRA
    model = setup_model_for_lora_training_base(config)
    
    # Load the processor
    processor = AutoProcessor.from_pretrained(
        config.model_id,
        trust_remote_code=True,
    )
    
    return model, processor


# =============================================================================
# Training
# =============================================================================

def train(
    model,
    processor,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    config: Optional[UltravoxLoraConfig] = None,
):
    """Train the model using HuggingFace Trainer with DDP support."""
    if config is None:
        config = UltravoxLoraConfig()
    
    # Create data collator
    data_collator = UltravoxDataCollator(
        processor=processor,
        max_audio_length_seconds=config.max_audio_length_seconds,
        max_text_length=config.max_text_length,
        sample_rate=config.sample_rate,
    )
    
    # Determine if we're in distributed mode
    is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    
    # Initialize WandB if enabled (only on main process)
    if "wandb" in config.report_to and is_main_process():
        try:
            import wandb
            wandb.init(
                project=config.wandb_project or "ultravox-lora-finetune",
                name=config.wandb_run_name,
                config=asdict(config),
                tags=config.wandb_tags,
                save_code=True,
                resume="allow",
            )
            log_info(f"WandB initialized: project={config.wandb_project}, run={wandb.run.name if wandb.run else 'N/A'}")
        except ImportError:
            log_warning("WandB not installed. Installing with: pip install wandb")
            config.report_to = [r for r in config.report_to if r != "wandb"]
    
    # Training arguments
    training_args_kwargs = dict(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio if config.warmup_steps == 0 else 0,
        warmup_steps=config.warmup_steps if config.warmup_steps > 0 else 0,
        lr_scheduler_type=config.lr_scheduler_type,
        max_grad_norm=config.max_grad_norm,
        logging_steps=config.logging_steps,
        logging_first_step=True,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps if eval_dataset is not None else None,
        fp16=config.fp16,
        bf16=config.bf16,
        # Gradient checkpointing settings - use_reentrant=False is required for LoRA compatibility
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if config.gradient_checkpointing else None,
        remove_unused_columns=False,
        report_to=config.report_to,
        run_name=config.wandb_run_name,
        save_total_limit=3,
        load_best_model_at_end=eval_dataset is not None,
        eval_strategy="steps" if eval_dataset is not None else "no",
        metric_for_best_model="eval_loss" if eval_dataset is not None else None,
        greater_is_better=False,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=config.dataloader_pin_memory,
        # Reproducibility
        seed=config.seed,
        data_seed=config.seed,
    )
    
    # Add DDP settings only if in distributed mode
    if is_distributed:
        training_args_kwargs["ddp_find_unused_parameters"] = config.ddp_find_unused_parameters
        training_args_kwargs["ddp_backend"] = config.ddp_backend
    
    training_args = TrainingArguments(**training_args_kwargs)
    
    # Create trainer (use custom UltravoxTrainer for proper eval_loss computation)
    trainer = UltravoxTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Note: Skipping initial evaluation as it can be very slow with large validation sets
    # The trainer will run evaluation at eval_steps intervals
    
    # Train
    log_info("Starting training...")
    log_info(f"  Num examples = {len(train_dataset)}")
    log_info(f"  Num epochs = {config.num_train_epochs}")
    log_info(f"  Batch size per device = {config.per_device_train_batch_size}")
    log_info(f"  Gradient accumulation steps = {config.gradient_accumulation_steps}")
    log_info(f"  Learning rate scheduler = {config.lr_scheduler_type}")
    log_info(f"  Warmup steps = {config.warmup_steps if config.warmup_steps > 0 else f'{config.warmup_ratio*100}% of total steps'}")
    log_info(f"  Reporting to: {config.report_to}")
    
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    
    # Save the final model
    log_info(f"Saving model to {config.output_dir}")
    trainer.save_model()
    processor.save_pretrained(config.output_dir)
    
    # Finish WandB run if active
    if "wandb" in config.report_to and is_main_process():
        try:
            import wandb
            if wandb.run:
                wandb.finish()
                log_info("WandB run finished successfully")
        except Exception as e:
            log_warning(f"Error finishing WandB run: {e}")
    
    return trainer


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Ultravox with LoRA")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--model_id", type=str, default=None, help="Model ID (overrides config)")
    parser.add_argument("--train_data", type=str, default=None, help="Path to training data")
    parser.add_argument("--eval_data", type=str, default=None, help="Path to eval data")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for DDP")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, 
                        help="Path to checkpoint to resume training from (Trainer checkpoint or LoRA adapter)")
    parser.add_argument("--load_adapter", type=str, default=None,
                        help="Path to pre-trained LoRA adapter to load (for continued fine-tuning)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    if args.config:
        log_info(f"Loading config from {args.config}")
        config = UltravoxLoraConfig.from_yaml(args.config)
    else:
        config = UltravoxLoraConfig()
    
    # Override config with command line arguments
    if args.model_id:
        config.model_id = args.model_id
    if args.train_data:
        config.train_data_path = args.train_data
    if args.eval_data:
        config.eval_data_path = args.eval_data
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.local_rank != -1:
        config.local_rank = args.local_rank
    if args.resume_from_checkpoint:
        config.resume_from_checkpoint = args.resume_from_checkpoint
    
    # Set seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Check if distributed training
    is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Initialize distributed process group before loading data
    if is_distributed and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend=config.ddp_backend)
    
    # Setup model - either from checkpoint or fresh
    if args.load_adapter:
        # Load pre-trained LoRA adapter for continued fine-tuning
        log_info(f"Loading pre-trained LoRA adapter from {args.load_adapter}")
        model, processor = load_model_from_checkpoint(args.load_adapter, config)
    else:
        # Fresh training
        model, processor = setup_model_for_lora_training(config)
    
    # Load datasets
    # In DDP, each process loads the same dataset, and Trainer handles distribution
    train_dataset = None
    eval_dataset = None
    
    if config.train_data_path:
        log_info(f"Loading training data from {config.train_data_path}")
        full_dataset = load_training_data(config.train_data_path)
        log_info(f"Loaded {len(full_dataset)} total samples")
        
        # Shuffle data if enabled
        if config.shuffle_data:
            log_info("Shuffling data...")
            full_dataset = full_dataset.shuffle(seed=config.seed)
        
        # Split into train/val if configured
        if config.train_val_split > 0 and config.eval_data_path is None:
            log_info(f"Splitting data: {1 - config.train_val_split:.0%} train, {config.train_val_split:.0%} validation")
            split = full_dataset.train_test_split(
                test_size=config.train_val_split,
                seed=config.seed,
            )
            train_dataset = split['train']
            eval_dataset = split['test']
            log_info(f"Train samples: {len(train_dataset)}, Validation samples: {len(eval_dataset)}")
        else:
            train_dataset = full_dataset
            log_info(f"Using all {len(train_dataset)} samples for training")
    
    if config.eval_data_path:
        log_info(f"Loading evaluation data from {config.eval_data_path}")
        eval_dataset = load_training_data(config.eval_data_path)
        log_info(f"Loaded {len(eval_dataset)} evaluation samples")
    
    if train_dataset is None:
        log_warning("No training data provided. Creating dummy dataset for testing...")
        dummy_audio = np.random.randn(16000).astype(np.float32)
        train_dataset = Dataset.from_dict({
            'audio': [{'array': dummy_audio, 'sampling_rate': 16000}] * 4,
            'instruction': ["Transcribe this audio."] * 4,
            'response': ["This is a test transcription."] * 4,
        })
        log_info("Created dummy dataset with 4 samples for testing")
    
    # Train
    trainer = train(
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=config,
    )
    
    log_info("Training complete!")


if __name__ == "__main__":
    main()
