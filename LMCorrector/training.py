"""
Fine-tuning script for Llama 3.2 1B model using LoRA for text-only ASR error correction.
This script uses only HuggingFace libraries (transformers, peft, datasets).

Model: meta-llama/Llama-3.2-1B (text-only LLM)

Usage:
    python training.py --config training_config.yaml
    
For multi-GPU:
    torchrun --nproc_per_node=4 training.py --config training_config.yaml

This is a text-only version of the SpeechLMCorrector. Instead of using audio input,
it relies solely on k-best ASR transcription candidates to correct errors.
"""

import os
import yaml
import argparse
import torch
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import logging

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)
from datasets import Dataset, load_dataset
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
class LlamaLoraConfig:
    """Configuration for LoRA fine-tuning of Llama model."""
    
    # Model
    model_id: str = "meta-llama/Llama-3.2-1B"
    
    # LoRA parameters for language model
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # Target modules for LoRA in the language model
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    
    # Training parameters
    output_dir: str = "./llama_lora_finetuned"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    # NOTE: bf16=True causes SIGFPE with Llama-3.2-1B + DDP on some systems
    # Use fp32 (both False) for multi-GPU training stability
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    
    # Data
    train_data_path: Optional[str] = None
    eval_data_path: Optional[str] = None
    max_text_length: int = 512
    
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
    wandb_project: Optional[str] = "llama-lora-finetune"
    wandb_run_name: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    
    # Learning rate scheduler
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 0  # If > 0, overrides warmup_ratio
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "LlamaLoraConfig":
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

class LlamaDataCollator:
    """
    Data collator for Llama model (text-only).
    Processes instruction + response pairs for causal language modeling.
    """
    
    def __init__(
        self,
        tokenizer,
        max_text_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        
        # Get pad token id - Llama doesn't have pad token by default
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.pad_token_id = self.tokenizer.pad_token_id
    
    def _process_single_sample(self, feature: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Process a single sample."""
        instruction = feature.get('instruction', '')
        response = feature.get('response', '')
        
        # Get special tokens
        bos_token = self.tokenizer.bos_token or ""
        eos_token = self.tokenizer.eos_token or ""
        
        # Build full text: <bos>{instruction}\n{response}<eos>
        full_text = f"{bos_token}{instruction}\n{response}{eos_token}"
        
        # Tokenize full text
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_text_length,
            add_special_tokens=False,  # We already added BOS/EOS manually
        )
        
        # Tokenize prompt only (for loss masking)
        prompt_only_text = f"{bos_token}{instruction}\n"
        prompt_inputs = self.tokenizer(
            prompt_only_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_text_length,
            add_special_tokens=False,
        )
        loss_mask_len = prompt_inputs["input_ids"].shape[-1]
        
        return inputs, loss_mask_len
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of features.
        
        Each feature should have:
        - 'instruction': the task instruction with k-best candidates
        - 'response': the expected output (correct transcription)
        """
        batch_size = len(features)
        
        # Process each sample
        all_inputs = []
        loss_mask_lens = []
        
        for feature in features:
            try:
                inputs, loss_mask_len = self._process_single_sample(feature)
                all_inputs.append(inputs)
                loss_mask_lens.append(loss_mask_len)
            except Exception as e:
                logger.error(f"Error processing sample: {e}")
                raise
        
        # Collate input_ids and attention_mask with RIGHT padding
        input_ids_list = [inp["input_ids"].squeeze(0) for inp in all_inputs]
        attention_mask_list = [inp["attention_mask"].squeeze(0) for inp in all_inputs]
        
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
        
        # Create labels with proper masking
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


class LlamaTrainer(Trainer):
    """
    Custom Trainer for Llama model that properly computes eval_loss.
    """
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss for training and evaluation."""
        outputs = model(**inputs)
        
        if hasattr(outputs, "loss") and outputs.loss is not None:
            loss = outputs.loss
        else:
            labels = inputs.get("labels", None)
            if labels is not None:
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                
                # Shift for causal LM
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
            else:
                raise ValueError("No loss returned by model and no labels provided")
        
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Override prediction_step to ensure loss is computed during evaluation."""
        inputs = self._prepare_inputs(inputs)
        
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.mean().detach()
        
        return (loss, None, None)


def _is_custom_error_correction_format(data_path: str) -> bool:
    """
    Check if the JSONL file is in the custom ASR error correction format.
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
                if all(key in sample for key in ['k_best_candidates', 'continuation_transcript']):
                    return True
                break
    except Exception:
        pass
    
    return False


def format_instruction_for_correction(
    k_best_candidates: List[str],
    previous_transcript: str,
    num_candidates: int = None,
) -> str:
    """
    Format the instruction prompt for the error correction task.
    """
    if num_candidates is None:
        num_candidates = len(k_best_candidates)
    
    candidates_str = "\n".join([
        f"  {i+1}. {candidate}" 
        for i, candidate in enumerate(k_best_candidates[:num_candidates])
    ])
    
    instruction = f"""Given the following ASR transcription candidates, predict the correct continuation.

Previous confirmed transcript: "{previous_transcript}"

K-best candidates for the continuation:
{candidates_str}

Based on the context and candidates, what is the correct text that should be appended after "{previous_transcript}"?"""
    
    return instruction


def create_dataset_from_jsonl(jsonl_path: str) -> Dataset:
    """
    Create a HuggingFace Dataset from the custom JSONL format for ASR error correction.
    Text-only version - no audio loading needed.
    """
    import json
    from tqdm import tqdm
    
    data = []
    
    log_info(f"Loading data from {jsonl_path}")
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        iterator = tqdm(f, total=total_lines, desc="Loading data", disable=not is_main_process())
        
        for line_num, line in enumerate(iterator, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                sample = json.loads(line)
                
                k_best_candidates = sample['k_best_candidates']
                previous_transcript = sample.get('previous_transcript', '')
                continuation_transcript = sample['continuation_transcript']
                num_candidates = sample.get('num_candidates', len(k_best_candidates))
                
                instruction = format_instruction_for_correction(
                    k_best_candidates=k_best_candidates,
                    previous_transcript=previous_transcript,
                    num_candidates=num_candidates,
                )
                
                data.append({
                    'instruction': instruction,
                    'response': continuation_transcript,
                })
                
            except Exception as e:
                if is_main_process():
                    log_warning(f"Error processing line {line_num}: {e}")
                continue
    
    log_info(f"Loaded {len(data)} samples from {jsonl_path}")
    
    dataset = Dataset.from_dict({
        'instruction': [d['instruction'] for d in data],
        'response': [d['response'] for d in data],
    })
    
    return dataset


def load_training_data(
    data_path: str,
    instruction_column: str = "instruction",
    response_column: str = "response",
) -> Dataset:
    """
    Load training data from various formats.
    """
    # Check if this is the custom error correction format
    if _is_custom_error_correction_format(data_path):
        log_info("Detected custom ASR error correction format")
        return create_dataset_from_jsonl(data_path)
    
    # Standard format handling
    if data_path.endswith('.json') or data_path.endswith('.jsonl'):
        dataset = load_dataset('json', data_files=data_path, split='train')
    elif data_path.endswith('.csv'):
        dataset = load_dataset('csv', data_files=data_path, split='train')
    else:
        dataset = load_dataset(data_path, split='train')
    
    # Rename columns if needed
    if instruction_column != "instruction" and instruction_column in dataset.column_names:
        dataset = dataset.rename_column(instruction_column, "instruction")
    if response_column != "response" and response_column in dataset.column_names:
        dataset = dataset.rename_column(response_column, "response")
    
    return dataset


# =============================================================================
# Model Setup
# =============================================================================

def load_model_from_checkpoint(checkpoint_path: str, config: LlamaLoraConfig):
    """
    Load a model from a saved checkpoint for continued fine-tuning.
    """
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
    
    adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
    is_lora_checkpoint = os.path.exists(adapter_config_path)
    
    if is_lora_checkpoint:
        log_info("Detected LoRA adapter checkpoint")
        
        # For DDP with HuggingFace Trainer, we should NOT use device_map
        # The Trainer will handle device placement
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map=None if is_distributed else "auto",
        )
        
        if config.gradient_checkpointing:
            base_model.gradient_checkpointing_enable()
        
        log_info("Loading LoRA adapter weights...")
        model = PeftModel.from_pretrained(
            base_model,
            checkpoint_path,
            is_trainable=True,
        )
        
        model.print_trainable_parameters()
        
    else:
        log_info("Checkpoint appears to be a Trainer checkpoint, will load base model and let Trainer resume")
        model = setup_model_for_lora_training_base(config)
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        trust_remote_code=True,
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return model, tokenizer


def setup_model_for_lora_training_base(config: LlamaLoraConfig):
    """Base model setup without tokenizer."""
    
    # Determine dtype
    if config.bf16:
        dtype = torch.bfloat16
    elif config.fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32
    
    is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Load the model
    # For DDP with HuggingFace Trainer, we should NOT use device_map
    # The Trainer will handle device placement
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=None if is_distributed else "auto",
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    log_info("Applying LoRA to language model...")
    
    # Enable gradient checkpointing
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    model.print_trainable_parameters()
    
    return model


def setup_model_for_lora_training(config: LlamaLoraConfig):
    """Load the Llama model and apply LoRA for fine-tuning."""
    log_info(f"Loading model: {config.model_id}")
    
    model = setup_model_for_lora_training_base(config)
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        trust_remote_code=True,
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return model, tokenizer


# =============================================================================
# Training
# =============================================================================

def train(
    model,
    tokenizer,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    config: Optional[LlamaLoraConfig] = None,
):
    """Train the model using HuggingFace Trainer with DDP support."""
    if config is None:
        config = LlamaLoraConfig()
    
    # Create data collator
    data_collator = LlamaDataCollator(
        tokenizer=tokenizer,
        max_text_length=config.max_text_length,
    )
    
    is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    
    # Initialize WandB if enabled (only on main process)
    if "wandb" in config.report_to and is_main_process():
        try:
            import wandb
            wandb.init(
                project=config.wandb_project or "llama-lora-finetune",
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
        gradient_checkpointing=config.gradient_checkpointing,
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
        seed=config.seed,
        data_seed=config.seed,
    )
    
    if is_distributed:
        training_args_kwargs["ddp_find_unused_parameters"] = config.ddp_find_unused_parameters
        training_args_kwargs["ddp_backend"] = config.ddp_backend
    
    training_args = TrainingArguments(**training_args_kwargs)
    
    trainer = LlamaTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    log_info("Starting training...")
    log_info(f"  Num examples = {len(train_dataset)}")
    log_info(f"  Num epochs = {config.num_train_epochs}")
    log_info(f"  Batch size per device = {config.per_device_train_batch_size}")
    log_info(f"  Gradient accumulation steps = {config.gradient_accumulation_steps}")
    log_info(f"  Learning rate scheduler = {config.lr_scheduler_type}")
    log_info(f"  Warmup steps = {config.warmup_steps if config.warmup_steps > 0 else f'{config.warmup_ratio*100}% of total steps'}")
    log_info(f"  Reporting to: {config.report_to}")
    
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    
    log_info(f"Saving model to {config.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)
    
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
    parser = argparse.ArgumentParser(description="Fine-tune Llama with LoRA")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--model_id", type=str, default=None, help="Model ID (overrides config)")
    parser.add_argument("--train_data", type=str, default=None, help="Path to training data")
    parser.add_argument("--eval_data", type=str, default=None, help="Path to eval data")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for DDP")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, 
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--load_adapter", type=str, default=None,
                        help="Path to pre-trained LoRA adapter to load")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    if args.config:
        log_info(f"Loading config from {args.config}")
        config = LlamaLoraConfig.from_yaml(args.config)
    else:
        config = LlamaLoraConfig()
    
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
    
    is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    
    if is_distributed and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend=config.ddp_backend)
    
    # Setup model
    if args.load_adapter:
        log_info(f"Loading pre-trained LoRA adapter from {args.load_adapter}")
        model, tokenizer = load_model_from_checkpoint(args.load_adapter, config)
    else:
        model, tokenizer = setup_model_for_lora_training(config)
    
    # Load datasets
    train_dataset = None
    eval_dataset = None
    
    if config.train_data_path:
        log_info(f"Loading training data from {config.train_data_path}")
        full_dataset = load_training_data(config.train_data_path)
        log_info(f"Loaded {len(full_dataset)} total samples")
        
        if config.shuffle_data:
            log_info("Shuffling data...")
            full_dataset = full_dataset.shuffle(seed=config.seed)
        
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
        train_dataset = Dataset.from_dict({
            'instruction': ["Correct this transcription: hello wrold"] * 4,
            'response': ["hello world"] * 4,
        })
        log_info("Created dummy dataset with 4 samples for testing")
    
    # Train
    trainer = train(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=config,
    )
    
    log_info("Training complete!")


if __name__ == "__main__":
    main()
