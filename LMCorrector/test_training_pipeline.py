"""
Test script for LMCorrector training pipeline.
Tests data loading, collation, and model setup without full training.

Usage:
    python test_training_pipeline.py
"""

import os
import sys
import torch
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training import (
    LlamaLoraConfig,
    LlamaDataCollator,
    load_training_data,
    create_dataset_from_jsonl,
    format_instruction_for_correction,
    setup_model_for_lora_training,
    _is_custom_error_correction_format,
)


def test_config_creation():
    """Test that we can create and serialize configs."""
    print("\n" + "="*60)
    print("Testing: Config Creation")
    print("="*60)
    
    config = LlamaLoraConfig(
        model_id="meta-llama/Llama-3.2-1B",
        lora_r=8,
        lora_alpha=16,
        per_device_train_batch_size=2,
    )
    
    assert config.model_id == "meta-llama/Llama-3.2-1B"
    assert config.lora_r == 8
    assert config.lora_alpha == 16
    assert config.per_device_train_batch_size == 2
    
    # Test default values
    assert config.lora_dropout == 0.05
    # NOTE: bf16=False by default for DDP compatibility
    assert config.bf16 == False
    assert config.gradient_checkpointing == True
    
    print("✓ Config creation successful")
    print(f"  Model ID: {config.model_id}")
    print(f"  LoRA rank: {config.lora_r}")
    print(f"  LoRA alpha: {config.lora_alpha}")
    print(f"  Target modules: {config.lora_target_modules}")
    
    return config


def test_instruction_formatting():
    """Test the instruction formatting for error correction task."""
    print("\n" + "="*60)
    print("Testing: Instruction Formatting")
    print("="*60)
    
    k_best_candidates = [
        "hello world",
        "hello word",
        "helo world",
    ]
    previous_transcript = "Good morning,"
    
    instruction = format_instruction_for_correction(
        k_best_candidates=k_best_candidates,
        previous_transcript=previous_transcript,
    )
    
    assert "hello world" in instruction
    assert "hello word" in instruction
    assert "helo world" in instruction
    assert "Good morning," in instruction
    assert "K-best candidates" in instruction
    
    print("✓ Instruction formatting successful")
    print("  Generated instruction:")
    print("-" * 40)
    print(instruction)
    print("-" * 40)
    
    return instruction


def test_data_collator():
    """Test the data collator with mock data."""
    print("\n" + "="*60)
    print("Testing: Data Collator")
    print("="*60)
    
    from transformers import AutoTokenizer
    
    # Use a small, fast tokenizer for testing
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    except Exception as e:
        print(f"⚠ Could not load Llama tokenizer: {e}")
        print("  Trying GPT-2 tokenizer as fallback...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    collator = LlamaDataCollator(
        tokenizer=tokenizer,
        max_text_length=256,
    )
    
    # Create test batch
    features = [
        {
            "instruction": "Correct this transcription: hello wrold",
            "response": "hello world"
        },
        {
            "instruction": "Correct this transcription: good mroning",
            "response": "good morning"
        }
    ]
    
    batch = collator(features)
    
    # Check batch structure
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "labels" in batch
    
    assert batch["input_ids"].shape[0] == 2  # batch size
    assert batch["attention_mask"].shape[0] == 2
    assert batch["labels"].shape[0] == 2
    
    # Check shapes match
    assert batch["input_ids"].shape == batch["attention_mask"].shape
    assert batch["input_ids"].shape == batch["labels"].shape
    
    # Check labels masking (prompt should be masked with -100)
    assert (batch["labels"] == -100).any()  # Some tokens should be masked
    
    print("✓ Data collator successful")
    print(f"  Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"  Batch attention_mask shape: {batch['attention_mask'].shape}")
    print(f"  Batch labels shape: {batch['labels'].shape}")
    print(f"  Number of masked tokens (sample 0): {(batch['labels'][0] == -100).sum().item()}")
    
    return batch


def test_label_masking():
    """Test that labels are correctly masked for the instruction portion."""
    print("\n" + "="*60)
    print("Testing: Label Masking")
    print("="*60)
    
    from transformers import AutoTokenizer
    
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    collator = LlamaDataCollator(
        tokenizer=tokenizer,
        max_text_length=256,
    )
    
    # Test with a single sample where we know the expected structure
    features = [
        {
            "instruction": "This is the instruction part",
            "response": "This is the response"
        }
    ]
    
    batch = collator(features)
    
    # Check that labels have some masked (-100) and some unmasked portions
    labels = batch["labels"][0]
    masked_count = (labels == -100).sum().item()
    unmasked_count = (labels != -100).sum().item()
    
    assert masked_count > 0, "No tokens were masked (instruction should be masked)"
    assert unmasked_count > 0, "All tokens were masked (response should not be masked)"
    
    print("✓ Label masking successful")
    print(f"  Masked tokens: {masked_count}")
    print(f"  Unmasked tokens: {unmasked_count}")
    
    # Decode and show what's masked vs unmasked
    input_ids = batch["input_ids"][0]
    
    # Find first unmasked position
    unmasked_positions = (labels != -100).nonzero(as_tuple=True)[0]
    if len(unmasked_positions) > 0:
        first_unmasked = unmasked_positions[0].item()
        print(f"  First unmasked position: {first_unmasked}")
        
        masked_text = tokenizer.decode(input_ids[:first_unmasked], skip_special_tokens=False)
        unmasked_text = tokenizer.decode(input_ids[first_unmasked:], skip_special_tokens=False)
        print(f"  Masked portion: '{masked_text[:50]}...' (loss ignored)")
        print(f"  Unmasked portion: '{unmasked_text[:50]}' (loss computed)")


def test_custom_format_detection():
    """Test detection of custom error correction format."""
    print("\n" + "="*60)
    print("Testing: Custom Format Detection")
    print("="*60)
    
    import tempfile
    import json
    
    # Test with custom format
    custom_data = [
        {
            "k_best_candidates": ["hello world", "hello word"],
            "previous_transcript": "Good morning,",
            "continuation_transcript": "hello world"
        }
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for item in custom_data:
            f.write(json.dumps(item) + '\n')
        temp_path = f.name
    
    try:
        is_custom = _is_custom_error_correction_format(temp_path)
        assert is_custom == True, "Should detect custom format"
        print("✓ Custom format detection successful")
        print(f"  Detected as custom format: {is_custom}")
    finally:
        os.unlink(temp_path)
    
    # Test with standard format
    standard_data = [
        {
            "instruction": "Hello",
            "response": "World"
        }
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for item in standard_data:
            f.write(json.dumps(item) + '\n')
        temp_path = f.name
    
    try:
        is_custom = _is_custom_error_correction_format(temp_path)
        assert is_custom == False, "Should not detect standard format as custom"
        print(f"  Standard format detected as custom: {is_custom}")
    finally:
        os.unlink(temp_path)


def test_dataset_creation():
    """Test creating a dataset from custom JSONL format."""
    print("\n" + "="*60)
    print("Testing: Dataset Creation from Custom Format")
    print("="*60)
    
    import tempfile
    import json
    
    # Create test data
    test_data = [
        {
            "k_best_candidates": ["hello world", "hello word", "helo world"],
            "previous_transcript": "Good morning,",
            "continuation_transcript": "hello world",
            "num_candidates": 3
        },
        {
            "k_best_candidates": ["how are you", "how are yuo", "haw are you"],
            "previous_transcript": "Hello,",
            "continuation_transcript": "how are you",
            "num_candidates": 3
        }
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')
        temp_path = f.name
    
    try:
        dataset = create_dataset_from_jsonl(temp_path)
        
        assert len(dataset) == 2
        assert "instruction" in dataset.column_names
        assert "response" in dataset.column_names
        
        print("✓ Dataset creation successful")
        print(f"  Number of samples: {len(dataset)}")
        print(f"  Columns: {dataset.column_names}")
        print(f"  Sample instruction (truncated): {dataset[0]['instruction'][:100]}...")
        print(f"  Sample response: {dataset[0]['response']}")
        
    finally:
        os.unlink(temp_path)


def test_model_setup(skip_if_no_gpu: bool = True):
    """Test model setup with LoRA (requires GPU and model access)."""
    print("\n" + "="*60)
    print("Testing: Model Setup with LoRA")
    print("="*60)
    
    if skip_if_no_gpu and not torch.cuda.is_available():
        print("⚠ Skipping model setup test (no GPU available)")
        return
    
    config = LlamaLoraConfig(
        model_id="meta-llama/Llama-3.2-1B",
        lora_r=8,
        lora_alpha=16,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        gradient_checkpointing=True,
    )
    
    try:
        print("Setting up model with LoRA...")
        model, tokenizer = setup_model_for_lora_training(config)
        
        # Check model is a PEFT model
        from peft import PeftModel
        assert isinstance(model, PeftModel) or hasattr(model, 'peft_config')
        
        # Check tokenizer
        assert tokenizer is not None
        assert tokenizer.pad_token is not None
        
        print("✓ Model setup successful")
        print(f"  Model type: {type(model).__name__}")
        print(f"  Tokenizer vocab size: {tokenizer.vocab_size}")
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable %: {100 * trainable_params / total_params:.2f}%")
        
    except Exception as e:
        print(f"⚠ Model setup test failed: {e}")
        print("  This may be due to missing model access or insufficient resources.")
        raise


def test_forward_pass(skip_if_no_gpu: bool = True):
    """Test a forward pass through the model."""
    print("\n" + "="*60)
    print("Testing: Forward Pass")
    print("="*60)
    
    if skip_if_no_gpu and not torch.cuda.is_available():
        print("⚠ Skipping forward pass test (no GPU available)")
        return
    
    config = LlamaLoraConfig(
        model_id="meta-llama/Llama-3.2-1B",
        lora_r=8,
        lora_alpha=16,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        max_text_length=128,
    )
    
    try:
        print("Setting up model...")
        model, tokenizer = setup_model_for_lora_training(config)
        
        # Create data collator
        collator = LlamaDataCollator(
            tokenizer=tokenizer,
            max_text_length=config.max_text_length,
        )
        
        # Create a test batch
        features = [
            {"instruction": "Correct: hello wrold", "response": "hello world"},
            {"instruction": "Correct: good mroning", "response": "good morning"},
        ]
        
        batch = collator(features)
        
        # Move to device
        device = next(model.parameters()).device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        print("Running forward pass...")
        model.train()
        outputs = model(**batch)
        
        # Check outputs
        assert hasattr(outputs, 'loss')
        assert outputs.loss is not None
        assert not torch.isnan(outputs.loss)
        assert outputs.loss.requires_grad
        
        print("✓ Forward pass successful")
        print(f"  Loss: {outputs.loss.item():.4f}")
        print(f"  Logits shape: {outputs.logits.shape}")
        
        # Test backward pass
        print("Testing backward pass...")
        outputs.loss.backward()
        
        # Check gradients exist
        has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        
        assert has_grad, "No gradients computed"
        print("✓ Backward pass successful")
        
    except Exception as e:
        print(f"⚠ Forward pass test failed: {e}")
        raise


def run_all_tests(test_model: bool = False):
    """Run all tests."""
    print("\n" + "="*60)
    print("LMCorrector Training Pipeline Tests")
    print("="*60)
    
    # Basic tests (no GPU required)
    test_config_creation()
    test_instruction_formatting()
    test_custom_format_detection()
    test_dataset_creation()
    test_data_collator()
    test_label_masking()
    
    # Model tests (GPU required, optional)
    if test_model:
        test_model_setup(skip_if_no_gpu=True)
        test_forward_pass(skip_if_no_gpu=True)
    
    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test LMCorrector training pipeline")
    parser.add_argument("--test-model", action="store_true", help="Also test model loading and forward pass")
    args = parser.parse_args()
    
    run_all_tests(test_model=args.test_model)
