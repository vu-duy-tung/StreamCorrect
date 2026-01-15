#!/usr/bin/env python
"""
Test script to validate the training pipeline before running actual training.

Tests:
1. Data loading and preprocessing
2. Batch collation and padding
3. Label masking correctness
4. Model forward pass
5. Loss computation sanity checks
6. Gradient flow verification

Usage:
    python test_training_pipeline.py --config training_config.yaml
    python test_training_pipeline.py --config training_config.yaml --num_samples 10
"""

import os
import argparse
import torch
import numpy as np
from transformers import AutoProcessor, AutoModel
from training import (
    UltravoxLoraConfig,
    UltravoxDataCollator,
    UltravoxTrainer,
    load_training_data,
    setup_model_for_lora_training,
)


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_pass(msg: str):
    print(f"  ✅ {msg}")


def print_fail(msg: str):
    print(f"  ❌ {msg}")


def print_warn(msg: str):
    print(f"  ⚠️  {msg}")


def print_info(msg: str):
    print(f"  ℹ️  {msg}")


def test_data_loading(config: UltravoxLoraConfig, num_samples: int = 5):
    """Test that data loads correctly."""
    print_header("TEST 1: Data Loading")
    
    try:
        dataset = load_training_data(config.train_data_path)
        print_pass(f"Dataset loaded: {len(dataset)} samples")
        print_info(f"Columns: {dataset.column_names}")
        
        # Check required columns
        required_cols = ['instruction', 'response']
        has_audio = 'audio' in dataset.column_names
        has_lazy = 'audio_path' in dataset.column_names and 'timestamp' in dataset.column_names
        
        if has_audio:
            print_pass("Has 'audio' column (pre-loaded audio)")
        elif has_lazy:
            print_pass("Has 'audio_path' and 'timestamp' columns (lazy loading)")
        else:
            print_fail("Missing audio data columns!")
            return None
        
        for col in required_cols:
            if col in dataset.column_names:
                print_pass(f"Has '{col}' column")
            else:
                print_fail(f"Missing '{col}' column!")
                return None
        
        # Check a few samples
        print_info(f"Checking first {num_samples} samples...")
        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            response = sample.get('response', '')
            if not response:
                print_warn(f"Sample {i} has empty response")
            elif len(response) > 500:
                print_warn(f"Sample {i} has very long response ({len(response)} chars)")
        
        return dataset
    
    except Exception as e:
        print_fail(f"Data loading failed: {e}")
        return None


def test_collation(processor, collator, dataset, num_samples: int = 4):
    """Test batch collation and padding."""
    print_header("TEST 2: Batch Collation")
    
    try:
        # Get samples with varying lengths
        samples = [dataset[i] for i in range(min(num_samples, len(dataset)))]
        batch = collator(samples)
        
        print_pass(f"Collated batch of {len(samples)} samples")
        
        # Check batch structure
        expected_keys = ['input_ids', 'attention_mask', 'labels', 'audio_values']
        for key in expected_keys:
            if key in batch:
                shape = batch[key].shape if isinstance(batch[key], torch.Tensor) else "N/A"
                print_pass(f"Has '{key}': shape={shape}")
            else:
                print_fail(f"Missing '{key}' in batch!")
        
        # Check shapes consistency
        batch_size = batch['input_ids'].shape[0]
        seq_len = batch['input_ids'].shape[1]
        
        if batch['attention_mask'].shape == (batch_size, seq_len):
            print_pass(f"attention_mask shape matches: {batch['attention_mask'].shape}")
        else:
            print_fail(f"attention_mask shape mismatch!")
        
        if batch['labels'].shape == (batch_size, seq_len):
            print_pass(f"labels shape matches: {batch['labels'].shape}")
        else:
            print_fail(f"labels shape mismatch!")
        
        return batch
    
    except Exception as e:
        print_fail(f"Collation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_label_masking(processor, batch):
    """Test that label masking is correct."""
    print_header("TEST 3: Label Masking")
    
    try:
        labels = batch['labels']
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        all_passed = True
        
        for i in range(labels.shape[0]):
            # Count masked vs unmasked
            masked_count = (labels[i] == -100).sum().item()
            total_count = labels[i].shape[0]
            response_count = total_count - masked_count
            
            # Get actual sequence length (non-padding)
            real_len = attention_mask[i].sum().item()
            
            # Decode response tokens
            response_ids = labels[i][labels[i] != -100]
            if len(response_ids) > 0:
                response_text = processor.tokenizer.decode(response_ids, skip_special_tokens=True)
            else:
                response_text = "<EMPTY>"
            
            print_info(f"Sample {i}: {response_count} response tokens, {masked_count} masked, {real_len} real tokens")
            print_info(f"  Response: '{response_text[:50]}{'...' if len(response_text) > 50 else ''}'")
            
            # Validations
            if response_count == 0:
                print_warn(f"  Sample {i} has NO response tokens (all masked)!")
                all_passed = False
            
            # Check padding is masked
            padding_positions = (attention_mask[i] == 0)
            padding_labels = labels[i][padding_positions]
            if len(padding_labels) > 0 and not (padding_labels == -100).all():
                print_fail(f"  Sample {i}: padding tokens not properly masked!")
                all_passed = False
        
        if all_passed:
            print_pass("All label masking checks passed")
        
        return all_passed
    
    except Exception as e:
        print_fail(f"Label masking test failed: {e}")
        return False


def test_model_forward(model, batch, device='cuda'):
    """Test model forward pass."""
    print_header("TEST 4: Model Forward Pass")
    
    try:
        model.eval()
        
        # Move batch to device
        batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        with torch.no_grad():
            outputs = model(**batch_gpu)
        
        print_pass("Model forward pass successful")
        
        # Check outputs
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            print_pass(f"Model returned loss: {outputs.loss.item():.4f}")
        else:
            print_warn("Model did not return loss directly")
        
        if hasattr(outputs, 'logits'):
            logits_shape = outputs.logits.shape
            print_pass(f"Logits shape: {logits_shape}")
            
            # Check logits are reasonable
            if torch.isnan(outputs.logits).any():
                print_fail("Logits contain NaN values!")
                return False
            if torch.isinf(outputs.logits).any():
                print_fail("Logits contain Inf values!")
                return False
            
            print_pass("Logits are finite (no NaN/Inf)")
        
        return True
    
    except Exception as e:
        print_fail(f"Model forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss_computation(model, batch, device='cuda'):
    """Test loss computation sanity."""
    print_header("TEST 5: Loss Computation Sanity")
    
    try:
        model.eval()
        batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        with torch.no_grad():
            outputs = model(**batch_gpu)
            loss = outputs.loss
        
        if loss is None:
            print_warn("Model returned None loss, computing manually...")
            logits = outputs.logits
            labels = batch_gpu['labels']
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        loss_value = loss.item()
        print_info(f"Loss value: {loss_value:.4f}")
        
        # Sanity checks
        if loss_value < 0:
            print_fail(f"Loss is negative ({loss_value})! This should not happen.")
            return False
        
        if loss_value > 100:
            print_warn(f"Loss is very high ({loss_value}). Model may need warmup or lower LR.")
        
        if torch.isnan(loss):
            print_fail("Loss is NaN!")
            return False
        
        if torch.isinf(loss):
            print_fail("Loss is Inf!")
            return False
        
        # Test that loss changes with different inputs
        print_info("Testing loss sensitivity to different responses...")
        
        # Create a batch with random labels (should have higher loss)
        batch_random = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch_gpu.items()}
        vocab_size = outputs.logits.shape[-1]
        random_labels = torch.randint(0, vocab_size, batch_random['labels'].shape, device=device)
        # Keep masking pattern
        random_labels[batch_random['labels'] == -100] = -100
        batch_random['labels'] = random_labels
        
        with torch.no_grad():
            outputs_random = model(**batch_random)
            loss_random = outputs_random.loss
            if loss_random is None:
                logits = outputs_random.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = random_labels[..., 1:].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                loss_random = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        print_info(f"Loss with random labels: {loss_random.item():.4f}")
        
        # Random labels should generally have higher loss (unless original was also random)
        if loss_random.item() > loss_value:
            print_pass("Loss is lower for correct labels than random labels (good!)")
        else:
            print_warn("Loss for random labels is not higher - may be expected for untrained model")
        
        print_pass("Loss computation sanity checks passed")
        return True
    
    except Exception as e:
        print_fail(f"Loss computation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gradient_flow(model, collator, dataset, device='cuda'):
    """Test that gradients flow correctly through trainable parameters."""
    print_header("TEST 6: Gradient Flow")
    
    try:
        model.train()
        
        # Get a fresh batch
        samples = [dataset[i] for i in range(min(2, len(dataset)))]
        batch = collator(samples)
        batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Zero gradients
        model.zero_grad()
        
        # Forward pass
        outputs = model(**batch_gpu)
        loss = outputs.loss
        
        if loss is None:
            logits = outputs.logits
            labels = batch_gpu['labels']
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Backward pass
        loss.backward()
        
        print_pass(f"Backward pass successful (loss={loss.item():.4f})")
        
        # Check gradients on trainable parameters
        trainable_with_grad = 0
        trainable_without_grad = 0
        lora_params_with_grad = 0
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None and param.grad.abs().sum() > 0:
                    trainable_with_grad += 1
                    if 'lora' in name.lower():
                        lora_params_with_grad += 1
                else:
                    trainable_without_grad += 1
                    if trainable_without_grad <= 3:  # Only show first few
                        print_warn(f"  No gradient for: {name}")
        
        print_info(f"Trainable params with gradients: {trainable_with_grad}")
        print_info(f"Trainable params without gradients: {trainable_without_grad}")
        print_info(f"LoRA params with gradients: {lora_params_with_grad}")
        
        if trainable_with_grad > 0 and lora_params_with_grad > 0:
            print_pass("Gradients flow to LoRA parameters")
        else:
            print_fail("No gradients reaching LoRA parameters!")
            return False
        
        # Check for NaN gradients
        nan_grads = 0
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                nan_grads += 1
                print_fail(f"  NaN gradient in: {name}")
        
        if nan_grads == 0:
            print_pass("No NaN gradients detected")
        else:
            print_fail(f"{nan_grads} parameters have NaN gradients!")
            return False
        
        return True
    
    except Exception as e:
        print_fail(f"Gradient flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_trainer_eval_loss(model, collator, dataset, config, device='cuda'):
    """Test that UltravoxTrainer computes eval_loss correctly."""
    print_header("TEST 7: Trainer Eval Loss")
    
    try:
        from transformers import TrainingArguments
        
        # Create minimal training args for testing
        training_args = TrainingArguments(
            output_dir="/tmp/test_trainer",
            per_device_eval_batch_size=2,
            remove_unused_columns=False,
            report_to=[],
        )
        
        # Create small eval dataset
        eval_samples = min(4, len(dataset))
        eval_dataset = dataset.select(range(eval_samples))
        
        trainer = UltravoxTrainer(
            model=model,
            args=training_args,
            eval_dataset=eval_dataset,
            data_collator=collator,
        )
        
        # Run evaluation
        print_info("Running trainer.evaluate()...")
        metrics = trainer.evaluate()
        
        if 'eval_loss' in metrics:
            print_pass(f"eval_loss computed: {metrics['eval_loss']:.4f}")
        else:
            print_fail("eval_loss not in metrics!")
            print_info(f"Available metrics: {list(metrics.keys())}")
            return False
        
        # Sanity check
        if metrics['eval_loss'] < 0:
            print_fail("eval_loss is negative!")
            return False
        
        if np.isnan(metrics['eval_loss']):
            print_fail("eval_loss is NaN!")
            return False
        
        print_pass("Trainer eval_loss computation works correctly")
        return True
    
    except Exception as e:
        print_fail(f"Trainer eval loss test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests(config_path: str, num_samples: int = 5):
    """Run all tests."""
    print_header("ULTRAVOX TRAINING PIPELINE TESTS")
    print(f"Config: {config_path}")
    print(f"Num samples to test: {num_samples}")
    
    # Load config
    config = UltravoxLoraConfig.from_yaml(config_path)
    print_info(f"Model: {config.model_id}")
    print_info(f"Train data: {config.train_data_path}")
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print_info(f"Device: {device}")
    
    results = {}
    
    # Test 1: Data loading
    dataset = test_data_loading(config, num_samples)
    results['data_loading'] = dataset is not None
    
    if not results['data_loading']:
        print("\n❌ Aborting: Data loading failed")
        return results
    
    # Load processor and create collator
    print_info("\nLoading processor...")
    processor = AutoProcessor.from_pretrained(config.model_id, trust_remote_code=True)
    collator = UltravoxDataCollator(
        processor=processor,
        max_audio_length_seconds=config.max_audio_length_seconds,
        max_text_length=config.max_text_length,
        sample_rate=config.sample_rate,
    )
    
    # Test 2: Collation
    batch = test_collation(processor, collator, dataset, num_samples)
    results['collation'] = batch is not None
    
    if not results['collation']:
        print("\n❌ Aborting: Collation failed")
        return results
    
    # Test 3: Label masking
    results['label_masking'] = test_label_masking(processor, batch)
    
    # Load model for remaining tests
    print_info("\nLoading model with LoRA (single GPU for testing)...")
    
    # Force single GPU loading for testing
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Temporarily modify config to avoid device_map="auto" spreading across GPUs
    original_bf16 = config.bf16
    original_fp16 = config.fp16
    
    # Load model directly without device_map for testing
    from peft import LoraConfig, get_peft_model, TaskType
    
    dtype = torch.bfloat16 if config.bf16 else (torch.float16 if config.fp16 else torch.float32)
    
    base_model = AutoModel.from_pretrained(
        config.model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map={"": device},  # Put everything on single device
    )
    
    # Freeze audio encoder
    if hasattr(base_model, 'audio_tower'):
        for param in base_model.audio_tower.parameters():
            param.requires_grad = False
    
    # Enable projector training
    if hasattr(base_model, 'multi_modal_projector'):
        for param in base_model.multi_modal_projector.parameters():
            param.requires_grad = config.train_projector
    
    # Apply LoRA
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, lora_config)
    print_info(f"Model loaded on {device}")
    
    # Test 4: Model forward pass
    results['forward_pass'] = test_model_forward(model, batch, device)
    
    if not results['forward_pass']:
        print("\n❌ Aborting: Forward pass failed")
        return results
    
    # Test 5: Loss computation
    results['loss_computation'] = test_loss_computation(model, batch, device)
    
    # Test 6: Gradient flow
    results['gradient_flow'] = test_gradient_flow(model, collator, dataset, device)
    
    # Test 7: Trainer eval loss
    results['trainer_eval'] = test_trainer_eval_loss(model, collator, dataset, config, device)
    
    # Summary
    print_header("TEST SUMMARY")
    total = len(results)
    passed = sum(results.values())
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Ready for training.")
    else:
        print("\n⚠️  Some tests failed. Please fix issues before training.")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test training pipeline")
    parser.add_argument("--config", type=str, default="training_config.yaml", help="Path to config")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to test")
    args = parser.parse_args()
    
    run_all_tests(args.config, args.num_samples)


if __name__ == "__main__":
    main()
