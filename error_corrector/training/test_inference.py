#!/usr/bin/env python3
"""
Simple inference script to test the trained error corrector model.

Usage:
    cd /home/duy/PlayWithMino/SimulStreaming
    PYTHONPATH=. python error_corrector/training/test_inference.py --checkpoint runs/exp--2025-12-29--12-17-10/checkpoint-980

This script loads a few samples from the training data and runs inference to check
the model's output quality.

IMPORTANT: This script follows the preprocessing from 
CorrectorDataproc._process() in error_corrector/model/corrector_data_proc.py
to ensure the prompt format matches what the model was finetuned on.

Training preprocessing creates sequence:
    [BOS, prompt_tokens..., audio_padding..., continuation_tokens..., EOS]

For inference, we provide:
    [BOS, prompt_tokens..., audio_padding...]
and let the model generate the continuation.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import transformers

# Import the exact template used during training
from error_corrector.data.types import ERROR_CORRECTOR_TEMPLATE


def load_samples(samples_path: str, num_samples: int = 5) -> list:
    """Load a few samples from the training data."""
    samples = []
    with open(samples_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            samples.append(json.loads(line))
    return samples


def main():
    parser = argparse.ArgumentParser(description="Test inference on trained model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="runs/exp--2025-12-29--19-04-25/checkpoint-688",
        help="Path to the checkpoint directory"
    )
    parser.add_argument(
        "--samples",
        type=str,
        default="error_corrector/data/sample_custom_data/samples.jsonl",
        help="Path to samples file"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples to test"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--use_audio",
        action="store_true",
        default=True,
        help="Use audio embeddings (if available)"
    )
    parser.add_argument(
        "--legacy_mode",
        action="store_true",
        default=False,
        help="Use legacy mode (add BOS after audio) for models trained before the fix"
    )
    args = parser.parse_args()

    print(f"Loading model from: {args.checkpoint}")
    print(f"Using device: {args.device}")
    print(f"Using audio: {args.use_audio}")
    print(f"Legacy mode (BOS after audio): {args.legacy_mode}")

    # Load the model - use float32 to avoid fp16 issues
    model = transformers.AutoModel.from_pretrained(
        args.checkpoint,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model = model.to(args.device)
    model.eval()

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.checkpoint,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    print(f"Model loaded successfully!")
    print(f"Model type: {type(model).__name__}")

    # Load samples
    samples = load_samples(args.samples, args.num_samples)
    print(f"\nLoaded {len(samples)} samples for testing\n")

    # Process each sample
    for i, sample in enumerate(samples):
        print(f"{'=' * 60}")
        print(f"Sample {i + 1}:")
        print(f"  Previous: '{sample['previous_transcript']}'")
        print(f"  Candidates: {sample['k_best_candidates']}")
        print(f"  Ground truth: '{sample['continuation_transcript']}'")

        # ============================================================
        # Build prompt text using the EXACT same format as 
        # CorrectorDataproc._preprocess() in corrector_data_proc.py
        # ============================================================
        prompt_text = ERROR_CORRECTOR_TEMPLATE.format(
            prev_display=sample['previous_transcript'],
            prompt_lines="\n".join(sample['k_best_candidates']),
        )

        # Tokenize prompt (this is what the model sees as input)
        prompt_inputs = tokenizer(prompt_text, return_tensors="pt")
        prompt_ids = prompt_inputs["input_ids"].to(args.device)
        prompt_attention_mask = prompt_inputs["attention_mask"].to(args.device)

        # Optional: load audio embeddings and compute audio token length
        audio_values = None
        audio_token_start_idx = None
        audio_token_len = None
        audio_batch_size = None
        
        if args.use_audio:
            audio_embed_path = sample.get('audio_embed_path', '')
            if audio_embed_path:
                try:
                    # Load audio embedding from npy file
                    # Note: saved embeddings have shape (1, T, C), but we need (B, T, C)
                    # where B=1 for single sample inference. The leading dim is from the
                    # encoder, but we treat it as batch dim directly.
                    audio_embed = np.load(audio_embed_path)
                    # Squeeze if there's an extra batch dim from saving
                    if audio_embed.ndim == 3 and audio_embed.shape[0] == 1:
                        # Shape is (1, T, C) - keep it as is for batch_size=1
                        pass
                    audio_values = torch.from_numpy(audio_embed).float().to(args.device)
                    
                    # ============================================================
                    # Compute audio token length using the SAME formula as
                    # CorrectorDataproc._process()
                    # ============================================================
                    stack_factor = 8
                    frames = audio_values.shape[1]
                    audio_token_len_value = max(1, math.ceil(frames / stack_factor))
                    
                    # Audio metadata tensors - must be 1D tensors matching training format
                    # In training: audio_token_start_idx = torch.full((1,), prompt_ids.shape[0], ...)
                    # Note: prompt_ids.shape[1] because we have batched tensor (1, seq_len)
                    audio_token_start_idx = torch.tensor(
                        [prompt_ids.shape[1]],  # audio starts after prompt
                        dtype=torch.long,
                        device=args.device,
                    )
                    audio_token_len = torch.tensor(
                        [audio_token_len_value],
                        dtype=torch.long,
                        device=args.device,
                    )
                    audio_batch_size = torch.tensor(
                        [1],  # 1 audio segment per batch item
                        dtype=torch.long,
                        device=args.device,
                    )
                    
                    # Build sequence with audio padding like _preprocess does:
                    # prompt → audio PAD → (continuation for training, but for inference we generate)
                    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                    audio_padding = torch.full(
                        (1, audio_token_len_value),
                        pad_token_id,
                        dtype=prompt_ids.dtype,
                        device=args.device,
                    )
                    
                    # Concatenate prompt and audio padding for input
                    input_ids = torch.cat([prompt_ids, audio_padding], dim=1)
                    
                    # LEGACY MODE: For models trained before the fix, add BOS token after audio
                    # The old training code tokenized continuation WITH special tokens, adding an extra BOS
                    # So the model learned: prompt → audio_pad → BOS → continuation → EOS
                    if args.legacy_mode:
                        bos_token = torch.full(
                            (1, 1),
                            tokenizer.bos_token_id,
                            dtype=prompt_ids.dtype,
                            device=args.device,
                        )
                        input_ids = torch.cat([input_ids, bos_token], dim=1)
                        print(f"  [LEGACY] Added BOS token after audio padding")
                    
                    attention_mask = torch.ones_like(input_ids)
                    
                    print(f"  Audio shape: {audio_values.shape}")
                    print(f"  Audio token len: {audio_token_len_value}")
                except Exception as e:
                    print(f"  Warning: Could not load audio: {e}")
                    # Fall back to text-only
                    input_ids = prompt_ids
                    attention_mask = prompt_attention_mask
            else:
                # No audio file available
                input_ids = prompt_ids
                attention_mask = prompt_attention_mask
        else:
            # Text-only mode
            input_ids = prompt_ids
            attention_mask = prompt_attention_mask

        # Generate
        with torch.no_grad():
            try:
                if audio_values is not None:
                    # Use model's generate with audio
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        audio_values=audio_values,
                        audio_token_start_idx=audio_token_start_idx,
                        audio_token_len=audio_token_len,
                        audio_batch_size=audio_batch_size,
                        max_new_tokens=20,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                else:
                    # Text-only: use the language model directly
                    outputs = model.language_model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=20,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                # Decode only new tokens
                new_tokens = outputs[0, input_ids.shape[1]:]
                generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                print(f"  Generated: '{generated_text.strip()}'")
                
                # Check match
                expected = sample['continuation_transcript']
                match = generated_text.strip() == expected
                print(f"  Match: {'✓' if match else '✗'}")

            except Exception as e:
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()

        print()

    print("=" * 60)
    print("Inference test completed!")


if __name__ == "__main__":
    main()
