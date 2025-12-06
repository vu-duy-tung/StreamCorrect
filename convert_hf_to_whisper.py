#!/usr/bin/env python3
"""
Convert HuggingFace Whisper model to OpenAI Whisper format.

Usage:
    python convert_hf_to_whisper.py \
        --hf-model JackyHoCL/whisper-large-v3-turbo-cantonese-yue-english \
        --output-path ./cantonese-yue-en.pt
"""

import argparse
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def convert_hf_to_whisper_format(hf_model_name, output_path):
    """
    Convert a HuggingFace Whisper model to OpenAI Whisper checkpoint format.
    
    Args:
        hf_model_name: HuggingFace model name or path
        output_path: Path to save the converted .pt file
    """
    print(f"Loading HuggingFace model: {hf_model_name}")
    
    # Load the model from HuggingFace
    model = WhisperForConditionalGeneration.from_pretrained(hf_model_name)
    processor = WhisperProcessor.from_pretrained(hf_model_name)
    
    # Get model dimensions
    config = model.config
    
    dims = {
        "n_mels": config.num_mel_bins,
        "n_audio_ctx": config.max_source_positions,
        "n_audio_state": config.d_model,
        "n_audio_head": config.encoder_attention_heads,
        "n_audio_layer": config.encoder_layers,
        "n_vocab": config.vocab_size,
        "n_text_ctx": config.max_target_positions,
        "n_text_state": config.d_model,
        "n_text_head": config.decoder_attention_heads,
        "n_text_layer": config.decoder_layers,
    }
    
    print(f"Model dimensions: {dims}")
    
    # Create the checkpoint in OpenAI Whisper format
    checkpoint = {
        "dims": dims,
        "model_state_dict": model.state_dict(),
    }
    
    print(f"Saving converted model to: {output_path}")
    torch.save(checkpoint, output_path)
    print(f"✓ Conversion complete!")
    print(f"\nYou can now use this model with:")
    print(f"  python simulstreaming_whisper.py --model_path {output_path} --lan yue <audio_path>")


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace Whisper model to OpenAI Whisper format"
    )
    parser.add_argument(
        "--hf-model",
        type=str,
        required=True,
        help="HuggingFace model name (e.g., khleeloo/whisper-large-v3-cantonese)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output path for the converted .pt file (e.g., ./cantonese-yue-en.pt)",
    )
    
    args = parser.parse_args()
    
    convert_hf_to_whisper_format(args.hf_model, args.output_path)


if __name__ == "__main__":
    main()
