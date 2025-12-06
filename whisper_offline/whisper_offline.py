#!/usr/bin/env python3
"""
Batch inference script for Whisper Cantonese models with CER evaluation.

Usage:
    python whisper_offline.py --audio_dir <path> --reference_file <path> --output_dir <path>
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import librosa
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from tqdm import tqdm
from evaluate import calculate_cer
import time



def load_references(reference_file: str) -> Dict[str, str]:
    """
    Load reference transcriptions from JSON file.
    
    Args:
        reference_file: Path to JSON file with references
        
    Returns:
        Dictionary mapping file IDs to reference text
    """
    with open(reference_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    references = {}
    for item in data:
        # Handle different file ID keys
        file_id = item.get("file") or item.get("audio_id") or item.get("id")
        
        # If no file_id, try to extract from audio_path
        if not file_id and "audio_path" in item:
            file_id = os.path.splitext(os.path.basename(item["audio_path"]))[0]
        
        # Handle different reference text keys (including language-specific ones)
        text = (item.get("text") or 
                item.get("reference") or 
                item.get("transcription") or
                item.get("text_yue") or  # Cantonese
                item.get("text_en") or   # English
                item.get("text_zh") or   # Chinese
                item.get("text_ja"))     # Japanese
        
        if file_id and text:
            # Remove file extension if present
            file_id = os.path.splitext(file_id)[0]
            references[file_id] = text
    
    return references


class WhisperBatchInference:
    """Batch inference handler for Whisper models."""
    
    def __init__(self, model_name: str, device: str = None, batch_size: int = 1, language: str = None):
        """
        Initialize the inference handler.
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            batch_size: Batch size for processing (1 for sequential)
            language: Language code (e.g., 'yue' for Cantonese, 'en' for English, None for auto-detect)
        """
        print(f"Loading model: {model_name}")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.model.eval()
        self.batch_size = batch_size
        self.language = language
        
        if self.language:
            print(f"Model loaded on: {self.device} with language: {self.language}")
        else:
            print(f"Model loaded on: {self.device}")
    
    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe a single audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        # Load audio
        speech, sr = librosa.load(audio_path, sr=16000)
        
        # Process audio
        inputs = self.processor(speech, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Prepare generation kwargs
        generate_kwargs = {}
        if self.language:
            # Set language for generation
            generate_kwargs["language"] = self.language
        
        # Generate transcription
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **generate_kwargs)
        
        # Decode
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return transcription
    
    def process_directory(
        self, 
        audio_dir: str, 
        references: Dict[str, str],
        output_dir: str = None,
        num_audios: int = None
    ) -> Tuple[List[Dict], float]:
        """
        Process all audio files in a directory and evaluate.
        
        Args:
            audio_dir: Directory containing audio files
            references: Dictionary of reference transcriptions
            output_dir: Optional directory to save results
            num_audios: Limit number of files to process (None = all files)
            
        Returns:
            Tuple of (results list, average CER)
        """
        audio_dir = Path(audio_dir)
        
        # Find all audio files
        audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(audio_dir.glob(f"**/*{ext}"))
        
        audio_files = sorted(audio_files)
        print(f"Found {len(audio_files)} audio files")
        
        # Limit number of files if specified
        if num_audios is not None and num_audios > 0:
            audio_files = audio_files[:num_audios]
            print(f"Limiting to first {num_audios} audio files")
        
        # Process files
        results = []
        total_cer = 0.0
        matched_count = 0
        unmatched_count = 0
        
        for audio_file in tqdm(audio_files, desc="Processing audio files"):
            file_id = audio_file.stem
            
            # Check if reference exists
            if file_id not in references:
                unmatched_count += 1
                continue
            
            reference_text = references[file_id]
            
            # Transcribe
            start_time = time.time()
            try:
                generated_text = self.transcribe(str(audio_file))
                inference_time = (time.time() - start_time) * 1000  # Convert to ms
                
                # Calculate CER
                cer = calculate_cer(reference_text, generated_text, language=self.language)
                total_cer += cer
                matched_count += 1
                
                # Store result
                result = {
                    "file": file_id,
                    "reference": reference_text,
                    "generated": generated_text,
                    "cer": cer,
                    "ref_length": len(reference_text.replace(" ", "")),
                    "gen_length": len(generated_text.replace(" ", "")),
                    "inference_time_ms": inference_time
                }
                results.append(result)
                
            except Exception as e:
                print(f"\nError processing {file_id}: {str(e)}")
                continue
        
        # Calculate average CER
        avg_cer = total_cer / matched_count if matched_count > 0 else 0.0
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Evaluation Summary")
        print(f"{'='*60}")
        print(f"Total audio files: {len(audio_files)}")
        print(f"Matched files: {matched_count}")
        print(f"Unmatched files: {unmatched_count}")
        print(f"Average CER: {avg_cer:.4f} ({avg_cer*100:.2f}%)")
        print(f"{'='*60}")
        
        # Save results
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            results_file = output_dir / "evaluation_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "total_files": len(audio_files),
                    "matched_files": matched_count,
                    "unmatched_files": unmatched_count,
                    "average_cer": avg_cer,
                    "per_file_results": results
                }, f, ensure_ascii=False, indent=2)
            
            print(f"Results saved to: {results_file}")
        
        return results, avg_cer


def main():
    parser = argparse.ArgumentParser(
        description="Batch inference and evaluation for Whisper Cantonese models"
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        required=True,
        help="Directory containing audio files"
    )
    parser.add_argument(
        "--reference_file",
        type=str,
        required=True,
        help="JSON file with reference transcriptions"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save evaluation results (default: ./results)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="openai/whisper-large-v3",
        help="HuggingFace model name (default: khleeloo/whisper-large-v3-cantonese)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help="Device to use (default: auto-detect)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for processing (default: 1)"
    )
    parser.add_argument(
        "--num-audios",
        type=int,
        default=None,
        help="Limit the number of audio files to process (useful for testing). Process all files if not specified."
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language code for transcription (e.g., 'yue' for Cantonese, 'en' for English, 'zh' for Chinese). If not specified, auto-detection is used."
    )
    
    args = parser.parse_args()
    
    # Load references
    print(f"Loading references from: {args.reference_file}")
    references = load_references(args.reference_file)
    print(f"Loaded {len(references)} references")
    
    # Initialize inference handler
    inference = WhisperBatchInference(
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        language=args.language
    )
    
    # Process directory
    results, avg_cer = inference.process_directory(
        audio_dir=args.audio_dir,
        references=references,
        output_dir=args.output_dir,
        num_audios=args.num_audios
    )
    
    print(f"\nEvaluation complete! Average CER: {avg_cer:.4f}")


if __name__ == "__main__":
    main()
