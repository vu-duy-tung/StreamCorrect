#!/usr/bin/env python3
"""
Evaluation module for transcription quality using CER (Character Error Rate).
Can be used as a library or standalone script.
"""

import os
import re
import json
import cn2an
import string
import logging
import argparse
import unicodedata
from typing import List, Optional, Tuple
from pathlib import Path
from jiwer import cer 
from opencc import OpenCC
from jiwer import wer

# Module logger - will use the parent logger when imported
logger = logging.getLogger(__name__)

converter_zh = OpenCC('t2s')
converter_yue = OpenCC('s2hk')


_CJK_RANGES = (
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0x20000, 0x2A6DF),  # Extension B
    (0x2A700, 0x2B73F),  # Extension C
    (0x2B740, 0x2B81F),  # Extension D
    (0x2B820, 0x2CEAF),  # Extension E
    (0x2CEB0, 0x2EBEF),  # Extensions F/G
)


def _contains_cjk(text: str) -> bool:
    for char in text:
        codepoint = ord(char)
        if any(start <= codepoint <= end for start, end in _CJK_RANGES):
            return True
    return False


def _resolve_normalization_language(language: str, text: str) -> Optional[str]:
    """Infer whether Cantonese/Mandarin normalization should be applied."""
    if not text:
        return None

    lang = (language or '').strip().lower()
    if lang in {'yue', 'zh'}:
        return lang
    if 'yue' in lang or 'canton' in lang or 'hongkong' in lang or 'hk' in lang:
        return 'yue'
    if any(token in lang for token in ('zh', 'cmn', 'mandarin', 'chi', 'cn', 'zh-cn')):
        return 'zh'

    if _contains_cjk(text):
        # Default to Mandarin normalization when we detect CJK without explicit hint
        return 'zh'

    return None


def normalize_utterances(text: str, language: str = "yue") -> str:
    # ---- Normalize script ----
    if language == 'yue':
        text = converter_yue.convert(text)
    elif language == 'zh':
        text = converter_zh.convert(text)

    # ---- Normalize percentages and numbers ----
    # Handle % first
    percent_pattern = r"(\d+(\.\d+)?)\s*%"

    for match in re.finditer(percent_pattern, text):
        num = match.group(1)
        try:
            chinese_num = cn2an.transform(num, "an2cn")
        except Exception:
            continue
        # Convert to "百分之X"
        percent_text = f"百分之{chinese_num}"
        if language == 'yue':
            percent_text = converter_yue.convert(percent_text)
        else:
            percent_text = converter_zh.convert(percent_text)
        text = text.replace(match.group(0), percent_text)

    # ---- Normalize plain digits ----
    number_pattern = r"\d+(\.\d+)?"
    for match in re.finditer(number_pattern, text):
        num = match.group(0)
        try:
            chinese_num = cn2an.transform(num, "an2cn")
        except Exception:
            continue
        if language == 'yue':
            chinese_num = converter_yue.convert(chinese_num)
        else:
            chinese_num = converter_zh.convert(chinese_num)
        text = text.replace(num, chinese_num)

    # ---- Normalize numeric variants ----
    replacements = {
        "两": "二",
        "兩": "二",
        "俩": "二",
        "萬": "万",
        "億": "亿"
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # ---- Lowercase English letters ----
    text = ''.join(c.lower() if ord(c) < 128 and c.isalpha() else c for c in text)

    return text

def remove_punctuation_unicode(s: str) -> str:
    # Remove characters whose Unicode category starts with 'P' (punctuation)
    return ''.join(ch for ch in s if not unicodedata.category(ch).startswith('P'))


def replace_punctuation_with_space(s: str) -> str:
    if not s:
        return s

    # Replace ASCII punctuation with spaces
    translation_table = str.maketrans({ch: ' ' for ch in string.punctuation})
    s = s.translate(translation_table)

    # Replace Unicode punctuation categories with spaces
    s = ''.join(' ' if unicodedata.category(ch).startswith('P') else ch for ch in s)
    return s


def _normalize_for_mer(text: str, language: str) -> Tuple[str, Optional[str]]:
    """Normalize text while preserving spaces required for mixed tokenization."""
    normalized = text.strip()
    if not normalized:
        return '', None

    normalization_language = _resolve_normalization_language(language, normalized)

    if normalization_language:
        normalized = normalize_utterances(normalized, language=normalization_language)
    else:
        normalized = normalized.lower()

    normalized = replace_punctuation_with_space(normalized)

    # Collapse repeated whitespace and trim
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized, normalization_language


def mixed_tokenize(text: str, language: str) -> List[str]:
    """
    Tokenize text into a sequence that mixes single CJK characters and Latin words.
    """
    normalized, normalization_language = _normalize_for_mer(text, language)
    if not normalized:
        return []

    token_language = normalization_language or (language if language in ['yue', 'zh'] else None)

    if token_language in ['yue', 'zh']:
        tokens: List[str] = []
        ascii_buffer: List[str] = []

        for char in normalized:
            if char.isspace():
                if ascii_buffer:
                    tokens.append(''.join(ascii_buffer))
                    ascii_buffer.clear()
                continue

            if ord(char) < 128:
                ascii_buffer.append(char)
                continue

            if ascii_buffer:
                tokens.append(''.join(ascii_buffer))
                ascii_buffer.clear()

            tokens.append(char)

        if ascii_buffer:
            tokens.append(''.join(ascii_buffer))

        return tokens

    return normalized.split()

def calculate_cer(reference: str, hypothesis: str, language: str) -> float:
    """
    Calculate Character Error Rate (CER) between reference and hypothesis strings.
    """
    # Normalize strings
    ref = reference.strip()
    hyp = hypothesis.strip()

    normalization_language = _resolve_normalization_language(language, ref + hyp)

    if normalization_language:
        ref = normalize_utterances(ref, language=normalization_language)
        hyp = normalize_utterances(hyp, language=normalization_language)

        # Remove spaces and punctuation for Cantonese/Yue
        ref = ref.translate(str.maketrans('', '', string.punctuation + ' '))
        hyp = hyp.translate(str.maketrans('', '', string.punctuation + ' '))
        ref = remove_punctuation_unicode(ref)
        hyp = remove_punctuation_unicode(hyp)
    else:
        # For other languages, just lower case and strip
        ref = ref.lower()
        hyp = hyp.lower()
        ref = ref.translate(str.maketrans('', '', string.punctuation))
        hyp = hyp.translate(str.maketrans('', '', string.punctuation))

    print("Ref:", ref)
    print("Hyp:", hyp)
    return cer(ref, hyp)

def calculate_mer(reference: str, hypothesis: str, language: str, char_weight: float = 0.5) -> float:
    """
    Calculate Mixed Error Rate (MER) using mixed character/word tokenization.

    The optional ``char_weight`` argument is kept for backwards compatibility but
    no longer affects the computation.
    """
    ref_tokens = mixed_tokenize(reference, language)
    hyp_tokens = mixed_tokenize(hypothesis, language)

    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0

    ref_seq = ' '.join(ref_tokens)
    hyp_seq = ' '.join(hyp_tokens)

    print("[PLAY WITH MINO] - Mixed tokenization:")
    print(ref_tokens)
    print(hyp_tokens)

    return wer(ref_seq, hyp_seq)

def load_references(reference_file, language='en'):
    """
    Load reference transcriptions from JSON file.
    
    Args:
        reference_file (str): Path to JSON file containing references
        language (str): Language code for reference text field (default: 'en')
    
    Returns:
        dict: Dictionary mapping audio file basenames to reference texts
    """
    if reference_file.endswith('/WSYue-ASR-eval/Short/content.json'):
        language = 'yue'
        
    with open(reference_file, 'r', encoding='utf-8') as f:
        references = json.load(f)
    
    # Create mapping from audio filename to reference text
    ref_map = {}
    text_field = f'text_{language}'
    
    for item in references:
        audio_path = item.get('audio_path', '')
        basename = os.path.basename(audio_path)
        # Remove extension for matching
        basename_noext = os.path.splitext(basename)[0]
        
        # Get reference text for specified language
        ref_text = item.get(text_field, '')
        
        if ref_text:
            ref_map[basename] = ref_text
            ref_map[basename_noext] = ref_text
    
    logger.info(f"Loaded {len(ref_map)} reference transcriptions for language '{language}'")
    return ref_map


def load_generated_transcriptions(transcription_file):
    """
    Load generated transcriptions from JSON file (batch output).
    
    Args:
        transcription_file (str): Path to batch_transcriptions.json
    
    Returns:
        dict: Dictionary mapping filenames to generated texts
    """
    with open(transcription_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    gen_map = {}
    for result in data.get('results', []):
        filename = result.get('file', '')
        text = result.get('transcription', '')
        
        basename_noext = os.path.splitext(filename)[0]
        gen_map[filename] = text
        gen_map[basename_noext] = text
    
    logger.info(f"Loaded {len(gen_map)} generated transcriptions")
    return gen_map


def load_individual_transcriptions(logdir):
    """
    Load individual transcription files from directory.
    
    Args:
        logdir (str): Directory containing *_transcription.txt files
    
    Returns:
        dict: Dictionary mapping filenames to generated texts
    """
    gen_map = {}
    logdir_path = Path(logdir)
    
    for txt_file in logdir_path.glob('*_transcription.txt'):
        # Extract original filename
        filename = txt_file.stem.replace('_transcription', '')
        
        with open(txt_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        gen_map[filename] = text
        # Also store with .wav extension for matching
        gen_map[f"{filename}.wav"] = text
    
    logger.info(f"Loaded {len(gen_map)} individual transcription files")
    return gen_map


def evaluate_transcriptions(references, generated, language):
    """
    Evaluate generated transcriptions against references using CER and MER.

    Args:
        references (dict): Reference transcriptions
        generated (dict): Generated transcriptions

    Returns:
        dict: Evaluation results including per-file metrics and averages
    """
    results = []
    total_cer = 0.0
    total_mer = 0.0
    matched_count = 0
    
    for filename, ref_text in references.items():
        if filename in generated:
            gen_text = generated[filename]
            cer_score = calculate_cer(ref_text, gen_text, language)
            mer_score = calculate_mer(ref_text, gen_text, language)
            
            results.append({
                'file': filename,
                'reference': ref_text,
                'generated': gen_text,
                'cer': cer_score,
                'mer': mer_score,
                'ref_length': len(ref_text),
                'gen_length': len(gen_text)
            })
            
            total_cer += cer_score
            total_mer += mer_score
            matched_count += 1
            
            logger.debug(f"{filename}: CER={cer_score:.4f} MER={mer_score:.4f}")
    
    # Calculate average CER
    avg_cer = total_cer / matched_count if matched_count > 0 else 0.0
    avg_mer = total_mer / matched_count if matched_count > 0 else 0.0
    
    summary = {
        'total_files': len(references),
        'matched_files': matched_count,
        'unmatched_files': len(references) - matched_count,
        'average_cer': avg_cer,
        'average_mer': avg_mer,
        'per_file_results': results
    }
    
    return summary


def main():
    # Setup logging for standalone execution
    logging.basicConfig(level=logging.INFO, format='%(levelname)s\t%(message)s')
    
    parser = argparse.ArgumentParser(
        description='Evaluate transcription quality using CER metric'
    )
    
    parser.add_argument(
        'reference_file',
        type=str,
        help='Path to JSON file containing reference transcriptions (with audio_path and text_<language> fields)'
    )
    
    parser.add_argument(
        '--logdir',
        type=str,
        required=True,
        help='Directory containing generated transcriptions (batch_transcriptions.json or individual files)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save evaluation results JSON (default: <logdir>/evaluation_results.json)'
    )
    
    parser.add_argument(
        '--batch-file',
        type=str,
        default='batch_transcriptions.json',
        help='Name of batch transcription file (default: batch_transcriptions.json)'
    )
    
    parser.add_argument(
        '--language',
        type=str,
        default='en',
        help='Language code for reference text field (default: en)'
    )
    
    parser.add_argument(
        '-l', '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set the log level'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    logger.setLevel(args.log_level)
    
    # Load references
    logger.info(f"Loading references from: {args.reference_file}")
    references = load_references(args.reference_file, language=args.language)
    
    # Load generated transcriptions
    logger.info(f"Loading generated transcriptions from: {args.logdir}")
    
    # Try to load batch file first
    batch_file = os.path.join(args.logdir, args.batch_file)
    if os.path.exists(batch_file):
        logger.info(f"Found batch transcription file: {batch_file}")
        generated = load_generated_transcriptions(batch_file)
    else:
        logger.info(f"Batch file not found, loading individual transcription files")
        generated = load_individual_transcriptions(args.logdir)
    
    if not generated:
        logger.error("No generated transcriptions found!")
        return 1
    
    # Evaluate
    logger.info("Evaluating transcriptions...")
    evaluation_results = evaluate_transcriptions(references, generated, args.language)
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Total reference files: {evaluation_results['total_files']}")
    print(f"Matched files: {evaluation_results['matched_files']}")
    print(f"Unmatched files: {evaluation_results['unmatched_files']}")
    print(f"\nAverage CER: {evaluation_results['average_cer']:.4f} ({evaluation_results['average_cer']*100:.2f}%)")
    print(f"Average MER: {evaluation_results['average_mer']:.4f} ({evaluation_results['average_mer']*100:.2f}%)")
    print("=" * 80)
    
    # Print per-file results
    if args.log_level == 'DEBUG' or args.log_level == 'INFO':
        print("\nPer-file results:")
        print("-" * 80)
        for result in evaluation_results['per_file_results']:
            print(f"{result['file']:40s} CER: {result['cer']:.4f} ({result['cer']*100:.2f}%)  MER: {result['mer']:.4f} ({result['mer']*100:.2f}%)")
    
    # Save results
    output_file = args.output or os.path.join(args.logdir, 'evaluation_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Evaluation results saved to: {output_file}")
    
    return 0


if __name__ == '__main__':
    exit(main())
