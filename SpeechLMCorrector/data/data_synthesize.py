import os
import re
import subprocess
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Dict, Any, Optional
import random
import json
import threading
from tqdm import tqdm

FAILED_REF_HYP_LOG = "SpeechLMCorrector/data/sample_custom_data/failed_ref_hyp.txt"
random.seed(21)

# Thread-safe lock for writing to output file
_write_lock = threading.Lock()

def _log_failed_audio_path(audio_path: str) -> None:
    os.makedirs(os.path.dirname(FAILED_REF_HYP_LOG), exist_ok=True)
    with open(FAILED_REF_HYP_LOG, "a", encoding="utf-8") as f:
        f.write(audio_path + "\n")


def extract_ref_hyp(stdout: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract reference and hypothesis from Stdout."""
    ref = hyp = None
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("Ref:"):
            ref = line[len("Ref:"):].strip()
        elif line.startswith("Hyp:"):
            hyp = line[len("Hyp:"):].strip()
    return ref, hyp


def parse_stderr_topk(stderr: str, k: int = 4) -> List[Dict[str, Any]]:
    """
    For each chunk (between 'The system received audio from ... to ...'):
      - 'start', 'end' : time range (float seconds)
      - 'topk': list[str] = last k DEBUG <|startoftranscript|>... candidates
                from the *last decoding loop* in that chunk
      - 'outputs': list[str] = all 'INFO    Output: ...' in that chunk
    """

    chunk_header = re.compile(
        r"INFO\s+The system received audio from\s+([\d.]+)\s+s to\s+([\d.]+)\s+s"
    )
    # Accept any language token (e.g. <|zh|>, <|yue|>) instead of hardcoding one.
    debug_line = re.compile(
        r'^DEBUG\s+<\|startoftranscript\|\><\|[^|>]+\|\><\|transcribe\|\><\|notimestamps\|\>(.*)$'
    )

    chunks: List[Dict[str, Any]] = []
    current_chunk: Optional[Dict[str, Any]] = None
    current_loop_debugs: List[str] = []

    for line in stderr.splitlines():
        if re.match(r"^INFO\s+Finish", line):
            break

        # Start of a new chunk
        m = chunk_header.search(line)
        if m:
            # close previous chunk
            if current_chunk is not None:
                if current_loop_debugs:
                    current_chunk["topk"] = current_loop_debugs[-k:]
                chunks.append(current_chunk)

            current_chunk = {
                "start": float(m.group(1)),
                "end": float(m.group(2)),
                "audio_embed_path": "",
                "previous_transcript" : "",
                "continuation_transcript" : "",
                "topk": [],
                "outputs": [],
            }
            current_loop_debugs = []
            continue

        if current_chunk is None:
            continue

        # New decoding loop: reset per-loop buffer
        if re.search(r"INFO\s+Decoding loop starts", line):
            current_loop_debugs = []
            continue

        # Collect transcript DEBUG lines
        m_dbg = debug_line.match(line)
        if m_dbg:
            current_loop_debugs.append(m_dbg.group(1))
            continue

        if re.search(r"INFO\s+Saved encoder_feature to", line):
            audio_embed_path = re.split(r"INFO\s+Saved encoder_feature to", line, 1)[1].strip()
            current_chunk["audio_embed_path"] = audio_embed_path

        # Collect outputs
        if re.search(r"INFO\s+Output:", line):
            current_chunk["outputs"].append(
                re.split(r"INFO\s+Output:", line, 1)[1].strip()
            )
            continue

        if re.search(r"INFO\s+Previous confirmed transcript:", line):
            prev_transcript = re.split(r"INFO\s+Previous confirmed transcript:", line, 1)[1].strip()
            current_chunk["previous_transcript"] = prev_transcript
            continue

    # Close last chunk
    if current_chunk is not None:
        if current_loop_debugs:
            current_chunk["topk"] = current_loop_debugs[-k:]
        chunks.append(current_chunk)

    return chunks


def prepare_error_correction_data(
    audio_path: str,
    reference_file: str,
    streaming_asr_script: str,
    chunk_size: int = 500,
    number_of_candidates: int = 4,
    cuda_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    env = os.environ.copy()
    if cuda_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_id

    env["REFERENCE_FILE"] = reference_file
    env["AUDIO_PATH"] = audio_path  # override AUDIO_PATH inside bash
    env["VAC_CHUNK_SIZE"] = str(chunk_size / 1000.0)  # in seconds
    env["BEAM_SIZE"] = str(number_of_candidates)
    # env["MODEL_PATH"] = "large-v2.pt"
    # env["MODEL_PATH"] = "medium.pt"
    env["USE_ERROR_CORRECTOR"] = "false"

    result = subprocess.run(
        ["bash", streaming_asr_script],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    ref, hyp = extract_ref_hyp(result.stdout)
    # print("Reference:", ref)
    # print("Hypothesis:", hyp)

    if ref is None or hyp is None:
        print("Failed to extract Ref/Hyp from ASR output.")
        _log_failed_audio_path(audio_path)
        print(result.stdout)
        print(result.stderr)
        return []

    # Convert English characters to uppercase, remove punctuation, and handle spaces
    def normalize_text(text: str) -> str:
        """Convert English to uppercase, remove punctuation, and add spaces between English and Mandarin characters."""
        import unicodedata
        import string

        # Remove punctuation (both ASCII and Unicode)
        text = ''.join(ch for ch in text if not unicodedata.category(ch).startswith('P') and ch not in string.punctuation)

        # Convert English characters to uppercase
        processed = []
        for char in text:
            if char.isascii() and char.isalpha():
                processed.append(char.upper())
            else:
                processed.append(char)
        result = ''.join(processed)
        
        # Add spaces between English and Mandarin characters if not present
        result = re.sub(r'([A-Za-z])([\u4e00-\u9fff])', r'\1 \2', result)
        result = re.sub(r'([\u4e00-\u9fff])([A-Za-z])', r'\1 \2', result)
        return result
    
    print(ref)
    ref = normalize_text(ref)
    print(ref)
    print('-----')
    print("STDERR START")
    print(result.stderr)
    print("STDERR END")

    chunks = parse_stderr_topk(result.stderr, k=number_of_candidates)
    for i, ch in enumerate(chunks):
        # Preprocess
        while ch['previous_transcript'].endswith('\uFFFD'):
            ch['previous_transcript'] = ch['previous_transcript'][:-1]

        for j in range(len(ch["topk"])):
            while ch["topk"][j].endswith('\uFFFD'):
                ch["topk"][j] = ch["topk"][j][:-1]

        ch['previous_transcript'] = normalize_text(ch['previous_transcript'])
        ch['topk'] = [normalize_text(t) for t in ch['topk']]

        def split_into_syllables(s: str) -> list:
            # Matches Chinese characters, spaces, punctuation or splits English words into crude syllables
            tokens = []
            for word in re.findall(r'[A-Za-z]+|[\u4e00-\u9fff]|\s+|\S', s):
                if re.fullmatch(r'[A-Za-z]+', word):
                    # Basic heuristic for English syllables: consonants + vowels + optional consonant (if another follows)
                    chunks = re.findall(r'(?i)[^aeiouy]*[aeiouy]+(?:[^aeiouy](?=[^aeiouy]))?|[^aeiouy]+', word)
                    tokens.extend(chunks if chunks else [word])
                else:
                    tokens.append(word)
            return tokens

        prev_tokens = split_into_syllables(ch['previous_transcript'])
        ref_tokens = split_into_syllables(ref)

        continuation_transcript = ""

        offsets = [0] + [j for j in range(-6, 7) if j != 0]  # widened offset range for syllables
        max_count, cand_offset = -1, 0
        for offset in offsets:
            count = 0
            for p_prev in range(len(prev_tokens) - 1, -1, -1):
                p_ref = p_prev + offset
                if p_ref < 0 or p_ref >= len(ref_tokens):
                    continue
                if prev_tokens[p_prev] == ref_tokens[p_ref]:
                    count += 1
                if count > max_count:
                    max_count = count
                    cand_offset = offset

        if len(ch['topk']) > 0:
            # Prefer the first non-empty candidate; some beams can be empty strings.
            top_candidate = next((cand for cand in ch["topk"] if cand.strip()), "")
            topk_tokens = split_into_syllables(top_candidate)
            num_pred_tokens = len(topk_tokens) - len(prev_tokens)
            range_l = len(prev_tokens) + cand_offset
            range_r = min(range_l + num_pred_tokens, len(ref_tokens))
            
            # Join the extracted syllables back into a string
            continuation_transcript = "".join(ref_tokens[range_l:range_r])

        ch['continuation_transcript'] = continuation_transcript

        # print(f"\nChunk {i}: {ch['start']}s -> {ch['end']}s")
        # print("  Previous confirmed transcript:", ch["previous_transcript"])
        # print("  Continuation transcript:", ch["continuation_transcript"])
        # print("  Last top-k candidates:", ch["topk"])
        # print("  Outputs:", ch["outputs"])
        # print("  Audio embed path:", ch["audio_embed_path"])

    return chunks


def _write_samples_to_file(samples: List[Dict[str, Any]], output_path: str) -> None:
    """Thread-safe write of samples to jsonl file."""
    if not samples:
        return
    with _write_lock:
        with open(output_path, "a", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def load_existing_samples(jsonl_path: str) -> List[str]:
    """
    Load existing samples from jsonl file.
    Returns:
        - List of unique audio paths
    """
    audio_paths_set = set()
    
    if not os.path.exists(jsonl_path):
        return []
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line.strip())
            audio_path = sample["audio_path"]
            audio_paths_set.add(audio_path)

    return list(audio_paths_set)    


def _process_single_audio(
    audio_path: str,
    reference_file: str,
    cuda_id: str,
    streaming_asr_script: str,
    chunk_size: int,
    number_of_candidates: int,
    output_path: str,
) -> int:
    """Worker entry: run ASR on a single audio file pinned to a GPU and write samples immediately."""

    chunks = prepare_error_correction_data(
        audio_path,
        reference_file,
        streaming_asr_script,
        chunk_size=chunk_size,
        number_of_candidates=number_of_candidates,
        cuda_id=cuda_id,
    )

    synthesized_samples = []
    for chunk in chunks:
        continuation_transcript = chunk["continuation_transcript"]
        
        if len(continuation_transcript) > 0:
            synthesized_samples.append(
                {
                    "k_best_candidates": chunk["topk"],
                    "num_candidates": number_of_candidates,
                    "chunk_size": chunk_size,
                    "previous_transcript": chunk["previous_transcript"],
                    "continuation_transcript": continuation_transcript,
                    # "audio_embed_path": chunk["audio_embed_path"],
                    "audio_path": audio_path,
                    "timestamp": chunk["end"]
                }
            )

    # Write samples immediately after processing
    _write_samples_to_file(synthesized_samples, output_path)

    return len(synthesized_samples)


if __name__ == "__main__":
    # Configuration
    WORKERS_PER_GPU = 4  # safe upper bound given ~50GB free per GPU
    # GPU_LIST = [str(i) for i in range(7)]  # cuda:0 to cuda:6
    GPU_LIST = ["4", "5", "6", "7"]  # cuda:0 to cuda:7

    chunk_size_options = [100, 500, 1000, 1500]
    number_of_candidates_options = [1, 2, 4, 8]

    streaming_asr_script = "runs/run_single_eval_aishell.sh"
    reference_file = "/data/mino/StreamCorrect/qwen3-asr-ft-dataset/chinese_companies_2000/transcript.json"
    train_aishell_folder = "/data/mino/StreamCorrect/qwen3-asr-ft-dataset/chinese_companies_2000/wav/train"

    existing_samples_path = "SpeechLMCorrector/data/sample_custom_data/chinese_companies.jsonl"

    # Prepare output file path
    output_dir = "SpeechLMCorrector/data/sample_custom_data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "chinese_companies.jsonl")

    existing_audio_paths = load_existing_samples(existing_samples_path)
    print(f"Loaded {len(existing_audio_paths)} unique audio paths from {existing_samples_path}")

    # Collect audio files from train folder
    all_audio_files = [os.path.join(train_aishell_folder, f) for f in os.listdir(train_aishell_folder) if f.endswith(".wav")]
    print(f"Found {len(all_audio_files)} audio files in {train_aishell_folder}")
    
    existing_audio_basenames = {os.path.basename(p) for p in existing_audio_paths}
    
    audio_paths = [ap for ap in all_audio_files if os.path.basename(ap) not in existing_audio_basenames]
    print(f"After filtering existing samples: {len(audio_paths)} new audio paths to process")
    
    # Randomize noise levels for each file if the folder includes "snr"
    if "snr" in train_aishell_folder:
        snr_levels = ["snr-5", "snr0", "snr5", "snr10"]
        randomized_paths = []
        for ap in audio_paths:  
            chosen_snr = random.choice(snr_levels)
            # We assume the base folder we read from has snr-5 since that's what was hardcoded previously,
            # but we can do a generic regex replace for any snr suffix in the path just in case.
            new_ap = re.sub(r'snr[-]?\d+', chosen_snr, ap)
            randomized_paths.append(new_ap)
        audio_paths = randomized_paths
    
    # Optionally limit the number of files
    max_files = 10000
    if len(audio_paths) > max_files:
        audio_paths = random.sample(audio_paths, max_files)
        print(f"Randomly selected {len(audio_paths)} audio paths")
    
    if len(audio_paths) == 0:
        print("No new audio files to process. Exiting.")
        exit(0)

    # Round-robin assign audio files to GPUs to cap concurrency per device
    device_to_audio: Dict[str, List[str]] = {gpu: [] for gpu in GPU_LIST}
    for idx, ap in enumerate(audio_paths):
        device_to_audio[GPU_LIST[idx % len(GPU_LIST)]].append(ap)

    total_samples = 0
    executors: List[ProcessPoolExecutor] = []
    futures = []

    try:
        for gpu_id, paths in device_to_audio.items():
            if not paths:
                continue
            ex = ProcessPoolExecutor(max_workers=WORKERS_PER_GPU)
            executors.append(ex)
            for ap in paths:
                # Randomly select chunk_size and number_of_candidates for each audio
                chunk_size = random.choice(chunk_size_options)
                number_of_candidates = random.choice(number_of_candidates_options)

                chunk_size = 500
                number_of_candidates = 4
                
                futures.append(
                    ex.submit(
                        _process_single_audio,
                        ap,
                        reference_file,
                        gpu_id,
                        streaming_asr_script,
                        chunk_size,
                        number_of_candidates,
                        output_path,
                    )
                )

        # Progress bar for tracking completed audio files
        with tqdm(total=len(futures), desc="Processing audio files", unit="file") as pbar:
            for fut in as_completed(futures):
                total_samples += fut.result()
                pbar.update(1)
                pbar.set_postfix(samples=total_samples)
    finally:
        for ex in executors:
            ex.shutdown(wait=True)

    print(f"Total {total_samples} samples written to {output_path}")

