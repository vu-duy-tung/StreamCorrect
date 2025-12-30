import os
import re
import subprocess
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Dict, Any, Optional
import random
import json

FAILED_REF_HYP_LOG = "error_corrector/data/sample_custom_data/failed_ref_hyp.txt"
random.seed(21)

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
        r"INFO\t+The system received audio from\s+([\d.]+)\s+s to\s+([\d.]+)\s+s"
    )
    debug_line = re.compile(
        r'^DEBUG\t+<\|startoftranscript\|\><\|zh\|\><\|transcribe\|\><\|notimestamps\|\>(.*)$'
    )

    chunks: List[Dict[str, Any]] = []
    current_chunk: Optional[Dict[str, Any]] = None
    current_loop_debugs: List[str] = []

    for line in stderr.splitlines():
        if line.startswith("INFO\tFinish"):
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
        if "INFO\tDecoding loop starts" in line:
            current_loop_debugs = []
            continue

        # Collect transcript DEBUG lines
        m_dbg = debug_line.match(line)
        if m_dbg:
            current_loop_debugs.append(m_dbg.group(1))
            continue

        if line.startswith("INFO\tSaved encoder_feature to"):
            audio_embed_path = line.split("INFO\tSaved encoder_feature to", 1)[1].strip()
            current_chunk["audio_embed_path"] = audio_embed_path

        # Collect outputs
        if line.startswith("INFO\tOutput:"):
            current_chunk["outputs"].append(
                line.split("INFO\tOutput:", 1)[1].strip()
            )
            continue

        if line.startswith("INFO\tPrevious confirmed transcript:"):
            prev_transcript = line.split("INFO\tPrevious confirmed transcript:", 1)[1].strip()
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
    streaming_asr_script: str,
    chunk_size: int = 500,
    number_of_candidates: int = 4,
    cuda_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    env = os.environ.copy()
    if cuda_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_id
    env["AUDIO_PATH"] = audio_path  # override AUDIO_PATH inside bash
    env["VAC_CHUNK_SIZE"] = str(chunk_size / 1000.0)  # in seconds
    env["BEAM_SIZE"] = str(number_of_candidates)

    result = subprocess.run(
        ["bash", streaming_asr_script],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    ref, hyp = extract_ref_hyp(result.stdout)
    print("Reference:", ref)
    print("Hypothesis:", hyp)

    if ref is None or hyp is None:
        print("Failed to extract Ref/Hyp from ASR output.")
        _log_failed_audio_path(audio_path)
        print(result.stdout)
        print(result.stderr)
        return []

    chunks = parse_stderr_topk(result.stderr, k=number_of_candidates)
    for i, ch in enumerate(chunks):
        # Preprocess
        while ch['previous_transcript'].endswith('\uFFFD'):
            ch['previous_transcript'] = ch['previous_transcript'][:-1]

        for j in range(len(ch["topk"])):
            while ch["topk"][j].endswith('\uFFFD'):
                ch["topk"][j] = ch["topk"][j][:-1]

        continuation_transcript = ""

        offsets = [0] + [j for j in range(-6, 7) if j != 0]
        max_count, cand_offset = -1, 0
        for offset in offsets:
            count = 0
            for p_prev in range(len(ch['previous_transcript']) - 1, -1, -1):
                p_ref = p_prev + offset
                if p_ref < 0 or p_ref >= len(ref):
                    continue
                if ch['previous_transcript'][p_prev] == ref[p_ref]:
                    count += 1
                if count > max_count:
                    max_count = count
                    cand_offset = offset

        if len(ch['topk']) > 0:
            num_pred_tokens = len(ch['topk'][0]) - len(ch['previous_transcript'])
            range_l = len(ch['previous_transcript']) + cand_offset
            range_r = min(range_l + num_pred_tokens, len(ref))
            continuation_transcript = ref[range_l:range_r]

        ch['continuation_transcript'] = continuation_transcript

        print(f"\nChunk {i}: {ch['start']}s -> {ch['end']}s")
        print("  Previous confirmed transcript:", ch["previous_transcript"])
        print("  Continuation transcript:", ch["continuation_transcript"])
        print("  Last top-k candidates:", ch["topk"])
        print("  Outputs:", ch["outputs"])
        print("  Audio embed path:", ch["audio_embed_path"])

    return chunks


def _process_single_audio(
    audio_path: str,
    cuda_id: str,
    streaming_asr_script: str,
    chunk_size: int,
    number_of_candidates: int,
) -> List[Dict[str, Any]]:
    """Worker entry: run ASR on a single audio file pinned to a GPU and return synthesized samples."""

    chunks = prepare_error_correction_data(
        audio_path,
        streaming_asr_script,
        chunk_size=chunk_size,
        number_of_candidates=number_of_candidates,
        cuda_id=cuda_id,
    )

    synthesized_samples = []
    for chunk in chunks:
        if len(chunk["continuation_transcript"]) > 0:
            synthesized_samples.append(
                {
                    "k_best_candidates": chunk["topk"],
                    "num_candidates": number_of_candidates,
                    "chunk_size": chunk_size,
                    "previous_transcript": chunk["previous_transcript"],
                    "continuation_transcript": chunk["continuation_transcript"],
                    "audio_embed_path": chunk["audio_embed_path"],
                }
            )

    return synthesized_samples


if __name__ == "__main__":
    # Configuration
    WORKERS_PER_GPU = 6  # safe upper bound given ~50GB free per GPU
    GPU_LIST = [str(i) for i in range(7)]  # cuda:0 to cuda:6

    current_time_stamp_options = [1, 2, 3, 4, 5, 6]
    chunk_size_options = [500, 1000]
    number_of_candidates_options = [2, 4, 8]

    streaming_asr_script = "runs/run_single_eval_aishell.sh"
    train_aishell_folder = "/data/mino/AISHELL-1/data_aishell/wav/train"

    chunk_size = 500
    number_of_candidates = 4

    audio_files = []
    # for sub_folder in os.listdir(train_aishell_folder):
    #     pth_to_sub_folder = os.path.join(train_aishell_folder, sub_folder)
    #     for audio_file in os.listdir(pth_to_sub_folder):
    #         if audio_file.endswith(".wav"):
    #             full_path = os.path.join(pth_to_sub_folder, audio_file)
    #             audio_files.append(full_path)
    train_aishell_folder = "/data/mino/AISHELL-1/data_aishell/wav/test/testset"
    for audio_file in os.listdir(train_aishell_folder):
        if audio_file.endswith(".wav"):
            full_path = os.path.join(train_aishell_folder, audio_file)
            audio_files.append(full_path)

    audio_paths = random.sample(audio_files, min(3000, len(audio_files)))

    # Get already processed audio basenames from audio_embeds folder
    audio_embeds_folder = "/data/mino/AISHELL-1/audio_embeds/"
    processed_basenames = set()
    if os.path.exists(audio_embeds_folder):
        for filename in os.listdir(audio_embeds_folder):
            filename = filename.split("_")[0]
            processed_basenames.add(filename)

    # Filter out audio paths whose basename is already in processed files
    audio_paths = [
        ap for ap in audio_paths
        if not any(os.path.basename(ap).replace(".wav", "") in processed_file 
                   for processed_file in processed_basenames)
    ]

    # Round-robin assign audio files to GPUs to cap concurrency per device
    device_to_audio: Dict[str, List[str]] = {gpu: [] for gpu in GPU_LIST}
    for idx, ap in enumerate(audio_paths):
        device_to_audio[GPU_LIST[idx % len(GPU_LIST)]].append(ap)

    synthesized_samples: List[Dict[str, Any]] = []
    executors: List[ProcessPoolExecutor] = []
    futures = []

    try:
        for gpu_id, paths in device_to_audio.items():
            if not paths:
                continue
            ex = ProcessPoolExecutor(max_workers=WORKERS_PER_GPU)
            executors.append(ex)
            for ap in paths:
                futures.append(
                    ex.submit(
                        _process_single_audio,
                        ap,
                        gpu_id,
                        streaming_asr_script,
                        chunk_size,
                        number_of_candidates,
                    )
                )

        for fut in as_completed(futures):
            synthesized_samples.extend(fut.result())
    finally:
        for ex in executors:
            ex.shutdown(wait=True)

    # Append synthesized_samples to jsonl file
    output_dir = "error_corrector/data/sample_custom_data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "samples.jsonl")

    with open(output_path, "a", encoding="utf-8") as f:
        for sample in synthesized_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Appended {len(synthesized_samples)} samples to {output_path}")

