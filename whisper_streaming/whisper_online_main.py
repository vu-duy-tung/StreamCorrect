#!/usr/bin/env python3

# This code is retrieved from the original WhisperStreaming whisper_online.py.
# It is refactored and simplified. Only the code that is needed for the
# SimulWhisper backend is kept.

import os
import sys
import json
import time
import logging
import random
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import torch
import numpy as np
import librosa

logger = logging.getLogger(__name__)


@lru_cache(10**6)
def load_audio(fname):
    a, _ = librosa.load(fname, sr=16000, dtype=np.float32)
    return a


def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_audio_chunk(fname, beg, end):
    audio = load_audio(fname)
    beg_s = int(beg * 16000)
    end_s = int(end * 16000)
    return audio[beg_s:end_s]


def processor_args(parser):
    """Shared args for the online processors.

    Args:
        parser: argparse.ArgumentParser object
    """
    group = parser.add_argument_group(
        "WhisperStreaming processor arguments (shared for simulation from file and for the server)"
    )
    group.add_argument(
        '--min-chunk-size',
        type=float,
        default=1.2,
        help=(
            'Minimum audio chunk size in seconds. It waits up to this time to do processing. '
            'If the processing takes shorter time, it waits, otherwise it processes the whole segment '
            'that was received by this time.'
        ),
    )

    group.add_argument(
        '--lan',
        '--language',
        type=str,
        default="en",
        help="Source language code, e.g. en, de, cs, or auto for automatic language detection from speech."
    )
    group.add_argument(
        '--task',
        type=str,
        default='transcribe',
        choices=["transcribe", "translate"],
        help="Transcribe or translate."
    )

    group.add_argument(
        '--vac',
        action="store_true",
        default=False,
        help='Use VAC = voice activity controller. Recommended. Requires torch.'
    )
    group.add_argument(
        '--vac-chunk-size',
        type=float,
        default=0.04,
        help='VAC sample size in seconds.'
    )

    parser.add_argument(
        "-l",
        "--log-level",
        dest="log_level",
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help="Set the log level",
        default='DEBUG'
    )

    parser.add_argument(
        "--logdir",
        help="Directory to save audio segments and generated texts for debugging.",
        default=None
    )


def asr_factory(args, factory=None):
    """
    Creates and configures an ASR and online processor object through a factory implemented in the backend.
    """
    asr, online = factory(args)

    # Create the OnlineASRProcessor
    if args.vac:
        from whisper_streaming.vac_online_processor import VACOnlineASRProcessor
        online = VACOnlineASRProcessor(
            args.min_chunk_size,
            online,
            use_error_corrector=getattr(args, 'use_error_corrector', False),
            error_corrector_ckpt=getattr(args, 'error_corrector_ckpt', None),
            error_corrector_base_model=getattr(args, 'error_corrector_base_model', None),
            error_corrector_type=getattr(args, 'error_corrector_type', 'speechlm'),
        )

    if args.task == "translate":
        if args.model_path.endswith(".en.pt"):
            logger.error(
                f"The model {args.model_path} is English only. Translation is not available. Terminating."
            )
            sys.exit(1)
        asr.set_translate_task()

    return asr, online


def set_logging(args, logger):
    logging.basicConfig(
        format='%(levelname)s\t%(message)s'
    )
    logger.setLevel(args.log_level)
    logging.getLogger("simul_whisper").setLevel(args.log_level)
    logging.getLogger("whisper_streaming").setLevel(args.log_level)


def simulation_args(parser):
    simulation_group = parser.add_argument_group("Arguments for simulation from file")
    simulation_group.add_argument(
        'audio_path',
        type=str,
        help="Filename of 16kHz mono channel wav, or directory containing multiple wav files for batch inference."
    )
    simulation_group.add_argument(
        '--start_at',
        type=float,
        default=0.0,
        help='Start processing audio at this time.'
    )
    simulation_group.add_argument(
        '--comp_unaware',
        action="store_true",
        default=False,
        help='Computationally unaware simulation.'
    )
    simulation_group.add_argument(
        '--batch',
        action="store_true",
        default=False,
        help='Enable batch processing for directory input.'
    )
    simulation_group.add_argument(
        '--audio-extensions',
        type=str,
        default='wav,mp3,flac,m4a',
        help='Comma-separated list of audio file extensions to process in batch mode.'
    )
    simulation_group.add_argument(
        '--num-audios',
        type=int,
        default=None,
        help='Limit the number of audio files to process (useful for testing). Process all files if not specified.'
    )
    simulation_group.add_argument(
        '--num-workers',
        type=int,
        default=1,
        help='Number of parallel workers for batch processing. Each worker gets its own GPU and model instance.'
    )
    simulation_group.add_argument(
        '--gpus',
        type=str,
        default='0,1,2,3,4,5,6,7',
        help='Comma-separated list of GPU IDs to use for parallel processing (e.g., "0,1,2,3").'
    )

    eval_group = parser.add_argument_group("Evaluation arguments")
    eval_group.add_argument(
        '--reference-file',
        type=str,
        default=None,
        help='Path to JSON file containing reference transcriptions (with audio_path and text_en fields). If provided, evaluation will be performed automatically.'
    )
    eval_group.add_argument(
        '--eval-output',
        type=str,
        default=None,
        help='Path to save evaluation results JSON (default: <logdir>/evaluation_results.json)'
    )


def get_audio_files(path, extensions):
    """Get list of audio files from a path (file or directory)."""
    if os.path.isfile(path):
        return [path]
    elif os.path.isdir(path):
        audio_files = []
        ext_list = [ext.strip().lower() for ext in extensions.split(',')]
        for root, dirs, files in os.walk(path):
            for file in sorted(files):
                if any(file.lower().endswith(f'.{ext}') for ext in ext_list):
                    audio_files.append(os.path.join(root, file))
        return audio_files
    else:
        raise ValueError(f"Path does not exist: {path}")


def _worker_init(gpu_id):
    """Initialize worker process with specific GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Clear any cached audio from parent process
    load_audio.cache_clear()


def _worker_process_files(worker_id, gpu_id, audio_files, args_dict, factory_module, factory_name):
    """
    Worker function that processes a list of audio files on a specific GPU.
    Each worker has its own ASR model and online processor instance.
    
    Args:
        worker_id: Worker identifier for logging
        gpu_id: GPU ID to use (will be set as CUDA_VISIBLE_DEVICES)
        audio_files: List of audio file paths to process
        args_dict: Dictionary of arguments (converted from argparse.Namespace)
        factory_module: Module name containing the factory function
        factory_name: Name of the factory function
    
    Returns:
        List of results for each processed file
    """
    import argparse
    import importlib
    
    # Set GPU for this worker BEFORE importing torch/loading models
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Clear LRU cache to avoid sharing memory with parent
    load_audio.cache_clear()
    
    # Reconstruct args from dict
    args = argparse.Namespace(**args_dict)
    
    # Import and get the factory function
    module = importlib.import_module(factory_module)
    factory = getattr(module, factory_name)
    
    # Set up logging for this worker
    worker_logger = logging.getLogger(f"worker_{worker_id}")
    logging.basicConfig(format=f'[Worker {worker_id} GPU {gpu_id}] %(levelname)s\t%(message)s')
    worker_logger.setLevel(args.log_level)
    
    worker_logger.info(f"Worker {worker_id} starting on GPU {gpu_id} with {len(audio_files)} files")
    
    # Determine min_chunk
    if args.vac:
        min_chunk = args.vac_chunk_size
    else:
        min_chunk = args.min_chunk_size
    
    # Initialize ASR and online processor for this worker (isolated instance)
    asr, online = asr_factory(args, factory)
    
    # Warm up the ASR
    if audio_files:
        a = load_audio_chunk(audio_files[0], 0, 1)
        asr.warmup(a)
        worker_logger.info(f"Worker {worker_id} ASR warmup complete")
    
    # Process all assigned files
    results = []
    for idx, audio_file in enumerate(audio_files, 1):
        worker_logger.info(f"Worker {worker_id}: Processing file {idx}/{len(audio_files)}: {os.path.basename(audio_file)}")
        
        try:
            result = process_single_audio_file(audio_file, args, asr, online, min_chunk, factory)
            results.append(result)
        except Exception as e:
            worker_logger.error(f"Worker {worker_id}: Error processing {audio_file}: {e}")
            import traceback
            traceback.print_exc()
            # Continue with next file instead of failing completely
            results.append({
                'file': audio_file,
                'duration': 0,
                'segments': [],
                'final_text': '',
                'first_token_latency': None,
                'last_token_latency': None,
                'error': str(e)
            })
    
    worker_logger.info(f"Worker {worker_id} completed {len(results)} files")
    return results


def run_parallel_batch_processing(audio_files, args, factory_module, factory_name, num_workers, gpu_list):
    """
    Run batch processing in parallel using multiple workers on different GPUs.
    
    Args:
        audio_files: List of all audio files to process
        args: Parsed arguments
        factory_module: Module name containing the factory function
        factory_name: Name of the factory function  
        num_workers: Number of parallel workers
        gpu_list: List of GPU IDs to use
    
    Returns:
        List of all results from all workers
    """
    # Distribute files across workers (round-robin assignment)
    worker_files = [[] for _ in range(num_workers)]
    for idx, audio_file in enumerate(audio_files):
        worker_idx = idx % num_workers
        worker_files[worker_idx].append(audio_file)
    
    # Assign GPUs to workers (round-robin if more workers than GPUs)
    worker_gpus = [gpu_list[i % len(gpu_list)] for i in range(num_workers)]
    
    # Convert args to dict for pickling (Namespace objects can have issues)
    args_dict = vars(args).copy()
    
    logger.info(f"Starting parallel batch processing with {num_workers} workers on GPUs: {worker_gpus}")
    for i in range(num_workers):
        logger.info(f"  Worker {i}: GPU {worker_gpus[i]}, {len(worker_files[i])} files")
    
    all_results = []
    
    # Use 'spawn' context to ensure clean process state (important for CUDA)
    ctx = get_context('spawn')
    
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
        # Submit all worker tasks
        futures = {}
        for worker_id in range(num_workers):
            if worker_files[worker_id]:  # Only submit if worker has files
                future = executor.submit(
                    _worker_process_files,
                    worker_id,
                    worker_gpus[worker_id],
                    worker_files[worker_id],
                    args_dict,
                    factory_module,
                    factory_name
                )
                futures[future] = worker_id
        
        # Collect results as they complete
        for future in as_completed(futures):
            worker_id = futures[future]
            try:
                results = future.result()
                all_results.extend(results)
                logger.info(f"Worker {worker_id} returned {len(results)} results")
            except Exception as e:
                logger.error(f"Worker {worker_id} failed with exception: {e}")
                import traceback
                traceback.print_exc()
    
    return all_results


def process_single_audio_file(audio_path, args, asr, online, min_chunk, factory):
    """Process a single audio file and return transcriptions."""
    if args.vac:
        online.is_currently_final = False

    SAMPLING_RATE = 16000
    duration = len(load_audio(audio_path)) / SAMPLING_RATE
    logger.info(f"Processing: {os.path.basename(audio_path)} - Duration: {duration:.2f}s")

    beg = args.start_at
    start_time = None
    start = time.time() - beg  # Offset start for elapsed time calculation

    # List to store all transcription segments for this file
    all_transcriptions = []

    def output_transcript(iteration_output, now=None):
        # output format in stdout is like:
        # 4186.3606 0 1720 Takhle to je
        # - the first three words are:
        #    - emission time from beginning of processing, in milliseconds
        #    - beg and end timestamp of the text segment, as estimated by Whisper model. The timestamps are not accurate, but they're useful anyway
        # - the next words: segment transcript
        if now is None:
            now = time.time() - start

        if 'start' in iteration_output:
            start_ts = iteration_output['start']
            end_ts = iteration_output['end']
            text = iteration_output['text']
            logger.debug(f"{now * 1000:.4f} {start_ts * 1000:.0f} {end_ts * 1000:.0f} {text}")
            print(f"{now * 1000:.4f} {start_ts * 1000:.0f} {end_ts * 1000:.0f} {text}", flush=True)

            # Store the transcription segment
            all_transcriptions.append({
                'emission_time': now,
                'start': start_ts,
                'end': end_ts,
                'text': text.strip()
            })
        else:
            logger.debug("No text in this segment")

    first_token_latency = None
    last_token_latency = None
    last_speech_end = None

    if args.offline:  # offline mode processing (for testing/debugging)
        a = load_audio(audio_path)
        online.insert_audio_chunk(a)
        try:
            o = online.process_iter(start_time=start_time)
        except AssertionError as e:
            logger.error(f"assertion error: {repr(e)}")
        else:
            output_transcript(o)
        now = None
    elif args.comp_unaware:  # computational unaware mode
        end = beg + min_chunk
        while True:
            a = load_audio_chunk(audio_path, beg, end)
            logger.info(f"The system received audio from {beg:.2f} s to {end:.2f} s")
            online.insert_audio_chunk(a)
            if first_token_latency is None:
                start_time = time.time()

            last_speech_end = time.time()
            try:
                o = online.process_iter(start_time=start_time)
                if first_token_latency is None and 'first_token_latency' in o and o['first_token_latency'] is not None:
                    first_token_latency = o['first_token_latency']
                    logger.info(f"First Token Latency captured: {first_token_latency*1000:.2f} ms")
            except AssertionError as e:
                logger.error(f"assertion error: {repr(e)}")
                pass
            else:
                output_transcript(o, now=end)
            if 'text' in o:
                last_token_latency = time.time() - last_speech_end
                logger.info(f"Last Token Latency updated: {last_token_latency*1000:.2f} ms")
            logger.info(f"## last processed {end:.2f}s\n")

            if end >= duration:
                break

            beg = end

            if end + min_chunk > duration:
                end = duration
            else:
                end += min_chunk
        now = duration
    else:  # online = simultaneous mode
        end = 0
        while True:
            now = time.time() - start
            if now < min(end + min_chunk, duration):
                time.sleep(min(end + min_chunk, duration) - now)
            end = time.time() - start
            logger.info(f"The system received audio from {beg:.2f} s to {end:.2f} s")
            a = load_audio_chunk(audio_path, beg, end)
            beg = end
            online.insert_audio_chunk(a)
            if first_token_latency is None:
                start_time = time.time()
            try:
                o = online.process_iter(start_time=start_time)
                if first_token_latency is None and 'first_token_latency' in o and o['first_token_latency'] is not None:
                    first_token_latency = o['first_token_latency']
                    logger.info(f"First Token Latency captured: {first_token_latency*1000:.2f} ms")
            except AssertionError as e:
                logger.error(f"assertion error: {e}")
                pass
            else:
                output_transcript(o)
            now = time.time() - start
            logger.info(f"## last processed {end:.2f} s, now is {now:.2f}, the latency is {now-end:.2f}\n")
            if 'text' in o:
                last_token_latency = now - end
            if end >= duration or (args.vac and online.is_currently_final):
                break
        now = None

    # Refresh
    # print("[PLAY WITH MINO] - Finalizing transcription...")
    print(online.online.frame_delay)
    if args.vac and online.online.frame_delay:
        get_remained_trans = True
        last_speech_end = time.time()
    else:
        get_remained_trans = False
    o = online.finish(start_time=start_time)
    if args.vac:
        online.is_currently_final = False
    if get_remained_trans == True:
        # Add last infered speech
        output_transcript(o, now=now)
        if 'text' in o:
            last_token_latency = time.time() - last_speech_end
            logger.info(f"Last Token Latency updated: {last_token_latency*1000:.2f} ms")

    # Get First Token Latency if available
    if first_token_latency is not None:
        print(f"\nFirst Token Latency: {first_token_latency*1000:.2f} ms")
    if last_token_latency is not None:
        print(f"Last Token Latency: {last_token_latency*1000:.2f} ms")

    # Concatenate all transcriptions into final output
    final_transcription = ""
    if all_transcriptions:
        final_transcription = " ".join([segment['text'] for segment in all_transcriptions])
        print("\n" + "=" * 80)
        print("FINAL TRANSCRIPTION:")
        print(final_transcription)
        print("=" * 80)

    return {
        'file': audio_path,
        'duration': duration,
        'segments': all_transcriptions,
        'final_text': final_transcription,
        'first_token_latency': first_token_latency,
        'last_token_latency': last_token_latency
    }


def main_simulation_from_file(factory, add_args=None):
    '''
    factory: function that creates the ASR and online processor object from args and logger.  
            or in the default WhisperStreaming local agreement backends (not implemented but could be).
    add_args: add specific args for the backend
    '''

    import argparse
    parser = argparse.ArgumentParser()

    processor_args(parser)
    if add_args is not None:
        add_args(parser)

    simulation_args(parser)

    args = parser.parse_args()
    args.offline = False  # TODO: offline mode is not implemented in SimulStreaming yet

    if args.offline and args.comp_unaware:
        logger.error("No or one option from --offline and --comp_unaware are available, not both. Exiting.")
        sys.exit(1)

    set_logging(args,logger)

    random_seed(21)

    audio_path = args.audio_path

    # Check if batch processing is needed
    is_directory = os.path.isdir(audio_path)

    if is_directory or args.batch:
        # Batch processing mode
        audio_files = get_audio_files(audio_path, args.audio_extensions)

        if not audio_files:
            logger.error(f"No audio files found in: {audio_path}")
            sys.exit(1)

        # Limit number of files if specified
        if args.num_audios is not None and args.num_audios > 0:
            audio_files = audio_files[:args.num_audios]
            logger.info(f"Limiting to first {args.num_audios} audio files")

        logger.info(f"Found {len(audio_files)} audio files for batch processing")

        # Parse GPU list
        gpu_list = [g.strip() for g in args.gpus.split(',')]
        num_workers = min(args.num_workers, len(audio_files))  # Don't create more workers than files
        
        if num_workers > 1:
            # Parallel processing mode
            logger.info(f"Using parallel processing with {num_workers} workers on GPUs: {gpu_list}")
            
            # Get factory module and function name for subprocess import
            factory_module = factory.__module__
            factory_name = factory.__name__
            
            # Run parallel batch processing
            batch_results = run_parallel_batch_processing(
                audio_files, args, factory_module, factory_name, num_workers, gpu_list
            )
        else:
            # Sequential processing mode (original behavior)
            logger.info("Using sequential processing (single worker)")
            
            # Initialize ASR and online processor once
            if args.vac:
                min_chunk = args.vac_chunk_size
            else:
                min_chunk = args.min_chunk_size
            asr, online = asr_factory(args, factory)

            # Warm up the ASR with first file
            a = load_audio_chunk(audio_files[0], 0, 1)
            asr.warmup(a)
            print("ASR warmup complete.\n\n")

            # Process all files sequentially
            batch_results = []
            for idx, audio_file in enumerate(audio_files, 1):
                logger.info(f"\n{'='*80}")
                logger.info(f"Processing file {idx}/{len(audio_files)}: {os.path.basename(audio_file)}")
                logger.info(f"{'='*80}")

                try:
                    result = process_single_audio_file(audio_file, args, asr, online, min_chunk, factory)
                    batch_results.append(result)
                except Exception as e:
                    logger.error(f"Error processing {audio_file}: {e}")
                    import traceback
                    traceback.print_exc()
                    raise e

        # Save batch results
        if args.logdir and batch_results:
            os.makedirs(args.logdir, exist_ok=True)

            # Calculate average FTL
            ftl_values = [r['first_token_latency'] for r in batch_results if r.get('first_token_latency') is not None]
            avg_ftl = sum(ftl_values) / len(ftl_values) if ftl_values else None

            # Calculate average LTL (last token latency)
            ltl_values = [r['last_token_latency'] for r in batch_results if r.get('last_token_latency') is not None]
            avg_ltl = sum(ltl_values) / len(ltl_values) if ltl_values else None

            # Save summary JSON
            summary_file = os.path.join(args.logdir, "batch_transcriptions.json")
            summary_data = {
                'total_files': len(audio_files),
                'processed_files': len(batch_results),
                'average_first_token_latency_ms': avg_ftl * 1000 if avg_ftl is not None else None,
                'average_last_token_latency_ms': avg_ltl * 1000 if avg_ltl is not None else None,
                'results': [
                    {
                        'file': os.path.basename(r['file']),
                        'duration': r['duration'],
                        'transcription': r['final_text'],
                        'first_token_latency_ms': r['first_token_latency'] * 1000 if r.get('first_token_latency') else None,
                        'last_token_latency_ms': r['last_token_latency'] * 1000 if r.get('last_token_latency') else None
                    }
                    for r in batch_results
                ]
            }
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Batch summary saved to: {summary_file}")


        # Print final summary
        print("\n" + "=" * 80)
        print("BATCH PROCESSING COMPLETE")
        print("=" * 80)
        print(f"Total files processed: {len(batch_results)}/{len(audio_files)}")

        # Print average FTL if available
        if batch_results:
            ftl_values = [r['first_token_latency'] for r in batch_results if r.get('first_token_latency') is not None]
            if ftl_values:
                avg_ftl_ms = (sum(ftl_values) / len(ftl_values)) * 1000
                print(f"Average First Token Latency: {avg_ftl_ms:.2f} ms")

            # Print average LTL if available
            ltl_values = [r['last_token_latency'] for r in batch_results if r.get('last_token_latency') is not None]
            if ltl_values:
                avg_ltl_ms = (sum(ltl_values) / len(ltl_values)) * 1000
                print(f"Average Last Token Latency: {avg_ltl_ms:.2f} ms")

        print("=" * 80)

        # Run evaluation if reference file is provided
        if args.reference_file and batch_results:
            logger.info(f"\nRunning evaluation against reference file: {args.reference_file}")
            try:
                # Import evaluation module
                import sys
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from evaluate import load_references, evaluate_transcriptions

                # Load references
                references = load_references(args.reference_file, language=args.lan)

                # Create generated transcriptions map from batch results
                generated = {}
                ftl_map = {}  # Map filename to FTL
                ltl_map = {}  # Map filename to LTL
                for result in batch_results:
                    filename = os.path.basename(result['file'])
                    generated[filename] = result['final_text']
                    # Store FTL for this file
                    if result.get('first_token_latency') is not None:
                        ftl_map[filename] = result['first_token_latency']
                    # Store LTL for this file
                    if result.get('last_token_latency') is not None:
                        ltl_map[filename] = result['last_token_latency']

                # Evaluate
                evaluation_results = evaluate_transcriptions(references, generated, args.lan)

                # Add FTL and LTL information to evaluation results
                # Calculate average FTL and LTL for matched files
                ftl_values = [r['first_token_latency'] for r in batch_results if r.get('first_token_latency') is not None]
                avg_ftl = sum(ftl_values) / len(ftl_values) if ftl_values else None

                ltl_values = [r['last_token_latency'] for r in batch_results if r.get('last_token_latency') is not None]
                avg_ltl = sum(ltl_values) / len(ltl_values) if ltl_values else None

                evaluation_results['average_first_token_latency_ms'] = avg_ftl * 1000 if avg_ftl is not None else None
                evaluation_results['average_last_token_latency_ms'] = avg_ltl * 1000 if avg_ltl is not None else None

                # Add per-file FTL and LTL to per_file_results
                for result in evaluation_results['per_file_results']:
                    filename = result['file']
                    if filename in ftl_map:
                        result['first_token_latency_ms'] = ftl_map[filename] * 1000
                    if filename in ltl_map:
                        result['last_token_latency_ms'] = ltl_map[filename] * 1000

                # Print evaluation summary
                print("\n" + "=" * 80)
                print("EVALUATION RESULTS")
                print("=" * 80)
                print(f"Total reference files: {evaluation_results['total_files']}")
                print(f"Matched files: {evaluation_results['matched_files']}")
                print(f"Unmatched files: {evaluation_results['unmatched_files']}")
                print(f"\nAverage CER: {evaluation_results['average_cer']:.4f} ({evaluation_results['average_cer']*100:.2f}%)")
                print(f"Average MER: {evaluation_results['average_mer']:.4f} ({evaluation_results['average_mer']*100:.2f}%)")
                if evaluation_results.get('average_first_token_latency_ms') is not None:
                    print(f"Average First Token Latency: {evaluation_results['average_first_token_latency_ms']:.2f} ms")
                if evaluation_results.get('average_last_token_latency_ms') is not None:
                    print(f"Average Last Token Latency: {evaluation_results['average_last_token_latency_ms']:.2f} ms")
                print("=" * 80)

                # Print per-file results if in INFO or DEBUG mode
                if args.log_level in ['DEBUG', 'INFO']:
                    print("\nPer-file CER/MER:")
                    print("-" * 80)
                    for result in evaluation_results['per_file_results']:
                        ftl_str = f" FTL: {result['first_token_latency_ms']:.2f}ms" if result.get('first_token_latency_ms') else ""
                        ltl_str = f" LTL: {result['last_token_latency_ms']:.2f}ms" if result.get('last_token_latency_ms') else ""
                        print(
                            f"{result['file']:40s} CER: {result['cer']:.4f} ({result['cer']*100:.2f}%)  "
                            f"MER: {result['mer']:.4f} ({result['mer']*100:.2f}%){ftl_str}{ltl_str}"
                        )

                # Save evaluation results
                eval_output = args.eval_output or os.path.join(args.logdir, 'evaluation_results.json')
                with open(eval_output, 'w', encoding='utf-8') as f:
                    json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
                logger.info(f"\nEvaluation results saved to: {eval_output}")

            except Exception as e:
                logger.error(f"Error during evaluation: {e}")
                import traceback
                traceback.print_exc()

    else:
        # Single file processing mode (original behavior)
        SAMPLING_RATE = 16000
        duration = len(load_audio(audio_path)) / SAMPLING_RATE
        logger.info("Audio duration is: %2.2f seconds" % duration)

        if args.vac:
            # args.min_chunk_size = args.vac_chunk_size
            min_chunk = args.vac_chunk_size
        else:
            min_chunk = args.min_chunk_size
        asr, online = asr_factory(args, factory)

        # load the audio into the LRU cache before we start the timer
        a = load_audio_chunk(audio_path, 0, 1)

        # warm up the ASR because the very first transcribe takes much more time than the other
        asr.warmup(a)
        print("ASR warmup complete.\n\n")

        # Process the single file
        result = process_single_audio_file(audio_path, args, asr, online, min_chunk, factory)

        # Save single file result if logdir is specified
        if args.logdir:
            os.makedirs(args.logdir, exist_ok=True)
            transcript_file = os.path.join(args.logdir, "final_transcription.txt")
            with open(transcript_file, 'w', encoding='utf-8') as f:
                f.write(result['final_text'])
            
            # Save segments with emission times for video demo generation
            segments_file = os.path.join(args.logdir, "segments_with_timing.json")
            segments_data = {
                'audio_file': audio_path,
                'duration': result['duration'],
                'segments': result['segments']
            }
            with open(segments_file, 'w', encoding='utf-8') as f:
                json.dump(segments_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Segments with timing saved to: {segments_file}")
            logger.info(f"Transcription saved to: {transcript_file}")

        # Run evaluation if reference file is provided
        if args.reference_file and result['final_text']:
            logger.info(f"\nRunning evaluation against reference file: {args.reference_file}")
            try:
                # Import evaluation module
                import sys
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from evaluate import load_references, calculate_cer, calculate_mer

                # Load references
                references = load_references(args.reference_file, language=args.lan)

                # Find matching reference
                filename = os.path.basename(audio_path)

                ref_text = None
                if filename in references:
                    ref_text = references[filename]

                if ref_text:
                    cer = calculate_cer(ref_text, result['final_text'], args.lan)
                    mer = calculate_mer(ref_text, result['final_text'], args.lan)

                    # Print evaluation result
                    print("\n" + "=" * 80)
                    print("EVALUATION RESULT")
                    print("=" * 80)
                    print(f"File: {filename}")
                    print(f"CER: {cer:.4f} ({cer*100:.2f}%)")
                    print(f"MER: {mer:.4f} ({mer*100:.2f}%)")
                    print("=" * 80)

                    # Save evaluation result if logdir is specified
                    if args.logdir:
                        eval_output = args.eval_output or os.path.join(args.logdir, 'evaluation_result.json')
                        eval_data = {
                            'file': filename,
                            'reference': ref_text,
                            'generated': result['final_text'],
                            'cer': cer,
                            'mer': mer,
                            'ref_length': len(ref_text),
                            'gen_length': len(result['final_text']),
                            'first_token_latency_ms': result['first_token_latency'] * 1000 if result.get('first_token_latency') else None,
                            'last_token_latency_ms': result['last_token_latency'] * 1000 if result.get('last_token_latency') else None
                        }
                        with open(eval_output, 'w', encoding='utf-8') as f:
                            json.dump(eval_data, f, indent=2, ensure_ascii=False)
                        logger.info(f"Evaluation result saved to: {eval_output}")
                else:
                    logger.warning(f"No reference found for {filename} in reference file")

            except Exception as e:
                logger.error(f"Error during evaluation: {e}")
                import traceback
                traceback.print_exc()


