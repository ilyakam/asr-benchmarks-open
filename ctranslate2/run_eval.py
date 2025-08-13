"""Run evaluation for ctranslate2 whisper models."""""
import argparse
import io
import os
import time

import numpy as np
import soundfile
from tqdm import tqdm

import evaluate
from faster_whisper import BatchedInferencePipeline, WhisperModel

from normalizer import data_utils


def download_audio_files(batch, args):
    """
    Downloads audio files from the dataset, caches them locally,
    and adds file paths and durations to the batch.
    Inspired by `./nemo_asr/run_eval.py`
    """
    DATA_CACHE_DIR = '/root/.cache/audio_cache'
    DATASET_NAME = args.dataset
    SPLIT_NAME = args.split

    CACHE_DIR = os.path.join(DATA_CACHE_DIR, DATASET_NAME, SPLIT_NAME)

    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    audio_filepaths = []
    durations = []
    references = []

    # Iterate with id, audio sample, and duration from the dataset
    for id, sample, duration, text in zip(batch["id"], batch["audio"], batch["audio_length_s"], batch["norm_text"]):
        id = id.replace('/', '_').removesuffix('.wav')
        audio_path = os.path.join(CACHE_DIR, f"{id}.wav")

        if not os.path.exists(audio_path):
            if "array" in sample:
                audio_array = np.float32(sample["array"])
                sample_rate = 16000
            elif "bytes" in sample:
                with io.BytesIO(sample["bytes"]) as audio_file:
                    audio_array, sample_rate = soundfile.read(audio_file, dtype="float32")
            else:
                raise ValueError("Sample must have either 'array' or 'bytes' key")

            os.makedirs(os.path.dirname(audio_path), exist_ok=True)
            soundfile.write(audio_path, audio_array, sample_rate)

        audio_filepaths.append(audio_path)
        durations.append(duration)
        references.append(text)

    batch["reference"] = references
    batch["audio_filepath"] = audio_filepaths
    batch["duration"] = durations

    return batch

def main(args):
    """Main function to run evaluation on a dataset."""
    wer_metric = evaluate.load("wer")

    whisper_model = WhisperModel(model_size_or_path=args.model_id,
                                 compute_type='float16',
                                 device='cuda',
                                 device_index=args.device)

    asr_model = BatchedInferencePipeline(model=whisper_model)

    print("Loading and preparing dataset...")
    dataset = data_utils.load_data(args)
    dataset = data_utils.prepare_data(dataset)

    if args.max_eval_samples and args.max_eval_samples > 0:
        print(f"Subsampling dataset to first {args.max_eval_samples} sample(s)!")
        dataset = dataset.take(args.max_eval_samples)

    print("Caching audio files...")
    dataset = dataset.map(download_audio_files,
                          batch_size=args.batch_size,
                          batched=True,
                          remove_columns=["audio"],
                          fn_kwargs={'args': args})

    all_predictions = []
    all_references = []
    all_durations = []
    all_transcription_times = []

    for sample in tqdm(iter(dataset), desc="Samples..."):
        audio_path = sample['audio_filepath']
        audio_duration = sample['duration']
        reference_text = sample['reference']

        print(f"--- Processing: {os.path.basename(audio_path)} (Duration: {audio_duration:.2f}s) ---")

        transcription_time_start = time.perf_counter()
        segments, _ = asr_model.transcribe(audio_path, language="en", batch_size=args.batch_size)
        transcription_time = time.perf_counter() - transcription_time_start

        transcription = "".join(segment.text for segment in segments)
        normalized_prediction = data_utils.normalizer(transcription).strip()

        all_predictions.append(normalized_prediction)
        all_references.append(reference_text)
        all_durations.append(audio_duration)
        all_transcription_times.append(transcription_time)

        print(f"Time taken: {transcription_time:.2f}s | RTFx: {round(audio_duration / transcription_time, 2)}")

    print("\n--- Benchmark Complete ---")
    total_audio_duration = sum(all_durations)
    total_transcription_time = sum(all_transcription_times)

    wer = wer_metric.compute(references=all_references, predictions=all_predictions)
    wer = round(100 * wer, 2)
    rtfx = round(total_audio_duration / total_transcription_time, 2)

    print(f"Total audio duration: {total_audio_duration:.2f}s")
    print(f"Total transcription time: {total_transcription_time:.2f}s")
    print(f"Overall RTFx: {rtfx}")
    print(f"Word Error Rate (WER): {wer}%")

    # Write manifest results (WER and RTFX)
    manifest_path = data_utils.write_manifest(
        all_references,
        all_predictions,
        args.model_id,
        args.dataset_path,
        args.dataset,
        args.split,
        audio_length=all_durations,
        transcription_time=all_transcription_times,
    )

    print("Results saved at path:", os.path.abspath(manifest_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model identifier. Should be loadable with faster-whisper",
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        default='hf-audio/esb-datasets-test-only-sorted',
        help='Dataset path. By default, it is `hf-audio/esb-datasets-test-only-sorted`.'
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="tedlium",
        required=True,
        help="Dataset name from Hugging Face (e.g., `tedlium`).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split of the dataset. *E.g.* `'validation`' for the dev split, or `'test'` for the test split.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="The device to run the pipeline on. -1 for CPU (default), 0 for the first GPU and so on.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for segment-level parallel processing."
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Number of samples to be evaluated. Put a lower number e.g. 64 for testing this script.",
    )
    parser.add_argument(
        "--no-streaming",
        dest='streaming',
        action="store_false",
        help="Choose whether you'd like to download the entire dataset or stream it during the evaluation.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=5,
        help="Number of warm-up steps to run before launching the timed runs.",
    )

    args = parser.parse_args()
    parser.set_defaults(streaming=False)

    main(args)
