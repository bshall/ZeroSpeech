import argparse
from pathlib import Path
import json
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
from utils import process_wav


def preprocess(args, params):
    out_dir = Path(args.output) / "ZeroSpeech2019"
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_dirs = Path(args.data_path) / args.language / args.split

    executor = ProcessPoolExecutor(max_workers=args.num_workers)
    futures = []
    for wav_path in filter(lambda path: "parallel" not in path.parts, wav_dirs.rglob("*.wav")):
        speaker = wav_path.stem.split("_")[0]
        speaker_out_dir = out_dir.joinpath(args.language, args.split, speaker)
        speaker_out_dir.mkdir(parents=True, exist_ok=True)
        out_path = speaker_out_dir / wav_path.stem
        futures.append((speaker, executor.submit(partial(process_wav, wav_path, out_path, params))))

    results = [(speaker, *future.result()) for speaker, future in tqdm(futures)]
    write_metadata(results, out_dir, args, params)


def write_metadata(results, out_dir, args, params):
    metadata = dict()
    for speaker, path, length in results:
        metadata.setdefault(speaker, []).append((str(path), length))

    metadata_path = out_dir / args.language / args.split
    with metadata_path.with_suffix(".json").open("w") as file:
        json.dump(metadata, file)

    lengths = [x[-1] for x in results]
    frames = sum(lengths)
    frame_shift_ms = params["preprocessing"]["hop_length"] / params["preprocessing"]["sample_rate"]
    hours = frames * frame_shift_ms / 3600
    print("Wrote {} utterances, {} frames ({:.2f} hours)".format(len(lengths), frames, hours))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="processed")
    parser.add_argument("--num-workers", type=int, default=cpu_count())
    parser.add_argument("--language", type=str, default="english")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--data-path", type=str)
    with open("config.json") as file:
        params = json.load(file)
    args = parser.parse_args()
    preprocess(args, params)


if __name__ == "__main__":
    main()
