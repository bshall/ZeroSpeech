import argparse
import os
import numpy as np
import json
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
from utils import load_wav, melspectrogram, mulaw_encode
import random
import glob
from itertools import chain


def process_wav(wav_path, audio_path, mel_path, params):
    wav = load_wav(wav_path, sample_rate=params["preprocessing"]["sample_rate"])
    wav /= np.abs(wav).max() * 0.999
    mel = melspectrogram(wav, sample_rate=params["preprocessing"]["sample_rate"],
                         preemph=params["preprocessing"]["preemph"],
                         num_mels=params["preprocessing"]["num_mels"],
                         num_fft=params["preprocessing"]["num_fft"],
                         min_level_db=params["preprocessing"]["min_level_db"],
                         hop_length=params["preprocessing"]["hop_length"],
                         win_length=params["preprocessing"]["win_length"],
                         fmin=params["preprocessing"]["fmin"])

    length_diff = len(mel) * params["preprocessing"]["hop_length"] - len(wav)
    wav = np.pad(wav, (0, length_diff), "constant")

    pad = (40 - 8) // 2
    mel = np.pad(mel, ((pad,), (0,)), "constant")
    wav = np.pad(wav, (pad * params["preprocessing"]["hop_length"],), "constant")
    wav = mulaw_encode(wav, mu=256)

    speaker = os.path.splitext(os.path.split(wav_path)[-1])[0].split("_")[0]
    np.save(audio_path, wav)
    np.save(mel_path, mel)
    return speaker, audio_path, mel_path, len(mel)


def preprocess(wav_dirs, out_dir, num_workers, params):
    mel_out_dir = os.path.join(out_dir, "mels")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(mel_out_dir, exist_ok=True)

    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    wav_paths = chain.from_iterable(glob.iglob("{}/*.wav".format(dir), recursive=True) for dir in wav_dirs)
    for wav_path in wav_paths:
        fid = os.path.basename(wav_path).replace(".wav", ".npy")
        mel_path = os.path.join(mel_out_dir, fid)
        futures.append(executor.submit(partial(process_wav, wav_path, mel_path, params)))

    metadata = [future.result() for future in tqdm(futures)]
    write_metadata(metadata, out_dir, params)


def write_metadata(metadata, out_dir, params):
    random.shuffle(metadata)
    test = metadata[-params["preprocessing"]["num_evaluation_utterances"]:]
    train = metadata[:-params["preprocessing"]["num_evaluation_utterances"]]

    speakers = set([m[0] for m in metadata])
    with open(os.path.join(out_dir, "speakers.txt"), "w", encoding="utf-8") as f:
        for speaker in speakers:
            f.write(speaker + "\n")

    with open(os.path.join(out_dir, "test.txt"), "w", encoding="utf-8") as f:
        for m in test:
            f.write("|".join([str(x) for x in m]) + "\n")

    with open(os.path.join(out_dir, "train.txt"), "w", encoding="utf-8") as f:
        for m in train:
            f.write("|".join([str(x) for x in m]) + "\n")

    frames = sum([m[2] for m in metadata])
    frame_shift_ms = params["preprocessing"]["hop_length"] / params["preprocessing"]["sample_rate"]
    hours = frames * frame_shift_ms / 3600
    print('Wrote %d utterances, %d frames (%.2f hours)' % (len(metadata), frames, hours))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data")
    parser.add_argument("--num-workers", type=int, default=cpu_count())
    parser.add_argument("--language", type=str, default="./english")
    with open("config.json") as f:
        params = json.load(f)
    args = parser.parse_args()
    wav_dirs = [os.path.join(args.language, "train", "unit"), os.path.join(args.language, "train", "voice")]
    preprocess(wav_dirs, args.output, args.num_workers, params)


if __name__ == "__main__":
    main()