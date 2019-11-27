import argparse
from pathlib import Path
import librosa
import scipy
import json
import csv
import numpy as np
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm


def preemphasis(x, preemph):
    return scipy.signal.lfilter([1, -preemph], [1], x)


def process_wav(wav_path, out_path, params, offset=0.0, duration=None):
    wav, _ = librosa.load(wav_path.with_suffix(".wav"), sr=params["preprocessing"]["sample_rate"],
                          offset=offset, duration=duration)
    wav = wav / np.abs(wav).max() * 0.999

    mel = librosa.feature.melspectrogram(preemphasis(wav, params["preprocessing"]["preemph"]),
                                         sr=params["preprocessing"]["sample_rate"],
                                         n_fft=params["preprocessing"]["num_fft"],
                                         n_mels=params["preprocessing"]["num_mels"],
                                         hop_length=params["preprocessing"]["hop_length"],
                                         win_length=params["preprocessing"]["win_length"],
                                         fmin=params["preprocessing"]["fmin"],
                                         power=1)
    logmel = librosa.amplitude_to_db(mel, top_db=params["preprocessing"]["top_db"])

    mfccs = librosa.feature.mfcc(S=logmel, n_mfcc=params["preprocessing"]["num_mfccs"])
    deltas = librosa.feature.delta(mfccs, width=5, order=1)
    ddeltas = librosa.feature.delta(mfccs, width=5, order=2)
    mfccs = np.vstack((mfccs, deltas, ddeltas))

    logmel = logmel / params["preprocessing"]["top_db"] + 1
    mfccs = (mfccs - mfccs.mean(axis=-1, keepdims=True)) / mfccs.std(axis=-1, keepdims=True)

    wav = mulaw_encode(wav, mu=2 ** params["preprocessing"]["bits"])

    np.save(out_path.with_suffix(".wav.npy"), wav)
    np.save(out_path.with_suffix(".mel.npy"), logmel)
    np.save(out_path.with_suffix(".mfcc.npy"), mfccs)
    return out_path, logmel.shape[-1]


def mulaw_encode(x, mu):
    mu = mu - 1
    fx = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    return np.floor((fx + 1) / 2 * mu + 0.5)


def mulaw_decode(y, mu):
    mu = mu - 1
    x = np.sign(y) / mu * ((1 + mu) ** np.abs(y) - 1)
    return x


def preprocess(args, params):
    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    executor = ProcessPoolExecutor(max_workers=args.num_workers)
    futures = []
    with open(args.manifest_path) as file:
        reader = csv.reader(file)
        for in_path, start, duration, out_path in reader:
            wav_path = in_dir / in_path
            out_path = out_dir / out_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            futures.append(executor.submit(partial(process_wav, wav_path, out_path, params,
                                                   float(start), float(duration))))

    results = [future.result() for future in tqdm(futures)]

    lengths = [x[-1] for x in results]
    frames = sum(lengths)
    frame_shift_ms = params["preprocessing"]["hop_length"] / params["preprocessing"]["sample_rate"]
    hours = frames * frame_shift_ms / 3600
    print("Wrote {} utterances, {} frames ({:.2f} hours)".format(len(lengths), frames, hours))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=str)
    parser.add_argument("--out-dir", type=str)
    parser.add_argument("--manifest-path", type=str)
    parser.add_argument("--num-workers", type=int, default=cpu_count())
    with open("config.json") as file:
        params = json.load(file)
    args = parser.parse_args()
    preprocess(args, params)


if __name__ == "__main__":
    main()
