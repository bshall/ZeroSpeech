import argparse
from pathlib import Path
import librosa
import numpy as np
import scipy
import json
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


def preemphasis(x, preemph):
    return scipy.signal.lfilter([1, -preemph], [1], x)


def mulaw_encode(x, mu):
    mu = mu - 1
    fx = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    return np.floor((fx + 1) / 2 * mu + 0.5)


def process_wav(wav_path, speaker_out_dir, language, split, speaker, params):
    wav, _ = librosa.load(wav_path, sr=params["preprocessing"]["sample_rate"])
    wav = wav / np.abs(wav).max() * 0.999
    mel = librosa.feature.melspectrogram(preemphasis(wav, params["preprocessing"]["preemph"]),
                                         n_mels=params["preprocessing"]["num_mels"],
                                         n_fft=params["preprocessing"]["num_fft"],
                                         hop_length=params["preprocessing"]["hop_length"],
                                         win_length=params["preprocessing"]["win_length"],
                                         fmin=params["preprocessing"]["fmin"],
                                         power=1)
    logmel = librosa.amplitude_to_db(mel, top_db=params["preprocessing"]["top_db"])
    mfcc = librosa.feature.mfcc(S=logmel, n_mfcc=params["preprocessing"]["num_mfcc"])
    delta = librosa.feature.delta(mfcc, order=1, width=7)
    ddelta = librosa.feature.delta(mfcc, order=2, width=7)
    mfcc = np.vstack((mfcc, delta, ddelta))

    logmel = logmel / params["preprocessing"]["top_db"] + 1
    wav = mulaw_encode(wav, mu=2 ** params["preprocessing"]["bits"])

    out_path = speaker_out_dir / wav_path.stem
    np.save(out_path.with_suffix(".wav.npy"), wav)
    np.save(out_path.with_suffix(".mel.npy"), logmel)
    np.save(out_path.with_suffix(".mfcc.npy"), mfcc)
    return language, split, speaker, out_path, logmel.shape[-1]


def preprocess(data_dir, out_dir, num_workers, params):
    out_dir.mkdir(parents=True, exist_ok=True)

    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    for language_path in [path for path in data_dir.glob("*") if path.is_dir()]:
        for split_path in [path for path in language_path.glob("*") if path.is_dir()]:
            for wav_path in split_path.rglob("*.wav"):
                relative_path = wav_path.relative_to(split_path)
                language, split = split_path.parts[-2:]
                speaker = relative_path.stem.split("_")[0]
                speaker_out_dir = out_dir.joinpath(language, split, *relative_path.parts[:-1], speaker)
                speaker_out_dir.mkdir(parents=True, exist_ok=True)
                futures.append(executor.submit(partial(process_wav, wav_path, speaker_out_dir,
                                                       language, split, speaker, params)))

    results = [future.result() for future in tqdm(futures)]
    speaker_standardization(results)
    write_metadata(results, out_dir, params)


def speaker_standardization(results):
    print("Calculating mean and variance per speaker")
    scalers = dict()
    for language, _, speaker, path, _ in tqdm(results):
        scaler = scalers.setdefault(language, {}).setdefault(speaker, StandardScaler())
        mfcc = np.load(Path(path).with_suffix(".mfcc.npy"))
        scaler.partial_fit(mfcc.T)

    print("Standardizing and writing outputs")
    for language, _, speaker, path, _ in tqdm(results):
        scaler = scalers[language][speaker]
        path = Path(path).with_suffix(".mfcc.npy")
        mfcc = np.load(path)
        mfcc = scaler.transform(mfcc.T).T
        np.save(path, mfcc)


def write_metadata(results, out_dir, params):
    metadata = dict()
    for language, split, speaker, path, length in results:
        metadata.setdefault(language, {}).setdefault(split, {}).setdefault(speaker, []).append((str(path), length))

    for language, data in metadata.items():
        for split, content in data.items():
            matadata_path = out_dir / language / split
            with matadata_path.with_suffix(".json").open("w") as file:
                json.dump(content, file)

    lengths = [x[-1] for x in results]
    frames = sum(lengths)
    frame_shift_ms = params["preprocessing"]["hop_length"] / params["preprocessing"]["sample_rate"]
    hours = frames * frame_shift_ms / 3600
    print("Wrote {} utterances, {} frames ({:.2f} hours)".format(len(lengths), frames, hours))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data")
    parser.add_argument("--num-workers", type=int, default=cpu_count())
    parser.add_argument("--language", type=str, default="./english")
    with open("config.json") as f:
        params = json.load(f)
    args = parser.parse_args()
    wav_dirs = Path("../../Datasets/ZeroSpeech2019/")
    preprocess(wav_dirs, Path(args.output), args.num_workers, params)


if __name__ == "__main__":
    main()
