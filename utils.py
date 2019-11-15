import librosa
import numpy as np
import scipy


def preemphasis(x, preemph):
    return scipy.signal.lfilter([1, -preemph], [1], x)


def process_wav(wav_path, out_path, params, offset=0.0, duration=None):
    wav, _ = librosa.load(wav_path.with_suffix(".wav"), sr=params["preprocessing"]["sample_rate"],
                          offset=offset, duration=duration)
    wav = wav / np.abs(wav).max() * 0.999
    mel = librosa.feature.melspectrogram(preemphasis(wav, params["preprocessing"]["preemph"]),
                                         sr=params["preprocessing"]["sample_rate"],
                                         n_mels=params["preprocessing"]["num_mels"],
                                         n_fft=params["preprocessing"]["num_fft"],
                                         hop_length=params["preprocessing"]["hop_length"],
                                         win_length=params["preprocessing"]["win_length"],
                                         fmin=params["preprocessing"]["fmin"],
                                         power=1)
    logmel = librosa.amplitude_to_db(mel, top_db=params["preprocessing"]["top_db"])
    logmel = logmel / params["preprocessing"]["top_db"] + 1
    wav = mulaw_encode(wav, mu=2 ** params["preprocessing"]["bits"])

    np.save(out_path.with_suffix(".wav.npy"), wav)
    np.save(out_path.with_suffix(".mel.npy"), logmel)
    return out_path, logmel.shape[-1]


def mulaw_encode(x, mu):
    mu = mu - 1
    fx = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    return np.floor((fx + 1) / 2 * mu + 0.5)


def mulaw_decode(y, mu):
    mu = mu - 1
    x = np.sign(y) / mu * ((1 + mu) ** np.abs(y) - 1)
    return x
