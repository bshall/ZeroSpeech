import librosa
import numpy as np
import scipy


def load_wav(path, sample_rate):
    return librosa.load(path, sr=sample_rate)[0]


def save_wav(path, wav, sample_rate):
    librosa.output.write_wav(path, wav.astype(np.float32), sr=sample_rate)


def mulaw_encode(x, mu):
    mu = mu - 1
    fx = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    return np.floor((fx + 1) / 2 * mu + 0.5)


def mulaw_decode(y, mu):
    mu = mu - 1
    x = np.sign(y) / mu * ((1 + mu) ** np.abs(y) - 1)
    return x


def preemphasis(x, preemph):
    return scipy.signal.lfilter([1, -preemph], [1], x)


def deemphasis(x, preemph):
    return scipy.signal.lfilter([1], [1, -preemph], x)


def melspectrogram(y, preemph, sample_rate, num_mels, num_fft, min_level_db, hop_length, win_length, fmin):
    y = preemphasis(y, preemph)
    S = np.abs(librosa.stft(y, n_fft=num_fft, hop_length=hop_length, win_length=win_length))
    mel_basis = librosa.filters.mel(sample_rate, num_fft, n_mels=num_mels, fmin=fmin)
    S = np.dot(mel_basis, S)
    mel = amp_to_db(S, min_level_db=min_level_db)
    return normalize(mel, min_level_db=min_level_db).T


def amp_to_db(x, min_level_db):
    min_level = 10**(min_level_db / 20)
    return 20 * np.log10(np.maximum(min_level, x))


def normalize(S, min_level_db):
    return np.clip((S - min_level_db) / -min_level_db, 0, 1)


def load_wav(path, sample_rate):
    return librosa.load(path, sr=sample_rate)[0]


def save_wav(path, wav, sample_rate):
    librosa.output.write_wav(path, wav.astype(np.float32), sr=sample_rate)


def mulaw_encode(x, mu):
    mu = mu - 1
    fx = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    return np.floor((fx + 1) / 2 * mu + 0.5)


def mulaw_decode(y, mu):
    mu = mu - 1
    x = np.sign(y) / mu * ((1 + mu) ** np.abs(y) - 1)
    return x


def preemphasis(x, preemph):
    return scipy.signal.lfilter([1, -preemph], [1], x)


def deemphasis(x, preemph):
    return scipy.signal.lfilter([1], [1, -preemph], x)


def melspectrogram(y, preemph, sample_rate, num_mels, num_fft, min_level_db, hop_length, win_length, fmin):
    y = preemphasis(y, preemph)
    S = np.abs(librosa.stft(y, n_fft=num_fft, hop_length=hop_length, win_length=win_length))
    mel_basis = librosa.filters.mel(sample_rate, num_fft, n_mels=num_mels, fmin=fmin)
    S = np.dot(mel_basis, S)
    mel = amp_to_db(S, min_level_db=min_level_db)
    return normalize(mel, min_level_db=min_level_db).T


def amp_to_db(x, min_level_db):
    min_level = 10**(min_level_db / 20)
    return 20 * np.log10(np.maximum(min_level, x))


def normalize(S, min_level_db):
    return np.clip((S - min_level_db) / -min_level_db, 0, 1)
