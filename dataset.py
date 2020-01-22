import numpy as np
import torch
from torch.utils.data import Dataset
import csv
from random import randint
from pathlib import Path


class SpeechDataset(Dataset):
    def __init__(self, root, sample_frames, hop_length, sample_rate):
        self.root = Path(root)
        self.sample_frames = sample_frames
        self.hop_length = hop_length

        with open(self.root / "speakers.csv") as file:
            reader = csv.reader(file)
            self.speakers = sorted([speaker for speaker, in reader])

        min_duration = (sample_frames + 2) * hop_length / sample_rate
        with open(self.root / "train.csv") as file:
            reader = csv.reader(file)
            self.manifest = [Path(out_path) for _, _, duration, out_path in reader if float(duration) > min_duration]

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, index):
        path = self.manifest[index]
        path = self.root.parent / path

        audio = np.load(path.with_suffix(".wav.npy"))
        mel = np.load(path.with_suffix(".mel.npy"))

        pos = randint(1, mel.shape[-1] - self.sample_frames - 2)
        mel = mel[:, pos - 1:pos + self.sample_frames + 1]

        audio = audio[pos * self.hop_length:(pos + self.sample_frames) * self.hop_length + 1]

        speaker = self.speakers.index(path.parts[-2])

        return torch.LongTensor(audio), torch.FloatTensor(mel), speaker

