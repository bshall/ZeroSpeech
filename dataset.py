import numpy as np
import torch
from torch.utils.data import Dataset
from random import randint


class MelDataset(Dataset):
    def __init__(self, meta_file, speakers_file, sample_frames):
        self.sample_frames = sample_frames

        with open(meta_file, encoding="utf-8") as f:
            self.metadata = [line.strip().split("|") for line in f]
        self.metadata = [m for m in self.metadata if int(m[2]) > sample_frames + 1]

        with open(speakers_file, encoding="utf-8") as f:
            self.speakers = [line.strip() for line in f]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        speaker_id, mel_path, _ = self.metadata[index]

        mel = np.load(mel_path)

        pos = randint(0, len(mel) - self.sample_frames - 1)
        mel = mel[pos:pos + self.sample_frames + 1, :]

        speaker = self.speakers.index(speaker_id)

        return torch.FloatTensor(mel), speaker
