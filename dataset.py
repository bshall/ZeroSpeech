import numpy as np
import torch
from torch.utils.data import Dataset
import json
from random import randint
from pathlib import Path


class SpeechDataset(Dataset):
    def __init__(self, metadata_path, sample_frames, audio_slice_frames, hop_length):
        self.sample_frames = sample_frames
        self.audio_slice_frames = audio_slice_frames
        self.pad = (sample_frames - audio_slice_frames) // 2
        self.hop_length = hop_length

        with metadata_path.open() as file:
            metadata = json.load(file)

        self.speakers = sorted(metadata.keys())
        self.metadata = list()
        for speaker, paths in metadata.items():
            self.metadata.extend([(speaker, path) for path, length in paths if length > sample_frames])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        speaker, path = self.metadata[index]
        path = Path(path)

        audio = np.load(path.with_suffix(".wav.npy"))
        mel = np.load(path.with_suffix(".mel.npy"))
        mfcc = np.load(path.with_suffix(".mfcc.npy"))

        pos = randint(0, mel.shape[-1] - self.sample_frames - 1)
        mel = mel[:, pos:pos + self.sample_frames]
        mfcc = mfcc[:, pos:pos + self.sample_frames]

        p, q = pos + self.pad, pos + self.pad + self.audio_slice_frames
        audio = audio[p * self.hop_length:q * self.hop_length + 1]

        speaker = self.speakers.index(speaker)

        return torch.LongTensor(audio), torch.FloatTensor(mel), torch.FloatTensor(mfcc), speaker
