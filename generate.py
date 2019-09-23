import argparse
import json
from pathlib import Path
import torch
import numpy as np
import librosa
from model import Model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="Checkpoint path to resume")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--language", type=str, default="english")
    parser.add_argument("--gen-dir", type=str, default="./generated")
    parser.add_argument("--mel-path", type=str)
    parser.add_argument("--speaker", type=str)
    args = parser.parse_args()
    with Path("config.json").open() as f:
        params = json.load(f)

    metadata_path = Path(args.data_dir) / args.language / "train.json"
    with metadata_path.open() as f:
        metadata = json.load(f)
    speakers = sorted(metadata.keys())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model()
    model.to(device)

    print("Load checkpoint from: {}:".format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["model"])
    model_step = checkpoint["step"]

    mel_path = Path(args.mel_path)
    mel = np.load(mel_path)
    mel = torch.FloatTensor(mel).unsqueeze(0).to(device)
    speaker = torch.LongTensor([speakers.index(args.speaker)]).to(device)
    output = model.generate(mel, speaker)

    gen_dir = Path(args.gen_dir)
    gen_dir.mkdir(exist_ok=True)

    utterance_id = mel_path.stem
    path = Path(gen_dir) / "gen_{}_model_steps_{}.wav".format(utterance_id, model_step)
    librosa.output.write_wav(path, output.astype(np.float32), sr=16000)
