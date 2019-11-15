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
    parser.add_argument("--gen-dir", type=str, default="./generated")
    parser.add_argument("--speaker-list", type=str)
    parser.add_argument("--mel-path", type=str)
    parser.add_argument("--speaker", type=str)
    args = parser.parse_args()

    with Path("config.json").open() as file:
        params = json.load(file)

    with open(args.speaker_list) as file:
        metadata = json.load(file)
    speakers = sorted(metadata.keys())

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    model = Model(in_channels=params["preprocessing"]["num_mels"],
                  encoder_channels=params["model"]["encoder"]["channels"],
                  num_codebook_embeddings=params["model"]["codebook"]["num_embeddings"],
                  codebook_embedding_dim=params["model"]["codebook"]["embedding_dim"],
                  num_speakers=params["model"]["vocoder"]["num_speakers"],
                  speaker_embedding_dim=params["model"]["vocoder"]["speaker_embedding_dim"],
                  conditioning_channels=params["model"]["vocoder"]["conditioning_channels"],
                  embedding_dim=params["model"]["vocoder"]["embedding_dim"],
                  rnn_channels=params["model"]["vocoder"]["rnn_channels"],
                  fc_channels=params["model"]["vocoder"]["fc_channels"],
                  bits=params["preprocessing"]["bits"],
                  hop_length=params["preprocessing"]["hop_length"])
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
    path = Path(gen_dir) / "gen_{}_to_{}_model_steps_{}.wav".format(utterance_id, args.speaker, model_step)
    librosa.output.write_wav(path, output.astype(np.float32), sr=16000)
