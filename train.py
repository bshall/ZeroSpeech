import argparse
import os
import json

import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import MelDataset
from model import Model


def save_checkpoint(model, optimizer, step, checkpoint_dir):
    checkpoint_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step}
    checkpoint_path = os.path.join(
        checkpoint_dir, "model.ckpt-{}.pt".format(step))
    torch.save(checkpoint_state, checkpoint_path)
    print("Saved checkpoint: {}".format(checkpoint_path))


def train_fn(args, params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=4e-4)

    if args.resume is not None:
        print("Resume checkpoint from: {}:".format(args.resume))
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = checkpoint["step"]
    else:
        global_step = 0

    dataset = MelDataset(meta_file=os.path.join(args.data_dir, "train.txt"),
                         speakers_file=os.path.join(args.data_dir, "speakers.txt"),
                         sample_frames=40,
                         audio_slice_frames=8,
                         hop_length=200)

    dataloader = DataLoader(dataset, batch_size=params["model"]["batch_size"],
                            shuffle=True, num_workers=args.num_workers,
                            pin_memory=True)

    num_epochs = params["model"]["num_steps"] // len(dataloader) + 1
    start_epoch = global_step // len(dataloader) + 1

    for epoch in range(start_epoch, num_epochs + 1):
        running_recon_loss = 0
        running_vq_loss = 0
        running_perplexity = 0

        for i, (audio, mels, speakers) in enumerate(tqdm(dataloader), 1):
            audio, mels, speakers = audio.to(device), mels.to(device), speakers.to(device)

            output, vq_loss, perplexity = model(audio[:, :-1], mels, speakers)
            recon_loss = F.cross_entropy(output.transpose(1, 2), audio[:, 1:])
            loss = recon_loss + vq_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_recon_loss += recon_loss.item()
            average_recon_loss = running_recon_loss / i
            running_vq_loss += vq_loss.item()
            average_vq_loss = running_vq_loss / i
            running_perplexity += perplexity.item()
            average_perplexity = running_perplexity / i

            global_step += 1

            if global_step % params["model"]["checkpoint_interval"] == 0:
                save_checkpoint(model, optimizer, global_step, args.checkpoint_dir)

        print("epoch:{}, recon loss:{:.2E}, vq loss:{:.2E}, perpexlity:{:.3f}"
              .format(epoch, average_recon_loss, average_vq_loss, average_perplexity))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/", help="Directory to save checkpoints.")
    parser.add_argument("--data_dir", type=str, default="./data")
    args = parser.parse_args()
    with open("config.json") as f:
        params = json.load(f)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    train_fn(args, params)
