import argparse
from pathlib import Path
import json

from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import SpeechDataset
from model import Model


def save_checkpoint(model, optimizer, step, checkpoint_dir):
    checkpoint_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step}
    checkpoint_path = checkpoint_dir / "model.ckpt-{}.pt".format(step)
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

    print(optimizer.defaults)
    dataset = SpeechDataset(metadata_path=Path("./data/english/train.json"),
                            sample_frames=40,
                            audio_slice_frames=8,
                            hop_length=200)

    dataloader = DataLoader(dataset, batch_size=32,
                            shuffle=True, num_workers=args.num_workers,
                            pin_memory=True)

    num_epochs = 250000 // len(dataloader) + 1
    start_epoch = global_step // len(dataloader) + 1

    for epoch in range(start_epoch, num_epochs + 1):
        average_recon_loss = average_vq_loss = average_perplexity = 0

        for i, (audio, mels, mfccs, speakers) in enumerate(tqdm(dataloader), 1):
            audio, mfccs, speakers = audio.to(device), mfccs.to(device), speakers.to(device)

            output, vq_loss, perplexity = model(audio[:, :-1], mfccs, speakers)
            recon_loss = F.cross_entropy(output.transpose(1, 2), audio[:, 1:])
            loss = recon_loss + vq_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            average_recon_loss += (recon_loss.item() - average_recon_loss) / i
            average_vq_loss += (vq_loss.item() - average_vq_loss) / i
            average_perplexity += (perplexity.item() - average_perplexity) / i

            global_step += 1

            if global_step % 25000 == 0:
                save_checkpoint(model, optimizer, global_step, Path(args.checkpoint_dir))

        print("epoch:{}, recon loss:{:.2E}, vq loss:{:.2E}, perpexlity:{:.3f}"
              .format(epoch, average_recon_loss, average_vq_loss, average_perplexity))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=6, help="Number of dataloader workers.")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/", help="Directory to save checkpoints.")
    parser.add_argument("--data_dir", type=str, default="./data")
    args = parser.parse_args()
    with open("config.json") as f:
        params = json.load(f)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    train_fn(args, params)
