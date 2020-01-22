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
import apex.amp as amp

from torch.utils.tensorboard import SummaryWriter


def save_checkpoint(model, optimizer, amp, step, checkpoint_dir):
    checkpoint_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "amp": amp.state_dict(),
        "step": step}
    checkpoint_path = checkpoint_dir / "model.ckpt-{}.pt".format(step)
    torch.save(checkpoint_state, checkpoint_path)
    print("Saved checkpoint: {}".format(checkpoint_path))


def train_fn(args, params):
    writer = SummaryWriter(Path("./runs") / args.checkpoint_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
                  hop_length=params["preprocessing"]["hop_length"],
                  jitter=params["model"]["codebook"]["jitter"])
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=params["training"]["learning_rate"])

    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    if args.resume is not None:
        print("Resume checkpoint from: {}:".format(args.resume))
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        amp.load_state_dict(checkpoint["amp"])
        global_step = checkpoint["step"]
    else:
        global_step = 0

    dataset = SpeechDataset(root=args.data_dir,
                            sample_frames=params["training"]["sample_frames"],
                            hop_length=params["preprocessing"]["hop_length"],
                            sample_rate=params["preprocessing"]["sample_rate"])

    dataloader = DataLoader(dataset, batch_size=params["training"]["batch_size"],
                            shuffle=True, num_workers=args.num_workers,
                            pin_memory=True)

    num_epochs = params["training"]["num_steps"] // len(dataloader) + 1
    start_epoch = global_step // len(dataloader) + 1

    for epoch in range(start_epoch, num_epochs + 1):
        average_recon_loss = average_vq_loss = average_perplexity = 0

        for i, (audio, mels, speakers) in enumerate(tqdm(dataloader), 1):
            audio, mels, speakers = audio.to(device), mels.to(device), speakers.to(device)

            output, vq_loss, perplexity = model(audio[:, :-1], mels, speakers)
            recon_loss = F.cross_entropy(output.transpose(1, 2), audio[:, 1:])
            loss = recon_loss + vq_loss

            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1)
            optimizer.step()

            average_recon_loss += (recon_loss.item() - average_recon_loss) / i
            average_vq_loss += (vq_loss.item() - average_vq_loss) / i
            average_perplexity += (perplexity.item() - average_perplexity) / i

            global_step += 1

            if global_step % params["training"]["checkpoint_interval"] == 0:
                save_checkpoint(model, optimizer, amp, global_step, Path(args.checkpoint_dir))

        writer.add_scalar("recon_loss/train", average_recon_loss, global_step)
        writer.add_scalar("vq_loss/train", average_vq_loss, global_step)
        writer.add_scalar("average_perplexity", average_perplexity, global_step)
        # writer.add_embedding(model.codebook.embedding, tag="codebook", global_step=global_step)
        # writer.add_embedding(model.decoder.speaker_embedding.weight, tag="speaker_embedding", global_step=global_step)

        print("epoch:{}, recon loss:{:.2E}, vq loss:{:.2E}, perpexlity:{:.3f}"
              .format(epoch, average_recon_loss, average_vq_loss, average_perplexity))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=6, help="Number of dataloader workers.")
    parser.add_argument("--checkpoint-dir", type=str, help="Directory to save checkpoints.")
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume")
    args = parser.parse_args()
    with open("config.json") as file:
        params = json.load(file)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    train_fn(args, params)
