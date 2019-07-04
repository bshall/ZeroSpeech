import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class VQEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VQEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1. / self.num_embeddings, 1. / self.num_embeddings)
        self.commitment_cost = commitment_cost

    def forward(self, x):
        x_flatten = x.view(-1, self.embedding_dim)

        # Compute the distances to the codebook
        distances = torch.addmm(torch.sum(self.embedding.weight ** 2, dim=1) +
                                torch.sum(x_flatten ** 2, dim=1, keepdim=True),
                                x_flatten, self.embedding.weight.t(),
                                alpha=-2.0, beta=1.0)

        _, indices = torch.min(distances, dim=1)
        quantized = self.embedding(indices)
        quantized = quantized.view_as(x)

        e_latent_loss = F.mse_loss(x, quantized.detach())
        q_latent_loss = F.mse_loss(quantized, x.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = x + (quantized - x).detach()

        encodings = torch.zeros(indices.shape[0], self.num_embeddings, device=x.device)
        encodings.scatter_(1, indices.unsqueeze(1), 1)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity


class Encoder(nn.Module):
    def __init__(self, mel_channels, encoder_channels, latent_channels):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(mel_channels, encoder_channels, 5, 1, 2, bias=False),
            nn.BatchNorm1d(encoder_channels),
            nn.ReLU(True),
            nn.Conv1d(encoder_channels, encoder_channels, 5, 1, 2, bias=False),
            nn.BatchNorm1d(encoder_channels),
            nn.ReLU(True),
            nn.Conv1d(encoder_channels, encoder_channels, 4, 2, 1, bias=False),
            nn.BatchNorm1d(encoder_channels),
            nn.ReLU(True),
            nn.Conv1d(encoder_channels, encoder_channels, 5, 1, 2, bias=False),
            nn.BatchNorm1d(encoder_channels),
            nn.ReLU(True),
            nn.Conv1d(encoder_channels, encoder_channels, 4, 2, 1, bias=False),
            nn.BatchNorm1d(encoder_channels),
            nn.ReLU(True),
            nn.Conv1d(encoder_channels, encoder_channels, 5, 1, 2, bias=False),
            nn.BatchNorm1d(encoder_channels),
            nn.ReLU(True),
            nn.Conv1d(encoder_channels, encoder_channels, 5, 1, 2, bias=False),
            nn.BatchNorm1d(encoder_channels),
            nn.ReLU(True),
        )
        self.proj = nn.Linear(encoder_channels, latent_channels)

    def forward(self, mels):
        x = self.conv(mels.transpose(1, 2))
        x = self.proj(x.transpose(1, 2))
        return x


def get_gru_cell(gru):
    cell = nn.GRUCell(gru.input_size, gru.hidden_size)
    cell.weight_hh.data = gru.weight_hh_l0.data
    cell.weight_ih.data = gru.weight_ih_l0.data
    cell.bias_hh.data = gru.bias_hh_l0.data
    cell.bias_ih.data = gru.bias_ih_l0.data
    return cell


class Decoder(nn.Module):
    def __init__(self, mel_channels, prenet_channels, num_speakers, speaker_embedding_dim,
                 latent_channels, decoder_channels, condition_channels):
        super(Decoder, self).__init__()
        self.mel_channels = mel_channels
        self.decoder_channels = decoder_channels

        self.prenet = nn.Sequential(
            nn.Linear(mel_channels, prenet_channels, bias=False),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(prenet_channels, prenet_channels, bias=False),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )
        self.speaker_embedding = nn.Embedding(num_speakers, speaker_embedding_dim)
        self.rnn1 = nn.GRU(latent_channels + speaker_embedding_dim, condition_channels, batch_first=True, bidirectional=True)
        self.rnn2 = nn.GRU(prenet_channels + 2*condition_channels, decoder_channels, batch_first=True)
        self.proj = nn.Linear(decoder_channels, mel_channels)

        nn.init.uniform_(self.speaker_embedding.weight, -1./512, 1./512)

    def forward(self, x, m, speakers):
        batch_size, frames, _ = x.size()
        speakers = self.speaker_embedding(speakers)
        speakers = speakers.unsqueeze(1).expand(-1, frames, -1)
        x = torch.cat((x, speakers), dim=2)
        x = F.interpolate(x.transpose(1, 2), scale_factor=4)
        x, _ = self.rnn1(x.transpose(1, 2))

        m = self.prenet(m)
        x, _ = self.rnn2(torch.cat((m, x), dim=2))
        mels = self.proj(x)
        return mels

    def generate(self, x, speaker):
        self.eval()
        cell = get_gru_cell(self.rnn2)
        output = []

        with torch.no_grad():
            speaker = self.speaker_embedding(speaker)
            speaker = speaker.unsqueeze(1).expand(-1, x.size(1), -1)
            x = torch.cat((x, speaker), dim=2)
            x = F.interpolate(x.transpose(1, 2), scale_factor=4)
            x, _ = self.rnn1(x.transpose(1, 2))

            h = torch.zeros(1, self.decoder_channels, device=x.device)
            m = torch.zeros(1, self.mel_channels, device=x.device)

            for z in tqdm(torch.unbind(x, dim=1), leave=False):
                m = self.prenet(m)
                h = cell(torch.cat((m, z), dim=1), h)
                m = self.proj(h)

                output.append(m.squeeze(0).cpu().numpy())

            output = np.vstack(output)
            self.train()
            return output


class Model(nn.Module):
    def __init__(self, mel_channels, encoder_channels, num_vq_embeddings, vq_embedding_dim, prenet_channels,
                 num_speakers, speaker_embedding_dim, decoder_channels, condition_channels):
        super(Model, self).__init__()
        self.encoder = Encoder(mel_channels, encoder_channels, vq_embedding_dim)
        self.codebook = VQEmbedding(num_vq_embeddings, vq_embedding_dim)
        self.decoder = Decoder(mel_channels, prenet_channels, num_speakers, speaker_embedding_dim,
                               vq_embedding_dim, decoder_channels, condition_channels)

    def forward(self, mels, speakers):
        x = self.encoder(mels[:, 1:, :])
        x, loss, perplexity = self.codebook(x)
        mels = self.decoder(x, mels[:, :-1, :], speakers)
        return mels, loss, perplexity

    def generate(self, mels, speaker):
        self.eval()
        with torch.no_grad():
            x = self.encoder(mels)
            x, _, _ = self.codebook(x)
            output = self.decoder.generate(x, speaker)
        self.train()
        return output
