import hydra
import hydra.utils as utils

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch

from model import Encoder


@hydra.main(config_path="config/encode.yaml")
def encode_dataset(cfg):
    out_dir = Path(utils.to_absolute_path(cfg.out_dir))
    out_dir.mkdir(exist_ok=True, parents=True)

    if cfg.save_auxiliary:
        aux_path = out_dir.parent / "auxiliary_embedding1"
        aux_path.mkdir(exist_ok=True, parents=True)

    root_path = Path(utils.to_absolute_path("datasets")) / cfg.dataset.path
    with open(root_path / "test.json") as file:
        metadata = json.load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder(**cfg.model.encoder)
    encoder.to(device)

    print("Load checkpoint from: {}:".format(cfg.checkpoint))
    checkpoint_path = utils.to_absolute_path(cfg.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    encoder.load_state_dict(checkpoint["encoder"])

    encoder.eval()

    if cfg.save_auxiliary:
        auxiliary = []

        def hook(module, input, output):
            auxiliary.append(output.clone().transpose(1, 2))

        encoder.encoder[-1].register_forward_hook(hook)

    for _, _, _, path in tqdm(metadata):
        path = root_path.parent / path
        mel = torch.from_numpy(np.load(path.with_suffix(".mel.npy"))).unsqueeze(0).to(device)
        with torch.no_grad():
            z, indices = encoder.encode(mel)

        z = z.squeeze().cpu().numpy()

        out_path = out_dir / path.stem
        with open(out_path.with_suffix(".txt"), "w") as file:
            np.savetxt(file, z, fmt="%.16f")

        if cfg.save_auxiliary:
            out_path = aux_path / path.stem
            aux = auxiliary.pop().squeeze().cpu().numpy()
            with open(out_path.with_suffix(".txt"), "w") as file:
                np.savetxt(file, aux, fmt="%.16f")


if __name__ == "__main__":
    encode_dataset()
