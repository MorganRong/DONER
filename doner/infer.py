import argparse
from pathlib import Path

import numpy as np
import torch

from .images import read_image
from .model import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', required=True, choices=('stm', 'ir', 'confocal'))
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.output.suffix.lower() != '.npy':
        raise ValueError('Choose a .npy output path to preserve unclipped float32 values.')
    image = torch.from_numpy(read_image(args.input)).unsqueeze(0).unsqueeze(0).to(args.device)
    model = load_model(args.domain, args.checkpoint, device=args.device)
    with torch.inference_mode():
        output = model(image)[0, 0].cpu().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, output, allow_pickle=False)


if __name__ == '__main__':
    main()
