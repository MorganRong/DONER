import argparse
from pathlib import Path

import numpy as np
import torch
from doner.images import read_image
from torch.nn import functional as F

from .censure import CENSURE
from .clipdenoising import CLIPDenoising
from .deal import NE
from .drunet_light import DRUNet_light
from .idf import IDFNet
from .rninet import RNINet


def network(name):
    if name == "DEAL":
        return NE()
    if name == "RNINet":
        return RNINet(apply_prob=0.5, ep=1e-6, in_nc=1, out_nc=1,
                      nc=[64, 128, 256, 512], nb=2, act_mode="BR")
    if name == "CLIPDenoising":
        return CLIPDenoising(num_blocks=[3, 4, 6, 3], inp_channels=1,
                             out_channels=1, depth=5, wf=64, slope=0.2,
                             bias=False, aug_level=0.025)
    if name == "CENSURE":
        return CENSURE(DRUNet_light(in_nc=1), color=False, n=4)
    if name == "IDF":
        return IDFNet(num_iter=10, kernel_size=3, num_channels=3, hidden_channels=64,
                      power_alpha=3.0, halt_threshold=0.015)
    raise ValueError(name)


def load(name, checkpoint, device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = network(name)
    if name == "IDF":
        state = payload["state_dict"]
        if not all(key.startswith("model.") for key in state):
            raise ValueError("Expected an official IDF checkpoint.")
        state = {key.removeprefix("model."): value for key, value in state.items()}
        norm = payload["hyper_parameters"]["data_config"]["norm"]
        mean = np.asarray(norm["raw_mean"], dtype=np.float32)
        std = np.asarray(norm["raw_std"], dtype=np.float32)
        scale = np.float32(norm["sigma_data"]) / std
        bias = np.float32(norm["mu_data"]) - mean * scale
        model.load_state_dict(state, strict=True)
        model.register_buffer("input_scale", torch.from_numpy(scale).view(1, 3, 1, 1))
        model.register_buffer("input_bias", torch.from_numpy(bias).view(1, 3, 1, 1))
    else:
        if name == "DEAL" and "models" in payload:
            state = payload["models"]["NE"]
        else:
            state = payload.get("model", payload)
        model.load_state_dict(state, strict=True)
    return model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)


def restore(name, model, x):
    if x.ndim != 4 or x.shape[:2] != (1, 1):
        raise ValueError("Expected one grayscale image in NCHW format.")
    with torch.no_grad():
        if name == "CENSURE":
            with torch.inference_mode(False):
                return model(x.detach().clone()).detach()
        if name == "IDF":
            rgb = x.repeat(1, 3, 1, 1) * model.input_scale + model.input_bias
            pred = model(rgb, adaptive_iter=False, max_iter=10, alpha_schedule=None)
            return ((pred - model.input_bias) / model.input_scale).mean(1, keepdim=True)
        if name == "DEAL":
            return model(x.repeat(1, 3, 1, 1)).mean(1, keepdim=True)
        factor = 8 if name == "RNINet" else 32
        h, w = x.shape[-2:]
        padding = (0, (-w) % factor, 0, (-h) % factor)
        return model(F.pad(x, padding, mode="reflect"))[..., :h, :w]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["DEAL", "RNINet", "CLIPDenoising", "IDF", "CENSURE"])
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    x = torch.from_numpy(read_image(args.input))[None, None].to(args.device)
    model = load(args.method, args.checkpoint, args.device)
    y = restore(args.method, model, x)
    if y.shape != x.shape or not torch.isfinite(y).all():
        raise ValueError("Invalid output shape or non-finite prediction.")
    if args.output.suffix.lower() != ".npy":
        raise ValueError("Use an .npy output path to preserve the floating-point result.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, y[0, 0].detach().cpu().numpy(), allow_pickle=False)


if __name__ == "__main__":
    main()
