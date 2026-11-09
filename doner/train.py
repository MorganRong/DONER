import argparse
import importlib
import json
import math
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel

from . import objectives, pairwise_loss
from .confocal_data import DomainData
from .ir_objective import Objective as InfraredObjective
from .model import build_model

SEED = 20261101
TOTAL_STEPS = 21000
ROOT = Path(__file__).resolve().parent


def training_stage(domain, directory, step):
    prefix = domain
    variant = 'native'
    phase = 'native'
    if domain == 'confocal':
        phase = 'r2' if step <= 2048 else 'r3' if step <= 6156 else 'r5'
        prefix = 'confocal_early' if phase == 'r2' else 'confocal_late'
        variant = 'confocal_final' if phase == 'r5' else prefix
    data_module = importlib.import_module(f'.{prefix}_data', __package__)
    geometry = {'ir': None, 'stm': 300, 'confocal': 128}[domain]
    sampler = data_module.TrainingData(seed=SEED, patch_size=geometry, train_manifest=directory)
    if domain == 'confocal':
        sampler = DomainData(sampler)
    config = json.loads((ROOT / 'settings' / (variant + '.json')).read_text())
    losses_name = 'native' if variant == 'confocal_final' else variant
    losses = importlib.import_module(f'.{losses_name}_losses', __package__)
    detail = importlib.import_module(f'.{domain}_detail', __package__)
    return SimpleNamespace(phase=phase, cfg=config, data=sampler, losses=losses,
                           detail=detail, pairs=pairwise_loss)


def infrared_step(model, stage, step, rank, device, identity_calls):
    batch = stage.data.get_batch(step, device='cpu', pair_rank=rank)
    clean = batch['clean'].to(device)
    noisy = batch['noisy'].to(device)
    known = analytic = signal = None
    if step % stage.cfg['fidelity_every'] == 0 or step == 1:
        identity_calls += 1
        known, _ = stage.detail.detail_signals(clean, 2*(step-1)+rank+1, SEED+2701)
        templates, metadata = stage.detail.make_detail_clean(torch.zeros(2, 1, 768, 1024),
                                                            identity_calls, SEED+7303)
        analytic = templates[rank:rank+1].to(device)
        signal = analytic - float(metadata[rank]['background'])
    loss, _, _, _ = model(noisy, clean, batch['metadata'], batch['pair_records'][0],
                          known, analytic, signal)
    return loss, identity_calls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', choices=('stm','ir','confocal'), required=True)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--prior', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--steps', type=int, default=TOTAL_STEPS)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if not 1 <= args.steps <= TOTAL_STEPS:
        raise ValueError('Use 1 to 21000 updates.')
    ddp = args.domain == 'ir'
    rank = int(os.environ.get('RANK', 0))
    if ddp:
        if int(os.environ.get('WORLD_SIZE', 1)) != 2:
            raise ValueError('Run infrared training using torchrun --standalone --nproc_per_node=2.')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        torch.distributed.init_process_group('nccl')
    else:
        device = torch.device(args.device)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    stage = training_stage(args.domain, args.data, 1)
    net = build_model(args.domain, prior=args.prior).to(device).train()
    trainable = [p for p in net.parameters() if p.requires_grad]
    model = net
    if ddp:
        model = DistributedDataParallel(InfraredObjective(net, stage.cfg), device_ids=[device.index],
                                        output_device=device.index, broadcast_buffers=False)
    optimizer = torch.optim.AdamW(trainable, lr=stage.cfg['learning_rate'],
                                  weight_decay=stage.cfg['weight_decay'])
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    if ddp:
        torch.distributed.barrier()
    identity_calls = 0
    for step in range(1, args.steps+1):
        if args.domain == 'confocal' and step in (2049, 6157):
            stage = training_stage(args.domain, args.data, step)
        cfg = stage.cfg
        lr = cfg['min_learning_rate'] + .5*(cfg['learning_rate'] - cfg['min_learning_rate'])*(1+math.cos(math.pi*(step-1)/TOTAL_STEPS))
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        if ddp:
            loss, identity_calls = infrared_step(model, stage, step, rank, device, identity_calls)
        else:
            batch = stage.data.get_batch(step, device)
            parts, identity_calls = objectives.base_objectives(stage, net, batch, step,
                                                               identity_calls, True)
            loss = sum(parts.values())
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Non-finite training objective at update {step}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, cfg['gradient_clip_norm'], error_if_nonfinite=True)
        optimizer.step()
        if rank == 0 and (step == 1 or step % 100 == 0 or step == args.steps):
            print(f'update={step} objective={float(loss.detach()):.6f}', flush=True)
        if rank == 0 and (step % 500 == 0 or step == args.steps):
            name = 'final.pt' if step == TOTAL_STEPS else 'latest.pt'
            target = args.output / name
            temporary = target.with_suffix('.tmp')
            torch.save({'model': net.state_dict(), 'step':step, 'domain':args.domain}, temporary)
            temporary.replace(target)
    if ddp:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
