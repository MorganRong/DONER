import torch
from torch.nn import functional as F
SEED=20261101
def base_objectives(stage, net, batch, step, identity_calls, enable_encoder):
    cfg, ops = stage.cfg, stage.losses
    noisy, clean = batch['noisy'], batch['clean']
    pred, aux = net.forward_with_aux(noisy)
    restore = ops.reconstruction(pred, clean)
    contrast = pred.sum()*0
    for pair_idx, row in enumerate(batch['pair_records']):
        sl = slice(2*pair_idx,2*pair_idx+2)
        if enable_encoder:
            value,_ = stage.pairs.pairwise_contrastive_loss(aux['projection'][sl],batch['metadata'][sl],
                           bool(row['pair_same']), bool(row['pair_valid']), margin=.5)
            contrast = contrast + value / len(batch['pair_records'])
    noise = noisy-clean
    stats = F.smooth_l1_loss(aux['noise_statistics'], stage.pairs.noise_statistics_target(noise))
    parts = {'reconstruction':restore, 'contrast':cfg['contrast_weight'] * contrast,
             'statistics':cfg['statistics_weight'] * stats}
    with torch.no_grad():
        interaction = noise-noise.mean(-1,keepdim=True)-noise.mean(-2,keepdim=True)+noise.mean((-2,-1),keepdim=True)
        pure = (interaction.square().mean((1,2,3)).sqrt()<1e-6) & (noise.square().mean((1,2,3)).sqrt()>1e-6)
    if bool(pure.any()):
        parts['stripe_teacher'] = cfg['stripe_teacher_weight'] * pure.float().mean() * ops.reconstruction(pred[pure],aux['baseline'][pure])
    if step % cfg['fidelity_every'] == 0 or step in cfg['gradient_audit_steps']:
        known,_ = stage.detail.detail_signals(clean, step, SEED+2701)
        plus,_ = net.forward_with_aux(noisy+known)
        parts['reconstruction'] = .5*(restore+ops.reconstruction(plus,clean+known))
        parts['response'] = cfg['response_weight']*ops.response_loss(pred,plus,known)
        parts['identity'] = cfg['identity_weight']*ops.reconstruction(net(clean),clean)
        if stage.phase != 'r2':
            identity_calls += 1
            analytic, records = stage.detail.make_detail_clean(torch.zeros_like(clean[:2]),identity_calls,SEED+7303)
            bg = analytic.new_tensor([r['background'] for r in records])[:,None,None,None]
            parts['analytic_identity'] = cfg['analytic_identity_weight']*ops.analytic_identity_loss(net(analytic),analytic,analytic-bg)
    return parts, identity_calls
