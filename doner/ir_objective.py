from torch import nn
import torch
from torch.nn import functional as F
from . import pairwise_loss
from .ir_losses import reconstruction, response_loss, analytic_identity_loss
def check(x,b):
    assert x.shape==(b,1,*[768,1024]) and x.dtype==torch.float32 and bool(torch.isfinite(x).all())
class Objective(nn.Module):
    def __init__(self,net,cfg):
        super().__init__();self.net=net;self.cfg=cfg
    def forward(self,noisy,clean,metadata,pair_record,known=None,analytic_clean=None,analytic_signal=None):
        check(noisy,2);check(clean,2)
        pred,aux=self.net.forward_with_aux(noisy)
        check(pred,2)
        restore=reconstruction(pred,clean)
        contrast,pair_info=pairwise_loss.pairwise_contrastive_loss(aux['projection'],metadata,
            bool(pair_record['pair_same']),bool(pair_record['pair_valid']),margin=.5)
        target=pairwise_loss.noise_statistics_target(noisy-clean)
        losses={'reconstruction':restore,'contrast':self.cfg['contrast_weight']*contrast,
                'statistics':self.cfg['statistics_weight']*F.smooth_l1_loss(aux['noise_statistics'],target)}
        noise=noisy-clean
        with torch.no_grad():
            local=noise-noise.mean(-1,keepdim=True)-noise.mean(-2,keepdim=True)+noise.mean((-2,-1),keepdim=True)
            pure=(local.square().mean((1,2,3)).sqrt()<1e-6)&(noise.square().mean((1,2,3)).sqrt()>1e-6)
        if bool(pure.any()):
            losses['stripe_teacher']=self.cfg['stripe_teacher_weight']*pure.float().mean()*reconstruction(pred[pure],aux['baseline'][pure])
        if known is not None:
            check(known,2);check(analytic_clean,1)
            plus=self.net(noisy+known)
            losses['reconstruction']=.5*(restore+reconstruction(plus,clean+known))
            losses['response']=self.cfg['response_weight']*response_loss(pred,plus,known)
            identity=self.net(clean)
            losses['identity']=self.cfg['identity_weight']*reconstruction(identity,clean)
            analytic_pred=self.net(analytic_clean)
            losses['analytic_identity']=self.cfg['analytic_identity_weight']*analytic_identity_loss(analytic_pred,analytic_clean,analytic_signal)
        total=sum(losses.values())
        return total,{k:v.detach() for k,v in losses.items()},pair_info,int(pure.sum())
