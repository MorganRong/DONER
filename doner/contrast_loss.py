import math
import torch
import torch.nn.functional as F


def pair_masks(metadata, device):
    n=len(metadata)
    positive=torch.zeros((n,n),dtype=torch.bool,device=device)
    negative=torch.zeros_like(positive)
    required={'domain','content_id','style_id','signature','realization_seed'}
    for m in metadata:
        if not required.issubset(m) or m['domain'] not in ('ir','stm'):
            raise ValueError('Missing or invalid contrast pairing provenance')
    for i,a in enumerate(metadata):
        for j,b in enumerate(metadata):
            if i==j or a['domain']!=b['domain'] or not a.get('contrast_valid',True) or not b.get('contrast_valid',True):
                continue
            positive[i,j]=(a['style_id']==b['style_id'] and a['content_id']!=b['content_id']
                           and a['realization_seed']!=b['realization_seed'])


            amplitude_match=True
            if 'actual_rms' in a and 'actual_rms' in b:
                ratio=float(a['actual_rms'])/max(float(b['actual_rms']),1e-12)
                amplitude_match=.5<=ratio<=2.
            negative[i,j]=(a['signature']!=b['signature'] and a['style_id']!=b['style_id'] and amplitude_match)
    return positive,negative


def noise_contrastive_loss(projection, metadata, temperature=.15, allow_empty=False):
    if projection.ndim!=2 or len(projection)!=len(metadata) or not 0<temperature<10:
        raise ValueError('Expected NxD embeddings, N provenance records and positive temperature')
    if not bool(torch.isfinite(projection).all()):raise ValueError('Nonfinite embedding')
    positive,negative=pair_masks(metadata,projection.device)
    valid=positive.any(1)&negative.any(1)
    stats={'anchors':len(metadata),'valid_anchors':int(valid.sum()),
           'positive_pairs_directed':int(positive.sum()),'negative_pairs_directed':int(negative.sum()),
           'cross_domain_pairs':0,'temperature':temperature,'negative_actual_RMS_ratio_range':[.5,2.]}
    if not bool(valid.any()):
        if not allow_empty:raise ValueError('No anchor has both independent positive and distinct-noise negative')
        return projection.sum()*0.,stats
    z=F.normalize(projection,p=2,dim=1,eps=1e-8)
    logits=z@z.T/temperature
    support=positive|negative

    selected=logits[valid]
    denominator=torch.logsumexp(selected.masked_fill(~support[valid],-torch.inf),dim=1)
    positive_mean=(selected*positive[valid]).sum(1)/positive[valid].sum(1)
    loss=(denominator-positive_mean).mean()
    cosine=z@z.T
    stats.update(positive_cosine=float(cosine[positive].detach().mean()),
                 negative_cosine=float(cosine[negative].detach().mean()),
                 retrieval_accuracy=float(positive[valid].gather(1,selected.masked_fill(~support[valid],-torch.inf).argmax(1)[:,None]).float().mean()),
                 embedding_coordinate_std=float(z.detach().std(0,unbiased=False).mean()))
    return loss,stats


@torch.no_grad()
def noise_statistics_target(noise):

    x=noise.mean(1,keepdim=True)
    x=x-x.mean((-2,-1),keepdim=True)
    power=x.square().mean((1,2,3)).clamp_min(1e-12)
    rms=power.sqrt()
    row=x.mean(-1).square().mean((1,2))/power
    column=x.mean(-2).square().mean((1,2))/power
    lag_y=(x[:,:,1:,:]*x[:,:,:-1,:]).mean((1,2,3))/power
    lag_x=(x[:,:,:,1:]*x[:,:,:,:-1]).mean((1,2,3))/power
    spectrum=torch.fft.fft2(x,norm='ortho').abs().square()
    h,w=x.shape[-2:]
    fy=torch.fft.fftfreq(h,device=x.device)[:,None]
    fx=torch.fft.fftfreq(w,device=x.device)[None,:]
    low=(fy.square()+fx.square() <= (1/16)**2)&((fy!=0)|(fx!=0))
    fraction=(spectrum*low).sum((1,2,3))/spectrum.sum((1,2,3)).clamp_min(1e-12)
    return torch.stack((rms.log()/math.log(10)/3,row,column,
                        lag_y.clamp(-1,1),lag_x.clamp(-1,1),fraction),1)


def gradient_l1(a,b):
    return ((torch.diff(a,dim=-1)-torch.diff(b,dim=-1)).abs().mean()
            +(torch.diff(a,dim=-2)-torch.diff(b,dim=-2)).abs().mean())


def reconstruction_loss(output,clean):
    return (output-clean).abs().mean()+.2*gradient_l1(output,clean)
