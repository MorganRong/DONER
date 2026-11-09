from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

MODEL_ID = 'stabilityai/stable-diffusion-xl-base-1.0'
REVISION = '462165984030d82259a11f4367a4eed129e94a7b'
NEGATIVE = ('photograph, objects, stones, pebbles, crystal, atoms, lattice, leaves, '
            'swirl, diagonal lines, grid, tiles, shadows, perspective, text, watermark, color')
PROMPTS = {
    'short_streaks': 'Flat two dimensional grayscale noise map. Sparse isolated thin horizontal bright and dark scan errors, short finite straight segments with distinct endpoints, irregular spacing, neutral uniform background.',
    'grain_streaks': 'Flat two dimensional grayscale electronic noise map. Fine irregular grain and sparse isolated thin short horizontal bright and dark scan errors, distinct endpoints, neutral background.',
    'scan_bands': 'Flat two dimensional grayscale scan noise map. Irregular thin horizontal bright and dark scan lines, nonperiodic spacing, intensity changes along each line, no sample structures.',
    'row_bands': 'Flat two dimensional grayscale detector noise map. Irregular horizontal brightness bands, uneven spacing, nonperiodic offsets, intensity varies along each row, no scene.',
    'column_bands': 'Flat two dimensional grayscale detector noise map. Irregular vertical brightness bands, uneven spacing, nonperiodic offsets, intensity varies along each column, no scene.',
    'grain': 'Flat two dimensional grayscale electronic noise map. Fine irregular bright and dark pixel fluctuations, uneven local strength, no objects or coherent shapes.',
    'smooth': 'Flat two dimensional grayscale detector residual map. Subtle smooth irregular brightness variation without edges, no objects or scene.',
}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def seed_for(dataset, family, split, index, role='guide'):
    key = f'20261028_guided_v3|{dataset}|{family}|{split}|{index}|{role}'
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big') % (2**63-1)


def unit(a):
    a = np.asarray(a, dtype=np.float64)
    a = a - a.mean()
    rms = float(np.sqrt(np.mean(a*a)))
    if not np.isfinite(rms) or rms < 1e-9:
        raise ValueError('Degenerate field')
    return np.asarray(a/rms, dtype=np.float32)


def smooth_noise(rng, shape, coarse=16):

    a = rng.standard_normal((coarse, coarse)).astype(np.float32)
    return np.asarray(Image.fromarray(a, mode='F').resize((shape[1],shape[0]),Image.Resampling.BILINEAR))


def make_guide(kind, options, seed, size=256, return_metadata=False):
    rng = np.random.default_rng(seed)
    mask = np.zeros((size,size), dtype=bool)
    events = []
    grain = rng.standard_normal((size,size))
    correlation = float(options.get('grain_correlation', 0))
    if correlation:
        grain = (1-correlation)*grain + correlation*(np.roll(grain,1,0)+np.roll(grain,1,1))/2
    envelope = 1 + float(options.get('local_variation', .25))*unit(smooth_noise(rng,(size,size),8))
    envelope = np.clip(envelope, .25, 2)
    grain = unit(grain*envelope)
    if kind in ('short_streaks','grain_streaks'):
        field = np.zeros((size,size), dtype=np.float32)
        density = float(options.get('events_per_128', 6))
        count = max(1, int(round(density*(size/128)**2)))
        lengths = options.get('length_px', [12,48])
        widths = options.get('width_px', [1,2])
        for index in range(count):
            length = int(rng.integers(lengths[0],lengths[1]+1))
            width = int(rng.integers(widths[0],widths[1]+1))
            for attempt in range(1000):
                y,x = int(rng.integers(3,size-width-3)),int(rng.integers(3,size-length-3))
                if not mask[y-2:y+width+2,x-2:x+length+2].any(): break
            else: raise ValueError('Cannot place requested sparse guide events')
            polarity = 1 if index%2==0 else -1
            contrast = polarity*float(rng.uniform(.14,.27))
            modulation = 1+.15*np.sin(np.linspace(0,rng.uniform(.5,2)*np.pi,length)+rng.uniform(0,6))
            field[y:y+width,x:x+length]=contrast*modulation[None,:]
            mask[y:y+width,x:x+length]=True
            events.append({'y':y,'x':x,'length':length,'width':width,'polarity':polarity,'contrast':contrast})
        field += (float(options.get('grain_std',.018)) if kind=='grain_streaks' else .002)*grain
    elif kind in ('scan_bands','row_bands','column_bands'):
        widths=options.get('width_px',[1,3]); profile=np.zeros(size); cursor=0
        while cursor<size:
            width=int(rng.integers(widths[0],widths[1]+1));value=float(rng.normal(0,.07))
            profile[cursor:cursor+width]=value;cursor+=width
        along = smooth_noise(rng,(size,size),int(options.get('envelope_grid',12)))
        field=profile[:,None]*(1+float(options.get('line_variation',.35))*unit(along))
        if kind=='column_bands': field=field.T
        field+=float(options.get('grain_std',.006))*grain
    elif kind=='grain':
        field=float(options.get('grain_std',.055))*grain
    elif kind=='smooth':
        field=.06*unit(smooth_noise(rng,(size,size),int(options.get('envelope_grid',8))))
        field+=.003*grain
    else: raise ValueError('Unknown guide kind '+kind)
    if kind in ('short_streaks','grain_streaks') and options.get('axis','row')=='column':
        field=field.T;mask=mask.T
        events=[dict(ev,y=ev['x'],x=ev['y'],axis='column') for ev in events]
    else:
        events=[dict(ev,axis='row') for ev in events]
    peak=float(np.max(np.abs(field)))
    scale=min(1.,.45/max(peak,1e-12))
    gray=np.asarray(.5+field*scale,dtype=np.float32)
    guide_meta={'global_amplitude_downscale':scale,'original_residual_peak':peak,
                'pixel_clipping':False,'support_fraction':float(mask.mean()),
                'events_per_128_actual':len(events)/(size/128)**2,
                'guide_grid':size,'generation_grid':1024}
    if return_metadata:return gray,mask,events,guide_meta
    return gray,mask,events


def morphology_metrics(guide, output, mask, events, kind):
    a,b=unit(guide).astype(np.float64),unit(output).astype(np.float64)
    gx=float(np.mean(np.diff(b,axis=1)**2));gy=float(np.mean(np.diff(b,axis=0)**2))
    metrics={'guide_output_correlation':float(np.mean(a*b)),
             'generated_vs_guide_unit_RMSE':float(np.sqrt(np.mean((a-b)**2))),
             'horizontal_structure_fraction':gy/max(gx+gy,1e-30),
             'bank_shape':list(output.shape), 'events_requested':len(events)}
    failures=[]
    if kind!='grain' and metrics['guide_output_correlation']<.6: failures.append('guide_shape_lost')
    if metrics['generated_vs_guide_unit_RMSE']<.001: failures.append('generated_output_identical_to_guide')
    if events:
        axis=events[0].get('axis','row')
        work=output.T if axis=='column' else output
        local_events=[dict(ev,y=ev['x'],x=ev['y']) if axis=='column' else ev for ev in events]
        dilated=mask.copy()
        for dy in (-1,0,1):
            for dx in (-1,0,1): dilated|=np.roll(mask,(dy,dx),(0,1))
        field=output-output.mean()
        energies=field**2
        metrics['event_neighborhood_energy_fraction']=float(energies[dilated].sum()/max(energies.sum(),1e-30))
        detected=[];endpoint_errors=[];fwhm=[];lengths=[]
        for ev in local_events:
            y,x,L,w=ev['y'],ev['x'],ev['length'],ev['width']
            inside=float(work[y:y+w,x:x+L].mean())
            background=float(np.concatenate([work[y-3:y-1,x:x+L].ravel(),work[y+w+1:y+w+3,x:x+L].ravel()]).mean())
            detected.append((inside-background)*ev['polarity'])

            left,right=max(0,x-5),min(work.shape[1],x+L+5)
            along=(work[y:y+w,left:right].mean(0)-background)*ev['polarity']
            mid=x+L//2-left;threshold=max(detected[-1]*.5,.005)
            above=along>=threshold
            lo=hi=mid
            while lo>0 and above[lo-1]:lo-=1
            while hi+1<len(above) and above[hi+1]:hi+=1
            measured_length=hi-lo+1 if above[mid] else 0
            lengths.append(measured_length)
            endpoint_errors.append(max(abs((left+lo)-x),abs((left+hi+1)-(x+L))) if measured_length else L)
            top,bottom=max(0,y-5),min(work.shape[0],y+w+5)
            across=(work[top:bottom,x+L//4:x+3*L//4].mean(1)-background)*ev['polarity']
            center=int(np.argmax(across[max(0,y-top-1):min(len(across),y-top+w+1)]))+max(0,y-top-1)
            above_y=across>=max(float(across[center])*.5,.005)
            lo=hi=center
            while lo>0 and above_y[lo-1]:lo-=1
            while hi+1<len(above_y) and above_y[hi+1]:hi+=1
            fwhm.append(hi-lo+1 if above_y[center] else 0)
        metrics['event_polarity_retained_fraction']=float(np.mean(np.asarray(detected)>0.02))
        metrics['event_contrast_min_median_max']=[float(min(detected)),float(np.median(detected)),float(max(detected))]
        if metrics['event_polarity_retained_fraction']<.9: failures.append('finite_streak_contrast_lost')
        metrics['event_axis']=axis
        metrics['output_endpoint_error_max_px']=float(max(endpoint_errors))
        metrics['output_endpoint_error_p90_px']=float(np.quantile(endpoint_errors,.9))
        metrics['output_length_min_median_max_px']=[int(min(lengths)),float(np.median(lengths)),int(max(lengths))]
        metrics['output_width_FWHM_min_median_max_px']=[int(min(fwhm)),float(np.median(fwhm)),int(max(fwhm))]
        metrics['output_endpoint_within_3px_fraction']=float(np.mean(np.asarray(endpoint_errors)<=3))
        metrics['output_width_within_target_plus_2px_fraction']=float(np.mean([0<v<=e['width']+2 for v,e in zip(fwhm,local_events)]))
        directional=metrics['horizontal_structure_fraction'] if axis=='row' else 1-metrics['horizontal_structure_fraction']
        if directional<.70: failures.append('scan_direction_lost')
        if metrics['output_endpoint_within_3px_fraction']<.90: failures.append('finite_endpoints_lost')
        if metrics['output_width_within_target_plus_2px_fraction']<.90: failures.append('thin_line_width_lost')
        if kind=='short_streaks' and metrics['event_neighborhood_energy_fraction']<.60: failures.append('sparse_streak_support_lost')
    elif kind in ('scan_bands','row_bands') and metrics['horizontal_structure_fraction']<.70:
        failures.append('horizontal_direction_lost')
    elif kind=='column_bands' and metrics['horizontal_structure_fraction']>.30:
        failures.append('vertical_direction_lost')
    elif kind=='grain':
        fy,fx=np.fft.fftfreq(output.shape[0])[:,None],np.fft.rfftfreq(output.shape[1])[None,:]
        power=np.abs(np.fft.rfft2(b))**2;power[0,0]=0
        radius=np.hypot(fy,fx);total=max(float(power.sum()),1e-30)
        metrics['grain_high_frequency_power_fraction']=float(power[radius>.125].sum()/total)
        metrics['grain_low_frequency_power_fraction']=float(power[radius<.03125].sum()/total)
        metrics['grain_max_frequency_power_fraction']=float(power.max()/total)
        metrics['grain_lag1_row_correlation']=float(np.mean(b*np.roll(b,1,0)))
        metrics['grain_lag1_column_correlation']=float(np.mean(b*np.roll(b,1,1)))
        if not .25<metrics['horizontal_structure_fraction']<.75:failures.append('grain_axis_dominance')
        if metrics['grain_high_frequency_power_fraction']<.2:failures.append('fine_grain_power_lost')
        if metrics['grain_low_frequency_power_fraction']>.3:failures.append('broad_structures_dominate_grain')
        if metrics['grain_max_frequency_power_fraction']>.02:failures.append('periodic_grain_pattern')
    metrics['numeric_passed']=not failures;metrics['failures']=failures
    metrics['semantic_validity_certified']=False
    return metrics


def verify_model(model):
    pin=json.loads((model/'pinned_model.json').read_text())
    if pin['model_id']!=MODEL_ID or pin['revision']!=REVISION: raise ValueError('Unexpected pretrained model')
    for name,expected in pin['files'].items():
        p=model/name
        if p.stat().st_size!=expected['bytes'] or sha(p)!=expected['sha256']: raise ValueError('Model pin failed '+name)
    return sha(model/'pinned_model.json')


def load_pipeline(model,device):
    import torch
    from diffusers import StableDiffusionXLImg2ImgPipeline
    pipe=StableDiffusionXLImg2ImgPipeline.from_pretrained(str(model),torch_dtype=torch.float16,
        variant='fp16',use_safetensors=True,local_files_only=True,add_watermarker=False).to(device)
    for m in (pipe.unet,pipe.vae,pipe.text_encoder,pipe.text_encoder_2):m.eval().requires_grad_(False)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate(pipe,guide,prompt,seed,strength,device,steps=30,guidance=3.0):
    import torch

    enlarged=np.asarray(Image.fromarray(guide,mode='F').resize((1024,1024),Image.Resampling.NEAREST))
    rgb=np.repeat(enlarged[:,:,None],3,axis=2).astype(np.float32)
    for tokenizer in (pipe.tokenizer,pipe.tokenizer_2):
        if any(len(tokenizer(t,truncation=False)['input_ids'])>tokenizer.model_max_length for t in (prompt,NEGATIVE)):
            raise ValueError('Prompt would be truncated')
    with torch.inference_mode():
        raw=pipe(prompt=prompt,negative_prompt=NEGATIVE,image=rgb,strength=strength,
                 num_inference_steps=steps,guidance_scale=guidance,
                 generator=torch.Generator(device='cpu').manual_seed(seed),output_type='np').images[0]
    raw=np.asarray(raw,dtype=np.float32)
    if raw.shape!=(1024,1024,3) or not np.isfinite(raw).all():raise ValueError('Invalid SDXL output')
    gray=raw@np.asarray([.2126,.7152,.0722],dtype=np.float32)
    scale=1024//guide.shape[0]
    if scale*guide.shape[0]!=1024: raise ValueError('Guide resolution must divide1024')
    reduced=gray.reshape(guide.shape[0],scale,guide.shape[1],scale).mean((1,3))
    return raw,np.asarray(reduced,dtype=np.float32)


def save_preview(path,guide,generated,texture,label):
    size=guide.shape[0];canvas=Image.new('RGB',(size*3,size+40),'white');draw=ImageDraw.Draw(canvas)
    for index,(array,title) in enumerate(((guide,'analytic guide'),(generated,'SDXL generated'),(.5+texture/6,'bank normalized'))):
        img=Image.fromarray(np.rint(np.clip(array,0,1)*255).astype('uint8')).convert('RGB')
        canvas.paste(img,(size*index,0));draw.text((size*index+2,size+2),title,fill='black')
    draw.text((2,size+20),label,fill='black');canvas.save(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--model-path',type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--probe',action='store_true')
    p.add_argument('--text-probe-only',action='store_true')
    p.add_argument('--profiles',type=Path);p.add_argument('--domain',choices=('ir','stm','all'),default='all')
    args=p.parse_args();root=args.root.resolve();root.mkdir(parents=True,exist_ok=True)
    model=args.model_path.resolve();pin=verify_model(model);pipe=load_pipeline(model,args.device)
    if args.text_probe_only:
        directory=root/'guided_probe';report=[]
        for kind in ('short_streaks','grain'):
            dataset='stm_hq50';guide,mask,events=make_guide(kind,{},seed_for(dataset,kind,'probe',0))
            seed=seed_for(dataset,kind,'probe',0,'diffusion')
            target=np.load(directory/f'stm_{kind}_probe_000_s12_generated.npy')
            for label,prompt in [('empty',''),('mismatched',PROMPTS['column_bands'])]:
                _,changed=generate(pipe,guide,prompt,seed,.12,args.device)
                np.save(directory/f'{kind}_{label}_prompt_generated.npy',changed)
                report.append({'kind':kind,'changed_prompt':label,'same_guide_and_diffusion_seed':True,
                               'pixel_RMS_difference':float(np.sqrt(np.mean((target.astype(float)-changed)**2)))})
        write_json(directory/'text_effect_probe_extended.json',{'comparisons':report,'interpretation':'Text affects output; no claim of restoration benefit.'})
        for meta_path in directory.glob('stm_*_s[0-9][0-9].json'):
            meta=json.loads(meta_path.read_text());task=meta['recipe']
            guide=np.load(directory/meta['files']['guide']['path']);out=np.load(directory/meta['files']['generated']['path'])
            mask=np.load(directory/meta['files']['mask']['path']);metrics=morphology_metrics(guide,out,mask,meta['guide_events'],task['kind'])
            write_json(directory/(meta_path.stem+'_reaudit.json'),metrics)
        return
    if args.probe:
        tasks=[]
        for kind in ('short_streaks','grain_streaks','scan_bands','grain'):
            for strength in (.12,.22):
                tasks.append({'domain':'stm','dataset_id':'stm_hq50','id':kind,'kind':kind,'options':{},
                              'split':'probe','index':0,'strength':strength,'prompt':PROMPTS[kind]})
    else:
        profiles=json.loads((args.profiles or root/'guided_profiles.json').read_text());tasks=[]
        for domain,profile in profiles['datasets'].items():
            if args.domain not in ('all',domain):continue
            for family in profile['families']:
                for split,count in (('train',profile['train_seeds_per_pattern']),('validation',profile['validation_seeds_per_pattern'])):
                    for index in range(count):tasks.append(dict(domain=domain,dataset_id=profile['dataset_id'],
                        split=split,index=index,strength=profile.get('strength',.12),**family))
    directory=root/('guided_probe' if args.probe else 'guided_candidates');directory.mkdir(exist_ok=True)
    summary=[]
    for task in tasks:
        kind=task['kind'];strength=task['strength'];index=task['index'];split=task['split'];dataset=task['dataset_id']
        ident=f"{task['domain']}_{task['id']}_{split}_{index:03d}_s{int(strength*100):02d}"
        meta_path=directory/(ident+'.json')
        if meta_path.exists():
            previous=json.loads(meta_path.read_text())
            if (previous['recipe']!=task or previous['source_sha256']!=sha(Path(__file__))
                    or previous['model_pin_sha256']!=pin):
                raise ValueError('Existing generated recipe/source/model differs '+ident)
            for value in previous['files'].values():
                if sha(directory/value['path'])!=value['sha256']:raise ValueError('Existing file hash differs '+ident)
            summary.append(previous)
            continue
        if any(directory.glob(ident+'_*.npy')):
            raise ValueError('Partial outputs without completion metadata; preserve and inspect '+ident)
        guide,mask,events,guide_meta=make_guide(kind,task.get('options',{}),seed_for(dataset,task['id'],split,index),return_metadata=True)
        start=time.monotonic();generation_seed=seed_for(dataset,task['id'],split,index,'diffusion')
        raw,reduced=generate(pipe,guide,task.get('prompt',PROMPTS[kind]),generation_seed,strength,args.device)
        texture=unit(reduced);metrics=morphology_metrics(guide,reduced,mask,events,kind)
        files={}
        for suffix,array in (('guide',guide),('raw_rgb',raw),('generated',reduced),('texture',texture),('mask',mask)):
            path=directory/f'{ident}_{suffix}.npy';np.save(path,array,allow_pickle=False)
            files[suffix]={'path':path.name,'sha256':sha(path)}
        metadata={'id':ident,'recipe':task,'guide_events':events,'guide_statistics':guide_meta,'metrics':metrics,'files':files,
                  'generation_seconds':time.monotonic()-start,'model_pin_sha256':pin,'model_revision':REVISION,
                  'source_sha256':sha(Path(__file__)),'generation_seed':generation_seed,
                  'method':'Frozen SDXL Img2Img conditioned on text and an independent randomized morphology guide',
                  'real_pixels_used':False,'postprocessing':'RGB luminance; exact4x4 area mean to256; global mean/RMS only',
                  'bank_not_admitted_automatically':True}
        save_preview(directory/f'{ident}.png',guide,reduced,texture,ident)
        write_json(directory/(ident+'.json'),metadata);summary.append(metadata)
        write_json(directory/f'status_{args.domain}.json',{'completed':len(summary),'total':len(tasks),'last':ident,'metrics':metrics})
        print(canonical({'id':ident,'metrics':metrics}),flush=True)
    if args.probe:
        first=tasks[0];guide,mask,events=make_guide(first['kind'],{},seed_for(first['dataset_id'],first['id'],'probe',0))
        raw_blank,reduced_blank=generate(pipe,guide,'',seed_for(first['dataset_id'],first['id'],'probe',0,'diffusion'),.12,args.device)
        reference=np.load(directory/summary[0]['files']['generated']['path'])
        text_difference=float(np.sqrt(np.mean((reference.astype(float)-reduced_blank)**2)))
        np.save(directory/'same_guide_empty_prompt_generated.npy',reduced_blank)
        write_json(directory/'text_effect_probe.json',{'same_guide_seed_and_diffusion_seed':True,'only_prompt_changed':True,
            'generated_pixel_RMS_difference':text_difference,'text_changes_output':text_difference>1e-7,
            'interpretation':'Nonzero difference proves text participates; it does not prove useful text semantics or restoration benefit.'})
    write_json(directory/f'summary_{args.domain}.json',summary)


if __name__=='__main__':main()
