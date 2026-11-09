import hashlib, math
import numpy as np
import torch

FAMILIES=('isolated_points','sparse_point_cloud','thin_lines','multiple_thin_lines',
 'narrow_gaussians','gaussian_ridges','step_edges','multidirectional_periodic',
 'irregular_windows','small_disks')
AMPLITUDE_RANGE=(.01,.12)

def _check_template(template):
 if not isinstance(template,torch.Tensor) or template.ndim!=4 or template.shape[1]!=1:
  raise ValueError('Expected a B1HW tensor')
 if template.dtype!=torch.float32 or not 1<=template.shape[0]<=len(FAMILIES) or min(template.shape[-2:])<32:
  raise ValueError('Expected float32 native geometry >=32 and batch1..10')

def _rng(seed,step,slot,stream):
 if not isinstance(seed,int) or isinstance(seed,bool) or not isinstance(step,int) or isinstance(step,bool) or step<1:
  raise ValueError('Integer seed and positive integer step required')
 raw=f'stm_detail_train_v2|{seed}|{step}|{slot}|{stream}'.encode()
 return np.random.default_rng(int.from_bytes(hashlib.sha256(raw).digest()[:8],'little'))

def _angle(rng):
 return float(rng.choice([0.,math.pi/4,math.pi/2,3*math.pi/4])) if rng.random()<.2 else float(rng.uniform(0,math.pi))

def _field(h,w,rng,family):
 yy,xx=np.mgrid[:h,:w].astype(np.float32)
 field=np.zeros((h,w),np.float32);events=[]
 if family in ('isolated_points','sparse_point_cloud'):
  count=1 if family=='isolated_points' else int(rng.integers(5,61))
  positions=rng.choice(h*w,size=count,replace=False)
  for position in positions:
   cy,cx=divmod(int(position),w);polarity=int(rng.choice([-1,1]));value=float(rng.uniform(.5,1.))
   field[cy,cx]=polarity*value;events.append({'center_yx':[cy,cx],'width_px':1,'polarity':polarity,'weight':value})
 elif family in ('thin_lines','multiple_thin_lines','gaussian_ridges'):
  count=1 if family=='thin_lines' else int(rng.integers(2,7)) if family=='multiple_thin_lines' else int(rng.integers(1,5))
  for _ in range(count):
   cy,cx=float(rng.uniform(0,h)),float(rng.uniform(0,w));angle=_angle(rng)
   length=float(rng.uniform(5,min(h,w)*.85));polarity=int(rng.choice([-1,1]));weight=float(rng.uniform(.5,1.))
   dy,dx=yy-cy,xx-cx;along=dx*math.cos(angle)+dy*math.sin(angle);across=-dx*math.sin(angle)+dy*math.cos(angle)
   if family=='gaussian_ridges':
    width=float(rng.uniform(.4,1.6));part=np.exp(-.5*(across/width)**2)*np.exp(-.5*(along/(length/2))**8)
   else:
    width=float(rng.choice([1.,1.,1.,1.5,2.]));part=((abs(across)<=width/2)&(abs(along)<=length/2)).astype(np.float32)
   field+=np.float32(polarity*weight)*part.astype(np.float32)
   events.append({'center_yx':[cy,cx],'angle_rad':angle,'length_px':length,'width_px':width,'polarity':polarity,'weight':weight})
 elif family=='narrow_gaussians':
  for _ in range(int(rng.integers(1,13))):
   cy,cx=float(rng.uniform(0,h)),float(rng.uniform(0,w));angle=_angle(rng)
   sy,sx=float(rng.uniform(.4,1.5)),float(rng.uniform(.4,1.5));polarity=int(rng.choice([-1,1]));weight=float(rng.uniform(.5,1.))
   dy,dx=yy-cy,xx-cx;u=dx*math.cos(angle)+dy*math.sin(angle);v=-dx*math.sin(angle)+dy*math.cos(angle)
   field+=np.float32(polarity*weight)*np.exp(-.5*((u/sx)**2+(v/sy)**2)).astype(np.float32)
   events.append({'center_yx':[cy,cx],'angle_rad':angle,'sigma_yx':[sy,sx],'polarity':polarity,'weight':weight})
 elif family=='step_edges':
  angle=_angle(rng);cy,cx=float(rng.uniform(.15*h,.85*h)),float(rng.uniform(.15*w,.85*w))
  signed=(xx-cx)*math.cos(angle)+(yy-cy)*math.sin(angle)
  width=float(rng.choice([0.,.4,1.,2.,4.]));field=(signed>=0).astype(np.float32) if width==0 else (.5+.5*np.tanh(signed/width)).astype(np.float32)
  events=[{'center_yx':[cy,cx],'angle_rad':angle,'transition_width_px':width}]
 elif family=='multidirectional_periodic':
  for _ in range(int(rng.integers(1,4))):
   angle=_angle(rng);period=float(np.exp(rng.uniform(np.log(2.8),np.log(min(h,w)/4))))
   phase=float(rng.uniform(-math.pi,math.pi));weight=float(rng.uniform(.4,1.));polarity=int(rng.choice([-1,1]))
   field+=np.asarray(polarity*weight*np.sin(2*math.pi*(xx*math.cos(angle)+yy*math.sin(angle))/period+phase),np.float32)
   events.append({'angle_rad':angle,'period_px':period,'phase_rad':phase,'polarity':polarity,'weight':weight})
 elif family=='irregular_windows':
  for _ in range(int(rng.integers(3,20))):
   cy,cx=float(rng.uniform(0,h)),float(rng.uniform(0,w));angle=_angle(rng)
   height,width=float(rng.uniform(1,16)),float(rng.uniform(1,16));polarity=int(rng.choice([-1,1]));weight=float(rng.uniform(.4,1.))
   dy,dx=yy-cy,xx-cx;u=dx*math.cos(angle)+dy*math.sin(angle);v=-dx*math.sin(angle)+dy*math.cos(angle)
   field+=np.float32(polarity*weight)*((abs(u)<=width/2)&(abs(v)<=height/2)).astype(np.float32)
   events.append({'center_yx':[cy,cx],'angle_rad':angle,'size_hw':[height,width],'polarity':polarity,'weight':weight})
 elif family=='small_disks':
  for _ in range(int(rng.integers(1,13))):
   cy,cx=float(rng.uniform(0,h)),float(rng.uniform(0,w));radius=float(rng.uniform(.5,3.));polarity=int(rng.choice([-1,1]));weight=float(rng.uniform(.4,1.))
   field+=np.float32(polarity*weight)*(((yy-cy)**2+(xx-cx)**2)<=radius**2).astype(np.float32)
   events.append({'center_yx':[cy,cx],'radius_px':radius,'polarity':polarity,'weight':weight})
 else:raise ValueError(family)
 peak=float(abs(field).max())


 if peak>0:field=field/np.float32(peak)
 return np.asarray(field,np.float32),{'family':family,'events':events,'template_peak_before_unit_scaling':peak}

@torch.no_grad()
def _signals(clean,step,seed,stream):
 _check_template(clean)
 if not torch.isfinite(clean).all() or bool((clean<0).any()) or bool((clean>1).any()):
  raise ValueError('Clean supervision pixels must be finite and in [0,1]')
 h,w=map(int,clean.shape[-2:]);fields=[];records=[]
 offset=seed%len(FAMILIES)
 for slot in range(clean.shape[0]):
  family=FAMILIES[(step-1+offset+3*slot)%len(FAMILIES)]
  rng=_rng(seed,step,slot,stream);field,record=_field(h,w,rng,family)
  amplitude=float(rng.uniform(*AMPLITUDE_RANGE));sign=int(rng.choice([-1,1]))
  fields.append(np.asarray(field*(amplitude*sign),np.float32))
  records.append({**record,'sample_index':slot,'step':step,'seed':seed,'rng_stream':stream,
   'requested_peak':amplitude,'global_polarity':sign,'shape_HW':[h,w],
   'source':'fresh_analytic_training_signal','noise_added_to_target':False,'reads_validation_arrays_or_seeds':False,
   'original_clean_pixels_normalized_or_filtered':False,'crop_resize_tile':False})
 assert len({r['family'] for r in records})==len(records)
 proposed=torch.from_numpy(np.stack(fields)[:,None]).to(device=clean.device,dtype=clean.dtype)
 delta=torch.minimum(torch.maximum(proposed,-clean),1-clean)
 for slot,record in enumerate(records):
  record.update(actual_peak=float(delta[slot].abs().max()),actual_rms=float(delta[slot].square().mean().sqrt()),
   nonzero_pixels=int(torch.count_nonzero(delta[slot])),headroom_limited_fraction=float((delta[slot]!=proposed[slot]).float().mean()))
 return delta,records

@torch.no_grad()
def detail_signals(clean,step,seed):

 return _signals(clean,step,seed,211)

@torch.no_grad()
def make_detail_clean(template,step,seed):

 _check_template(template)
 background=torch.empty_like(template)
 values=[]
 for slot in range(template.shape[0]):
  value=float(_rng(seed,step,slot,312).uniform(.2,.8));background[slot].fill_(value);values.append(value)
 delta,records=_signals(background,step,seed,313)
 for record,value in zip(records,values):record.update(background=value,source='fresh_analytic_clean_training_image',uses_template_pixel_values=False)
 return background+delta,records

def self_check():

 torch.set_num_threads(4);torch.manual_seed(17)
 clean=torch.rand(2,1,300,300);original=clean.clone();seen=set();signed=set();counts={}
 for step in range(1,21):
  a,ra=detail_signals(clean,step,2026103211);b,rb=detail_signals(clean,step,2026103211)
  assert torch.equal(a,b) and ra==rb and torch.equal(clean,original)
  assert a.shape==clean.shape and a.dtype==clean.dtype and torch.isfinite(a).all()
  assert 0<=float((clean+a).min())<=float((clean+a).max())<=1 and float(a.abs().max())<=.120001
  assert ra[0]['family']!=ra[1]['family']
  for row in ra:seen.add(row['family']);signed.add(row['global_polarity']);counts[row['family']]=counts.get(row['family'],0)+1
  x,rx=make_detail_clean(clean,step,2026103211);y,ry=make_detail_clean(torch.zeros_like(clean),step,2026103211)
  assert torch.equal(x,y) and rx==ry and torch.equal(clean,original)
  assert torch.isfinite(x).all() and 0<=float(x.min())<=float(x.max())<=1
 assert seen==set(FAMILIES) and signed=={-1,1}
 return {'passed':True,'CPU_only':True,'native_BCHW':[2,1,300,300],'replay_steps':20,'families':counts,
  'source_clean_unchanged':True,'standalone_clean_independent_of_template_pixels':True,
  'no_file_reads_or_validation_dependencies':True,'additional_measured_training_images':0}

if __name__=='__main__':
 import json
 print(json.dumps(self_check()),flush=True)
