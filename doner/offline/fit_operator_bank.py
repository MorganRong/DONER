from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

PSD_BINS = 64
QUANTILES = [.005,.01,.025,.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.975,.99,.995]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def unit(array):
    a = np.asarray(array, dtype=np.float64)
    if a.ndim != 2 or min(a.shape) < 32 or not np.isfinite(a).all():
        raise ValueError('Expected finite two-dimensional noise field, each axis >=32')
    a = a-a.mean()
    scale = float(np.sqrt(np.mean(a*a)))
    if not np.isfinite(scale) or scale < 1e-9:
        raise ValueError('Degenerate noise field')
    return a/scale


def _bin_average(frequency, power, maximum, bins=PSD_BINS):
    edges = np.linspace(0., maximum+1e-12, bins+1)
    ids = np.minimum(np.searchsorted(edges, frequency.ravel(), side='right')-1, bins-1)
    ids = np.maximum(ids, 0)
    counts = np.bincount(ids, minlength=bins)
    sums = np.bincount(ids, weights=power.ravel(), minlength=bins)
    values = sums/np.maximum(counts, 1)
    present = counts > 0
    if not present.all():
        values = np.interp(np.arange(bins), np.flatnonzero(present), values[present])
    if values.mean() <= 1e-20:
        values = np.ones(bins)
    values = values/values.mean()
    return {'frequency_cpp':((edges[1:]+edges[:-1])/2).tolist(), 'power':values.tolist()}


def radial_psd(a):


    a = np.asarray(a, dtype=np.float64)
    window = np.outer(np.hanning(a.shape[0]),np.hanning(a.shape[1]))
    center = float(np.sum(a*window)/np.sum(window))
    tapered = (a-center)*window
    fy = np.fft.fftfreq(a.shape[0])[:,None]
    fx = np.fft.fftfreq(a.shape[1])[None,:]
    radius = np.sqrt(fx*fx+fy*fy)
    power = abs(np.fft.fft2(tapered, norm='ortho'))**2/np.mean(window*window)
    power[0,0] = 0.
    return _bin_average(radius, power, np.sqrt(.5))


def profile_psd(profile):
    profile = np.asarray(profile, dtype=np.float64)
    window = np.hanning(len(profile))
    center = float(np.sum(profile*window)/np.sum(window))
    power = abs(np.fft.rfft((profile-center)*window,norm='ortho'))**2/np.mean(window*window)
    power[0] = 0.
    return _bin_average(np.fft.rfftfreq(len(profile)), power, .5)


def correlation_length(a, axis, maximum=64):

    a = np.asarray(a, dtype=np.float64); a = a-a.mean()
    variance = float(np.mean(a*a))
    if variance < 1e-14: return 1.
    limit = min(maximum, a.shape[axis]//3)
    for lag in range(1, limit+1):
        left = [slice(None)]*2; right = left.copy()
        left[axis] = slice(None,-lag); right[axis] = slice(lag,None)
        covariance = float(np.mean(a[tuple(left)]*a[tuple(right)]))/variance
        if covariance <= np.exp(-1.): return float(lag)
    return float(limit)


def _common(a):
    local_rms = np.sqrt(np.maximum(ndimage.gaussian_filter(a*a, 8., mode='reflect'), 0.))
    envelope_cv = float(local_rms.std()/max(float(local_rms.mean()),1e-12))
    return {
        'radial_psd':radial_psd(a),
        'pixel_quantiles':{'probabilities':QUANTILES, 'values':np.quantile(a,QUANTILES).tolist()},
        'envelope_cv':envelope_cv,
        'envelope_correlation_px':[correlation_length(local_rms,0),correlation_length(local_rms,1)],
        'lag1_correlation':[float(np.mean(a[:-1]*a[1:])),float(np.mean(a[:,:-1]*a[:,1:]))],
    }


def _finite_events(a, axis):


    work = a if axis == 'row' else a.T
    along = ndimage.uniform_filter1d(work, 3, axis=1, mode='reflect')
    baseline = ndimage.median_filter(along, size=(9,1), mode='reflect')
    contrast = along-baseline
    robust = float(np.median(abs(contrast-np.median(contrast)))/.67448975)
    threshold = max(4.*robust, .12*float(work.std()), 1e-7)
    event_mask = np.zeros_like(work,dtype=bool)
    local_baseline = ndimage.median_filter(work, size=(9,1), mode='reflect')
    events, modulation = [], []
    censored = 0
    for polarity in (-1,1):
        selected = polarity*contrast > threshold
        selected = ndimage.binary_closing(selected, structure=np.ones((1,3),bool))
        labels, count = ndimage.label(selected, structure=np.ones((3,3),bool))
        for ident, slices in enumerate(ndimage.find_objects(labels),1):
            if slices is None: continue
            sy,sx = slices
            component = labels[sy,sx]==ident
            if component.sum()<6: continue
            coarse_contrast = polarity*contrast[sy,sx]


            half_contrast = max(threshold,.5*float(np.quantile(coarse_contrast[component],.95)))
            core = component & (coarse_contrast>=half_contrast)
            core_labels, _ = ndimage.label(core,structure=np.ones((3,3),bool))
            recognized = False
            for core_ident,core_slice in enumerate(ndimage.find_objects(core_labels),1):
                if core_slice is None: continue
                cy,cx=core_slice
                length=cx.stop-cx.start; width=cy.stop-cy.start
                mask=core_labels[cy,cx]==core_ident
                if length<6 or width>10 or length/max(width,1)<4 or mask.mean()<.45: continue
                yy=slice(sy.start+cy.start,sy.start+cy.stop)
                xx=slice(sx.start+cx.start,sx.start+cx.stop)
                signed=polarity*(work[yy,xx]-local_baseline[yy,xx])
                amplitude=float(np.median(signed[mask]))
                if amplitude<=0: continue
                recognized=True
                if (xx.start<=1 or xx.stop>=work.shape[1]-1
                        or yy.start==0 or yy.stop==work.shape[0]):
                    censored+=1
                    continue
                profile=np.mean(signed,axis=0)
                modulation.append(float(profile.std()/max(profile.mean(),1e-12)))
                events.append({'length_px':int(length),'width_px':int(width),
                               'amplitude':amplitude,'polarity':int(polarity)})
            if recognized:event_mask[sy,sx] |= component
    if not events:
        raise ValueError('No measurable finite streaks in the generated field')


    halo = ndimage.binary_dilation(event_mask, structure=np.ones((3,3),bool))
    background = work.copy()
    fill = ndimage.median_filter(work, size=(11,1), mode='reflect')
    background[halo] = fill[halo]
    clean_background = background if axis=='row' else background.T
    return events, clean_background, float(np.median(modulation)), float(event_mask.mean()), censored


def describe_field(array, kind, axis=None):

    if kind not in ('grain','smooth','axis','axis_grain','streak','streak_grain'):
        raise ValueError('Unsupported operator kind: '+str(kind))
    if kind in ('axis','axis_grain','streak','streak_grain') and axis not in ('row','column'):
        raise ValueError('A line-like operator requires row/column axis')
    a = unit(array)
    parameters = _common(a)
    if kind in ('axis','axis_grain'):
        work = a if axis=='row' else a.T
        profile = work.mean(1)
        profile_variance = float(profile.var())
        if profile_variance < .005:
            raise ValueError('Axis-labelled field lacks measurable shared scan component')
        smoothed = ndimage.gaussian_filter1d(work, 3., axis=1, mode='reflect')
        modulation = smoothed-profile[:,None]
        grain = work-smoothed
        modulation_cv = float(np.sqrt(np.mean(modulation*modulation)/profile_variance))
        background_fraction = float(np.clip(np.mean(grain*grain),0.,.99))
        background = grain if axis=='row' else grain.T
        parameters.update(
            axis_psd=profile_psd(profile),
            axis_variance_fraction=float(profile_variance),
            along_modulation_cv=modulation_cv,
            along_correlation_px=correlation_length(modulation,1),
            background_fraction=background_fraction,
            background_radial_psd=radial_psd(background),
        )
    elif kind in ('streak','streak_grain'):
        events, background, modulation, support, censored = _finite_events(a,axis)
        parameters.update(
            events=events,
            event_density_per_pixel=float((len(events)+censored)/a.size),
            event_expected_variance=float((len(events)+censored)/a.size*np.mean([e['length_px']*e['width_px']*e['amplitude']**2 for e in events])),
            boundary_censored_event_count=censored,
            complete_event_count=len(events),
            boundary_policy='Count visible censored components in density; exclude their shapes from event distribution',
            background_fraction=float(np.clip(np.var(background),0.,.99)),
            background_radial_psd=radial_psd(background),
            within_event_modulation_cv=modulation,
            measured_event_support_fraction=support,
        )
    return parameters


def family_schema(profile):

    kind=profile['kind']; options=profile.get('options',{})
    if kind=='grain': return 'grain',None
    if kind=='smooth': return 'smooth',None
    if kind in ('column_bands','row_bands','scan_bands'):
        return 'axis','column' if kind=='column_bands' else 'row'
    if kind in ('short_streaks','grain_streaks'):
        return ('streak_grain' if kind=='grain_streaks' else 'streak'),options.get('axis','row')
    raise ValueError('Unregistered morphology label: '+kind)


def fit_domain(source_root, domain, profiles):
    source_root=Path(source_root).resolve()
    path=source_root/'bank/t2i'/domain/'manifest.json'
    manifest=json.loads(path.read_text())
    if manifest['domain']!=domain or manifest['schema_version']<5:
        raise ValueError('Wrong source bank domain/schema')
    train=[r for r in manifest['records'] if r['split']=='train']
    heldout=[r for r in manifest['records'] if r['split']!='train']
    if not train or not heldout or {r['sha256'] for r in train}&{r['sha256'] for r in heldout}:
        raise ValueError('Missing or overlapping generated splits')
    by_name={p['id']:p for p in profiles['datasets'][domain]['families']}
    families=sorted({r['pattern'] for r in train})
    if set(families)!=set(by_name):
        raise ValueError('Every source family requires an explicit morphology label')
    operators=[]
    for family in families:
        kind,axis=family_schema(by_name[family]);records=[]
        for row in sorted((r for r in train if r['pattern']==family),key=lambda r:r['id']):
            file=(path.parent/row['path']).resolve()
            if not file.is_relative_to(path.parent) or sha(file)!=row['sha256']:
                raise ValueError('Invalid generated training artifact')
            array=np.load(file,allow_pickle=False)
            parameters=describe_field(array,kind,axis)
            records.append({'source_id':row['id'],'source_sha256':row['sha256'],
                            'source_shape_HW':list(array.shape),'parameters':parameters})
        operators.append({'name':family,'family':family,'kind':kind,'axis':axis,'records':records})
    return {
        'schema_version':1,'domain':domain,'dataset_id':manifest['dataset_id'],
        'source_manifest_sha256':sha(path),'source_manifest_path':str(path),
        'fit_split':'train','fit_script_sha256':sha(__file__),
        'operator_count':len(operators),'fit_records':len(train),
        'heldout_records_used_for_fit':0,'stores_pixels_or_positions':False,
        'operators':operators,
        'estimator':{'spectrum_bins':PSD_BINS,'spectrum_units':'cycles/pixel',
                     'spectrum_power':'mean power per Fourier coefficient; curve mean normalized to 1',
                     'spectrum_boundary_treatment':'separable Hann taper with window-weighted mean subtraction and mean-square-window power correction',
                     'normalization':'source zero mean/unit RMS before feature measurement',
                     'numerical_guide_parameters_used':False,
                     'event_method':'along3 smooth; transverse9 median;4*MAD candidate threshold; connected half-local95%-contrast elongated cores; full candidate halo excluded from background; boundary-censored shapes excluded',
                     'envelope_method':'RMS of Gaussian-smoothed squared field, sigma8 pixels',
                     'axes':'row=horizontal events, column=vertical events'},
        'limitations':[
            'Fitted generated-noise statistics are an engineering prior, not measured sensor calibration.',
            'Spectra, marginal tails and event statistics do not preserve exact generated phase/layout.',
            'Sparse event segmentation is threshold dependent; per-source statistics remain separate.',
            'Estimated background includes some filtering/segmentation error and line modulation.',
            'No test/validation array contributes to fitted parameters.',
        ],
    }


def main(source_root,output_root):
    source_root=Path(source_root).resolve();output_root=Path(output_root).resolve()
    profiles=json.loads((source_root/'guided_profiles.json').read_text())
    reports=[]
    for domain in ('ir','stm'):
        library=fit_domain(source_root,domain,profiles)
        path=output_root/'operators'/f'{domain}.json';write(path,library)
        reports.append({'domain':domain,'path':str(path),'sha256':sha(path),
                        'operator_count':library['operator_count'],'fit_records':library['fit_records'],
                        'families':[{'name':op['name'],'kind':op['kind'],'records':len(op['records'])} for op in library['operators']]})
    report={'passed':True,'created_utc':datetime.now(timezone.utc).isoformat(),
            'source_root':str(source_root),'script_sha256':sha(__file__),
            'source_profiles_sha256':sha(source_root/'guided_profiles.json'),
            'heldout_arrays_read':0,'operators':reports}
    write(output_root/'operators/fitreport.json',report)
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    args=parser.parse_args();main(args.source_root,args.output_root)
