from collections import defaultdict
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .operator_sampling import OperatorLibrary, _smooth_field


FAMILIES = ('axis_row', 'axis_column', 'axis_both', 't2i', 'grain', 'smooth')
MORPHOLOGIES = ('iid', 'wide', 'periodic', 'sparse')
DEFAULT_SAMPLING = {
    'ir': {
        'family_weights': dict(zip(FAMILIES, (.25, .25, .10, .30, .05, .05))),
        'strength_weights': {'weak': .25, 'medium': .35, 'strong': .40},
    },
    'stm': {
        'family_weights': dict(zip(FAMILIES, (.20, .05, .05, .60, .05, .05))),
        'strength_weights': {'weak': .40, 'medium': .40, 'strong': .20},
    },
}
for _settings in DEFAULT_SAMPLING.values():
    _settings.update(strength_ranges={'weak': [.002, .035], 'medium': [.035, .12], 'strong': [.12, .25]},
                     analytic_morphology_weights={'iid': .5, 'wide': .25, 'periodic': .125, 'sparse': .125})
SOURCE_DOMAINS = {'ir': 'ir', 'm3fd_ir': 'ir', 'stm': 'stm', 'sim_height300': 'stm'}


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:8], 'big')


def signature(kind, axis=None):
    if kind in ('axis', 'axis_grain'):
        return 'axis:'+axis
    if kind in ('streak', 'streak_grain'):
        return 'streak:'+axis
    if kind in ('grain', 'smooth'):
        return kind
    raise ValueError('Unknown noise signature: '+str(kind))


def compatible_negative(a, b):
    if a == b:
        return False


    return not ('axis:both' in (a, b) and ({a, b} & {'axis:row', 'axis:column'}))


def source_domain(source_id):
    if not isinstance(source_id, str) or not source_id:
        raise ValueError('Nonempty domain-qualified source IDs required')
    prefix = source_id.split('/', 1)[0].split(':', 1)[0]
    if prefix not in SOURCE_DOMAINS:
        raise ValueError('Unknown source domain; prefix content ID with ir: or stm:')
    return SOURCE_DOMAINS[prefix]


def rms(value):
    return float(np.sqrt(np.mean(np.asarray(value, dtype=np.float64)**2)))


def _weighted_cycle(weights, keys):
    if set(weights) != set(keys):
        raise ValueError('Sampling weight keys must exactly match '+str(tuple(keys)))
    values = [float(weights[k]) for k in keys]
    if any(not math.isfinite(x) or x <= 0 for x in values) or abs(sum(values)-1.) > 1e-8:
        raise ValueError('Sampling weights must be positive finite and sum to one')
    fractions = [Fraction(str(x)).limit_denominator(1000) for x in values]
    denominator = math.lcm(*(x.denominator for x in fractions))
    if denominator > 10000 or any(abs(float(f)-v) > 1e-8 for f, v in zip(fractions, values)):
        raise ValueError('Use rational schedule weights with a cycle of at most 10000')
    counts = [int(f*denominator) for f in fractions]

    events = [(j/count, k) for k, count in zip(keys, counts) for j in range(count)]
    return [k for _, k in sorted(events)]


def _axis_profile(name, length, p, rng):
    if name == 'iid':
        return rng.standard_normal(length)
    if name == 'wide':
        sigma = p['gaussian_sigma_px'];radius = int(np.ceil(4*sigma))
        k = np.exp(-.5*(np.arange(-radius, radius+1)/sigma)**2)
        k /= np.sqrt(np.square(k).sum())
        return np.convolve(np.pad(rng.standard_normal(length), radius, mode='reflect'), k, mode='valid')
    if name == 'periodic':
        phase = rng.uniform(0, 2*np.pi)

        return np.sqrt(2)*np.sin(2*np.pi*np.arange(length)/p['period_px']+phase)
    if name == 'sparse':
        profile = np.zeros(length, dtype=np.float64)
        count = int(rng.poisson(p['expected_bands']))
        lo, hi = p['band_width_px']
        for _ in range(count):
            width = int(rng.integers(lo, hi+1));center = int(rng.integers(length))
            left = center-width//2;right = left+width
            profile[max(left, 0):min(right, length)] += float(rng.normal())
        expected_energy = p['expected_bands']*(lo+hi)/2/length
        return profile/np.sqrt(expected_energy)
    raise ValueError('Unknown analytic axis morphology')


def analytic_noise(name, height, width, parameters, rng):
    if name == 'white_grain':
        field = rng.standard_normal((height, width))
    elif name == 'smooth':
        length = parameters['correlation_length_px']
        field = _smooth_field(height, width, (length, length), rng)
    else:
        morphology, axis = name.rsplit('_', 1)
        components = []
        if axis in ('row', 'both'):
            profile = _axis_profile(morphology, height, parameters, rng)
            components.append(np.broadcast_to(profile[:, None], (height, width)))
        if axis in ('column', 'both'):
            profile = _axis_profile(morphology, width, parameters, rng)
            components.append(np.broadcast_to(profile[None, :], (height, width)))
        if not components:
            raise ValueError('Unknown analytic axis')
        field = sum(components)/np.sqrt(len(components))
    field = np.asarray(field, dtype=np.float32)
    field -= np.float32(field.mean(dtype=np.float64))
    return field


class DomainPairSampler:
    def __init__(self, root, domain, seed, settings=None):
        if domain not in ('ir', 'stm'):
            raise ValueError('Separate ir or stm sampler required')
        self.root, self.domain, self.seed = Path(root), domain, int(seed)
        defaults = deepcopy(DEFAULT_SAMPLING[domain])
        config_path = self.root/'config.json'
        if settings is None and config_path.is_file():
            config = json.loads(config_path.read_text())
            if 'noise_sampling' in config:
                if domain not in config['noise_sampling']:
                    raise ValueError('Explicit noise_sampling must contain this domain')
                settings = config['noise_sampling'][domain]
        if settings is not None:
            if not isinstance(settings, dict) or set(settings)-set(defaults):
                raise ValueError('Unknown noise sampling settings')
            defaults.update(deepcopy(settings))
        self.settings = defaults
        self.family_cycle = _weighted_cycle(defaults['family_weights'], FAMILIES)
        self.strength_cycle = _weighted_cycle(defaults['strength_weights'], ('weak', 'medium', 'strong'))
        self.morphology_cycle = _weighted_cycle(defaults['analytic_morphology_weights'], MORPHOLOGIES)
        ranges = defaults['strength_ranges']
        if set(ranges) != {'weak', 'medium', 'strong'}:
            raise ValueError('Three explicit strength ranges required')
        prior = None
        for level in ('weak', 'medium', 'strong'):
            bounds = list(map(float, ranges[level]))
            if len(bounds) != 2 or not all(map(math.isfinite, bounds)) or not 0 < bounds[0] < bounds[1] <= .25:
                raise ValueError('Strength ranges must be finite, ordered and bounded by .25')
            if prior is not None and abs(prior-bounds[0]) > 1e-10:
                raise ValueError('Strength ranges must be contiguous')
            prior = bounds[1]
        self.library = OperatorLibrary(self.root/'operators'/(domain+'.json'), domain, domain+'_hq50')
        self.catalog = []
        for axis in ('row', 'column', 'both'):
            for morphology in MORPHOLOGIES:
                self.catalog.append({'name': morphology+'_'+axis, 'family': 'axis_'+axis,
                                     'origin': 'analytic_augmentation', 'signature': 'axis:'+axis,
                                     'morphology': morphology})
        self.catalog += [{'name': name, 'family': family, 'origin': 'analytic_augmentation', 'signature': sig}
                         for name, family, sig in [('white_grain', 'grain', 'grain'), ('smooth', 'smooth', 'smooth')]]
        self.catalog += [{'name': name, 'family': 't2i', 'origin': 'T2I_fitted_operator',
                          'signature': signature(entry['kind'], entry.get('axis'))}
                         for name, entry in self.library.operators.items()]
        self.groups = defaultdict(list);self.family_indices = defaultdict(list)
        for index, spec in enumerate(self.catalog):
            self.groups[spec['signature']].append(index);self.family_indices[spec['family']].append(index)
        self._family_counts = {f: self.family_cycle.count(f) for f in FAMILIES}
        self._occurrence = [self.family_cycle[:i].count(f) for i, f in enumerate(self.family_cycle)]

    def _describe(self, index, rng, record_ordinal=None):
        spec = dict(self.catalog[index])
        if spec['origin'] == 'T2I_fitted_operator':
            entry = self.library.operators[spec['name']]
            ri = int(rng.integers(len(entry['records']))) if record_ordinal is None else record_ordinal % len(entry['records'])
            record = entry['records'][ri]
            spec.update(record_index=ri, source_record=record['source_id'], source_record_sha256=record['source_sha256'], parameters={})
        else:
            params = {}
            if spec.get('morphology') == 'wide':params['gaussian_sigma_px'] = float(rng.uniform(4., 12.))
            if spec.get('morphology') == 'periodic':params['period_px'] = float(rng.uniform(32., 128.))
            if spec.get('morphology') == 'sparse':params.update(expected_bands=float(rng.uniform(2., 8.)), band_width_px=[1, 12])
            if spec['name'] == 'smooth':params['correlation_length_px'] = float(rng.uniform(6., 18.))
            spec.update(record_index=None, source_record=None, source_record_sha256=None, parameters=params)
        spec['style_id'] = self.domain+'|'+spec['origin']+'|'+spec['name']+'|'+str(spec['record_index'])
        return spec

    def plan(self, step):
        if not isinstance(step, int) or isinstance(step, bool) or step < 1:
            raise ValueError('Positive integer step required')
        rng = np.random.default_rng(stable_seed(self.seed, self.domain, 'pair2_formal_plan', step))
        anchor_number = (step-1)//2
        cycle, pos = divmod(anchor_number, len(self.family_cycle));family = self.family_cycle[pos]
        ordinal = cycle*self._family_counts[family]+self._occurrence[pos]
        indices = self.family_indices[family]
        if family.startswith('axis_'):
            morphology = self.morphology_cycle[ordinal % len(self.morphology_cycle)]
            index = next(i for i in indices if self.catalog[i]['morphology'] == morphology)
        else:index = indices[ordinal % len(indices)]
        first = self._describe(index, rng, ordinal//len(indices))
        first.update(anchor_family=family, anchor_number=anchor_number)
        if step % 2:return [first, deepcopy(first)]
        candidates = [i for i, s in enumerate(self.catalog) if compatible_negative(first['signature'], s['signature'])]
        weights = []
        for i in candidates:
            s = self.catalog[i];weight = self.settings['family_weights'][s['family']]
            if s['family'].startswith('axis_'):weight *= self.settings['analytic_morphology_weights'][s['morphology']]
            else:weight /= len(self.family_indices[s['family']])
            weights.append(weight)
        weights = np.asarray(weights);weights /= weights.sum()
        second = self._describe(int(rng.choice(candidates, p=weights)), rng)
        second.update(anchor_family=family, anchor_number=anchor_number)
        return [first, second]

    def _strength(self, step, sigma_override):
        rng = np.random.default_rng(stable_seed(self.seed, self.domain, 'pair2_formal_sigma', step))


        epoch, offset = divmod(step-1, len(self.strength_cycle))
        schedule = np.asarray(self.strength_cycle, dtype=object)
        permutation = np.random.default_rng(stable_seed(self.seed, self.domain, 'strength_cycle', epoch)).permutation(len(schedule))
        level = str(schedule[permutation[offset]])
        low, high = self.settings['strength_ranges'][level]
        sigma = float(np.exp(rng.uniform(np.log(low), np.log(high))))
        if sigma_override is not None:
            sigma = float(sigma_override)
            if not np.isfinite(sigma) or not 0 < sigma <= .25:
                raise ValueError('Sigma override must be in (0,.25]')
            level = 'override'
        return sigma, level

    def make_pair_batch(self, clean, source_ids, step, *, sigma_override=None):
        if (not isinstance(clean, torch.Tensor) or clean.ndim != 4 or tuple(clean.shape[:2]) != (2, 3)
                or min(clean.shape[-2:]) < 16 or clean.dtype != torch.float32
                or not bool(torch.isfinite(clean).all())):
            raise ValueError('Expected finite float32 [2,3,H,W] full-image pair with H,W >= 16')
        height, width = map(int, clean.shape[-2:])
        if not (torch.equal(clean[:, 0], clean[:, 1]) and torch.equal(clean[:, 0], clean[:, 2])):
            raise ValueError('Expected exact grayscale repeats')
        if len(source_ids) != 2 or source_ids[0] == source_ids[1]:raise ValueError('Two distinct source IDs required')
        if any(source_domain(s) != self.domain for s in source_ids):raise ValueError('Cross-domain pairs prohibited')
        if torch.equal(clean[0], clean[1]):raise ValueError('The two source contents are pixel-identical')
        styles = self.plan(step);pair_same = bool(step % 2)
        negative_same_content = not pair_same and step % 4 == 2
        content_slots = (0, 0) if negative_same_content else (0, 1)
        base_sigma, stratum = self._strength(step, sigma_override)
        rms_limit, peak_limit = max(.12, base_sigma*1.5), max(.5, base_sigma*6)
        noisy, targets, metadata = [], [], []
        for sample_slot, (spec, content_slot) in enumerate(zip(styles, content_slots)):
            realization_seed = stable_seed(self.seed, self.domain, 'pair2_formal_realization', step, sample_slot)
            rng = np.random.default_rng(realization_seed);jitter = float(rng.uniform(.9, 1.1))
            if spec['origin'] == 'T2I_fitted_operator':
                field, detail = self.library.sample_record(spec['name'], spec['record_index'], height, width, rng)
            else:
                field = analytic_noise(spec['name'], height, width, spec['parameters'], rng)
                detail = {'final_RMS_normalized': False, 'parameters': spec['parameters']}
            unit_rms = rms(field);field = np.asarray(field*base_sigma*jitter, dtype=np.float32)
            before_rms, before_peak = rms(field), float(np.max(np.abs(field)))
            scale = min(1., rms_limit/max(before_rms, 1e-15), peak_limit/max(before_peak, 1e-15))
            field = np.asarray(field*scale, dtype=np.float32)
            tensor = torch.from_numpy(field.copy())[None, None].to(clean.device)
            target = clean[content_slot:content_slot+1];item = target+tensor
            actual = (item-target).mean(1)[0].detach().cpu().numpy().astype(np.float64)
            actual_rms = rms(actual);dc = actual.mean();centered = actual-dc
            row = centered.mean(1, keepdims=True);column = centered.mean(0, keepdims=True)
            valid = actual_rms >= 1e-5;noisy.append(item);targets.append(target)
            metadata.append({**spec, 'domain': self.domain, 'content_id': source_ids[content_slot],
                'content_slot': content_slot, 'sample_slot': sample_slot, 'pair_slot': 0,
                'pair_same': pair_same, 'negative_same_content': negative_same_content,
                'realization_seed': realization_seed, 'batch_sigma': base_sigma, 'requested_stratum': stratum,
                'amplitude_jitter': jitter, 'requested_sigma': base_sigma*jitter,
                'raw_operator_rms': unit_rms, 'downward_scale': scale,
                'budget_rms_max': rms_limit, 'budget_peak_max': peak_limit,
                'actual_rms': actual_rms, 'actual_mean': float(dc), 'actual_peak': float(np.max(np.abs(actual))),
                'actual_row_rms': rms(row), 'actual_column_rms': rms(column), 'actual_local_rms': rms(centered-row-column),
                'field_sha256': hashlib.sha256(field.tobytes()).hexdigest(),
                'sample_noise_valid': valid, 'contrast_valid': valid, 'contrast_exclusion_reason': None,
                'field_shape_HW': [height, width],
                'source_noise_pixels_loaded': False, 'pixel_clipping': False,
                'realization_RMS_normalized': False, 'operator_detail': detail})
        a, b = metadata;valid = a['sample_noise_valid'] and b['sample_noise_valid']
        reason = None if valid else 'near_zero_realized_noise'
        if a['field_sha256'] == b['field_sha256']:valid, reason = False, 'identical_realization_degeneracy'
        ratio = a['actual_rms']/b['actual_rms'] if b['actual_rms'] > 0 else None
        if not pair_same and valid and not .5 <= ratio <= 2.:
            valid, reason = False, 'negative_realized_RMS_ratio_outside_0.5_2.0'
        for record in metadata:
            record.update(contrast_valid=bool(valid), contrast_exclusion_reason=reason, pair_rms_ratio=ratio)
        return {'noisy': torch.cat(noisy), 'clean': torch.cat(targets), 'metadata': metadata,
                'pair_same': pair_same, 'pair_valid': bool(valid)}
