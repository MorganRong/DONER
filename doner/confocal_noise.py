import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import fft

SIZE = 128
VERSION = 'confocal_text_operator_v1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:8], 'big')


def pixels(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


class ConfocalT2INoise:
    def __init__(self, path, split='train'):
        self.path = Path(path)
        self.document = json.loads(self.path.read_text())
        self.protocol_path = self.path.parent/'prompt_protocol.json'
        self.protocol = json.loads(self.protocol_path.read_text())
        self.sha256, self.protocol_sha256 = sha(self.path), sha(self.protocol_path)
        self.split = split
        if split not in ('train', 'holdout') or self.document.get('fit_split') != split:
            raise ValueError('Training and held-out generated descriptors must remain separate')
        if self.document.get('domain') != 'hagen' or self.document.get('dataset_id') != 'hagen_confocal_128':
            raise ValueError('Only the new confocal domain library is accepted')
        self.entries = {row['name']: row for row in self.document['operators']}
        self.recipes = {row['name']: row for row in self.protocol['families']}
        if set(self.entries) != set(self.recipes) or len(self.entries) != 3:
            raise ValueError('Expected the three independently described confocal textures')
        self.families = list(self.recipes) + ['mixed_grain']
        self.transfers = {}
        self.source_ids = set()
        for family, row in self.entries.items():
            if row['kind'] != 'grain' or not row['records']:
                raise ValueError('Unsupported or empty confocal noise family')
            for index, record in enumerate(row['records']):
                if record['source_id'] in self.source_ids or not record['source_sha256']:
                    raise ValueError('Missing or duplicate generated source')
                self.source_ids.add(record['source_id'])
                desc = record['parameters']['psd2d_rfft']
                power = np.asarray(desc['power'], dtype=np.float64)
                if (desc['shape_HW'] != [SIZE, SIZE] or power.shape != (SIZE, SIZE//2+1)
                        or not np.isfinite(power).all() or power.min() < 0):
                    raise ValueError('Invalid generated-texture two-dimensional power spectrum')


                reflected_y = (-np.arange(SIZE)) % SIZE
                power[:, [0, -1]] = .5*(power[:, [0, -1]] + power[reflected_y][:, [0, -1]])
                amplitude = np.sqrt(power)
                amplitude[0, 0] = 0.
                weights = np.full(SIZE//2+1, 2.)
                weights[[0, -1]] = 1.
                expectation = float(np.sum(amplitude**2 * weights)/SIZE**2)
                if expectation <= 1e-12:
                    raise ValueError('Generated descriptor has no non-DC energy')
                self.transfers[family, index] = amplitude / np.sqrt(expectation)

    def draw_setting(self, family, rng):
        if family not in self.families:
            raise ValueError('Unknown confocal family')
        if family == 'mixed_grain':
            return {'family': family, 'weights': rng.dirichlet([2., 2., 2.]).tolist(),
                    'components': [self.draw_setting(f, rng) for f in self.recipes]}
        recipe = self.recipes[family]['sampling']
        qlo, qhi = recipe['photon_gain']
        return {'family': family, 'record_index': int(rng.integers(len(self.entries[family]['records']))),
                'photon_gain': float(np.exp(rng.uniform(np.log(qlo), np.log(qhi)))),
                'background_count_level': float(rng.uniform(*recipe['background_count_level'])),
                'read_std': float(rng.uniform(*recipe['read_std']))}

    def _raw(self, clean, setting, rng):
        family = setting['family']
        if family == 'mixed_grain':
            components, provenance = [], []
            for component in setting['components']:
                field, detail = self._raw(clean, component, rng)
                field -= field.mean()
                energy = float(np.sqrt(np.mean(field**2)))
                if energy < 1e-12:
                    raise ValueError('Degenerate component')
                components.append(field/energy)
                provenance.extend(detail)
            field = sum(np.sqrt(w)*v for w, v in zip(setting['weights'], components))
            return field, provenance
        idx = setting['record_index']
        transfer = self.transfers[family, idx]
        source = self.entries[family]['records'][idx]
        q = setting['photon_gain']
        signal = np.maximum(clean, 0.) + setting['background_count_level']
        rate = signal/q


        count_white = (rng.poisson(rate)-rate)/np.sqrt(np.maximum(rate, 1e-12))
        count_colored = fft.irfft2(fft.rfft2(count_white, workers=1)*transfer,
                                  s=(SIZE, SIZE), workers=1)
        read_white = rng.standard_normal((SIZE, SIZE))
        read_colored = fft.irfft2(fft.rfft2(read_white, workers=1)*transfer,
                                 s=(SIZE, SIZE), workers=1)
        field = np.sqrt(q*signal)*count_colored + setting['read_std']*read_colored
        return field, [{'family': family, 'source_id': source['source_id'],
                        'source_sha256': source['source_sha256'], 'record_index': idx}]

    def sample(self, family, clean, seed, a, setting=None):
        clean = np.asarray(clean, dtype=np.float64)
        if clean.shape != (SIZE, SIZE) or not np.isfinite(clean).all() or not 0 < a <= .30:
            raise ValueError('Finite native 128x128 target and RMS in (0,.30] required')
        rng = np.random.default_rng(seed)
        setting = self.draw_setting(family, rng) if setting is None else setting
        if setting['family'] != family:
            raise ValueError('Mismatched family and parameters')
        raw, sources = self._raw(clean, setting, rng)
        raw -= raw.mean()
        rms = float(np.sqrt(np.mean(raw**2)))
        if not np.isfinite(rms) or rms <= 1e-12:
            raise ValueError('Degenerate noise realization')
        field = np.asarray(raw*(a/rms), dtype=np.float32)
        detail = {'version': VERSION, 'family': family, 'setting': setting,
                  't2i_sources': sources, 'library_sha256': self.sha256,
                  'prompt_protocol_sha256': self.protocol_sha256, 'descriptor_split': self.split,
                  'requested_rms': float(a), 'actual_rms': float(np.sqrt(np.mean(field.astype(float)**2))),
                  'unscaled_rms': rms, 'field_sha256': pixels(field), 'realization_seed': int(seed),
                  'signal_dependent': True, 'T2I_spectrum_used': True,
                  'measured_noise_parameter_fit': False, 'source_noise_pixels_loaded': False,
                  'pixel_clipping': False, 'training_HW': [SIZE, SIZE]}
        return field, detail


class ConfocalT2IPairs:
    def __init__(self, noise, seed):
        if noise.split != 'train':
            raise ValueError('A held-out generated source cannot train a model')
        self.noise, self.seed = noise, seed

    def make_pair_batch(self, clean, source_ids, pair_number):
        import torch
        if clean.shape != (2, 3, SIZE, SIZE) or clean.dtype != torch.float32 or len(set(source_ids)) != 2:
            raise ValueError('Two distinct native 128-pixel grayscale-repeat references required')
        rng = np.random.default_rng(seed_for(VERSION, self.seed, 'training-pair', pair_number))
        same = bool(pair_number % 2)
        index = int(rng.choice(3, p=self.noise.protocol['training_rms_bin_weights']))
        a = float(rng.uniform(*self.noise.protocol['training_rms_bins'][index]))
        family = str(rng.choice(self.noise.families, p=self.noise.protocol['family_weights']))
        other = family if same else str(rng.choice([f for f in self.noise.families if f != family]))
        setting0 = self.noise.draw_setting(family, rng)
        setting1 = setting0 if same else self.noise.draw_setting(other, rng)
        slots = (0, 0) if not same and pair_number % 4 == 2 else (0, 1)
        noisy, targets, records, fields = [], [], [], []
        for i, (f, setting, slot) in enumerate(zip((family, other), (setting0, setting1), slots)):
            target = clean[slot:slot+1]
            realization = seed_for(VERSION, self.seed, 'train-realization', pair_number, i)
            field, detail = self.noise.sample(f, target[0, 0].cpu().numpy(), realization, a, setting)
            style = hashlib.sha256(json.dumps(setting, sort_keys=True).encode()).hexdigest()
            records.append({'domain': 'hagen', 'image_domain': 'hagen', 'name': f, 'family': f,
                            'signature': 'confocal_text|'+f, 'style_id': 'confocal_text|'+style,
                            'content_id': source_ids[slot], 'content_slot': slot,
                            'realization_seed': realization, 'field_sha256': pixels(field),
                            'noise_parameters': detail, 'pair_same': same,
                            'legacy_STM_operator_used': False, 'source_noise_pixels_loaded': False,
                            'pixel_clipping': False, 'realization_RMS_normalized': True})
            targets.append(target)
            noisy.append(target + torch.from_numpy(field).to(target.device)[None, None])
            fields.append(field)
        energies = [float(np.sqrt(np.mean(f.astype(float)**2))) for f in fields]
        valid = (min(energies) > 1e-8 and not np.array_equal(fields[0], fields[1])
                 and 0.5 <= energies[0]/energies[1] <= 2.)
        if same:
            valid = valid and records[0]['content_id'] != records[1]['content_id']
        else:
            valid = valid and records[0]['signature'] != records[1]['signature']
        for record in records:
            record['contrast_valid'] = bool(valid)
            record['sample_noise_valid'] = bool(valid)
        return {'clean': torch.cat(targets), 'noisy': torch.cat(noisy), 'metadata': records,
                'pair_same': same, 'pair_valid': bool(valid)}
