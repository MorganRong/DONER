from .pair_sampling import compatible_negative
from .ir_pairs import DomainPairSampler
from . import ir_detail
from .images import load_training_images
import hashlib

import json

from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
ANCHOR_CYCLE = ['particle'] * 8 + ['streak_mixed'] * 6 + ['pure_axes'] * 5 + ['clean']
TYPES = {
    'particle': ['fine_grain', 'correlated_grain', 'heterogeneous_grain'],
    'streak_mixed': ['grain_columns', 'intermittent_columns', 'smooth_offset'],
    'pure_axes': ['variable_columns', 'variable_rows', 'iid_row',
                  'iid_column', 'iid_both', 'wide_row', 'wide_column',
                  'periodic_row', 'periodic_column', 'sparse_row', 'sparse_column'],
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ah(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def rms(array):
    return float(np.sqrt(np.mean(np.asarray(array, dtype=np.float64) ** 2)))


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:8], 'big')


class TrainingData:
    def __init__(self, seed=20261101, patch_size=None, batch_size=4, base_root=ROOT,
                 train_manifest=None):
        if batch_size != 4 or patch_size is not None:
            raise ValueError('Native V6 requires exactly two pairs, full768x1024 frames only')
        self.seed, self.patch_size, self.batch_size = int(seed), None, 4
        self.base = Path(base_root)

        self.rows, self.images = load_training_images(train_manifest, 'ir')
        self.source_hashes = [ah(a) for a in self.images]
        self.sampler = DomainPairSampler(self.base, 'ir', self.seed, settings={
            'family_weights': {'axis_row': .20, 'axis_column': .05, 'axis_both': .05,
                               't2i': .60, 'grain': .05, 'smooth': .05},
            'strength_weights': {'weak': .30, 'medium': .35, 'strong': .35},
            'strength_ranges': {'weak': [.002, .035], 'medium': [.035, .12], 'strong': [.12, .25]},
            'analytic_morphology_weights': {'iid': .50, 'wide': .25, 'periodic': .125, 'sparse': .125},
        })
        self.sampler.plan = self._plan
        self.detail = ir_detail

    def _anchor(self, step):
        block, offset = divmod(step - 1, len(ANCHOR_CYCLE))
        rng = np.random.default_rng(stable_seed(self.seed, 'v6_anchor_schedule', block))
        schedule = [ANCHOR_CYCLE[int(i)] for i in rng.permutation(len(ANCHOR_CYCLE))]
        group = schedule[offset]
        ordinal = block * ANCHOR_CYCLE.count(group) + schedule[:offset].count(group)
        name = 'clean' if group == 'clean' else TYPES[group][ordinal % len(TYPES[group])]
        return group, name, ordinal

    def _plan(self, pair_number):
        step = (pair_number + 1) // 2
        group, name, ordinal = self._anchor(step)
        if group == 'clean':
            raise ValueError('Clean batches bypass noisy-pair generation')
        rng = np.random.default_rng(stable_seed(self.seed, 'v6_exact_type_plan', pair_number))
        index = next(i for i, value in enumerate(self.sampler.catalog) if value['name'] == name)
        first = self.sampler._describe(index, rng, ordinal // len(TYPES[group]))
        first.update(anchor_family=group, anchor_number=step - 1)
        if pair_number % 2:
            return [first, deepcopy(first)]
        candidates, weights = [], []
        for other_group, names in TYPES.items():
            mass = {'particle': .40, 'streak_mixed': .30, 'pure_axes': .25}[other_group]
            for other_name in names:
                i = next(i for i, value in enumerate(self.sampler.catalog) if value['name'] == other_name)
                if compatible_negative(first['signature'], self.sampler.catalog[i]['signature']):
                    candidates.append(i)
                    weights.append(mass / len(names))
        weights = np.asarray(weights, np.float64)
        weights /= weights.sum()
        second = self.sampler._describe(int(rng.choice(candidates, p=weights)), rng)
        second.update(anchor_family=group, anchor_number=step - 1)
        return [first, second]

    def _strength(self, step):
        block,offset=divmod(step-1,3)
        rng=np.random.default_rng(stable_seed(self.seed,'IR_three_training_levels',block))
        return [.05,.10,.20][int(rng.permutation(3)[offset])]

    def _sources(self, step, pair_slot, stream='train'):
        rng = np.random.default_rng(stable_seed(self.seed, 'v6_sources', stream, step, pair_slot))
        indices = [int(i) for i in rng.choice(50, size=2, replace=False)]
        windows = [[0, 0, 1024, 768] for _ in indices]
        full = torch.from_numpy(np.stack([self.images[i] for i in indices])[:, None]).repeat(1, 3, 1, 1)
        return indices, windows, full

    def sample_clean(self, step, device='cpu'):
        result = []
        for pair_slot in range(2):
            indices, windows, _ = self._sources(step, pair_slot, 'identity')
            for index, (x, y, w, h) in zip(indices, windows):
                assert [x, y, w, h] == [0, 0, 1024, 768]
                result.append(self.images[index].copy())
        return torch.from_numpy(np.stack(result)[:, None]).to(device)

    def get_batch(self, step, device='cpu', pair_rank=None):
        if not isinstance(step, int) or isinstance(step, bool) or step < 1:
            raise ValueError('Positive integer step required')
        group, anchor_name, ordinal = self._anchor(step)
        clean_batch, noisy_batch, metadata, pairs, selection = [], [], [], [], []
        standalone = step % 5 == 0
        content_source = 'analytic_standalone' if standalone else 'HQ'
        overlay = ((step - 1) % 4 == 0) and not standalone
        analytic_records = []
        for pair_slot in (range(2) if pair_rank is None else [int(pair_rank)]):
            pair_number = 2 * (step - 1) + pair_slot + 1
            indices, windows, full = self._sources(step, pair_slot)
            ids = [self.rows[i]['source_id'] for i in indices]
            assert all(window == [0, 0, 1024, 768] for window in windows)
            sources = full[:, :1].contiguous()
            detail_records = []
            standalone_records = []
            if standalone:


                generation_index = 2 * (step // 5 - 1) + pair_slot + 1
                sources, standalone_records = self.detail.make_detail_clean(
                    torch.zeros_like(sources), generation_index, self.seed + 3301)
                ids_aug = [f'analytic_train/irv6native768/{self.seed}/{generation_index}/{i}' for i in range(2)]
            elif overlay:
                delta, detail_records = self.detail.detail_signals(sources, pair_number, self.seed)
                sources = sources + delta
                ids_aug = [name + '|v6_signal_' + str(pair_number) for name in ids]
            else:
                ids_aug = ids
            if group == 'clean':
                content_slots = [0, 1]
                full_records = [{'domain': 'ir', 'name': 'clean', 'style_id': 'ir|clean',
                    'signature': 'clean', 'realization_seed': stable_seed(self.seed, 'clean', pair_number, i),
                    'content_slot': i, 'contrast_valid': False} for i in range(2)]
                full_noise = torch.zeros(2, 1, 768, 1024)
                original_pair_valid = False
            else:
                original = self.sampler.make_pair_batch(full, ids, pair_number, sigma_override=self._strength(step))
                full_noise = (original['noisy'] - original['clean'])[:, :1].contiguous()
                full_records = original['metadata']
                content_slots = [r['content_slot'] for r in full_records]
                original_pair_valid = original['pair_valid']
            pair_meta = []
            for slot, (record, content_slot) in enumerate(zip(full_records, content_slots)):
                x, y, w, h = windows[content_slot]
                target = sources[content_slot:content_slot+1]
                assert [x, y, w, h] == [0, 0, 1024, 768]
                noise = full_noise[slot:slot+1].contiguous()
                assert target.shape == noise.shape == (1, 1, 768, 1024)
                output = target + noise
                actual = (output - target)[0, 0].numpy()
                field = full_noise[slot, 0].numpy()
                row = {**record, 'content_id': ids_aug[content_slot], 'pair_slot': pair_slot,
                    'sample_slot': 2 * pair_slot + slot, 'pair_number': pair_number,
                    'anchor_group': group, 'anchor_name': anchor_name,
                    'full_field_sha256': ah(field), 'parent_generated_field_sha256': record.get('field_sha256'),
                    'field_sha256': ah(actual), 'actual_rms': rms(actual),
                    'actual_peak': float(abs(actual).max()), 'actual_mean': float(actual.mean(dtype=np.float64)),
                    'full_field_actual_rms': rms(field), 'xywh': [x, y, w, h],
                    'full_field_HW': [768, 1024], 'field_shape_HW': [h, w],
                    'selected_source_index': indices[content_slot], 'selected_source_id': ids[content_slot],
                    'selected_source_pixel_sha256': self.source_hashes[indices[content_slot]],
                    'content_source': content_source,
                    'uses_selected_HQ_pixels_as_clean_target': not standalone,
                    'selected_HQ_used_for_parent_noise_arithmetic': True,
                    'analytic_record': standalone_records[content_slot] if standalone else None,
                    'clean_array_sha256': ah(target[0, 0].numpy()),
                    'noisy_array_sha256': ah(output[0, 0].numpy()),
                    'preaddition_full_noise_sha256': ah(noise[0, 0].numpy()),
                    'signal_overlay': overlay, 'signal_record': detail_records[content_slot] if overlay else None,
                    'sample_noise_valid': rms(actual) >= 1e-5, 'pixel_clipping': False,
                    'resized': False, 'cropped': False, 'tiled': False, 'recentered': False, 'realization_RMS_normalized': False,
                    'reads_evaluation_sources': False, 'source_noise_pixels_loaded': False,
                    'full_noise_extracted_as_parent_float32_noisy_minus_clean': True}
                pair_meta.append(row)
                if standalone:
                    analytic_records.append({'sample_slot': 2 * pair_slot + slot,
                        'content_id': ids_aug[content_slot], 'generation_index': generation_index,
                        'record': standalone_records[content_slot]})
                clean_batch.append(target)
                noisy_batch.append(output)
            valid = bool(original_pair_valid and all(r['sample_noise_valid'] for r in pair_meta))
            ratio = pair_meta[0]['actual_rms'] / max(pair_meta[1]['actual_rms'], 1e-30)
            same = bool(pair_number % 2)
            if not same and not .5 <= ratio <= 2:
                valid = False
            if pair_meta[0]['field_sha256'] == pair_meta[1]['field_sha256']:
                valid = False
            for row in pair_meta:
                row.update(contrast_valid=valid, pair_rms_ratio=ratio, pair_same=same)
            metadata.extend(pair_meta)
            pairs.append({'indices': [2*pair_slot, 2*pair_slot+1], 'pair_slot': pair_slot,
                          'pair_same': same, 'pair_valid': valid, 'metadata': pair_meta,
                          'margin': .5, 'original_full_pair_valid': original_pair_valid})
            selection.append({'pair_slot': pair_slot, 'selected_source_ids': ids,
                              'selected_source_indices': indices, 'windows': windows,
                              'used_content_slots': content_slots,
                              'HQ_pixels_exposed_to_model': not standalone})
        return {'clean': torch.cat(clean_batch).to(device), 'noisy': torch.cat(noisy_batch).to(device),
                'metadata': metadata, 'pair_records': pairs,
                'audit': {'step': step, 'anchor_group': group, 'anchor_name': anchor_name,
                          'anchor_ordinal': ordinal, 'clean_step': group == 'clean',
                          'content_source': content_source, 'analytic_records': analytic_records,
                          'HQ_pixel_exposures': 0 if standalone else len(metadata),
                          'standalone_analytic_exposures': len(metadata) if standalone else 0,
                          'signal_overlay': overlay, 'selection': selection, 'HQ_count': 50,
                          'source_HW': [768, 1024], 'training_HW': [768, 1024], 'native_full_frame': True, 'crop_resize_tile': False,
                          'reads_evaluation_sources': False}}

    def planned_exposure(self, steps=21000):
        groups, types, pair_types = Counter(), Counter(), Counter()
        for step in range(1, steps + 1):
            group, name, _ = self._anchor(step)
            groups[group] += 1
            if group == 'clean':
                types['clean'] += 4
            else:
                for pair_number in (2*step-1, 2*step):
                    for spec in self._plan(pair_number):
                        types[spec['name']] += 1
                    pair_types['positive' if pair_number % 2 else 'negative'] += 1
        return {'steps': steps, 'samples': 4*steps, 'anchor_step_counts': dict(groups),
                'planned_content_sample_counts': {'HQ': 4*(steps-steps//5), 'analytic_standalone': 4*(steps//5)},
                'planned_input_type_counts_after_negative_partner_sampling': dict(types),
                'contrast_pair_labels_before_realized_RMS_filter': dict(pair_types),
                'is_actual_training_exposure': False,
                'actual_full_frame_RMS_validity_and_signal_exposure_must_be_logged_during_training': True}
