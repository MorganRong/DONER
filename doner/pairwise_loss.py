import math
from collections.abc import Mapping

import torch
from torch.nn import functional as F


from .contrast_loss import noise_statistics_target, reconstruction_loss, gradient_l1


def pairwise_contrastive_loss(projection, metadata, pair_same, pair_valid, margin=.5):


    if (not isinstance(projection, torch.Tensor) or projection.ndim != 2
            or projection.shape[0] != 2 or projection.shape[1] < 1
            or not projection.is_floating_point()):
        raise ValueError('Exactly two floating-point projections [2,D] are required')
    if not bool(torch.isfinite(projection).all()):
        raise ValueError('Nonfinite projection')
    if not isinstance(metadata, (list, tuple)) or len(metadata) != 2:
        raise ValueError('Exactly two metadata records are required')
    if type(pair_same) is not bool or type(pair_valid) is not bool:
        raise ValueError('pair_same and pair_valid must be explicit booleans')
    if isinstance(margin, bool) or not isinstance(margin, (float, int)):
        raise ValueError('margin must be a finite scalar in (0,2]')
    if not math.isfinite(margin) or not 0 < margin <= 2:
        raise ValueError('margin must be a finite scalar in (0,2]')
    required = {'domain', 'content_id', 'style_id', 'signature', 'realization_seed'}
    for record in metadata:
        if not isinstance(record, Mapping) or not required.issubset(record):
            raise ValueError('Missing pair provenance')
        if record['domain'] not in ('ir', 'stm', 'hagen'):
            raise ValueError('Unknown noise domain')
        if any(not isinstance(record[key], str) or not record[key]
               for key in ('content_id', 'style_id', 'signature')):
            raise ValueError('Content, style and signature identifiers must be nonempty strings')
        if not isinstance(record['realization_seed'], int) or isinstance(record['realization_seed'], bool):
            raise ValueError('An integer realization seed is required')
    a, b = metadata
    if a['domain'] != b['domain']:
        raise ValueError('The two images must belong to the same domain')

    reasons = []
    if not pair_valid:
        reasons.append('pair_valid_false')
    for index, record in enumerate(metadata):
        if not record.get('contrast_valid', True):
            reasons.append(f'sample_{index}_contrast_invalid')
    if pair_same:
        if a['style_id'] != b['style_id']:
            reasons.append('positive_style_mismatch')
        if a['signature'] != b['signature']:
            reasons.append('positive_signature_mismatch')
        if a['content_id'] == b['content_id']:
            reasons.append('positive_content_not_distinct')
        if a['realization_seed'] == b['realization_seed']:
            reasons.append('positive_realization_seed_not_independent')
        if (a.get('field_sha256') is not None
                and a.get('field_sha256') == b.get('field_sha256')):
            reasons.append('positive_identical_realized_field')
    else:
        if a['signature'] == b['signature']:
            reasons.append('negative_coarse_signature_not_distinct')
        if a['style_id'] == b['style_id']:
            reasons.append('negative_style_not_distinct')

    ratio = None
    rms_values = []
    for index, record in enumerate(metadata):
        value = record.get('actual_rms')
        if value is None:
            if not pair_same:
                reasons.append(f'negative_sample_{index}_actual_RMS_missing')
            rms_values.append(None)
        elif (isinstance(value, bool) or not isinstance(value, (int, float))
              or not math.isfinite(value) or value <= 0):
            reasons.append(f'sample_{index}_actual_RMS_not_positive_finite')
            rms_values.append(None)
        else:
            rms_values.append(float(value))
    if all(value is not None for value in rms_values):
        ratio = rms_values[0]/rms_values[1]
        if not pair_same and not .5 <= ratio <= 2.:
            reasons.append('negative_actual_RMS_ratio_outside_[0.5,2]')

    norms = torch.linalg.vector_norm(projection, dim=1)
    norm_ok = bool(torch.isfinite(norms).all()) and bool((norms > 1e-8).all())
    if not norm_ok:
        reasons.append('degenerate_or_overflowed_projection_norm')
    normalized = F.normalize(projection, p=2, dim=1, eps=1e-8)
    distance = (1.-(normalized[0]*normalized[1]).sum()).clamp(0., 2.)
    valid = not reasons
    stats = {'images': 2, 'pairs': 1, 'valid_pairs': int(valid), 'valid_pair': valid,
             'pair_same': pair_same, 'pair_kind': 'positive' if pair_same else 'negative',
             'requested_pair_valid': pair_valid, 'domain': a['domain'],
             'distance': float(distance.detach()) if norm_ok else None,
             'cosine_distance': float(distance.detach()) if norm_ok else None,
             'margin': float(margin), 'exclusion_reasons': reasons,
             'negative_actual_RMS_ratio_range': [.5, 2.],
             'actual_RMS_ratio': ratio if ratio is None or math.isfinite(ratio) else None,
             'loss_form': 'positive:d^2; negative:relu(margin-d)^2; d=clamp(1-cosine,0,2)',
             'extra_views': 0, 'queue_entries': 0, 'memory_bank_entries': 0}
    if not valid:

        return (projection*0.).sum(), stats
    loss = distance.square() if pair_same else F.relu(margin-distance).square()
    stats['negative_margin_satisfied'] = (not pair_same and float(distance.detach()) >= margin)
    return loss, stats
