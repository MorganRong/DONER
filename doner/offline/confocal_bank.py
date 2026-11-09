from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

HERE = Path(__file__).resolve().parent

from . import guided_bank as guided
from .fit_operator_bank import describe_field
from ..operator_sampling import OperatorLibrary

VERSION = 'confocal_text_bank128_v1'
SIZE = 128
RENDER = 1024
STEPS = 30
STRENGTH = .12
GUIDANCE = 3.0
NEGATIVE = ('cells, nuclei, mitochondria, actin filaments, fibers, tubes, networks, '
            'organelles, anatomical structures, rings, periodic patterns, text, '
            'watermark, color, scene, photograph')
guided.NEGATIVE = NEGATIVE
VENDORS = ('guided_bank.py', 'fit_operator_bank.py', 'operator_sampling.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def save_array(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('wb') as handle:
        np.save(handle, value, allow_pickle=False)
    temporary.replace(path)


def code_pins():
    return {'wrapper': sha(__file__), 'vendor': {name: sha((HERE.parent if name == 'operator_sampling.py' else HERE) / name) for name in VENDORS}}


def seed_for(protocol, family, split, index, role):
    namespace = str(protocol.get('seed_namespace', VERSION))
    message = f'{namespace}|{family}|{split}|{index}|{role}'
    return int.from_bytes(hashlib.sha256(message.encode()).digest()[:8], 'big') % (2**63 - 1)


def load_protocol(path):
    global NEGATIVE
    document = read(path)
    if document.get('geometry', [SIZE, SIZE]) != [SIZE, SIZE]:
        raise ValueError('This run is restricted to 128x128 textures and training blocks')
    for key, expected in [('steps', STEPS), ('strength', STRENGTH), ('guidance', GUIDANCE),
                          ('model_id', guided.MODEL_ID), ('model_revision', guided.REVISION)]:
        if document.get(key, expected) != expected:
            raise ValueError('Protocol differs from fixed generation setting: ' + key)
    negative = document.get('negative_prompt', NEGATIVE)
    if not isinstance(negative, str) or not negative.strip():
        raise ValueError('An explicit confocal negative prompt is required')
    NEGATIVE = negative
    guided.NEGATIVE = negative
    families = document.get('families')
    if not isinstance(families, list) or len(families) < 2:
        raise ValueError('At least two explicitly described confocal families are required')
    names = set()
    for family in families:
        name = family.get('name', '')
        if not re.fullmatch(r'[a-z][a-z0-9_]*', name) or name in names:
            raise ValueError('Family names must be unique lower-case identifiers')
        names.add(name)
        if family.get('kind') != 'grain' or family.get('axis') not in ('none', None):
            raise ValueError('This audited wrapper only accepts grain families, axis=none')
        prompt = family.get('prompt')
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('Every family needs its own explicit confocal prompt')
        if prompt in guided.PROMPTS.values():
            raise ValueError('Old STM/infrared prompt reuse is forbidden')
        for split, fallback in [('train', 4), ('holdout', 2)]:
            count = family.get(split + '_seeds', document.get(split + '_seeds', fallback))
            if not isinstance(count, int) or count < (2 if split == 'train' else 1):
                raise ValueError('Insufficient generated train/holdout seed counts')
    return document


def tasks_for(protocol):
    tasks = []
    for family in protocol['families']:
        for split, fallback in [('train', 4), ('holdout', 2)]:
            count = family.get(split + '_seeds', protocol.get(split + '_seeds', fallback))
            for index in range(count):
                name = family['name']
                tasks.append({'id': f'{name}_{split}_{index:03d}', 'family': name,
                              'split': split, 'index': index, 'kind': 'grain', 'axis': None,
                              'prompt': family['prompt'], 'guide_options': family.get('guide_options', {}),
                              'guide_seed': seed_for(protocol, name, split, index, 'guide'),
                              'diffusion_seed': seed_for(protocol, name, split, index, 'diffusion')})
    return tasks


def make_guide(options, seed):
    allowed = {'innovation', 'poisson_lambda', 'readout_std', 'horizontal_neighbor_correlation',
               'grain_correlation', 'grain_std', 'local_variation', 'envelope_grid'}
    if set(options) - allowed:
        raise ValueError('Unsupported guide options: ' + ','.join(sorted(set(options) - allowed)))
    rng = np.random.default_rng(seed)
    innovation = options.get('innovation', 'gaussian')
    if innovation == 'gaussian':
        field = rng.standard_normal((SIZE, SIZE))
    elif innovation == 'poisson_gaussian':
        rate = float(options.get('poisson_lambda', .15))
        readout = float(options.get('readout_std', .08))
        if not 0 < rate <= 100 or not 0 <= readout <= 5:
            raise ValueError('Invalid engineering counting/readout guide range')
        field = rng.poisson(rate, (SIZE, SIZE)) - rate + readout * rng.standard_normal((SIZE, SIZE))
    else:
        raise ValueError('Unknown independent guide innovation')
    rho = float(options.get('horizontal_neighbor_correlation', 0))
    if not -.95 <= rho <= .95:
        raise ValueError('Invalid horizontal neighbor correlation')
    if rho:
        field = guided.unit(field).astype(np.float64)
        original = field.copy()
        for column in range(1, SIZE):
            field[:, column] = rho * field[:, column - 1] + np.sqrt(1 - rho * rho) * original[:, column]
    correlation = float(options.get('grain_correlation', 0))
    if not 0 <= correlation <= 1:
        raise ValueError('Invalid isotropic guide correlation mixture')
    if correlation:
        field = (1 - correlation) * field + correlation * (np.roll(field, 1, 0) + np.roll(field, 1, 1)) / 2
    variation = float(options.get('local_variation', .15))
    amplitude = float(options.get('grain_std', .055))
    grid = options.get('envelope_grid', 8)
    if not 0 <= variation <= 1 or not 0 < amplitude <= .2 or not isinstance(grid, int) or not 2 <= grid <= 32:
        raise ValueError('Invalid engineering guide amplitude/envelope range')
    envelope = np.clip(1 + variation * guided.unit(guided.smooth_noise(rng, (SIZE, SIZE), grid)), .25, 2)
    field = amplitude * guided.unit(field * envelope)
    scale = min(1., .45 / max(float(np.abs(field).max()), 1e-12))
    guide = np.asarray(.5 + field * scale, dtype=np.float32)
    assert guide.shape == (SIZE, SIZE) and guide.min() >= .05 - 1e-6 and guide.max() <= .95 + 1e-6
    return guide, {'source': 'Independent analytic random field; no measured pixels or fitted residual parameters',
                   'options': options, 'seed': seed, 'shape_HW': [SIZE, SIZE],
                   'global_amplitude_scale': scale, 'pixel_clipping': False}


def raw_to_generated(raw):
    if raw.shape != (RENDER, RENDER, 3) or not np.isfinite(raw).all():
        raise ValueError('Wrong SDXL RGB output')
    gray = raw @ np.asarray([.2126, .7152, .0722], dtype=np.float32)
    factor = RENDER // SIZE
    return np.asarray(gray.reshape(SIZE, factor, SIZE, factor).mean((1, 3)), dtype=np.float32)


def numeric_audit(guide, generated):
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    result = guided.morphology_metrics(guide, generated, mask, [], 'grain')
    result['semantic_admitted'] = False
    result['meaning'] = 'Numerical morphology/provenance only; visual review is required before fitting'
    return result


def preview(path, guide, generated, texture, title):
    panels = [('Independent guide', guide), ('Actual SDXL output', generated),
              ('Mean/RMS normalized', np.clip(.5 + texture / 6, 0, 1))]
    canvas = Image.new('RGB', (SIZE * 3, SIZE + 45), 'white')
    draw = ImageDraw.Draw(canvas)
    for index, (label, array) in enumerate(panels):
        picture = Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype('uint8')).convert('RGB')
        canvas.paste(picture, (index * SIZE, 0))
        draw.text((index * SIZE + 3, SIZE + 2), label, fill='black')
    draw.text((3, SIZE + 22), title, fill='black')
    canvas.save(path)


def expected_record(task, protocol_path, model_pin_sha):
    return {'version': VERSION, 'task': task, 'protocol_sha256': sha(protocol_path),
            'code_pins': code_pins(), 'model_pin_sha256': model_pin_sha,
            'model_id': guided.MODEL_ID, 'model_revision': guided.REVISION,
            'negative_prompt': NEGATIVE, 'num_inference_steps': STEPS,
            'strength': STRENGTH, 'guidance_scale': GUIDANCE,
            'generation_shape_HWC': [RENDER, RENDER, 3], 'texture_shape_HW': [SIZE, SIZE],
            'measured_pixels_used': False, 'measured_calibration_parameters_used': False}


def verify_record(root, task, protocol_path, model_pin_sha):
    directory = root / 'candidates'
    metadata = read(directory / (task['id'] + '.json'))
    expected = expected_record(task, protocol_path, model_pin_sha)
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError('Existing candidate recipe/code/model differs: ' + task['id'])
    files = metadata['files']
    if set(files) != {'raw_rgb', 'guide', 'generated', 'texture'}:
        raise ValueError('Incomplete candidate arrays')
    arrays = {}
    for key, spec in files.items():
        file = (directory / spec['path']).resolve()
        if not file.is_relative_to(directory.resolve()) or sha(file) != spec['sha256']:
            raise ValueError('Candidate path/hash mismatch')
        arrays[key] = np.load(file, allow_pickle=False)
    guide, _ = make_guide(task['guide_options'], task['guide_seed'])
    if not np.array_equal(arrays['guide'], guide):
        raise ValueError('Saved guide differs from independent random recipe')
    generated = raw_to_generated(arrays['raw_rgb'])
    if not np.array_equal(arrays['generated'], generated):
        raise ValueError('Generated pixels changed during RGB/area mapping')
    if not np.array_equal(arrays['texture'], guided.unit(generated)):
        raise ValueError('Texture differs from mean/RMS-normalized SDXL output')
    if metadata['numeric_audit'] != numeric_audit(guide, generated):
        raise ValueError('Numerical audit differs from saved generated pixels')
    return metadata


def check_tokens(pipe, tasks):
    output = []
    for name, tokenizer in [('clip_l', pipe.tokenizer), ('clip_g', pipe.tokenizer_2)]:
        maximum = int(tokenizer.model_max_length)
        for task in tasks:
            for role, prompt in [('positive', task['prompt']), ('negative', NEGATIVE)]:
                count = len(tokenizer(prompt, truncation=False)['input_ids'])
                if count > maximum:
                    raise ValueError(f'{task["id"]} {name}/{role}: {count} tokens exceeds {maximum}')
                output.append({'id': task['id'], 'encoder': name, 'role': role,
                               'tokens': count, 'maximum': maximum, 'truncated': False})
    return output


def generate_shard(args, protocol, protocol_path):
    root = args.root
    all_tasks = tasks_for(protocol)
    tasks = [task for index, task in enumerate(all_tasks) if index % args.shards == args.shard]
    if not tasks:
        raise ValueError('Empty generation shard')
    model_pin_sha = guided.verify_model(args.model_path)
    pin = read(args.model_path / 'pinned_model.json')
    saved_pin = root / 'model_pin.json'
    if saved_pin.exists() and read(saved_pin) != pin:
        raise ValueError('Another model pin already exists')
    write(saved_pin, pin)

    pin_original = root / 'model_pin_original.json'
    contents = (args.model_path / 'pinned_model.json').read_bytes()
    if pin_original.exists() and pin_original.read_bytes() != contents:
        raise ValueError('Original model-pin bytes changed')
    temporary = pin_original.with_name(pin_original.name + f'.{os.getpid()}.tmp')
    temporary.write_bytes(contents)
    temporary.replace(pin_original)
    directory = root / 'candidates'
    directory.mkdir(parents=True, exist_ok=True)
    pending = []
    for task in tasks:
        if (directory / (task['id'] + '.json')).exists():
            verify_record(root, task, protocol_path, model_pin_sha)
        else:
            if list(directory.glob(task['id'] + '_*.npy')):
                raise ValueError('Partial candidate needs explicit inspection: ' + task['id'])
            pending.append(task)
    pipe = None
    if pending:
        pipe = guided.load_pipeline(args.model_path, args.device)
        write(root / f'token_audit_shard_{args.shard}.json', check_tokens(pipe, tasks))
    for task in pending:
        start = time.monotonic()
        guide, guide_info = make_guide(task['guide_options'], task['guide_seed'])
        raw, generated = guided.generate(pipe, guide, task['prompt'], task['diffusion_seed'],
                                         STRENGTH, args.device, steps=STEPS, guidance=GUIDANCE)
        if not np.array_equal(generated, raw_to_generated(raw)):
            raise ValueError('Vendor output is not exact area reduction of SDXL RGB')
        texture = guided.unit(generated)
        files = {}
        for key, array in [('raw_rgb', raw), ('guide', guide), ('generated', generated), ('texture', texture)]:
            file = directory / f'{task["id"]}_{key}.npy'
            save_array(file, array)
            files[key] = {'path': file.name, 'sha256': sha(file), 'shape': list(array.shape), 'dtype': str(array.dtype)}
        audit = numeric_audit(guide, generated)
        metadata = expected_record(task, protocol_path, model_pin_sha)
        metadata.update({'guide_provenance': guide_info, 'files': files, 'numeric_audit': audit,
                         'generation_seconds': time.monotonic() - start,
                         'postprocessing': 'RGB luminance, exact 8x8 area mean to128, global mean/RMS only'})
        preview(directory / (task['id'] + '.png'), guide, generated, texture, task['id'])
        write(directory / (task['id'] + '.json'), metadata)
        write(root / f'generation_status_shard_{args.shard}.json', {
            'state': 'generating', 'last_id': task['id'], 'shard': args.shard, 'shards': args.shards,
            'completed': sum((directory / (row['id'] + '.json')).exists() for row in tasks),
            'target': len(tasks), 'numeric_audit': audit})
        print(canonical({'id': task['id'], 'seconds': metadata['generation_seconds'], 'numeric_audit': audit}), flush=True)
    records = [verify_record(root, task, protocol_path, model_pin_sha) for task in tasks]
    receipt = {'state': 'completed', 'shard': args.shard, 'shards': args.shards,
               'protocol_sha256': sha(protocol_path), 'code_pins': code_pins(),
               'candidate_ids': [row['task']['id'] for row in records],
               'numeric_passed': all(row['numeric_audit']['numeric_passed'] for row in records),
               'semantic_admitted': False}
    write(root / f'generation_shard_{args.shard}.json', receipt)
    if all((directory / (task['id'] + '.json')).exists() for task in all_tasks):
        aggregate(root, protocol, protocol_path)


def aggregate(root, protocol, protocol_path):
    tasks = tasks_for(protocol)
    pin_sha = sha(root / 'model_pin_original.json')
    records = [verify_record(root, task, protocol_path, pin_sha) for task in tasks]
    hashes = [record['files']['texture']['sha256'] for record in records]
    if len(hashes) != len(set(hashes)):
        raise ValueError('Duplicate textures across generated candidates')
    directory = root / 'candidates'
    manifest = {'version': VERSION, 'protocol_sha256': sha(protocol_path), 'code_pins': code_pins(),
                'model_pin_sha256': pin_sha, 'candidate_count': len(records),
                'numeric_passed_count': sum(row['numeric_audit']['numeric_passed'] for row in records),
                'semantic_admitted': False, 'measured_pixels_used': False,
                'records': [{'id': row['task']['id'], 'family': row['task']['family'], 'split': row['task']['split'],
                             'metadata_path': f'candidates/{row["task"]["id"]}.json',
                             'metadata_sha256': sha(directory / (row['task']['id'] + '.json')),
                             'texture_sha256': row['files']['texture']['sha256'],
                             'numeric_passed': row['numeric_audit']['numeric_passed']} for row in records]}
    write(root / 'generation_manifest.json', manifest)
    review = root / 'contact_sheets'
    review.mkdir(exist_ok=True)
    for family in protocol['families']:
        rows = [row for row in records if row['task']['family'] == family['name']]
        columns = 3
        tile_h = SIZE + 36
        canvas = Image.new('RGB', (columns * SIZE, ((len(rows) + columns - 1) // columns) * tile_h), 'white')
        draw = ImageDraw.Draw(canvas)
        for index, row in enumerate(rows):
            array = np.load(directory / row['files']['texture']['path'], allow_pickle=False)
            picture = Image.fromarray(np.rint(np.clip(.5 + array / 6, 0, 1) * 255).astype('uint8')).convert('RGB')
            x, y = index % columns * SIZE, index // columns * tile_h
            canvas.paste(picture, (x, y))
            draw.text((x + 2, y + SIZE + 2), f'{row["task"]["split"]} {row["task"]["index"]}', fill='black')
            draw.text((x + 2, y + SIZE + 17), 'numeric PASS' if row['numeric_audit']['numeric_passed'] else 'numeric FAIL', fill='black')
        canvas.save(review / (family['name'] + '.png'))
    template = {'schema_version': 1, 'generation_manifest_sha256': sha(root / 'generation_manifest.json'),
                'reviewer': '', 'records': {row['id']: {'passed': None, 'reason': ''} for row in manifest['records']}}
    write(root / 'admission_template.json', template)
    print(canonical({'state': 'generation_audited', 'candidates': len(records),
                     'numeric_passed': manifest['numeric_passed_count'],
                     'manifest_sha256': template['generation_manifest_sha256'],
                     'semantic_admission_required': True}), flush=True)
    return manifest


def text_probe(args, protocol, protocol_path):
    all_tasks = [task for task in tasks_for(protocol) if task['split'] == 'train' and task['index'] == 0]
    tasks = [task for index, task in enumerate(all_tasks) if index % args.shards == args.shard]
    pin_sha = guided.verify_model(args.model_path)
    pipe = guided.load_pipeline(args.model_path, args.device)
    check_tokens(pipe, tasks)
    output = args.root / 'text_probe'
    output.mkdir(exist_ok=True)
    for task in tasks:
        source = verify_record(args.root, task, protocol_path, pin_sha)
        guide = np.load(args.root / 'candidates' / source['files']['guide']['path'], allow_pickle=False)
        original = np.load(args.root / 'candidates' / source['files']['generated']['path'], allow_pickle=False)
        raw, blank = guided.generate(pipe, guide, '', task['diffusion_seed'], STRENGTH, args.device,
                                     steps=STEPS, guidance=GUIDANCE)
        files = {}
        for key, array in [('empty_raw_rgb', raw), ('empty_generated', blank)]:
            path = output / f'{task["family"]}_{key}.npy'
            save_array(path, array)
            files[key] = {'path': path.name, 'sha256': sha(path)}
        difference = float(np.sqrt(np.mean((original.astype(np.float64) - blank) ** 2)))
        write(output / (task['family'] + '.json'), {
            'family': task['family'], 'source_id': task['id'], 'source_metadata_sha256': sha(args.root / 'candidates' / (task['id'] + '.json')),
            'protocol_sha256': sha(protocol_path), 'code_pins': code_pins(), 'model_pin_sha256': pin_sha,
            'same_guide_and_diffusion_seed': True, 'same_negative_prompt_and_generation_parameters': True,
            'only_positive_prompt_changed': True, 'positive_prompt': task['prompt'], 'comparison_prompt': '',
            'generated_pixel_RMS_difference': difference, 'text_changes_output': difference > 1e-7,
            'files': files, 'meaning': 'Output dependence on text only; not semantic validity or restoration benefit'})
        print(canonical({'probe': task['family'], 'pixel_RMS_difference': difference, 'text_changes_output': difference > 1e-7}), flush=True)


def psd2d(array):
    array = np.asarray(array, dtype=np.float64)
    window = np.outer(np.hanning(SIZE), np.hanning(SIZE))
    centered = array - np.sum(array * window) / window.sum()
    power = np.abs(np.fft.rfft2(centered * window, norm='ortho')) ** 2 / np.mean(window * window)
    power = ndimage.gaussian_filter(power, sigma=2., mode=('wrap', 'nearest'))
    power[0, 0] = 0.
    weights = np.full(SIZE // 2 + 1, 2.)
    weights[[0, -1]] = 1.
    expectation = float(np.sum(power * weights) / (SIZE * SIZE))
    if not expectation > 1e-12 or not np.isfinite(power).all():
        raise ValueError('Degenerate generated 2D spectrum')
    power /= expectation
    return {'shape_HW': [SIZE, SIZE], 'power': power.tolist(),
            'layout': 'numpy rfft2: signed y frequency, nonnegative x frequency',
            'window': 'separable Hann; weighted mean removed; mean-square window correction',
            'smoothing': {'sigma_frequency_bins': 2., 'mode_y': 'wrap', 'mode_x': 'nearest'},
            'normalization': 'Hermitian-weighted mean spectral power is one; DC is zero',
            'source': 'This generated TRAIN texture only; no real residual or heldout data'}


def fitted_records(root, rows, split):
    records = []
    for row in rows:
        if row['split'] != split:
            raise ValueError('Fitting split contamination')
        metadata = read(root / row['metadata_path'])
        spec = metadata['files']['texture']
        file = root / 'candidates' / spec['path']
        if sha(file) != spec['sha256']:
            raise ValueError('Generated texture changed')
        texture = np.load(file, allow_pickle=False)
        parameters = describe_field(texture, 'grain', None)
        parameters['psd2d_rfft'] = psd2d(texture)
        parameters['psd2d_rfft']['source'] = f'This generated {split.upper()} texture only; no real residual data'
        records.append({'source_id': row['id'], 'source_sha256': spec['sha256'],
                        'source_shape_HW': [SIZE, SIZE], 'parameters': parameters})
    return records


def fit_only(root, protocol, protocol_path):
    manifest = aggregate(root, protocol, protocol_path)
    admission = read(root / 'admission.json')
    if admission.get('generation_manifest_sha256') != sha(root / 'generation_manifest.json') or not admission.get('reviewer'):
        raise ValueError('Missing/stale explicit visual admission')
    decisions = admission.get('records', {})
    if set(decisions) != {row['id'] for row in manifest['records']}:
        raise ValueError('Every candidate needs an explicit visual decision')
    for decision in decisions.values():
        if not isinstance(decision.get('passed'), bool) or not decision.get('reason'):
            raise ValueError('Each semantic decision needs a boolean and reason')
    operators = []
    holdout_operators = []
    accepted = [row for row in manifest['records'] if decisions[row['id']]['passed'] and row['numeric_passed']]
    probe_hashes = {}
    for family in protocol['families']:
        name = family['name']
        probe_path = root / 'text_probe' / (name + '.json')
        probe = read(probe_path)
        if probe.get('protocol_sha256') != sha(protocol_path) or probe.get('code_pins') != code_pins() or not probe.get('text_changes_output'):
            raise ValueError('Missing/stale positive-versus-empty-prompt effect probe: ' + name)
        for value in probe['files'].values():
            if sha(probe_path.parent / value['path']) != value['sha256']:
                raise ValueError('Text probe artifact hash mismatch')
        probe_hashes[name] = sha(probe_path)
        train = [row for row in accepted if row['family'] == name and row['split'] == 'train']
        holdout = [row for row in accepted if row['family'] == name and row['split'] == 'holdout']
        if len(train) < 2 or not holdout:
            raise ValueError('At least two admitted TRAIN and one HOLDOUT texture per family are required')
        operators.append({'name': name, 'family': name, 'kind': 'grain', 'axis': None,
                          'records': fitted_records(root, train, 'train')})
        holdout_operators.append({'name': name, 'family': name, 'kind': 'grain', 'axis': None,
                                  'records': fitted_records(root, holdout, 'holdout')})
    library = {'schema_version': 1, 'domain': protocol.get('domain', 'hagen'),
               'dataset_id': protocol.get('dataset_id', 'hagen_confocal_128'),
               'fit_split': 'train', 'fit_records': sum(len(op['records']) for op in operators),
               'operator_count': len(operators), 'heldout_records_used_for_fit': 0,
               'stores_pixels_or_positions': False, 'operators': operators,
               'source_manifest_sha256': sha(root / 'generation_manifest.json'),
               'prompt_protocol_sha256': sha(protocol_path), 'admission_sha256': sha(root / 'admission.json'),
               'model_pin_sha256': sha(root / 'model_pin_original.json'), 'code_pins': code_pins(),
               'text_effect_probe_sha256': probe_hashes,
               'measured_pixels_used': False, 'measured_calibration_parameters_used': False,
               'limitations': ['Generated engineering morphology prior, not measured detector calibration.',
                               'The standard radial sampler ignores psd2d_rfft; the new confocal sampler must explicitly consume it.',
                               'Signal dependence and counting/readout parameter ranges are separately declared analytic rules.']}
    path = root / 'noise_library.json'
    write(path, library)
    runtime = OperatorLibrary(path, library['domain'], library['dataset_id'])
    for name in runtime.names:
        field, detail = runtime.sample(name, SIZE, SIZE, np.random.default_rng(77193))
        if field.shape != (SIZE, SIZE) or not np.isfinite(field).all() or detail['source_pixels_loaded']:
            raise ValueError('Library compatibility preflight failed')
    holdout_library = dict(library)
    holdout_library.update(fit_split='holdout', operators=holdout_operators,
                           fit_records=sum(len(op['records']) for op in holdout_operators),
                           heldout_records_used_for_fit=sum(len(op['records']) for op in holdout_operators),
                           train_records_used_for_fit=0, permitted_use='Evaluation only; training must reject this library')
    holdout_path = root / 'holdout_noise_library.json'
    write(holdout_path, holdout_library)
    train_ids = {record['source_id'] for op in operators for record in op['records']}
    holdout_ids = {record['source_id'] for op in holdout_operators for record in op['records']}
    train_hashes = {record['source_sha256'] for op in operators for record in op['records']}
    holdout_hashes = {record['source_sha256'] for op in holdout_operators for record in op['records']}
    if train_ids & holdout_ids or train_hashes & holdout_hashes:
        raise ValueError('Training and heldout generated operators overlap')
    try:
        OperatorLibrary(holdout_path, library['domain'], library['dataset_id'])
    except ValueError:
        pass
    else:
        raise ValueError('Training OperatorLibrary did not reject holdout fit_split')
    write(root / 'fit_receipt.json', {'state': 'completed', 'library_sha256': sha(path),
                                     'holdout_library_sha256': sha(holdout_path),
                                     'families': runtime.names, 'fit_records': library['fit_records'],
                                     'heldout_records_used_for_fit': 0,
                                     'holdout_library_fit_records': holdout_library['fit_records'],
                                     'train_holdout_source_overlap': 0,
                                     'training_sampler_rejects_holdout_library': True,
                                     'operator_library_128_compatibility_passed': True,
                                     'semantic_admission_sha256': library['admission_sha256']})
    print(canonical(read(root / 'fit_receipt.json')), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=HERE)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--fit-only', action='store_true')
    modes.add_argument('--audit-only', action='store_true')
    modes.add_argument('--text-probe-only', action='store_true')
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.root.mkdir(parents=True, exist_ok=True)
    protocol_path = (args.protocol or args.root / 'prompt_protocol.json').resolve()
    protocol = load_protocol(protocol_path)
    if not args.shards > 0 or not 0 <= args.shard < args.shards:
        raise ValueError('Invalid shard index/count')
    if args.fit_only:
        fit_only(args.root, protocol, protocol_path)
    elif args.audit_only:
        aggregate(args.root, protocol, protocol_path)
    else:
        if args.model_path is None:
            parser.error('--model-path is required for generation and text probes')
        args.model_path = args.model_path.resolve()
        if args.text_probe_only:
            text_probe(args, protocol, protocol_path)
        else:
            generate_shard(args, protocol, protocol_path)


if __name__ == '__main__':
    main()
