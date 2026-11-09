from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import fft


SAMPLER_VERSION = 'native_operator_v1'
KINDS = {'grain', 'smooth', 'axis', 'axis_grain', 'streak', 'streak_grain'}


@lru_cache(maxsize=4)
def _grids(height, width):
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.rfftfreq(width)[None, :]
    return fy, fx, np.hypot(fy, fx)


def _expected_rfft_power(amplitude, width):

    weights = np.full(amplitude.shape[-1], 2., dtype=np.float64)
    weights[0] = 1.
    if width % 2 == 0:
        weights[-1] = 1.
    return float(np.sum(amplitude.astype(np.float64)**2 * weights) /
                 (width * (amplitude.shape[0] if amplitude.ndim == 2 else 1)))


@lru_cache(maxsize=24)
def _radial_transfer(height, width, frequency, power):
    _, _, radius = _grids(height, width)
    amplitude = np.sqrt(np.maximum(np.interp(radius, frequency, power), 0.))
    amplitude[0, 0] = 0.
    expectation = _expected_rfft_power(amplitude, width)
    if expectation <= 1e-20:
        raise ValueError('PSD has no non-DC power at the requested geometry')
    return np.asarray(amplitude / np.sqrt(expectation), dtype=np.float32)


@lru_cache(maxsize=24)
def _gaussian_transfer(height, width, length_y, length_x):


    fy, fx, _ = _grids(height, width)
    amplitude = np.exp(-.5 * (2*np.pi)**2 *
                       ((fy*length_y/2)**2 + (fx*length_x/2)**2))
    amplitude[0, 0] = 0.
    expectation = _expected_rfft_power(amplitude, width)
    if expectation <= 1e-30:


        return np.zeros_like(amplitude, dtype=np.float32)
    return np.asarray(amplitude / np.sqrt(expectation), dtype=np.float32)


def _quantile_population_moments(probabilities, values):
    p = np.r_[0., probabilities, 1.]
    q = np.r_[values[0], values, values[-1]]
    delta = np.diff(p)
    mean = float(np.sum(delta*(q[:-1]+q[1:])/2))
    second = float(np.sum(delta*(q[:-1]**2+q[:-1]*q[1:]+q[1:]**2)/3))
    return p, q, mean, max(0., second-mean*mean)


def _innovations(shape, rng, quantiles=None):
    if quantiles is None:
        return rng.standard_normal(shape, dtype=np.float32)
    p, q, mean, variance = _quantile_population_moments(
        quantiles['probabilities'], quantiles['values'])
    if variance <= 1e-12:
        raise ValueError('Degenerate innovation quantile distribution')
    values = np.interp(rng.random(shape), p, q)
    return np.asarray((values-mean)/np.sqrt(variance), dtype=np.float32)


def _spectrum_tuple(spec):
    return tuple(map(float, spec['frequency_cpp'])), tuple(map(float, spec['power']))


def _colored_field(height, width, spectrum, rng, quantiles=None):
    f, p = _spectrum_tuple(spectrum)
    transfer = _radial_transfer(height, width, f, p)
    white = _innovations((height, width), rng, quantiles)
    return fft.irfft2(fft.rfft2(white, workers=1)*transfer,
                     s=(height, width), workers=1).astype(np.float32, copy=False)


def _smooth_field(height, width, lengths, rng):
    ly, lx = map(float, lengths)
    transfer = _gaussian_transfer(height, width, ly, lx)
    white = _innovations((height, width), rng)
    return fft.irfft2(fft.rfft2(white, workers=1)*transfer,
                     s=(height, width), workers=1).astype(np.float32, copy=False)


def _envelope(height, width, cv, lengths, rng):
    if cv <= 1e-8:
        return 1.
    sigma = float(np.sqrt(np.log1p(cv*cv)))
    smooth = _smooth_field(height, width, lengths, rng)


    return np.exp(sigma*smooth-.5*sigma*sigma).astype(np.float32)


def _profile(length, spectrum, rng):
    f, p = _spectrum_tuple(spectrum)
    frequencies = np.fft.rfftfreq(length)
    transfer = np.sqrt(np.maximum(np.interp(frequencies, f, p), 0.))
    transfer[0] = 0.
    expectation = _expected_rfft_power(transfer, length)
    if expectation <= 1e-20:
        raise ValueError('Axis PSD has no resolved non-DC component')
    transfer = np.asarray(transfer/np.sqrt(expectation), dtype=np.float32)
    return fft.irfft(fft.rfft(_innovations(length, rng), workers=1)*transfer,
                    n=length, workers=1).astype(np.float32, copy=False)


def _validate_spectrum(spec, maximum):
    if not isinstance(spec, dict):
        raise ValueError('Missing fitted spectrum')
    f, p = _spectrum_tuple(spec)
    if (len(f) < 4 or len(f) != len(p) or not np.isfinite(f).all()
            or not np.isfinite(p).all() or min(f) < 0 or max(f) > maximum+1e-8
            or np.any(np.diff(f) <= 0) or min(p) < 0 or max(p) <= 0):
        raise ValueError('Invalid physical-frequency PSD descriptor')


def _validate_parameters(kind, params):
    _validate_spectrum(params['radial_psd'], np.sqrt(.5))
    for key in ('envelope_cv',):
        if not np.isfinite(params[key]) or params[key] < 0:
            raise ValueError('Invalid fitted modulation')
    lengths = params['envelope_correlation_px']
    if len(lengths) != 2 or not np.isfinite(lengths).all() or min(lengths) <= 0:
        raise ValueError('Invalid correlation lengths in pixels')
    quantiles = params['pixel_quantiles']
    probs, values = quantiles['probabilities'], quantiles['values']
    if (len(probs) < 3 or len(probs) != len(values) or min(probs) <= 0 or max(probs) >= 1
            or np.any(np.diff(probs) <= 0) or np.any(np.diff(values) < 0)
            or not np.isfinite(values).all()):
        raise ValueError('Invalid quantile descriptor')
    if kind in ('axis', 'axis_grain', 'streak', 'streak_grain'):
        fraction = params['background_fraction']
        if not np.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError('Invalid separated background fraction')
        _validate_spectrum(params['background_radial_psd'], np.sqrt(.5))
    if kind in ('axis', 'axis_grain'):
        _validate_spectrum(params['axis_psd'], .5)
        if not np.isfinite(params['along_modulation_cv']) or params['along_modulation_cv'] < 0:
            raise ValueError('Invalid along-line modulation')
        if not np.isfinite(params['along_correlation_px']) or params['along_correlation_px'] <= 0:
            raise ValueError('Invalid along-line correlation')
    if kind in ('streak', 'streak_grain'):
        density = params['event_density_per_pixel']
        if not np.isfinite(density) or density < 0 or not params['events']:
            raise ValueError('Invalid fitted event process')
        for event in params['events']:
            if (not np.isfinite([event['length_px'], event['width_px'], event['amplitude']]).all()
                    or event['length_px'] <= 0 or event['width_px'] <= 0
                    or event['amplitude'] < 0 or event['polarity'] not in (-1, 1)):
                raise ValueError('Invalid finite event descriptor')


class OperatorLibrary:

    def __init__(self, path, expected_domain, expected_dataset_id):
        self.path = Path(path)
        raw = self.path.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.document = json.loads(raw)
        m = self.document
        if (m.get('schema_version') != 1 or m.get('domain') != expected_domain
                or m.get('dataset_id') != expected_dataset_id or m.get('fit_split') != 'train'):
            raise ValueError('Wrong fitted operator schema/domain/dataset/split')
        entries = m.get('operators')
        if not isinstance(entries, list) or not entries:
            raise ValueError('Empty fitted operator registry')
        self.operators = {}
        for entry in entries:
            name, kind = entry['name'], entry['kind']
            if not isinstance(name, str) or not name or name in self.operators or kind not in KINDS:
                raise ValueError('Duplicate/invalid operator or unsupported implementation kind')
            if kind in ('axis', 'axis_grain', 'streak', 'streak_grain') and entry.get('axis') not in ('row', 'column'):
                raise ValueError('An axis is required for directed operators')
            records = entry.get('records')
            if not isinstance(records, list) or not records:
                raise ValueError('Operator has no measured training records')
            identifiers = set()
            for record in records:
                identifier = record['source_id']
                if identifier in identifiers or not record.get('source_sha256'):
                    raise ValueError('Missing or repeated fit provenance')
                identifiers.add(identifier)
                _validate_parameters(kind, record['parameters'])
            self.operators[name] = entry
        self.names = list(self.operators)
        self.domain, self.dataset_id = expected_domain, expected_dataset_id

    def sample(self, name, height, width, rng):
        entry = self.operators[name]
        index = int(rng.integers(len(entry['records'])))
        return self.sample_record(name, index, height, width, rng)

    def sample_record(self, name, record_index, height, width, rng):
        if (not isinstance(height, int) or not isinstance(width, int)
                or min(height, width) < 16 or not isinstance(rng, np.random.Generator)):
            raise ValueError('Expected native integer geometry >=16 and an explicit RNG')
        entry = self.operators[name]
        record = entry['records'][record_index]
        params, kind, axis = record['parameters'], entry['kind'], entry.get('axis')
        event_info = {}
        if kind in ('grain', 'smooth'):
            field = _colored_field(height, width, params['radial_psd'], rng, params['pixel_quantiles'])
            if kind == 'grain':
                cv = params['envelope_cv']
                field *= _envelope(height, width, cv, params['envelope_correlation_px'], rng)
                field /= np.sqrt(1+cv*cv)
        elif kind in ('axis', 'axis_grain'):
            profile = _profile(height if axis == 'row' else width, params['axis_psd'], rng)
            directed = np.broadcast_to(profile[:, None] if axis == 'row' else profile[None, :], (height, width)).copy()
            along = params['along_correlation_px']


            cross = params['envelope_correlation_px'][0 if axis == 'row' else 1]
            lengths = (cross, along) if axis == 'row' else (along, cross)
            cv = params['along_modulation_cv']
            directed *= _envelope(height, width, cv, lengths, rng)
            directed /= np.sqrt(1+cv*cv)
            fraction = params['background_fraction']
            background = _colored_field(height, width, params['background_radial_psd'], rng)
            field = np.sqrt(1-fraction)*directed + np.sqrt(fraction)*background
        else:
            field, event_info = self._finite_events(params, axis, height, width, rng)
            fraction = params['background_fraction']
            if fraction:
                field += np.sqrt(fraction)*_colored_field(height, width, params['background_radial_psd'], rng)
        field = np.asarray(field, dtype=np.float32)
        field -= np.float32(field.mean(dtype=np.float64))
        if field.shape != (height, width) or not np.isfinite(field).all():
            raise ValueError('Invalid native operator sample')
        metadata = {'sampler_version': SAMPLER_VERSION, 'operator': name, 'kind': kind, 'axis': axis,
                    'source_id': record['source_id'], 'source_sha256': record['source_sha256'],
                    'fit_record_index': int(record_index), 'shape_HW': [height, width],
                    'field_rms': float(np.sqrt(np.mean(field.astype(np.float64)**2))),
                    'field_peak': float(np.max(np.abs(field))),
                    'zero_event_field': bool(kind in ('streak','streak_grain') and event_info['event_count'] == 0),
                    'final_RMS_normalized': False, 'resized': False, 'tiled': False,
                    'source_pixels_loaded': False, **event_info}
        return field, metadata

    @staticmethod
    def _finite_events(params, axis, height, width, rng):
        field = np.zeros((height, width), dtype=np.float32)
        expected = float(params['event_density_per_pixel']*height*width)
        count = int(rng.poisson(expected))
        events = params['events']
        lengths, widths, amplitudes = [], [], []
        positive = 0
        for _ in range(count):
            event = events[int(rng.integers(len(events)))]
            length = max(1, int(round(event['length_px'])))
            thickness = max(1, int(round(event['width_px'])))
            amplitude = float(event['amplitude'])*int(event['polarity'])
            cy, cx = int(rng.integers(height)), int(rng.integers(width))
            eh, ew = (thickness, length) if axis == 'row' else (length, thickness)
            y, x = cy-eh//2, cx-ew//2
            y0, x0, y1, x1 = max(y,0), max(x,0), min(y+eh,height), min(x+ew,width)


            cv = float(params.get('within_event_modulation_cv', 0.))
            if cv > 0 and length > 2:
                phase = np.linspace(0,1,length, dtype=np.float32)
                anchors = rng.standard_normal(4).astype(np.float32)
                mod = 1 + cv*np.interp(phase, np.linspace(0,1,4), anchors)
                visible = mod[x0-x:x1-x][None,:] if axis == 'row' else mod[y0-y:y1-y][:,None]
                field[y0:y1,x0:x1] += np.asarray(amplitude*visible, dtype=np.float32)
            else:
                field[y0:y1,x0:x1] += amplitude
            lengths.append(length); widths.append(thickness); amplitudes.append(abs(amplitude))
            positive += amplitude > 0
        return field, {'event_count': count, 'expected_event_count': expected,
                       'event_density_per_pixel': params['event_density_per_pixel'],
                       'sampled_length_mean_px': None if not count else float(np.mean(lengths)),
                       'sampled_width_mean_px': None if not count else float(np.mean(widths)),
                       'sampled_event_amplitude_mean': None if not count else float(np.mean(amplitudes)),
                       'sampled_positive_fraction': None if not count else positive/count,
                       'boundary_clipping_possible': True}
