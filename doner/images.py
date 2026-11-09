import csv
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


def read_image(path):
    path = Path(path)
    if path.suffix.lower() == '.npy':
        value = np.load(path, allow_pickle=False)
        if np.issubdtype(value.dtype, np.integer):
            value = value.astype(np.float32) / np.iinfo(value.dtype).max
    else:
        with Image.open(path) as image:
            pixels = np.asarray(image)
            if pixels.ndim == 3 and pixels.shape[2] == 3 and np.all(pixels == pixels[..., :1]):
                pixels = pixels[..., 0]
            if pixels.ndim != 2 or pixels.dtype != np.uint8:
                raise ValueError('Use grayscale 8-bit PNG/TIFF or a floating point .npy array.')
            value = np.asarray(pixels, dtype=np.float32) / np.float32(255)
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 2 or min(value.shape) < 32 or not np.isfinite(value).all():
        raise ValueError('Expected a finite, single-channel image of at least 32 x 32.')
    return np.ascontiguousarray(value)


def load_training_images(directory, domain):
    directory = Path(directory).resolve()
    index = directory.parents[1] / 'index.csv'
    with index.open(newline='', encoding='utf-8') as handle:
        records = [r for r in csv.DictReader(handle) if r['domain'] == domain and r['split'] == 'train']
    records.sort(key=lambda row: int(row['split_order']))
    if [int(r['split_order']) for r in records] != list(range(50)) or len({r['source_id'] for r in records}) != 50:
        raise ValueError('Expected exactly 50 distinct, ordered training references in data/index.csv')
    images = []
    for row in records:
        path = (index.parent.parent / row['path']).resolve()
        if path.parent != directory:
            raise ValueError(f'Training image outside selected directory: {path}')
        value = read_image(path)
        if hashlib.sha256(value.tobytes()).hexdigest() != row['float32_pixel_sha256']:
            raise ValueError(f'Training pixels do not match index: {path.name}')
        images.append(value)
    return records, images
