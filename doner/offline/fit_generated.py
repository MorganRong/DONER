import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from .fit_operator_bank import main as fit_libraries


def fit_candidates(directory):
    directory = Path(directory).resolve()
    source = directory / 'guided_candidates'
    templates = json.loads(Path(__file__).with_name('admitted.json').read_text())
    for domain, selections in templates.items():
        destination = directory / 'bank' / 't2i' / domain
        textures = destination / 'textures'
        textures.mkdir(parents=True, exist_ok=True)
        records = []
        for selection in selections:
            identity = selection['id']
            metadata = json.loads((source / (identity + '.json')).read_text())
            if metadata['id'] != identity or metadata['recipe']['domain'] != domain:
                raise ValueError(f'Unexpected candidate identity: {identity}')
            spec = metadata['files']['texture']
            path = source / spec['path']
            if hashlib.sha256(path.read_bytes()).hexdigest() != spec['sha256']:
                raise ValueError(f'Changed generated texture: {path}')
            pixels = np.load(path, allow_pickle=False)
            if pixels.shape != (256, 256) or not np.isfinite(pixels).all():
                raise ValueError(f'Invalid generated texture: {path}')
            target = textures / f'{identity}.npy'
            shutil.copyfile(path, target)
            records.append({**selection, 'path': str(target.relative_to(destination)), 'sha256': spec['sha256']})
        manifest = {'schema_version':5, 'domain':domain, 'dataset_id':domain+'_hq50', 'records':records}
        (destination / 'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    fit_libraries(directory, directory)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    fit_candidates(args.root)


if __name__ == '__main__':
    main()
