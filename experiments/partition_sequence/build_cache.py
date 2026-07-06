"""Preprocess each dataset ONCE and cache to disk (config-driven, build-if-missing).

Reads config.yaml (via ps_config): `cache_dir` and the `datasets` block.  Each dataset is
loaded with its production loader, subsampled by its OWN detector_factor / view_factor,
optionally cropped/aligned exactly as its production script does, and cached via the
library's two-stage-workflow format (mjp.save_preprocessing -> one HDF5 with sinogram +
geometry and optional params) plus a small JSON sidecar for what that format excludes:

    cache_dir/<tag>.h5      sinogram + geometry/optional params (mjp.load_preprocessing)
    cache_dir/<tag>.json    model class, auto_set_recon_geometry flag, recon settings
                            (held CONSTANT across the study), provenance

run_study.py rebuilds the model from these; weights are regenerated there (cheap,
deterministic), NOT cached.

BUILD-IF-MISSING: an existing <tag>.h5 in cache_dir is reused (the shared depot cache means
the common case does zero preprocessing).  Pass --force to rebuild, or a list of tags to
restrict which datasets are (re)built.

Run on the cluster (data paths are /depot):
    python build_cache.py                 # build any missing dataset in the config
    python build_cache.py z62_v4x_d4x_nv201_nch512   # only this tag
    python build_cache.py --force z62_v4x_d4x_nv201_nch512  # rebuild even if cached
"""
import json
import os
import sys
import time

import numpy as np
import mbirjax as mj                       # must precede jax (env binding)
import mbirjax.preprocess as mjp

import ps_config

CFG = ps_config.load()
CACHE_DIR = CFG['cache_dir']


def save_cache(tag, sino, geometry_params, optional_params, sidecar):
    os.makedirs(CACHE_DIR, exist_ok=True)
    h5_path = os.path.join(CACHE_DIR, f'{tag}.h5')
    mjp.save_preprocessing(h5_path, sino, geometry_params, optional_params)
    with open(os.path.join(CACHE_DIR, f'{tag}.json'), 'w') as f:
        json.dump(sidecar, f, indent=1)
    print(f'[{tag}] cached: sino {sino.shape} '
          f'({sino.size * 4 / 1e9:.2f} GB) -> {h5_path} (+ .json sidecar)', flush=True)


def build_nsi(tag, spec):
    df, vf = spec['detector_factor'], spec['view_factor']
    rs = dict(spec['recon_settings'])
    sino, cone_beam_params, optional_params = mjp.nsi.compute_sino_and_params(
        spec['path'], downsample_factor=[df, df], subsample_view_factor=vf)
    if spec.get('auto_crop', True):
        sino, cone_beam_params, optional_params = mjp.auto_crop_sino_conebeam(
            sino, cone_beam_params, optional_params)
    sino = np.maximum(sino, 0.0)          # host clip, as in the production script
    rs.setdefault('positivity_flag', True)
    sidecar = {
        'model_class': 'ConeBeamModel',
        'auto_set_recon_geometry': False,
        'recon_settings': rs,
        'provenance': {'source': spec['path'], 'detector_factor': df,
                       'view_factor': vf, 'built': time.strftime('%Y-%m-%d')},
    }
    save_cache(tag, sino, cone_beam_params, optional_params, sidecar)


def build_zeiss(tag, spec):
    df, vf = spec['detector_factor'], spec['view_factor']
    sino, geometry_params, optional_params, metadata = mjp.zeiss.compute_sino_and_params(
        spec['path'], downsample_factor=(df, df), subsample_view_factor=vf)
    model_class = 'ParallelBeamModel' if metadata['scanner_type'] == 'ultra' else 'ConeBeamModel'
    # Alignment (SiC) needs a live model + direct recon; cache the ALIGNED sinogram so
    # run_study never repeats this step.
    if spec.get('view_alignment', False):
        model = getattr(mj, model_class)(**geometry_params)
        model.set_params(**optional_params)
        model.auto_set_recon_geometry()
        direct = model.direct_recon(sino)
        sino = mjp.align_sino_views(model, sino, direct)
        del model, direct
    sidecar = {
        'model_class': model_class,
        'auto_set_recon_geometry': True,  # the zeiss script resets recon geometry
        'recon_settings': dict(spec['recon_settings']),
        'provenance': {'source': spec['path'], 'detector_factor': df, 'view_factor': vf,
                       'aligned': spec.get('view_alignment', False),
                       'built': time.strftime('%Y-%m-%d')},
    }
    save_cache(tag, sino, geometry_params, optional_params, sidecar)


def build_synthetic(tag, spec):
    n, num_views = spec['size'], spec['num_views']
    angles = np.linspace(0, np.pi, num_views, endpoint=False)
    model = mj.ConeBeamModel((num_views, n, n), angles,
                             source_detector_dist=4.0 * n, source_iso_dist=2.0 * n)
    phantom = mj.gen_cube_phantom(model.get_params('recon_shape'))
    sino = np.asarray(model.forward_project(phantom), dtype=np.float32)
    rng = np.random.default_rng(0)
    sino = sino + (0.01 * sino.max() * rng.standard_normal(sino.shape)).astype(np.float32)
    sino = np.maximum(sino, 0.0)
    geometry_params = {'sinogram_shape': (num_views, n, n), 'angles': angles,
                       'source_detector_dist': 4.0 * n, 'source_iso_dist': 2.0 * n}
    sidecar = {
        'model_class': 'ConeBeamModel',
        'auto_set_recon_geometry': False,
        'recon_settings': dict(spec['recon_settings']),
        'provenance': {'source': 'synthetic cube phantom', 'built': time.strftime('%Y-%m-%d')},
    }
    save_cache(tag, sino, geometry_params, {}, sidecar)


BUILDERS = {'nsi': build_nsi, 'zeiss': build_zeiss, 'synthetic': build_synthetic}


def main(argv):
    force = '--force' in argv
    tags = [a for a in argv if not a.startswith('--')] or list(CFG['datasets'])
    for tag in tags:
        if tag not in CFG['datasets']:
            print(f'[{tag}] SKIPPED (not in config datasets)', flush=True)
            continue
        h5_path = os.path.join(CACHE_DIR, f'{tag}.h5')
        if os.path.exists(h5_path) and not force:
            print(f'=== {tag}: reusing cached {h5_path} (--force to rebuild) ===', flush=True)
            continue
        print(f'=== building {tag} ===', flush=True)
        spec = CFG['datasets'][tag]
        try:
            BUILDERS[spec['loader']](tag, spec)
        except FileNotFoundError as e:
            print(f'[{tag}] SKIPPED (data not found): {e}', flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
