"""Shared config loader for the partition-sequence pipeline (build_cache / run_study /
build_page).  All three read config.yaml through here so a dataset/experiment is described
ONCE.  Config path: $PS_CONFIG, else config.yaml next to this file.
"""
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path=None):
    path = path or os.environ.get('PS_CONFIG') or os.path.join(HERE, 'config.yaml')
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # Env overrides for the two storage paths -- lets a teammate point build_cache/run_study
    # at local scratch (e.g. the synthetic smoke test) without editing the shared config.
    cfg['cache_dir'] = os.environ.get('PS_CACHE_DIR', cfg['cache_dir'])
    cfg['output_dir'] = os.environ.get('PS_OUTPUT_DIR', cfg['output_dir'])
    return cfg


def experiment(cfg, name=None):
    """Resolve one experiment into a flat params dict: defaults overlaid by the experiment
    entry.  `candidates`, `datasets`, and `phases` come from the experiment; every scalar
    study param falls back to `defaults`.
    """
    name = name or os.environ.get('PS_EXPERIMENT') or cfg['default_experiment']
    if name not in cfg['experiments']:
        raise KeyError(f'experiment {name!r} not in config (have: '
                       f'{sorted(cfg["experiments"])})')
    exp = dict(cfg['defaults'])
    exp.update(cfg['experiments'][name])
    exp['name'] = name
    return exp
