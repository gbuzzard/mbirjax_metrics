"""Recon-export runner: reconstruct and SAVE VOLUMES (+ run logs) as slice_viewer h5.

For each (dataset, sequence) it runs a SINGLE continuous recon to max(snapshots) iterations
via the checkpointed vcd_recon (error-sinogram + Hessian preserved, global-RNG subset stream
NOT re-seeded -> bit-continuous), saving the recon at each snapshot iteration.  So N snapshots
cost one max(snapshots)-iteration run, not sum(snapshots).  No reference/NRMSE -- just recons
+ logs for visual inspection.

Each output h5 has dataset 'recon' with recon_dict attributes (incl. 'recon_log', the captured
run log) via TomographyModel.save_recon_hdf5 -> _write_hdf5_streaming; loadable by
mbirjax.viewer.slice_viewer / load_recon_hdf5.

Config: the `recon_exports.<name>` block in config.yaml.  Run (on the cluster, prerelease):
    PS_EXPORT=scale1k_snapshots python recon_and_save.py            # all datasets
    PS_EXPORT=... PS_DATASETS=<tag> python recon_and_save.py        # one dataset (parallel jobs)
    PS_TRIAL=1 PS_OUT_DIR=<scratch> python recon_and_save.py        # synthetic correctness trial
"""
import json
import os
import subprocess
import sys
import time

import ps_config

CFG = ps_config.load()
EXPORT_NAME = os.environ.get('PS_EXPORT') or next(iter(CFG['recon_exports']))
EXPORT = CFG['recon_exports'][EXPORT_NAME]
CACHE_DIR = CFG['cache_dir']
OUT_DIR = os.environ.get('PS_OUT_DIR', EXPORT['out_dir'])
SEQUENCES = EXPORT['sequences']
SNAPSHOTS = EXPORT['snapshots']
SEED = EXPORT.get('seed', 0)
DATASETS = os.environ['PS_DATASETS'].split(',') if os.environ.get('PS_DATASETS') else EXPORT['datasets']

_ROLE, _JOB = 'PS_EXPORT_ROLE', 'PS_EXPORT_JOB'


# ----------------------------------------------------------------------------------
# Shared recon setup + chunked continuous stepping (mirrors run_study's proven worker)
# ----------------------------------------------------------------------------------
def _setup(tag, seq, seed, max_iters):
    import numpy as np
    import mbirjax as mj
    import mbirjax.preprocess as mjp
    sino, gp, op, _ = mjp.load_preprocessing(os.path.join(CACHE_DIR, f'{tag}.h5'))
    sidecar = json.load(open(os.path.join(CACHE_DIR, f'{tag}.json')))
    model = getattr(mj, sidecar['model_class'])(**gp)
    if op:
        model.set_params(**op)
    if sidecar['auto_set_recon_geometry']:
        model.auto_set_recon_geometry()
    model.set_params(**sidecar['recon_settings'])
    model.set_params(partition_sequence=seq)
    weights = mj.gen_weights(sino, weight_type='transmission_root')
    recon_shape = model.get_params('recon_shape')
    # Production semantics: partitions generated ONCE under the seed; per-iteration subset
    # permutations then continue the global-RNG stream unbroken across chunks (no re-seeding).
    np.random.seed(seed)
    partitions = mj.gen_set_of_pixel_partitions(
        recon_shape, model.get_params('granularity'),
        output_device=model.recon_placement.devices[0],
        use_ror_mask=model.get_params('use_ror_mask'))
    seq_ext = np.asarray(mj.gen_partition_sequence(seq, max_iterations=max_iters))
    model._log_run_header(0, '~/.mbirjax/logs/recon.log', print_logs=False)
    model.auto_set_regularization_params(sino, weights=weights)
    return model, sino, weights, partitions, seq_ext, recon_shape


def _step_to(model, sino, weights, partitions, seq_ext, boundaries):
    """Step the recon through the sorted `boundaries` (cumulative iteration counts), yielding
    (iteration, recon_device, elapsed_s) at each -- one continuous checkpointed run."""
    recon_dev, ckpt, total, prev = None, {}, 0.0, 0
    for b in boundaries:
        t0 = time.perf_counter()
        recon_dev, _losses, ckpt = model.vcd_recon(
            sino, partitions, seq_ext[prev:b], 0.0, weights=weights,      # stop_pct 0 -> full run
            init_recon=recon_dev, first_iteration=prev,
            init_error_sinogram=ckpt.get('error_sinogram'),
            fm_hessian=ckpt.get('fm_hessian'), return_checkpoint=True)
        total += time.perf_counter() - t0
        prev = b
        yield b, recon_dev, total


# ----------------------------------------------------------------------------------
# Worker: one (dataset, sequence) -> snapshots
# ----------------------------------------------------------------------------------
def worker():
    job = json.loads(os.environ[_JOB])
    import numpy as np
    tag, seqname, seq = job['dataset'], job['seqname'], job['sequence']
    snaps, seed = job['snapshots'], job['seed']
    model, sino, weights, partitions, seq_ext, rs = _setup(tag, seq, seed, max(snaps))
    os.makedirs(OUT_DIR, exist_ok=True)
    for it, recon_dev, elapsed in _step_to(model, sino, weights, partitions, seq_ext, snaps):
        recon = np.asarray(recon_dev)[:, :, :rs[2]]                       # gather + crop pad slices
        info = {'dataset': tag, 'sequence': seq, 'sequence_name': seqname, 'iterations': it,
                'seed': seed, 'stop_threshold_change_pct': 0.0,
                'weights': 'transmission_root', 'branch': 'prerelease'}
        recon_dict = model.get_recon_dict(
            recon_params=info, notes=f'recon-export {tag} seq={seqname}{seq} iter={it}')
        out = os.path.join(OUT_DIR, f'{tag}_{seqname}_iter{it}.h5')
        model.save_recon_hdf5(out, recon, recon_dict)
        print(f'  saved {out}  (iter {it}, cum t={elapsed:.0f}s, shape {recon.shape})', flush=True)
    print(f'WORKER {tag} {seqname}: done', flush=True)


def run_job(job):
    print(f'--- export {job["dataset"]} seq={job["seqname"]}{job["sequence"]} '
          f'snaps={job["snapshots"]} ---', flush=True)
    env = dict(os.environ, **{_ROLE: 'worker', _JOB: json.dumps(job), 'PS_EXPORT': EXPORT_NAME})
    r = subprocess.run([sys.executable, os.path.abspath(__file__)], env=env)
    if r.returncode != 0:
        raise RuntimeError(f'{job["dataset"]} {job["seqname"]} failed (exit {r.returncode})')


def orchestrator():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f'=== recon_export {EXPORT_NAME}: datasets={DATASETS} seqs={list(SEQUENCES)} '
          f'snapshots={SNAPSHOTS} -> {OUT_DIR} ===', flush=True)
    for tag in DATASETS:
        for seqname, seq in SEQUENCES.items():
            run_job({'dataset': tag, 'seqname': seqname, 'sequence': seq,
                     'snapshots': SNAPSHOTS, 'seed': SEED})


# ----------------------------------------------------------------------------------
# Trial: verify the snapshot-restart is bit-continuous, on synthetic
# ----------------------------------------------------------------------------------
def trial():
    import numpy as np
    import mbirjax as mj
    tag, seq, seed, snaps = 'synthetic', [0, 2, 4, 6, 7], 0, [5, 10, 15]
    os.makedirs(OUT_DIR, exist_ok=True)

    # (a) chunked WITH snapshot saves
    model, sino, w, part, seqx, rs = _setup(tag, seq, seed, max(snaps))
    chunk_final, saved = None, []
    for it, recon_dev, _ in _step_to(model, sino, w, part, seqx, snaps):
        chunk_final = np.asarray(recon_dev)[:, :, :rs[2]]
        rd = model.get_recon_dict(recon_params={'iterations': it}, notes=f'trial iter{it}')
        out = os.path.join(OUT_DIR, f'TRIAL_{tag}_iter{it}.h5')
        model.save_recon_hdf5(out, chunk_final, rd)
        saved.append(out)

    # (b) monolithic 0->max in one call (fresh model, same seed => same partitions + RNG start)
    m2, s2, w2, p2, sx2, rs2 = _setup(tag, seq, seed, max(snaps))
    recon_dev2, _l, _c = m2.vcd_recon(s2, p2, sx2[0:max(snaps)], 0.0, weights=w2,
                                      first_iteration=0, return_checkpoint=True)
    mono = np.asarray(recon_dev2)[:, :, :rs2[2]]

    rel = float(np.max(np.abs(chunk_final - mono)) / max(np.max(np.abs(mono)), 1e-12))

    # (c) h5 round-trip through the loader slice_viewer uses
    r_load, d_load = mj.TomographyModel.load_recon_hdf5(saved[-1])
    has_log = 'recon_log' in d_load and len(str(d_load['recon_log'])) > 0
    shape_ok = tuple(np.asarray(r_load).shape) == tuple(mono.shape)

    print(f'TRIAL chunked-vs-monolithic rel_max {rel:.2e} (expect fp-noise, ~<=1e-3)', flush=True)
    print(f'TRIAL h5 round-trip {os.path.basename(saved[-1])}: shape {np.asarray(r_load).shape}, '
          f'recon_log present={has_log}', flush=True)
    ok = rel <= 1e-3 and has_log and shape_ok
    print(f'TRIAL: {"PASS" if ok else "FAIL"}', flush=True)
    for p in saved:
        os.remove(p)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    if os.environ.get('PS_TRIAL'):
        trial()
    elif os.environ.get(_ROLE) == 'worker':
        worker()
    else:
        orchestrator()
