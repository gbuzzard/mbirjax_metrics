"""Partition-sequence study runner (config-driven): references, noise floor, sweep.

Reads config.yaml (via ps_config): `cache_dir`, `output_dir`, and one named `experiment`
(select with PS_EXPERIMENT=<name>; default is config's default_experiment).  Runs one recon
per (dataset, job) in a FRESH SUBPROCESS (honest peak memory; JAX-free orchestrator).  Each
worker reproduces PRODUCTION VCD semantics exactly: partitions are generated ONCE under the
seed (so they match across chunks and candidates), then the recon is stepped in CHUNKS via
vcd_recon with NO re-seeding, so the per-iteration subset-permutation stream evolves exactly
as in a monolithic recon() call.  (Re-seeding every iteration freezes the permutation and
systematically slows convergence -- the fixed order is a real optimization handicap, not
noise.)  Per iteration it records masked NRMSE vs the dataset's reference recon, the native
change %, and wall time.  Results land in output_dir as one JSON per run plus, when relevant,
the reference recons (.npy) and a <tag>_floor.json summary.

Phases (set per experiment; run 'reference' first -- 'sweep'/'noise_floor' need it):
  reference   -- default sequence to a tight threshold; saves recon_<tag>_reference.npy
  chunk_check -- chunked-vs-monolithic sanity gate (same seed, N iters both ways)
  noise_floor -- reference config x noise_floor_seeds at fixed iterations: the partition-
                 choice variability every candidate separation must beat
  sweep       -- all candidates x datasets

Run (on the cluster):
    python run_study.py                       # config's default_experiment
    PS_EXPERIMENT=tail_4x4 python run_study.py
"""
import json
import os
import subprocess
import sys
import time

import ps_config

CFG = ps_config.load()
EXP = ps_config.experiment(CFG)
CACHE_DIR = CFG['cache_dir']
OUTPUT_DIR = CFG['output_dir']
CANDIDATES = EXP['candidates']
DATASETS = EXP['datasets']
PHASES = EXP['phases']

_ROLE_ENV = 'PS_STUDY_ROLE'
_JOB_ENV = 'PS_STUDY_JOB'                # JSON job spec passed to the worker


# ----------------------------------------------------------------------------------
# Worker: one recon (chunked), one subprocess
# ----------------------------------------------------------------------------------
def worker():
    job = json.loads(os.environ[_JOB_ENV])
    import numpy as np
    import mbirjax as mj                 # must precede jax (env binding)
    import jax
    import mbirjax.preprocess as mjp

    tag = job['dataset']
    sino, geometry_params, optional_params, _ = mjp.load_preprocessing(
        os.path.join(CACHE_DIR, f'{tag}.h5'))
    with open(os.path.join(CACHE_DIR, f'{tag}.json')) as f:
        sidecar = json.load(f)

    model = getattr(mj, sidecar['model_class'])(**geometry_params)
    if optional_params:
        model.set_params(**optional_params)
    if sidecar['auto_set_recon_geometry']:
        model.auto_set_recon_geometry()
    model.set_params(verbose=0, **sidecar['recon_settings'])
    model.set_params(partition_sequence=job['sequence'])
    weights = mj.gen_weights(sino, weight_type='transmission_root')

    # NRMSE mask: RoR ellipse + drop drop_slice_fraction of slices at each end.  Host-side.
    recon_shape = model.get_params('recon_shape')
    ror = mj.get_2d_ror_mask(recon_shape)[:, :, None]
    drop = max(1, int(round(EXP['drop_slice_fraction'] * recon_shape[2])))
    reference = None
    if job.get('reference_path'):
        reference = np.load(job['reference_path'])

    def masked_nrmse(recon):
        a = (recon * ror)[:, :, drop:-drop]
        b = (reference * ror)[:, :, drop:-drop]
        return float(np.linalg.norm(a - b) / np.linalg.norm(b))

    seed, chunk, max_iters = job['seed'], job['chunk'], job['max_iterations']
    stop_pct = job['stop_pct']

    # Production-semantics setup, mirroring TomographyModel.initialize_recon: partitions
    # generated ONCE under the seed (fixed across chunks; matched across candidates),
    # regularization set once, the sequence extended by repeating its last entry.
    np.random.seed(seed)
    use_ror_mask = model.get_params('use_ror_mask')
    granularity = model.get_params('granularity')
    partitions = mj.gen_set_of_pixel_partitions(
        recon_shape, granularity, output_device=model.recon_placement.devices[0],
        use_ror_mask=use_ror_mask)
    seq_ext = np.asarray(mj.gen_partition_sequence(job['sequence'], max_iterations=max_iters))
    # vcd_recon needs the run logger that recon()/initialize_recon normally sets up.
    model._log_run_header(0, '~/.mbirjax/logs/recon.log', print_logs=False)
    model.auto_set_regularization_params(sino, weights=weights)

    rows, recon_dev, ckpt, total_time = [], None, {}, 0.0
    for k0 in range(0, max_iters, chunk):
        k1 = min(k0 + chunk, max_iters)
        # NO re-seeding here: the per-iteration subset permutations must continue the
        # global-RNG stream exactly as a monolithic run would.  The checkpoint round-trip
        # (error sinogram + Hessian) makes each restart nearly free.
        t0 = time.perf_counter()
        recon_dev, loss_vectors, ckpt = model.vcd_recon(
            sino, partitions, seq_ext[k0:k1], stop_pct, weights=weights,
            init_recon=recon_dev, first_iteration=k0,
            init_error_sinogram=ckpt.get('error_sinogram'),
            fm_hessian=ckpt.get('fm_hessian'), return_checkpoint=True)
        total_time += time.perf_counter() - t0
        # vcd_recon returns the DEVICE form (slice axis possibly padded with inert zeros);
        # gather to host and crop to the real slice count for the NRMSE and the saved recon.
        recon = np.asarray(recon_dev)[:, :, :recon_shape[2]]
        changes = [100.0 * float(v) for v in loss_vectors[2]]   # nmae_update, fractional
        for i, change in enumerate(changes):
            rows.append({'iteration': k0 + i + 1, 'change_pct': change,
                         'time_s': total_time,
                         'nrmse_vs_ref': masked_nrmse(recon) if reference is not None else None})
        print(f'  iter {k0 + len(changes):3d}: change {changes[-1]:.4f}%  t={total_time:7.1f}s'
              + (f'  nrmse {rows[-1]["nrmse_vs_ref"]:.5f}' if reference is not None else ''),
              flush=True)
        if changes[-1] < stop_pct:
            break

    peaks = [d.memory_stats()['peak_bytes_in_use'] / 2**30 for d in jax.local_devices()
             if d.memory_stats() and 'peak_bytes_in_use' in (d.memory_stats() or {})]
    # sino_shape/recon_shape travel with the run so build_page auto-derives them (no config).
    result = {'label': job['label'], 'dataset': tag, 'sequence': job['sequence'],
              'seed': seed, 'chunk': chunk, 'rows': rows, 'total_time_s': total_time,
              'peak_gib_per_device': peaks,
              'sino_shape': list(sino.shape), 'recon_shape': list(recon_shape)}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, f'{job["label"]}.json'), 'w') as f:
        json.dump(result, f, indent=1)
    if job.get('save_recon_path'):
        np.save(job['save_recon_path'], recon)
    print(f'WORKER {job["label"]}: {len(rows)} iters, {total_time:.1f} s, '
          f'peaks {peaks}', flush=True)


# ----------------------------------------------------------------------------------
# Orchestrator: JAX-free
# ----------------------------------------------------------------------------------
def run_job(job):
    print(f'--- {job["label"]} (dataset={job["dataset"]}, seq={job["sequence"]}, '
          f'seed={job["seed"]}, chunk={job["chunk"]}, max={job["max_iterations"]}) ---',
          flush=True)
    env = dict(os.environ, **{_ROLE_ENV: 'worker', _JOB_ENV: json.dumps(job),
                              'PS_EXPERIMENT': EXP['name']})
    result = subprocess.run([sys.executable, os.path.abspath(__file__)], env=env)
    if result.returncode != 0:
        raise RuntimeError(f'{job["label"]} failed (exit {result.returncode})')


def ref_path(tag):
    return os.path.join(OUTPUT_DIR, f'recon_{tag}_reference.npy')


def orchestrator():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f'=== experiment {EXP["name"]}: datasets={DATASETS} phases={PHASES} '
          f'candidates={list(CANDIDATES)} ===', flush=True)
    save_recons = EXP['save_recons']
    for tag in DATASETS:
        if 'reference' in PHASES:
            run_job({'label': f'{tag}_reference', 'dataset': tag,
                     'sequence': CANDIDATES['default'], 'seed': EXP['seed'],
                     'chunk': EXP['reference_chunk'],
                     'max_iterations': EXP['reference_max_iterations'],
                     'stop_pct': EXP['reference_stop_pct'], 'save_recon_path': ref_path(tag)})
        if 'chunk_check' in PHASES:
            # Same seed, chunked vs one-big-chunk: with production permutation semantics the
            # two runs use the SAME partitions and permutation stream, so they must agree to
            # iterated-fp noise (~1e-4 class).  The time ratio is the per-chunk restart cost.
            n = EXP['chunk_check_iterations']
            for label, chunk in [(f'{tag}_chunk1', EXP['chunk']), (f'{tag}_mono', n)]:
                run_job({'label': label, 'dataset': tag,
                         'sequence': CANDIDATES['default'], 'seed': EXP['seed'], 'chunk': chunk,
                         'max_iterations': n, 'stop_pct': 0,
                         'reference_path': ref_path(tag) if os.path.exists(ref_path(tag)) else None,
                         'save_recon_path': os.path.join(OUTPUT_DIR, f'recon_{label}.npy')})
        if 'noise_floor' in PHASES:
            for s in EXP['noise_floor_seeds']:
                run_job({'label': f'{tag}_floor_seed{s}', 'dataset': tag,
                         'sequence': CANDIDATES['default'], 'seed': s,
                         'chunk': EXP['noise_floor_iterations'],
                         'max_iterations': EXP['noise_floor_iterations'], 'stop_pct': 0,
                         'reference_path': ref_path(tag),
                         'save_recon_path': os.path.join(OUTPUT_DIR, f'recon_{tag}_floor_seed{s}.npy')})
        if 'sweep' in PHASES:
            for name, seq in CANDIDATES.items():
                job = {'label': f'{tag}_{name}', 'dataset': tag, 'sequence': seq,
                       'seed': EXP['seed'], 'chunk': EXP['chunk'],
                       'max_iterations': EXP['sweep_max_iterations'],
                       'stop_pct': EXP['sweep_stop_pct'], 'reference_path': ref_path(tag)}
                if save_recons:
                    job['save_recon_path'] = os.path.join(OUTPUT_DIR, f'recon_{tag}_{name}.npy')
                run_job(job)
    summarize()


def summarize():
    """Console summary + a <tag>_floor.json build_page reads for the auto-derived floor."""
    import glob
    import numpy as np
    print('\n=== SUMMARY ===')
    targets = EXP['nrmse_targets']
    header = f'{"run":28s} {"iters":>5s} {"time_s":>8s} {"peakGiB":>8s}' + ''.join(
        f'  it@{t:g}/t@{t:g}' for t in targets)
    print(header)
    for path in sorted(glob.glob(os.path.join(OUTPUT_DIR, '*.json'))):
        with open(path) as f:
            r = json.load(f)
        if 'rows' not in r:                # skip the floor summary json
            continue
        cells = []
        for target in targets:
            hit = next((row for row in r['rows']
                        if row['nrmse_vs_ref'] is not None and row['nrmse_vs_ref'] <= target),
                       None)
            cells.append(f'  {hit["iteration"]:3d}/{hit["time_s"]:6.1f}' if hit else '     --/  --  ')
        peak = max(r['peak_gib_per_device']) if r['peak_gib_per_device'] else float('nan')
        print(f'{r["label"]:28s} {len(r["rows"]):5d} {r["total_time_s"]:8.1f} {peak:8.2f}'
              + ''.join(cells))
    # Chunk check: chunked and monolithic runs (same seed) must agree to fp noise.
    for tag in DATASETS:
        c_path = os.path.join(OUTPUT_DIR, f'recon_{tag}_chunk1.npy')
        m_path = os.path.join(OUTPUT_DIR, f'recon_{tag}_mono.npy')
        if os.path.exists(c_path) and os.path.exists(m_path):
            c, m = np.load(c_path), np.load(m_path)
            rel = float(np.max(np.abs(c - m)) / np.max(np.abs(m)))
            times = {}
            for kind in ('chunk1', 'mono'):
                with open(os.path.join(OUTPUT_DIR, f'{tag}_{kind}.json')) as f:
                    times[kind] = json.load(f)['total_time_s']
            print(f'{tag}: chunk check rel_max {rel:.1e} (expect iterated-fp class, ~1e-4); '
                  f'restart overhead x{times["chunk1"] / times["mono"]:.2f} '
                  f'(chunked {times["chunk1"]:.1f}s vs one-chunk {times["mono"]:.1f}s)')
    # Noise floor: pairwise NRMSE between the floor-seed recons; also persist for build_page.
    for tag in DATASETS:
        recons = sorted(glob.glob(os.path.join(OUTPUT_DIR, f'recon_{tag}_floor_seed*.npy')))
        if len(recons) >= 2:
            arrs = [np.load(p) for p in recons]
            pair = [float(np.linalg.norm(a - b) / np.linalg.norm(b))
                    for i, a in enumerate(arrs) for b in arrs[i + 1:]]
            median = sorted(pair)[len(pair) // 2]
            print(f'{tag}: partition-noise floor (pairwise NRMSE over {len(arrs)} seeds, '
                  f'{EXP["noise_floor_iterations"]} iters): '
                  f'min {min(pair):.5f}  median {median:.5f}  max {max(pair):.5f}')
            with open(os.path.join(OUTPUT_DIR, f'{tag}_floor.json'), 'w') as f:
                json.dump({'dataset': tag, 'floor_median': median, 'floor_min': min(pair),
                           'floor_max': max(pair), 'n_seeds': len(arrs),
                           'iterations': EXP['noise_floor_iterations']}, f, indent=1)


if __name__ == '__main__':
    if os.environ.get(_ROLE_ENV) == 'worker':
        worker()
    else:
        orchestrator()
