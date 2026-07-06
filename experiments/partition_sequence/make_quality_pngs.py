"""Run flat [7] recon, snapshot at iter 15 / 0.2%-stop / 0.1%-stop, montage vs the converged
reference — to judge how converged the default stop needs to be.  PNGs to quality_pngs/."""
import os, json
import numpy as np
import mbirjax as mj                       # before jax
import mbirjax.preprocess as mjp
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

CACHE = '/scratch/gautschi/buzzard/ps_study/ps_cache'
RESULTS = '/scratch/gautschi/buzzard/ps_study/ps_results_4x4'
OUT = '/scratch/gautschi/buzzard/ps_study/quality_pngs'
os.makedirs(OUT, exist_ok=True)
DATASETS = ['z62_4x4', 'sic_4x4']
SEQ = [7]
MAX_ITERS = 95
THRESHOLDS = [0.2, 0.1]                     # change-% snapshot points

def masked_nrmse(a, b, ror, drop):
    aa = (a*ror)[:,:,drop:-drop]; bb = (b*ror)[:,:,drop:-drop]
    return float(np.linalg.norm(aa-bb)/np.linalg.norm(bb))

for ds in DATASETS:
    sino, gp, op, _ = mjp.load_preprocessing(f'{CACHE}/{ds}.h5')
    sc = json.load(open(f'{CACHE}/{ds}.json'))
    model = getattr(mj, sc['model_class'])(**gp)
    if op: model.set_params(**op)
    if sc['auto_set_recon_geometry']: model.auto_set_recon_geometry()
    model.set_params(verbose=0, partition_sequence=SEQ, **sc['recon_settings'])
    weights = mj.gen_weights(sino, weight_type='transmission_root')
    rs = model.get_params('recon_shape')
    ror = mj.get_2d_ror_mask(rs)[:,:,None]; drop = max(1, int(0.05*rs[2]))

    np.random.seed(0)
    parts = mj.gen_set_of_pixel_partitions(rs, model.get_params('granularity'),
              output_device=model.recon_placement.devices[0], use_ror_mask=model.get_params('use_ror_mask'))
    seq_ext = np.asarray(mj.gen_partition_sequence(SEQ, max_iterations=MAX_ITERS))
    model._log_run_header(0, '~/.mbirjax/logs/recon.log', print_logs=False)
    model.auto_set_regularization_params(sino, weights=weights)

    ref = np.load(f'{RESULTS}/recon_{ds}_reference.npy')
    snaps = []                              # (label, recon, iter, change)
    ckpt, recon_dev = {}, None
    thr = list(THRESHOLDS)
    for k in range(MAX_ITERS):
        recon_dev, lv, ckpt = model.vcd_recon(sino, parts, seq_ext[k:k+1], 0, weights=weights,
                    init_recon=recon_dev, first_iteration=k,
                    init_error_sinogram=ckpt.get('error_sinogram'),
                    fm_hessian=ckpt.get('fm_hessian'), return_checkpoint=True)
        chg = 100*float(lv[2][-1]); rec = np.asarray(recon_dev)[:,:,:rs[2]]
        if k+1 == 15:
            snaps.append((f'iter 15 (chg {chg:.2f}%)', rec.copy(), 15, chg))
            np.save(f'{OUT}/recon_{ds}_snap_iter15.npy', rec)
        while thr and chg < thr[0]:
            t = thr.pop(0); snaps.append((f'stop {t}% (iter {k+1})', rec.copy(), k+1, chg))
            np.save(f'{OUT}/recon_{ds}_snap_stop{str(t).replace(".","")}_iter{k+1}.npy', rec)
        if not thr and k+1 > 15: break
    snaps.append(('reference (0.01%)', ref, '—', 0.01))

    z = rs[2]//2                            # central axial slice
    vals = ref[:,:,z][ror[:,:,0]]
    vmin, vmax = np.percentile(vals, 1), np.percentile(vals, 99)
    n = len(snaps)
    fig, ax = plt.subplots(2, n, figsize=(3.2*n, 6.6))
    for j,(lab,rec,it,chg) in enumerate(snaps):
        nr = masked_nrmse(rec, ref, ror, drop)
        ax[0,j].imshow(rec[:,:,z].T, cmap='gray', vmin=vmin, vmax=vmax)
        ax[0,j].set_title(f'{lab}\nNRMSE {nr:.3f}', fontsize=10); ax[0,j].axis('off')
        d = ((rec-ref)[:,:,z]*ror[:,:,0]).T
        m = ax[1,j].imshow(d, cmap='RdBu_r', vmin=-(vmax-vmin)*0.3, vmax=(vmax-vmin)*0.3)
        ax[1,j].set_title('difference from reference', fontsize=9); ax[1,j].axis('off')
    fig.suptitle(f'{ds}  —  flat [7] tail, convergence vs stop point (central axial slice)', fontsize=12)
    fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(f'{OUT}/{ds}_convergence.png', dpi=110); plt.close(fig)
    print(f'{ds}: wrote {OUT}/{ds}_convergence.png  ({n} panels)')
