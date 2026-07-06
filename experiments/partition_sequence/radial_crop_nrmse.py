"""Offline: how much does the FoV-edge flash inflate the study NRMSE?  Recompute masked
NRMSE for the flat-[7] snapshots at several radial crops (numpy only, no jax)."""
import numpy as np

SNAP = '/scratch/gautschi/buzzard/ps_study/quality_pngs'
REF  = '/scratch/gautschi/buzzard/ps_study/ps_results_4x4'
FRACS = [0.0, 0.05, 0.10, 0.15, 0.20]     # radial crop fraction (0 = current full RoR)

def ror(rows, cols, frac):
    rc, cc = (rows-1)/2, (cols-1)/2
    rr, cr = rc - int(rc*frac), cc - int(cc*frac)
    y, x = np.ogrid[:rows, :cols]
    return ((x-cc)/cr)**2 + ((y-rc)/rr)**2 <= 1.0

def nrmse(a, b, mask2d, drop):
    aa = a[:, :, drop:-drop] * mask2d[:, :, None]
    bb = b[:, :, drop:-drop] * mask2d[:, :, None]
    return float(np.linalg.norm(aa-bb) / np.linalg.norm(bb))

DATASETS = {
  'z62_v4x_d4x_nv201_nch512': [('iter15','snap_iter15'), ('0.2% (it44)','snap_stop02_iter44'), ('0.1% (it60)','snap_stop01_iter60')],
  'sic_v4x_d4x_nv401_nch512': [('iter15','snap_iter15'), ('0.2% (it47)','snap_stop02_iter47'), ('0.1% (it88)','snap_stop01_iter88')],
}
for ds, snaps in DATASETS.items():
    ref = np.load(f'{REF}/recon_{ds}_reference.npy')
    rows, cols, sl = ref.shape
    drop = max(1, int(0.05*sl))
    masks = {f: ror(rows, cols, f) for f in FRACS}
    frac_area = {f: masks[f].sum()/masks[0.0].sum() for f in FRACS}
    print(f"\n=== {ds}  (recon {ref.shape}, end-slice drop {drop}) ===")
    print("crop frac ->      " + "".join(f"{f:>10.0%}" for f in FRACS))
    print("   (RoR area kept) " + "".join(f"{frac_area[f]:>10.0%}" for f in FRACS))
    for label, key in snaps:
        rec = np.load(f'{SNAP}/recon_{ds}_{key}.npy')
        vals = [nrmse(rec, ref, masks[f], drop) for f in FRACS]
        print(f"{label:16s} " + "".join(f"{v:>10.4f}" for v in vals))
