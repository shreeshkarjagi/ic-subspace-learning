"""
Multi-seed LOO and LOPO for DAE baseline
"""
import os, json, time
import numpy as np
from multiprocessing import Pool

from forward_model import quantize, REGIMES
from method_dae import fit_dae, correct_dae
from run_loo import eval_one_spectrum, aggregate, _jsonable

Q = 0.11
DEFAULT_SEEDS = [42, 43, 44]


# ---------- fold workers (must be top-level for Pool) ----------

def _dae_loo_fold(args):
    """One recording-level LOO fold, DAE only."""
    fold_rec, allamps, qmat, freqs, rec_ids, seed = args
    test_mask = rec_ids == fold_rec
    train_mask = ~test_mask
    train_amps = allamps[train_mask]
    if train_amps.shape[0] < 20:
        return None
    try:
        sd = fit_dae(train_amps, Q, seed=seed)
    except Exception as e:
        return {'error': repr(e), 'indices': np.where(test_mask)[0].tolist()}
    test_idx = np.where(test_mask)[0]
    F = allamps.shape[1]
    records = []
    corrected = np.full((len(test_idx), F), np.nan)
    for local_i, si in enumerate(test_idx):
        truth = allamps[si]; qa = qmat[si]
        try:
            c = correct_dae(freqs, qa, Q, dae_state=sd)
        except Exception:
            c = qa.copy()
        records.append(eval_one_spectrum(freqs, truth, qa, c))
        corrected[local_i] = c
    return {
        'results': records,
        'corrected': corrected,
        'indices': test_idx.tolist(),
        'n_train': train_amps.shape[0],
        'val_mse': sd.get('val_mse'),
        'epochs_trained': sd.get('epochs_trained'),
    }


# ---------- per-seed runners ----------

def run_loo_one_seed(hemi_files, data_dir, seed, n_workers):
    """Returns per-hemisphere aggregates + cohort weighted mean for this seed."""
    per_hemi = {}
    for npz_file in hemi_files:
        key = npz_file.replace('.npz', '')
        d = np.load(os.path.join(data_dir, npz_file))
        freqs = d['freqs']; amps = d['amps']; rec_ids = d['recording_ids']
        if amps.shape[0] < 10:
            continue
        qmat = np.array([quantize(amps[i], Q) for i in range(amps.shape[0])])
        unique_recs = np.unique(rec_ids)
        fold_args = [(rec, amps, qmat, freqs, rec_ids, seed) for rec in unique_recs]

        if n_workers > 1 and len(unique_recs) > 1:
            with Pool(min(n_workers, len(unique_recs))) as pool:
                fold_outs = pool.map(_dae_loo_fold, fold_args)
        else:
            fold_outs = [_dae_loo_fold(a) for a in fold_args]

        records = []
        for fo in fold_outs:
            if fo is None or 'error' in fo:
                continue
            records.extend(fo['results'])
        if not records:
            continue
        per_hemi[key] = aggregate({'dae': records})['dae']
    return per_hemi


def run_lopo_one_seed(patients, seed):
    """Train one DAE on 6-patient pool per held-out patient."""
    per_patient = {}
    for test_pid in sorted(patients.keys()):
        # Pool training amps from the other 6 patients
        tr = []
        freqs_ref = None
        for pid, hemis in patients.items():
            if pid == test_pid:
                continue
            for hd in hemis:
                if freqs_ref is None:
                    freqs_ref = hd['freqs']
                F = min(hd['amps'].shape[1], len(freqs_ref))
                tr.append(hd['amps'][:, :F])
        train_amps = np.vstack(tr)
        try:
            sd = fit_dae(train_amps, Q, seed=seed)
        except Exception as e:
            per_patient[test_pid] = {'error': repr(e)}
            continue

        # Apply to held-out patient's hemispheres
        records = []
        for hd in patients[test_pid]:
            freqs = hd['freqs']; amps = hd['amps']
            for i in range(amps.shape[0]):
                truth = amps[i]
                qa = quantize(truth, Q)
                try:
                    c = correct_dae(freqs, qa, Q, dae_state=sd)
                except Exception:
                    c = qa.copy()
                records.append(eval_one_spectrum(freqs, truth, qa, c))
        if records:
            per_patient[test_pid] = aggregate({'dae': records})['dae']
    return per_patient


# ---------- cross-seed aggregation ----------

def cross_seed(per_seed_hemis):
    """per_seed_hemis: list (n_seeds) of {hemi_key: summary_dict}.
    Returns per-hemi {metric: (mean, std)} + cohort weighted mean across hemis, then across seeds.
    """
    # Gather per-regime values across seeds, per hemi
    hemi_keys = set()
    for h in per_seed_hemis:
        hemi_keys.update(h.keys())

    per_hemi_stats = {}
    for hk in hemi_keys:
        per_hemi_stats[hk] = {'regime_rmse': {}, 'ssim_mean': None, 'consistency': None, 'plateaus_median': None}
        for rn in REGIMES:
            vals = [s[hk]['regime_rmse'].get(rn) for s in per_seed_hemis
                    if hk in s and s[hk]['regime_rmse'].get(rn) is not None]
            if vals:
                per_hemi_stats[hk]['regime_rmse'][rn] = {
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    'n_seeds': len(vals),
                }
        for fld in ('ssim_mean', 'consistency', 'plateaus_median'):
            vals = [s[hk].get(fld) for s in per_seed_hemis
                    if hk in s and s[hk].get(fld) is not None]
            if vals:
                per_hemi_stats[hk][fld] = {
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                }

    # Cohort: weighted mean across hemis per seed, then mean/std across seeds
    cohort = {'regime_rmse': {}}
    for rn in REGIMES:
        seed_means = []
        for s in per_seed_hemis:
            vs = [(s[hk]['regime_rmse'].get(rn), s[hk]['n'])
                  for hk in s if s[hk]['regime_rmse'].get(rn) is not None]
            if vs:
                seed_means.append(float(np.average([v for v, _ in vs],
                                                   weights=[w for _, w in vs])))
        if seed_means:
            cohort['regime_rmse'][rn] = {
                'mean': float(np.mean(seed_means)),
                'std': float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0,
                'per_seed': seed_means,
            }
    return cohort, per_hemi_stats


# ---------- main ----------

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    p.add_argument('--K', type=int, default=5)   # unused for DAE; kept for CLI consistency
    p.add_argument('--skip_loo', action='store_true')
    p.add_argument('--skip_lopo', action='store_true')
    args = p.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'results', 'data')
    out_dir = os.path.join(script_dir, 'results', 'dae')
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(data_dir):
        print(f'ERROR: {data_dir} not found. Run prep_data.py first.')
        return

    npz_files = sorted(f for f in os.listdir(data_dir) if f.endswith('.npz'))
    if not npz_files:
        print(f'ERROR: no npz in {data_dir}.')
        return

    print(f'DAE multi-seed eval')
    print(f'  seeds: {args.seeds}')
    print(f'  hemis: {len(npz_files)}  (data_dir={data_dir})')
    print(f'  workers: {args.workers}')

    # ---------- LOO ----------
    if not args.skip_loo:
        print(f'\n{"=" * 70}\n  LOO\n{"=" * 70}')
        per_seed_loo = []
        for seed in args.seeds:
            t0 = time.time()
            print(f'\n  seed={seed}')
            per_hemi = run_loo_one_seed(npz_files, data_dir, seed, args.workers)
            per_seed_loo.append(per_hemi)
            print(f'    {len(per_hemi)} hemis in {time.time()-t0:.1f}s')

        cohort_loo, per_hemi_loo = cross_seed(per_seed_loo)

        print(f'\n  LOO COHORT (DAE, mean +/- std across {len(args.seeds)} seeds)')
        print(f'  {"regime":>15} {"mean (uVp)":>12} {"std":>10} {"per_seed":>30}')
        for rn in REGIMES:
            r = cohort_loo['regime_rmse'].get(rn)
            if r is None:
                continue
            ps = ' '.join(f'{v:.4f}' for v in r['per_seed'])
            print(f'  {rn:>15} {r["mean"]:>12.4f} {r["std"]:>10.4f} [{ps}]')

        with open(os.path.join(out_dir, 'loo_results.json'), 'w') as f:
            json.dump(_jsonable({'seeds': args.seeds, 'cohort': cohort_loo,
                                 'per_hemi': per_hemi_loo}), f, indent=2)
        print(f'\n  Saved {out_dir}/loo_results.json')

    # ---------- LOPO ----------
    if not args.skip_lopo:
        print(f'\n{"=" * 70}\n  LOPO\n{"=" * 70}')
        # Group files by patient (prefix before _left/_right)
        patients = {}
        for f in npz_files:
            key = f.replace('.npz', '')
            pid = key.split('_')[0]
            d = np.load(os.path.join(data_dir, f))
            patients.setdefault(pid, []).append({
                'key': key, 'freqs': d['freqs'], 'amps': d['amps'],
            })
        print(f'  patients: {sorted(patients.keys())}')

        per_seed_lopo = []
        for seed in args.seeds:
            t0 = time.time()
            print(f'\n  seed={seed}')
            per_patient = run_lopo_one_seed(patients, seed)
            per_seed_lopo.append(per_patient)
            print(f'    {len(per_patient)} patients in {time.time()-t0:.1f}s')

        cohort_lopo, per_patient_lopo = cross_seed(per_seed_lopo)

        print(f'\n  LOPO COHORT (DAE, mean +/- std across {len(args.seeds)} seeds)')
        print(f'  {"regime":>15} {"mean (uVp)":>12} {"std":>10} {"per_seed":>30}')
        for rn in REGIMES:
            r = cohort_lopo['regime_rmse'].get(rn)
            if r is None:
                continue
            ps = ' '.join(f'{v:.4f}' for v in r['per_seed'])
            print(f'  {rn:>15} {r["mean"]:>12.4f} {r["std"]:>10.4f} [{ps}]')

        with open(os.path.join(out_dir, 'lopo_results.json'), 'w') as f:
            json.dump(_jsonable({'seeds': args.seeds, 'cohort': cohort_lopo,
                                 'per_patient': per_patient_lopo}), f, indent=2)
        print(f'\n  Saved {out_dir}/lopo_results.json')


if __name__ == '__main__':
    main()
