#method evaluation with recording-level LOO

import os
import sys
import json
import time
import numpy as np
from multiprocessing import Pool
from forward_model import quantize, quantization_interval, to_db, REGIMES
from methods import (correct_raw, correct_sg, correct_sg_sel,
                     correct_svd, correct_sccd, correct_qppca, correct_qmf,
                     build_basis, fit_qppca, fit_qmf)
from method_dae import fit_dae, correct_dae

Q = 0.11
DAE_SEED = 42
METHODS_PERSPC = ['raw', 'sg', 'sg_sel']
METHODS_LOO = ['svd', 'sccd', 'qppca', 'qmf', 'dae']
ALL_METHODS = METHODS_PERSPC + METHODS_LOO


def rmse(t, e):
    d = np.asarray(t) - np.asarray(e)
    return float(np.sqrt(np.mean(d ** 2))) if len(d) > 0 else np.nan


def spectral_ssim(t, e, w=7):
    x = to_db(np.maximum(t, 1e-10))
    y = to_db(np.maximum(e, 1e-10))
    if len(x) < w:
        return np.nan
    L = max(float(np.ptp(x)), 1.0)
    c1, c2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    h = w // 2
    vals = np.empty(len(x) - 2 * h)
    for idx, i in enumerate(range(h, len(x) - h)):
        xw, yw = x[i - h:i + h + 1], y[i - h:i + h + 1]
        mx, my = xw.mean(), yw.mean()
        sx, sy = xw.var(), yw.var()
        sxy = np.mean((xw - mx) * (yw - my))
        vals[idx] = (2 * mx * my + c1) * (2 * sxy + c2) / (
            (mx ** 2 + my ** 2 + c1) * (sx + sy + c2))
    return float(vals.mean())


def consistency_frac(q_amp, corr, q):
    return float(np.mean(np.abs(quantize(corr, q) - q_amp) < q * 0.01))


def count_plateaus(amp):
    db = np.round(to_db(np.maximum(amp, 1e-10)), 4)
    n = i = 0
    while i < len(db):
        j = i + 1
        while j < len(db) and db[j] == db[i]:
            j += 1
        if j - i >= 2:
            n += 1
        i = j
    return n


def eval_one_spectrum(freqs, truth, qa, corr):
    Fm = min(len(truth), len(corr))
    t, e, q_a = truth[:Fm], corr[:Fm], qa[:Fm]
    regime_out = {}
    for name, (flo, fhi) in REGIMES.items():
        m = (freqs[:Fm] >= flo) & (freqs[:Fm] < fhi)
        if m.sum() == 0:
            regime_out[name] = {'rmse': np.nan, 'rmse_db': np.nan}
            continue
        regime_out[name] = {
            'rmse': rmse(t[m], e[m]),
            'rmse_db': rmse(to_db(t[m]), to_db(e[m])),
        }
    return {
        'regimes': regime_out,
        'ssim': spectral_ssim(t, e),
        'consistency': consistency_frac(q_a, e, Q),
        'plateaus': count_plateaus(e),
    }


#one fold holds out every segment belonging to recording fold_rec
def eval_fold(fold_rec, allamps, qmat, freqs, rec_ids, K):
    test_mask = rec_ids == fold_rec
    train_mask = ~test_mask
    test_idx = np.where(test_mask)[0]

    train_amps = allamps[train_mask]
    train_qmat = qmat[train_mask]

    if train_amps.shape[0] < K + 2:
        return None

    F = allamps.shape[1]

    #shared basis for svd and sccd
    mu, sig, B, ev = build_basis(train_amps, K=K)
    Fuse = min(F, len(mu))

    #q-ppca and qmf both train on the quantized matrix
    try:
        W_pp, mu_pp, s2_pp = fit_qppca(train_qmat, Q, K=K, max_iter=30)
    except Exception:
        W_pp, mu_pp, s2_pp = None, None, None

    try:
        _, V_qmf, mu_qmf, sig_qmf = fit_qmf(train_qmat, Q, K=K, max_iter=200)
    except Exception:
        V_qmf, mu_qmf, sig_qmf = None, None, None

    #dae fits on clean amps, quantized version goes in as input
    try:
        dae_state = fit_dae(train_amps, Q, seed=DAE_SEED)
    except Exception:
        dae_state = None

    fold_results = {mk: [] for mk in ALL_METHODS}
    fold_corrected = {mk: [] for mk in ALL_METHODS}
    fold_indices = []

    for si in test_idx:
        truth = allamps[si]
        qa = qmat[si]
        fold_indices.append(int(si))

        #per-spectrum methods, nothing to train
        corr_raw = correct_raw(freqs, qa, Q)
        corr_sg = correct_sg(freqs, qa, Q)
        corr_sg_sel = correct_sg_sel(freqs, qa, Q)

        for mk, c in [('raw', corr_raw), ('sg', corr_sg), ('sg_sel', corr_sg_sel)]:
            fold_results[mk].append(eval_one_spectrum(freqs, truth, qa, c))
            fold_corrected[mk].append(c)

        #svd
        try:
            corr = correct_svd(freqs[:Fuse], qa[:Fuse], Q,
                               mu=mu[:Fuse], sigma=sig[:Fuse], basis=B[:, :Fuse])
            full = qa.copy(); full[:Fuse] = corr[:Fuse]
        except Exception:
            full = qa.copy()
        fold_results['svd'].append(eval_one_spectrum(freqs, truth, qa, full))
        fold_corrected['svd'].append(full)

        #sccd
        try:
            corr = correct_sccd(freqs[:Fuse], qa[:Fuse], Q,
                                mu=mu[:Fuse], sigma=sig[:Fuse],
                                basis=B[:, :Fuse], eigenvalues=ev)
            full = qa.copy(); full[:Fuse] = corr[:Fuse]
        except Exception:
            full = qa.copy()
        fold_results['sccd'].append(eval_one_spectrum(freqs, truth, qa, full))
        fold_corrected['sccd'].append(full)

        #q-ppca
        if W_pp is not None:
            try:
                corr = correct_qppca(freqs, qa, Q,
                                     W=W_pp, mu_qppca=mu_pp, sigma2=s2_pp)
            except Exception:
                corr = qa.copy()
        else:
            corr = qa.copy()
        fold_results['qppca'].append(eval_one_spectrum(freqs, truth, qa, corr))
        fold_corrected['qppca'].append(corr)

        #qmf
        if V_qmf is not None:
            try:
                corr = correct_qmf(freqs, qa, Q,
                                   V_qmf=V_qmf, mu_qmf=mu_qmf, sigma_qmf=sig_qmf)
            except Exception:
                corr = qa.copy()
        else:
            corr = qa.copy()
        fold_results['qmf'].append(eval_one_spectrum(freqs, truth, qa, corr))
        fold_corrected['qmf'].append(corr)

        #dae
        if dae_state is not None:
            try:
                corr = correct_dae(freqs, qa, Q, dae_state=dae_state)
            except Exception:
                corr = qa.copy()
        else:
            corr = qa.copy()
        fold_results['dae'].append(eval_one_spectrum(freqs, truth, qa, corr))
        fold_corrected['dae'].append(corr)

    return {
        'results': fold_results,
        'corrected': fold_corrected,
        'indices': fold_indices,
    }


#Pool.map wants a single-arg callable
def _worker(args):
    return eval_fold(*args)


def evaluate_hemisphere(freqs, allamps, rec_ids, K=5, n_workers=1):
    N, F = allamps.shape
    qmat = np.array([quantize(allamps[i], Q) for i in range(N)])
    unique_recs = np.unique(rec_ids)
    n_folds = len(unique_recs)
    print(f'    {N} segments, {n_folds} recording folds, K={K}')

    fold_args = [(rec, allamps, qmat, freqs, rec_ids, K) for rec in unique_recs]

    if n_workers > 1 and n_folds > 1:
        with Pool(min(n_workers, n_folds)) as pool:
            fold_outputs = pool.map(_worker, fold_args)
    else:
        fold_outputs = [eval_fold(*a) for a in fold_args]

    #corrected spectra go back into original row order
    all_results = {mk: [] for mk in ALL_METHODS}
    corrected = {mk: np.full_like(allamps, np.nan) for mk in ALL_METHODS}

    for fo in fold_outputs:
        if fo is None:
            continue
        for mk in ALL_METHODS:
            all_results[mk].extend(fo['results'][mk])
            for local_i, global_i in enumerate(fo['indices']):
                c = fo['corrected'][mk][local_i]
                Fc = min(len(c), F)
                corrected[mk][global_i, :Fc] = c[:Fc]

    return all_results, corrected


def aggregate(results):
    summary = {}
    for mk, recs in results.items():
        if not recs:
            continue
        n = len(recs)
        rr, rr_db = {}, {}
        for rn in REGIMES:
            rv = [r['regimes'].get(rn, {}).get('rmse', np.nan) for r in recs]
            rv_db = [r['regimes'].get(rn, {}).get('rmse_db', np.nan) for r in recs]
            rv = [v for v in rv if np.isfinite(v)]
            rv_db = [v for v in rv_db if np.isfinite(v)]
            rr[rn] = float(np.mean(rv)) if rv else None
            rr_db[rn] = float(np.mean(rv_db)) if rv_db else None
        ss = [r['ssim'] for r in recs if np.isfinite(r['ssim'])]
        summary[mk] = {
            'n': n,
            'regime_rmse': rr,
            'regime_rmse_db': rr_db,
            'ssim_mean': float(np.mean(ss)) if ss else None,
            'ssim_std': float(np.std(ss)) if ss else None,
            'consistency': float(np.mean([r['consistency'] for r in recs])),
            'plateaus_median': float(np.median([r['plateaus'] for r in recs])),
        }
    return summary


def _jsonable(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating, float)):
        return None if (isinstance(o, float) and np.isnan(o)) else float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, dict): return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, list): return [_jsonable(v) for v in o]
    return o


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--K', type=int, default=5)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'results', 'data')
    eval_dir = os.path.join(script_dir, 'results', 'eval')
    os.makedirs(eval_dir, exist_ok=True)

    #one npz per hemisphere, written by prep_data.py
    npz_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])
    if not npz_files:
        print(f'No data in {data_dir}/. Run prep_data.py first.')
        return

    all_summ = {}
    t_start = time.time()

    for npz_file in npz_files:
        key = npz_file.replace('.npz', '')
        print(f'\n  {key}...')
        d = np.load(os.path.join(data_dir, npz_file))
        freqs = d['freqs']
        amps = d['amps']
        rec_ids = d['recording_ids']

        if amps.shape[0] < args.K + 3:
            print(f'    too few segments ({amps.shape[0]}), skipping')
            continue

        t0 = time.time()
        results, corrected = evaluate_hemisphere(
            freqs, amps, rec_ids, K=args.K, n_workers=args.workers)
        print(f'    {time.time() - t0:.1f}s')

        all_summ[key] = aggregate(results)

        #step3 reads these back
        save_dict = {'freqs': freqs, 'truth': amps}
        for mk in ALL_METHODS:
            save_dict[f'corrected_{mk}'] = corrected[mk]
        np.savez_compressed(os.path.join(eval_dir, f'{key}.npz'), **save_dict)

    #cohort aggregate, weighted by n
    cohort = {}
    for key, sm in all_summ.items():
        for mk, s in sm.items():
            cohort.setdefault(mk, []).append(s)

    coh_agg = {}
    for mk, slist in cohort.items():
        ntot = sum(s['n'] for s in slist)
        rr, rr_db = {}, {}
        for rn in REGIMES:
            vs = [(s['regime_rmse'].get(rn), s['n']) for s in slist
                  if s['regime_rmse'].get(rn) is not None]
            rr[rn] = float(np.average([v for v, _ in vs],
                                      weights=[w for _, w in vs])) if vs else None
            vs_db = [(s['regime_rmse_db'].get(rn), s['n']) for s in slist
                     if s.get('regime_rmse_db', {}).get(rn) is not None]
            rr_db[rn] = float(np.average([v for v, _ in vs_db],
                                         weights=[w for _, w in vs_db])) if vs_db else None
        ss = [s['ssim_mean'] for s in slist if s.get('ssim_mean') is not None]
        coh_agg[mk] = {
            'n': ntot,
            'regime_rmse': rr,
            'regime_rmse_db': rr_db,
            'ssim_mean': float(np.mean(ss)) if ss else None,
            'ssim_std': float(np.std(ss)) if ss else None,
            'consistency': float(np.mean([s['consistency'] for s in slist])),
            'plateaus_median': float(np.median([s['plateaus_median'] for s in slist])),
        }

    qr = Q / np.sqrt(12)
    print(f'\n{"=" * 100}')
    print(f'  COHORT (30s segments, recording-level LOO, K={args.K})')
    print(f'{"=" * 100}')
    hdr = f'{"Method":>10} {"n":>6}'
    for rn in REGIMES:
        hdr += f' {rn[:12]:>12}'
    hdr += f' {"SSIM":>7} {"Consist":>8} {"Plat":>5}'
    print(hdr)
    print('-' * len(hdr))
    for mk in ALL_METHODS:
        s = coh_agg.get(mk)
        if s is None:
            continue
        row = f'{mk:>10} {s["n"]:>6}'
        for rn in REGIMES:
            v = s['regime_rmse'].get(rn)
            row += f' {v:>12.5f}' if v is not None else f' {"---":>12}'
        ss = s.get('ssim_mean')
        row += f' {ss:>7.3f}' if ss else f' {"---":>7}'
        row += f' {s["consistency"]:>8.1%} {s["plateaus_median"]:>5.0f}'
        print(row)
    print(f'  q/sqrt(12) = {qr:.5f}')

    out = {'cohort': coh_agg, 'per_hemi': all_summ}
    with open(os.path.join(eval_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(_jsonable(out), f, indent=2)

    print(f'\n  Total: {time.time() - t_start:.0f}s')
    print(f'  Saved to {eval_dir}/')


if __name__ == '__main__':
    main()
