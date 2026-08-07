#spurious peak rate after correction
import os
import sys
import json
import numpy as np
from multiprocessing import Pool
from forward_model import to_db, quantize
from specparam_utils import fit_fooof, match_peaks, _BACKEND

Q = 0.11
FREQ_RANGE = [2, 45]
ALL_METHODS = ['raw', 'sg', 'sg_sel', 'svd', 'sccd', 'qppca', 'qmf', 'dae']


#clean vs corrected, one spectrum and one method at a time
def eval_one_pair(args):
    freqs, truth_amp, corr_amp, freq_range = args

    truth_power = np.maximum(truth_amp, 1e-10) ** 2
    corr_power = np.maximum(corr_amp, 1e-10) ** 2

    fit_clean = fit_fooof(freqs, truth_power, freq_range)
    fit_corr = fit_fooof(freqs, corr_power, freq_range)

    if fit_clean is None or fit_corr is None:
        return None

    pm = match_peaks(fit_clean['peaks'], fit_corr['peaks'])
    return {
        'exp_clean': fit_clean['exponent'],
        'exp_corr': fit_corr['exponent'],
        'n_truth_peaks': fit_clean['n_peaks'],
        'n_matched': pm['n_matched'],
        'n_missed': pm['n_missed'],
        'n_spurious': pm['n_spurious'],
        'spurious_powers': pm['spurious_powers'].tolist(),
    }


def run_method(freqs, truth, corrected, freq_range, n_workers):
    N = truth.shape[0]
    args = [(freqs, truth[i], corrected[i], freq_range) for i in range(N)]

    if n_workers > 1:
        with Pool(n_workers) as pool:
            results = pool.map(eval_one_pair, args)
    else:
        results = [eval_one_pair(a) for a in args]

    #aggregate
    valid = [r for r in results if r is not None]
    if not valid:
        return None

    total_truth = sum(r['n_truth_peaks'] for r in valid)
    total_matched = sum(r['n_matched'] for r in valid)
    total_missed = sum(r['n_missed'] for r in valid)
    total_spurious = sum(r['n_spurious'] for r in valid)
    exp_errors = [r['exp_corr'] - r['exp_clean'] for r in valid]
    all_spur_pw = []
    for r in valid:
        all_spur_pw.extend(r['spurious_powers'])

    n_detected = total_matched + total_spurious
    return {
        'n_spectra': len(valid),
        'n_failed': N - len(valid),
        'total_truth_peaks': total_truth,
        'total_matched': total_matched,
        'total_missed': total_missed,
        'total_spurious': total_spurious,
        'detection_rate': total_matched / max(total_truth, 1),
        'spurious_rate': total_spurious / max(n_detected, 1),
        'exponent_rmse': float(np.sqrt(np.mean(np.array(exp_errors) ** 2))),
        'exponent_bias': float(np.mean(exp_errors)),
        'spurious_power_median': float(np.median(all_spur_pw)) if all_spur_pw else None,
        'spurious_power_p90': float(np.percentile(all_spur_pw, 90)) if all_spur_pw else None,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--freq_range', type=float, nargs=2, default=FREQ_RANGE)
    args = parser.parse_args()

    if _BACKEND is None:
        print('ERROR: fooof or specparam not installed.')
        print('  pip install fooof==1.1.0')
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    eval_dir = os.path.join(script_dir, 'results', 'eval')
    out_dir = os.path.join(script_dir, 'results', 'specparam')
    os.makedirs(out_dir, exist_ok=True)

    npz_files = sorted([f for f in os.listdir(eval_dir) if f.endswith('.npz')])
    if not npz_files:
        print(f'No data in {eval_dir}/. Run run_loo.py first.')
        return

    freq_range = [int(args.freq_range[0]), int(args.freq_range[1])]
    print(f'Specparam after correction ({_BACKEND}, range={freq_range})')

    #accumulate across hemispheres per method
    method_agg = {mk: {'truth': [], 'corrected': [], 'freqs': None}
                  for mk in ALL_METHODS}

    for npz_file in npz_files:
        key = npz_file.replace('.npz', '')
        d = np.load(os.path.join(eval_dir, npz_file))
        freqs = d['freqs']
        truth = d['truth']
        for mk in ALL_METHODS:
            ckey = f'corrected_{mk}'
            if ckey not in d:
                continue
            corr = d[ckey]
            #nan rows are folds that never got filled in, drop them
            valid = ~np.any(np.isnan(corr), axis=1)
            method_agg[mk]['truth'].append(truth[valid])
            method_agg[mk]['corrected'].append(corr[valid])
            if method_agg[mk]['freqs'] is None:
                method_agg[mk]['freqs'] = freqs

    all_results = {}
    for mk in ALL_METHODS:
        if not method_agg[mk]['truth']:
            continue
        truth_all = np.vstack(method_agg[mk]['truth'])
        corr_all = np.vstack(method_agg[mk]['corrected'])
        freqs = method_agg[mk]['freqs']
        print(f'  {mk}: {truth_all.shape[0]} spectra...', end=' ', flush=True)

        r = run_method(freqs, truth_all, corr_all, freq_range, args.workers)
        if r is not None:
            all_results[mk] = r
            print(f'det={r["detection_rate"]:.1%} spur={r["spurious_rate"]:.1%} '
                  f'exp_rmse={r["exponent_rmse"]:.4f}')
        else:
            print('failed')

    #table 3
    print(f'\n{"=" * 80}')
    print(f'  TABLE 3: Specparam after correction, {freq_range}')
    print(f'{"=" * 80}')
    print(f'{"Method":>10} {"n":>6} {"Detect":>8} {"Spurious":>9} {"Exp RMSE":>10}')
    print('-' * 50)
    for mk in ALL_METHODS:
        r = all_results.get(mk)
        if r is None:
            continue
        print(f'{mk:>10} {r["n_spectra"]:>6} {r["detection_rate"]:>8.1%} '
              f'{r["spurious_rate"]:>9.1%} {r["exponent_rmse"]:>10.4f}')

    out_path = os.path.join(out_dir, 'correction_results.json')
    with open(out_path, 'w') as f:
        json.dump({'freq_range': freq_range, 'methods': all_results}, f, indent=2)
    print(f'\n  Saved {out_path}')


if __name__ == '__main__':
    main()
