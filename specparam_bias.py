#specparam bias on uncorrected quantized spectra
import os
import json
import numpy as np
from multiprocessing import Pool
from forward_model import quantize, to_db, BANDS
from specparam_utils import fit_fooof, match_peaks, _BACKEND

Q = 0.11
FREQ_RANGE = [2, 45]


def eval_one_spectrum(args):
    freqs, amp, freq_range = args
    amp_q = quantize(amp, Q)
    fc = fit_fooof(freqs, np.maximum(amp, 1e-10)**2, freq_range)
    fq = fit_fooof(freqs, np.maximum(amp_q, 1e-10)**2, freq_range)
    if fc is None:
        return None
    if fq is None:
        return {'failed_quant': True, 'n_truth_peaks': fc['n_peaks']}

    pm = match_peaks(fc['peaks'], fq['peaks'])
    tp = np.atleast_2d(fc['peaks']); qp = np.atleast_2d(fq['peaks'])
    M, N = len(tp), len(qp)
    matched_cfs, spurious_cfs, matched_truth_cfs = [], [], []
    uq = set()

    #greedy nearest-cf pairing, 2 hz tolerance, each peak used once
    if M > 0 and N > 0 and tp.shape[1] >= 1 and qp.shape[1] >= 1:
        dist = np.abs(tp[:, 0:1] - qp[:, 0:1].T)
        ut = set()
        for idx in np.argsort(dist.ravel()):
            ti, qi = idx // N, idx % N
            if ti in ut or qi in uq: continue
            if dist[ti, qi] > 2.0: break
            ut.add(ti); uq.add(qi)
            matched_cfs.append(float(qp[qi, 0]))
            matched_truth_cfs.append(float(tp[ti, 0]))
    if N > 0 and qp.shape[1] >= 1:
        for qi in range(N):
            if qi not in uq:
                spurious_cfs.append(float(qp[qi, 0]))

    bp = {}
    for bn, (flo, fhi) in BANDS.items():
        m = (freqs >= flo) & (freqs < fhi)
        if m.sum() > 0:
            bp[bn] = {'clean': float(to_db(np.mean(amp[m]**2))),
                       'quant': float(to_db(np.mean(amp_q[m]**2))),
                       'err': float(to_db(np.mean(amp_q[m]**2)) - to_db(np.mean(amp[m]**2)))}

    return {
        'exp_clean': fc['exponent'], 'exp_quant': fq['exponent'],
        'exp_err': fq['exponent'] - fc['exponent'],
        'r2_clean': fc['r_squared'], 'r2_quant': fq['r_squared'],
        'n_truth_peaks': fc['n_peaks'],
        'n_matched': pm['n_matched'], 'n_missed': pm['n_missed'],
        'n_spurious': pm['n_spurious'],
        'spurious_powers': pm['spurious_powers'].tolist(),
        'matched_cfs': matched_cfs, 'spurious_cfs': spurious_cfs,
        'matched_truth_cfs': matched_truth_cfs, 'band_power': bp,
    }


def run(data_dir, freq_range, n_workers):
    npz_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])
    all_freqs, all_amps = None, []
    for f in npz_files:
        d = np.load(os.path.join(data_dir, f))
        if all_freqs is None: all_freqs = d['freqs']
        all_amps.append(d['amps'])
    amps = np.vstack(all_amps); N = amps.shape[0]
    print(f'  {N} spectra, freq_range={freq_range}')

    args = [(all_freqs, amps[i], freq_range) for i in range(N)]
    if n_workers > 1:
        with Pool(n_workers) as pool: results = pool.map(eval_one_spectrum, args)
    else:
        results = [eval_one_spectrum(a) for a in args]

    #spectra where the quantized fit blew up are dropped, not counted as misses
    valid = [r for r in results if r is not None and 'failed_quant' not in r]
    n_fail = N - len(valid)
    ce = np.array([r['exp_clean'] for r in valid])
    qe = np.array([r['exp_quant'] for r in valid])
    ee = np.array([r['exp_err'] for r in valid])
    r2c = np.array([r['r2_clean'] for r in valid])
    r2q = np.array([r['r2_quant'] for r in valid])
    spur_pw, all_matched_cfs, all_spurious_cfs, all_matched_truth_cfs = [], [], [], []
    for r in valid:
        spur_pw.extend(r['spurious_powers'])
        all_matched_cfs.extend(r['matched_cfs'])
        all_spurious_cfs.extend(r['spurious_cfs'])
        all_matched_truth_cfs.extend(r['matched_truth_cfs'])
    spur_pw = np.array(spur_pw)

    tp_total = sum(r['n_truth_peaks'] for r in valid)
    tp_match = sum(r['n_matched'] for r in valid)
    tp_miss = sum(r['n_missed'] for r in valid)
    tp_spur = sum(r['n_spurious'] for r in valid)

    bp_agg = {}
    for bn in BANDS:
        errs = [r['band_power'][bn]['err'] for r in valid if bn in r['band_power']]
        if errs:
            bp_agg[bn] = {'rmse': float(np.sqrt(np.mean(np.array(errs)**2))),
                          'bias': float(np.mean(errs)), 'n': len(errs)}

    n_det = tp_match + tp_spur
    metrics = {
        'n_spectra': len(valid), 'n_failed': n_fail, 'freq_range': freq_range,
        'exponent_rmse': float(np.sqrt(np.mean(ee**2))),
        'exponent_bias': float(np.mean(ee)),
        'natural_exponent_std': float(np.std(ce)),
        'mean_r2_clean': float(np.mean(r2c)), 'mean_r2_quant': float(np.mean(r2q)),
        'r2_change': float(np.mean(r2q - r2c)),
        'detection_rate': tp_match / max(tp_total, 1),
        'spurious_rate': tp_spur / max(n_det, 1),
        'total_truth_peaks': tp_total, 'total_matched': tp_match,
        'total_missed': tp_miss, 'total_spurious': tp_spur,
        'spurious_peak_power_median': float(np.median(spur_pw)) if len(spur_pw) > 0 else None,
        'spurious_peak_power_p90': float(np.percentile(spur_pw, 90)) if len(spur_pw) > 0 else None,
        'band_power': bp_agg,
    }
    arrays = {
        'clean_exponents': ce, 'quant_exponents': qe, 'exponent_errors': ee,
        'r2_clean': r2c, 'r2_quant': r2q, 'spurious_powers': spur_pw,
        'matched_cfs': np.array(all_matched_cfs),
        'spurious_cfs': np.array(all_spurious_cfs),
        'matched_truth_cfs': np.array(all_matched_truth_cfs),
    }
    for bn in BANDS:
        bc = [r['band_power'][bn]['clean'] for r in valid if bn in r['band_power']]
        bq = [r['band_power'][bn]['quant'] for r in valid if bn in r['band_power']]
        if bc:
            arrays[f'{bn}_clean'] = np.array(bc)
            arrays[f'{bn}_quant'] = np.array(bq)
    return metrics, arrays


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--freq_range', type=float, nargs=2, default=FREQ_RANGE)
    args = parser.parse_args()
    if _BACKEND is None: print('ERROR: fooof or specparam not installed.'); return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'results', 'data')
    out_dir = os.path.join(script_dir, 'results', 'specparam')
    os.makedirs(out_dir, exist_ok=True)
    freq_range = [int(args.freq_range[0]), int(args.freq_range[1])]

    print(f'Specparam bias test ({_BACKEND}, {freq_range})')
    metrics, arrays = run(data_dir, freq_range, args.workers)
    m = metrics
    print(f'\n  n = {m["n_spectra"]}')
    print(f'  Exponent RMSE: {m["exponent_rmse"]:.4f} (SD = {m["natural_exponent_std"]:.2f})')
    print(f'  Detection:     {m["detection_rate"]:.1%}')
    print(f'  Spurious:      {m["spurious_rate"]:.1%}')
    print(f'  Spurious CFs:  {len(arrays["spurious_cfs"])}')

    #suffix filenames when freq_range differs from default, otherwise runs overwrite each other
    if freq_range == [2, 45]:
        suffix = ''
    else:
        suffix = f'_{freq_range[0]}_{freq_range[1]}'
    with open(os.path.join(out_dir, f'bias_results{suffix}.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    np.savez_compressed(os.path.join(out_dir, f'bias_arrays{suffix}.npz'), **arrays)
    print(f'  Saved to {out_dir}/ (suffix: "{suffix}")')


if __name__ == '__main__': main()
