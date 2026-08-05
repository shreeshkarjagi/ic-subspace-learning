"""
Paired significance of the Q-PPCA spurious rate reduction vs Raw
"""
import os, json
import numpy as np
from multiprocessing import Pool
from scipy.stats import wilcoxon
from specparam_correction import eval_one_pair, FREQ_RANGE
from specparam_utils import _BACKEND

A, B = 'raw', 'qppca'
N_BOOT = 5000
SEED = 0


def hemi_counts(npz_path, fr, workers):
    """Per-spectrum (matched, spurious) for Raw and Q-PPCA on one hemisphere same rows"""
    d = np.load(npz_path)
    freqs, truth, ca, cb = d['freqs'], d['truth'], d[f'corrected_{A}'], d[f'corrected_{B}']
    idx = np.where(~(np.any(np.isnan(ca), 1) | np.any(np.isnan(cb), 1)))[0]
    aa = [(freqs, truth[i], ca[i], fr) for i in idx]
    bb = [(freqs, truth[i], cb[i], fr) for i in idx]
    if workers > 1:
        with Pool(workers) as p:
            ra, rb = p.map(eval_one_pair, aa), p.map(eval_one_pair, bb)
    else:
        ra, rb = [eval_one_pair(a) for a in aa], [eval_one_pair(b) for b in bb]
    rows = [(x['n_matched'], x['n_spurious'], y['n_matched'], y['n_spurious'])
            for x, y in zip(ra, rb) if x and y]
    return np.array(rows, dtype=np.int64).reshape(-1, 4)


def rate(c, off):
    """Pooled spurious rate = spurious / detected off=0 Raw, 2 Q-PPCA"""
    s = c[:, off + 1].sum(); det = c[:, off].sum() + s
    return s / det if det else np.nan


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--freq_range', type=float, nargs=2, default=FREQ_RANGE)
    args = ap.parse_args()

    d = os.path.dirname(os.path.abspath(__file__))
    eval_dir = os.path.join(d, 'results', 'eval')
    out_dir = os.path.join(d, 'results', 'specparam'); os.makedirs(out_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(eval_dir) if f.endswith('.npz'))

    fr = [int(args.freq_range[0]), int(args.freq_range[1])]
    print(f'Paired {A} vs {B} spurious rate ({_BACKEND}, range={fr})')

    keys, ra, rb, counts = [], [], [], []
    for f in files:
        c = hemi_counts(os.path.join(eval_dir, f), fr, args.workers)
        if c.shape[0] == 0:
            continue
        keys.append(f[:-4]); ra.append(rate(c, 0)); rb.append(rate(c, 2)); counts.append(c)
        print(f'  {f[:-4]:>18}: n={c.shape[0]:>4}  {A}={ra[-1]:.3f}  {B}={rb[-1]:.3f}')

    ra, rb = np.array(ra), np.array(rb)
    counts = np.vstack(counts)
    pa, pb = rate(counts, 0), rate(counts, 2)

    stat, p2 = wilcoxon(ra, rb, alternative='two-sided')
    _, p1 = wilcoxon(ra, rb, alternative='greater')

    rng = np.random.default_rng(SEED); n = len(counts)
    boot = np.empty(N_BOOT)
    for i in range(N_BOOT):
        c = counts[rng.integers(0, n, n)]           # one resample, both rates on it
        boot[i] = (rate(c, 0) - rate(c, 2)) * 100
    lo, hi = np.percentile(boot, [2.5, 97.5])

    print(f'\n{"=" * 80}')
    print(f'{B} vs {A}: {len(keys)} hemispheres, {n} spectra')
    print(f'spurious {pa:.4f} -> {pb:.4f}  ({(pa - pb) * 100:+.2f} pp), lower in {int((rb < ra).sum())}/{len(keys)}')
    print(f'Wilcoxon p={p2:.2e} (two-sided), {p1:.2e} (greater)')
    print(f'bootstrap 95% CI [{lo:.2f}, {hi:.2f}] pp')

    json.dump({
        'a': A, 'b': B, 'freq_range': fr, 'n_hemi': len(keys), 'n_spectra': int(n),
        'rate_a': float(pa), 'rate_b': float(pb), 'reduction_pp': float((pa - pb) * 100),
        'n_lower': int((rb < ra).sum()), 'p_two': float(p2), 'p_greater': float(p1),
        'ci_pp': [float(lo), float(hi)], 'n_boot': N_BOOT,
        'per_hemi': {k: [float(x), float(y)] for k, x, y in zip(keys, ra, rb)},
    }, open(os.path.join(out_dir, 'spurious_paired_test.json'), 'w'), indent=2)
    print('  saved results/specparam/spurious_paired_test.json')


if __name__ == '__main__':
    main()