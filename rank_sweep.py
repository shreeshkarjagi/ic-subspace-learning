"""
K sensitivity sweep
"""
import os, json, time
import numpy as np
from run_loo import evaluate_hemisphere, aggregate, _jsonable, ALL_METHODS
from forward_model import REGIMES

Q = 0.11
K_VALUES = [3, 5, 7]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'results', 'data')
    out_dir = os.path.join(script_dir, 'results', 'k_sweep')
    os.makedirs(out_dir, exist_ok=True)

    # Find largest hemisphere
    npz_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])
    best_key, best_n = None, 0
    for f in npz_files:
        d = np.load(os.path.join(data_dir, f))
        n = d['amps'].shape[0]
        if n > best_n:
            best_n = n
            best_key = f.replace('.npz', '')

    if best_key is None:
        print('No data found. Run prep_data first.')
        return

    d = np.load(os.path.join(data_dir, f'{best_key}.npz'))
    freqs, amps, rec_ids = d['freqs'], d['amps'], d['recording_ids']
    print(f'K sweep on {best_key} ({amps.shape[0]} segments)')

    all_results = {}
    for K in K_VALUES:
        print(f'\n  K={K}...')
        t0 = time.time()
        results, _ = evaluate_hemisphere(freqs, amps, rec_ids, K=K,
                                         n_workers=args.workers)
        summ = aggregate(results)
        all_results[K] = summ
        print(f'    {time.time() - t0:.1f}s')

        for mk in ALL_METHODS:
            s = summ.get(mk)
            if s is None:
                continue
            rr = s['regime_rmse']
            vals = ' '.join(f'{rr.get(rn, 0):.4f}' for rn in REGIMES)
            print(f'    {mk:>10}: {vals}')

    with open(os.path.join(out_dir, 'k_sweep_results.json'), 'w') as f:
        json.dump(_jsonable({'hemisphere': best_key, 'K_values': K_VALUES,
                             'results': {str(k): v for k, v in all_results.items()}}),
                  f, indent=2)
    print(f'\n  Saved to {out_dir}/')


if __name__ == '__main__':
    main()
