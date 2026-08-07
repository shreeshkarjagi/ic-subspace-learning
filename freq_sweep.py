#specparam fit over a few upper frequency bounds
import os
import json
import numpy as np
from specparam_bias import run as run_bias
from specparam_utils import _BACKEND

FREQ_RANGES = [[2, 30], [2, 40], [2, 45], [2, 50], [2, 60], [2, 80]]

#recorded in the output json for reference, nothing here enforces it
GATE = {'exponent_rmse': 0.10, 'detection_rate': 0.90, 'spurious_rate': 0.10}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    if _BACKEND is None:
        print('ERROR: fooof or specparam not installed.')
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'results', 'data')
    out_dir = os.path.join(script_dir, 'results', 'specparam')
    os.makedirs(out_dir, exist_ok=True)

    print(f'Frequency range sweep ({_BACKEND})')
    all_metrics = []

    for fr in FREQ_RANGES:
        print(f'\n  {fr}...')
        metrics, _ = run_bias(data_dir, fr, args.workers)
        all_metrics.append(metrics)
        m = metrics
        print(f'    n={m["n_spectra"]}  exp_rmse={m["exponent_rmse"]:.4f}  '
              f'det={m["detection_rate"]:.1%}  spur={m["spurious_rate"]:.1%}')

    #summary table
    print(f'\n{"=" * 80}')
    print(f'{"Range":>10} {"n":>6} {"Exp RMSE":>10} {"Detect":>8} '
          f'{"Spurious":>9} {"dR2":>10}')
    print('-' * 80)
    for m in all_metrics:
        fr = m['freq_range']
        print(f'{"[" + str(fr[0]) + "," + str(fr[1]) + "]":>10} {m["n_spectra"]:>6} '
              f'{m["exponent_rmse"]:>10.4f} {m["detection_rate"]:>8.1%} '
              f'{m["spurious_rate"]:>9.1%} {m["r2_change"]:>+10.4f}')

    with open(os.path.join(out_dir, 'freq_sweep_results.json'), 'w') as f:
        json.dump({'ranges': all_metrics, 'gate': GATE}, f, indent=2)
    print(f'\n  Saved to {out_dir}/')


if __name__ == '__main__':
    main()
