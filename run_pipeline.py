"""
Run all analysis steps
"""
import os
import sys
import subprocess

STEPS = {
    1: ('Load data (30s segments)', 'prep_data.py'),
    2: ('Method evaluation (LOO)', 'run_loo.py'),
    3: ('Specparam after correction', 'specparam_correction.py'),
    4: ('Specparam bias (uncorrected)', 'specparam_bias.py'),
    5: ('Frequency range sweep', 'freq_sweep.py'),
    6: ('K sensitivity sweep', 'rank_sweep.py'),
    7: ('LOPO evaluation', 'run_lopo.py'),
    8: ('Generate figures', 'make_figures.py'),
    9: ('DAE multi-seed LOO+LOPO', 'run_dae.py'),
}


def run_step(step_num, data_dir, workers):
    name, script = STEPS[step_num]
    print(f'\n{"=" * 60}')
    print(f'  Step {step_num}: {name}')
    print(f'{"=" * 60}')

    #step 1 wants the raw data path, every other step takes --workers
    cmd = [sys.executable, script]
    if step_num == 1:
        cmd.append(data_dir)
    else:
        cmd.extend(['--workers', str(workers)])

    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    if result.returncode != 0:
        print(f'  Step {step_num} FAILED')
        return False
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('data_dir', help='Path to RawData')
    parser.add_argument('--steps', type=int, nargs='*', default=None,
                        help='Steps to run (default: all)')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    steps = args.steps or sorted(STEPS.keys())

    print(f'RECAST v8 Pipeline')
    print(f'  Data: {args.data_dir}')
    print(f'  Workers: {args.workers}')
    print(f'  Steps: {steps}')

    #steps are ordered and later ones read earlier outputs, so bail on first failure
    for s in steps:
        if s not in STEPS:
            print(f'  Unknown step {s}, skipping')
            continue
        ok = run_step(s, args.data_dir, args.workers)
        if not ok:
            print(f'\nStopping at step {s}.')
            break

    print('\nDone.')


if __name__ == '__main__':
    main()
