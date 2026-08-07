import os
import json
import time
import numpy as np
from forward_model import quantize, REGIMES
from methods import (correct_raw, correct_sg_sel, correct_svd,
                     correct_sccd, correct_qppca, correct_qmf,
                     build_basis, fit_qppca, fit_qmf)
from run_loo import eval_one_spectrum, aggregate, _jsonable

Q = 0.11
METHODS = ['raw', 'sg_sel', 'svd', 'sccd', 'qppca', 'qmf']


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--K', type=int, default=5)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'results', 'data')
    out_dir = os.path.join(script_dir, 'results', 'lopo')
    os.makedirs(out_dir, exist_ok=True)

    #load every hemisphere, group by patient
    npz_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])
    patients = {}  #pid -> list of (key, freqs, amps)
    for f in npz_files:
        key = f.replace('.npz', '')
        pid = key.split('_')[0]
        d = np.load(os.path.join(data_dir, f))
        patients.setdefault(pid, []).append({
            'key': key,
            'freqs': d['freqs'],
            'amps': d['amps'],
        })

    print(f'LOPO: {len(patients)} patients')
    for pid, hemis in sorted(patients.items()):
        n = sum(h['amps'].shape[0] for h in hemis)
        print(f'  {pid}: {len(hemis)} hemispheres, {n} segments')

    K = args.K
    all_results = {}

    for test_pid in sorted(patients.keys()):
        print(f'\n  Hold out {test_pid}...')
        t0 = time.time()

        #pool the other patients, truncate to the shortest freq axis we have seen
        train_amps_list = []
        train_qmat_list = []
        freqs_ref = None

        for pid, hemis in patients.items():
            if pid == test_pid:
                continue
            for hd in hemis:
                if freqs_ref is None:
                    freqs_ref = hd['freqs']
                F = min(hd['amps'].shape[1], len(freqs_ref))
                for i in range(hd['amps'].shape[0]):
                    a = hd['amps'][i, :F]
                    train_amps_list.append(a)
                    train_qmat_list.append(quantize(a, Q))

        train_amps = np.array(train_amps_list)
        train_qmat = np.array(train_qmat_list)
        F = train_amps.shape[1]
        print(f'    Train: {train_amps.shape[0]} segments from '
              f'{len(patients) - 1} patients')

        #fit once per fold, then reuse for every test segment
        mu, sig, B, ev = build_basis(train_amps, K=K)
        Fuse = min(F, len(mu))

        try:
            W_pp, mu_pp, s2_pp = fit_qppca(train_qmat, Q, K=K, max_iter=30)
        except Exception as e:
            print(f'    Q-PPCA fit failed: {e}')
            W_pp = None

        try:
            _, V_qmf, mu_qmf, sig_qmf = fit_qmf(train_qmat, Q, K=K, max_iter=200)
        except Exception as e:
            print(f'    QMF fit failed: {e}')
            V_qmf = None

        #test on the held-out patient, any method that blows up just returns the quantized input
        patient_results = {mk: [] for mk in METHODS}
        for hd in patients[test_pid]:
            freqs = hd['freqs']
            amps = hd['amps']
            for si in range(amps.shape[0]):
                truth = amps[si]
                qa = quantize(truth, Q)
                Ft = min(len(truth), F, Fuse)

                for mk in METHODS:
                    if mk == 'raw':
                        corr = correct_raw(freqs, qa, Q)
                    elif mk == 'sg_sel':
                        from methods import correct_sg_sel
                        corr = correct_sg_sel(freqs, qa, Q)
                    elif mk == 'svd':
                        corr = qa.copy()
                        try:
                            c = correct_svd(freqs[:Ft], qa[:Ft], Q,
                                            mu=mu[:Ft], sigma=sig[:Ft],
                                            basis=B[:, :Ft])
                            corr[:Ft] = c[:Ft]
                        except Exception:
                            pass
                    elif mk == 'sccd':
                        corr = qa.copy()
                        try:
                            c = correct_sccd(freqs[:Ft], qa[:Ft], Q,
                                             mu=mu[:Ft], sigma=sig[:Ft],
                                             basis=B[:, :Ft], eigenvalues=ev)
                            corr[:Ft] = c[:Ft]
                        except Exception:
                            pass
                    elif mk == 'qppca':
                        if W_pp is not None:
                            try:
                                corr = correct_qppca(freqs, qa, Q,
                                                     W=W_pp, mu_qppca=mu_pp,
                                                     sigma2=s2_pp)
                            except Exception:
                                corr = qa.copy()
                        else:
                            corr = qa.copy()
                    elif mk == 'qmf':
                        if V_qmf is not None:
                            try:
                                corr = correct_qmf(freqs, qa, Q,
                                                   V_qmf=V_qmf, mu_qmf=mu_qmf,
                                                   sigma_qmf=sig_qmf)
                            except Exception:
                                corr = qa.copy()
                        else:
                            corr = qa.copy()

                    patient_results[mk].append(
                        eval_one_spectrum(freqs, truth, qa, corr))

        all_results[test_pid] = aggregate(patient_results)
        n_test = sum(hd['amps'].shape[0] for hd in patients[test_pid])
        print(f'    Test: {n_test} segments, {time.time() - t0:.1f}s')

    #cohort average across held-out patients, weighted by segment count
    cohort = {}
    for pid, summ in all_results.items():
        for mk, s in summ.items():
            cohort.setdefault(mk, []).append(s)

    coh_agg = {}
    for mk, slist in cohort.items():
        ntot = sum(s['n'] for s in slist)
        rr = {}
        for rn in REGIMES:
            vs = [(s['regime_rmse'].get(rn), s['n']) for s in slist
                  if s['regime_rmse'].get(rn) is not None]
            rr[rn] = float(np.average([v for v, _ in vs],
                                      weights=[w for _, w in vs])) if vs else None
        ss = [s['ssim_mean'] for s in slist if s.get('ssim_mean') is not None]
        coh_agg[mk] = {
            'n': ntot,
            'regime_rmse': rr,
            'ssim_mean': float(np.mean(ss)) if ss else None,
            'consistency': float(np.mean([s['consistency'] for s in slist])),
            'plateaus_median': float(np.median([s['plateaus_median'] for s in slist])),
        }

    print(f'\n{"=" * 80}')
    print(f'  LOPO COHORT')
    print(f'{"=" * 80}')
    hdr = f'{"Method":>10} {"n":>6}'
    for rn in REGIMES:
        hdr += f' {rn[:12]:>12}'
    hdr += f' {"SSIM":>7} {"Consist":>8}'
    print(hdr)
    print('-' * len(hdr))
    for mk in METHODS:
        s = coh_agg.get(mk)
        if s is None:
            continue
        row = f'{mk:>10} {s["n"]:>6}'
        for rn in REGIMES:
            v = s['regime_rmse'].get(rn)
            row += f' {v:>12.5f}' if v is not None else f' {"---":>12}'
        ss = s.get('ssim_mean')
        row += f' {ss:>7.3f}' if ss else f' {"---":>7}'
        row += f' {s["consistency"]:>8.1%}'
        print(row)

    with open(os.path.join(out_dir, 'lopo_results.json'), 'w') as f:
        json.dump(_jsonable({'cohort': coh_agg, 'per_patient': all_results}),
                  f, indent=2)
    print(f'\n  Saved to {out_dir}/')


if __name__ == '__main__':
    main()
