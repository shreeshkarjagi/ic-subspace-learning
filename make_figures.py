#publication figures for IEEE MLSP

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from forward_model import to_db, quantize, REGIMES
from style import set_style, panel_label, save_fig, PAL, LABELS, ORDER, COL1, COL2

Q = 0.11
SD = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(SD, 'results')
FDIR = os.path.join(SD, 'figures')


def _json(path):
    with open(path) as f: return json.load(f)


def _mlist(summary):
    return [m for m in ORDER if m in summary]


#4 panels: exponent scatter, beta scatter, detection bars, CF histogram
def fig2_specparam_impact():
    set_style()
    bp = os.path.join(RES, 'specparam', 'bias_results.json')
    ap = os.path.join(RES, 'specparam', 'bias_arrays.npz')
    if not os.path.exists(bp): print('  fig2: no bias_results.json'); return
    m = _json(bp); a = np.load(ap)

    fig = plt.figure(figsize=(COL2, 4.5))  #figure* in LaTeX
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.40,
                           height_ratios=[1, 0.9])

    #(a) exponent scatter
    ax = fig.add_subplot(gs[0, 0])
    ce, qe = a['clean_exponents'], a['quant_exponents']
    n_pts = len(ce)
    sc_alpha = max(0.02, min(0.12, 500.0 / n_pts))  #scale alpha to point count
    ax.scatter(ce, qe, s=0.6, alpha=sc_alpha, color=PAL['sccd'], edgecolors='none', rasterized=True)
    lo, hi = min(ce.min(), qe.min()) - 0.05, max(ce.max(), qe.max()) + 0.05
    ax.plot([lo, hi], [lo, hi], '--', color=PAL['identity'], lw=0.5, alpha=0.4)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal', 'box')
    ax.set_xlabel('Clean exponent'); ax.set_ylabel('Quantized exponent')
    ax.text(0.05, 0.95, f'RMSE = {m["exponent_rmse"]:.3f}\n\u03c3 = {m["natural_exponent_std"]:.2f}',
            transform=ax.transAxes, fontsize=7, va='top',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=PAL['neutral'], alpha=0.8, lw=0.3))
    panel_label(ax, 'a')

    #(b) beta scatter
    ax = fig.add_subplot(gs[0, 1])
    if 'beta_clean' in a:
        bc, bq = a['beta_clean'], a['beta_quant']
        ax.scatter(bc, bq, s=0.6, alpha=sc_alpha, color=PAL['sg_sel'], edgecolors='none', rasterized=True)
        lo2, hi2 = min(bc.min(), bq.min()) - 0.3, max(bc.max(), bq.max()) + 0.3
        ax.plot([lo2, hi2], [lo2, hi2], '--', color=PAL['identity'], lw=0.5, alpha=0.4)
        ax.set_xlim(lo2, hi2); ax.set_ylim(lo2, hi2); ax.set_aspect('equal', 'box')
        br = m.get('band_power', {}).get('beta', {})
        ax.text(0.05, 0.95, f'RMSE = {br.get("rmse", 0):.3f} dB',
                transform=ax.transAxes, fontsize=7, va='top',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=PAL['neutral'], alpha=0.8, lw=0.3))
    ax.set_xlabel('Clean beta power (dB)'); ax.set_ylabel('Quantized beta power (dB)')
    panel_label(ax, 'b')

    #(c) detection bars
    ax = fig.add_subplot(gs[0, 2])
    det = m['detection_rate'] * 100
    missed = (1 - m['detection_rate']) * 100
    spur = m['spurious_rate'] * 100
    vals = [det, missed, spur]
    cols = [PAL['sg_sel'], PAL['qppca'], PAL['sg']]
    bars = ax.bar(['Det.', 'Miss.', 'Spur.'], vals, color=cols, alpha=0.8, width=0.55, edgecolor='none')
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.5,
                f'{v:.1f}%', ha='center', fontsize=7, color=PAL['text'])
    ax.set_ylabel('Rate (%)'); ax.set_ylim(0, 110)
    panel_label(ax, 'c')

    #(d) spurious + matched CFs along the bottom, spanning all 3 cols. prefer
    #the [2,80] arrays when they exist so the full spectrum is visible
    ax = fig.add_subplot(gs[1, :3])
    wide_path = os.path.join(RES, 'specparam', 'bias_arrays_2_80.npz')
    cf_src = np.load(wide_path) if os.path.exists(wide_path) else a
    cf_range = [2, 80] if os.path.exists(wide_path) else m['freq_range']
    matched_cfs = cf_src.get('matched_cfs', np.array([]))
    spurious_cfs = cf_src.get('spurious_cfs', np.array([]))
    if len(matched_cfs) > 0 or len(spurious_cfs) > 0:
        bins = np.arange(cf_range[0], cf_range[1] + 2, 2)
        ax.hist(matched_cfs, bins=bins, alpha=0.6, color=PAL['sg_sel'],
                label=f'Matched (n={len(matched_cfs)})', edgecolor='white', lw=0.3)
        ax.hist(spurious_cfs, bins=bins, alpha=0.7, color=PAL['sg'],
                label=f'Spurious (n={len(spurious_cfs)})', edgecolor='white', lw=0.3)
        ax.axvline(8, color=PAL['neutral'], lw=0.5, ls='--', alpha=0.5)
        ax.axvline(30, color=PAL['neutral'], lw=0.5, ls='--', alpha=0.5)
        ax.text(8, ax.get_ylim()[1]*0.9, ' 8 Hz', fontsize=6, color=PAL['neutral'])
        ax.text(30, ax.get_ylim()[1]*0.9, ' 30 Hz', fontsize=6, color=PAL['neutral'])
        ax.legend(fontsize=7, loc='upper right')
    ax.set_xlabel('Center frequency (Hz)'); ax.set_ylabel('Count')
    panel_label(ax, 'd')

    save_fig(fig, os.path.join(FDIR, 'fig2_specparam_impact'))


#1x3: exponent RMSE, detection+spurious, dR2
def fig3_freq_sweep():
    set_style()
    fp = os.path.join(RES, 'specparam', 'freq_sweep_results.json')
    if not os.path.exists(fp): print('  fig3: no freq_sweep_results.json'); return
    data = _json(fp); all_m = data['ranges']; gate = data['gate']
    uppers = [m['freq_range'][1] for m in all_m]; x = np.arange(len(uppers))

    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.4))  #figure* in LaTeX
    fig.subplots_adjust(wspace=0.42)

    ax = axes[0]
    rmses = [m['exponent_rmse'] for m in all_m]
    ax.bar(x, rmses, color=PAL['sccd'], alpha=0.8, width=0.5, edgecolor='none')
    ax.axhline(gate['exponent_rmse'], color=PAL['qppca'], lw=0.6, ls='--',
               label=f'Gate = {gate["exponent_rmse"]}')
    for i, r in enumerate(rmses):
        ax.text(i, r + 0.003, f'{r:.3f}', ha='center', fontsize=6, color=PAL['text'])
    ax.set_xticks(x); ax.set_xticklabels([f'[2,{u}]' for u in uppers])
    ax.set_xlabel('Fit range (Hz)'); ax.set_ylabel('Exponent RMSE')
    ax.set_ylim(0, max(max(rmses)*1.4, gate['exponent_rmse']*1.3))
    ax.legend(fontsize=6.5); panel_label(ax, 'a')

    ax = axes[1]
    det = [m['detection_rate']*100 for m in all_m]
    spur = [m['spurious_rate']*100 for m in all_m]
    w = 0.25
    ax.bar(x - w/2, det, width=w, color=PAL['sg_sel'], alpha=0.8, label='Detected', edgecolor='none')
    ax.bar(x + w/2, spur, width=w, color=PAL['sg'], alpha=0.8, label='Spurious', edgecolor='none')
    ax.set_xticks(x); ax.set_xticklabels([f'[2,{u}]' for u in uppers])
    ax.set_xlabel('Fit range (Hz)'); ax.set_ylabel('Rate (%)')
    ax.set_ylim(0, 105); ax.legend(fontsize=6.5); panel_label(ax, 'b')

    ax = axes[2]
    dr2 = [m['r2_change'] for m in all_m]
    cols = [PAL['sg_sel'] if d > -0.005 else PAL['qppca'] for d in dr2]
    ax.bar(x, dr2, color=cols, alpha=0.8, width=0.5, edgecolor='none')
    for i, d in enumerate(dr2):
        ax.text(i, d - 0.0003 if d < 0 else d + 0.0003, f'{d:+.4f}',
                ha='center', fontsize=6, va='top' if d < 0 else 'bottom', color=PAL['text'])
    ax.axhline(0, color=PAL['neutral'], lw=0.4, ls=':')
    ax.set_xticks(x); ax.set_xticklabels([f'[2,{u}]' for u in uppers])
    ax.set_xlabel('Fit range (Hz)'); ax.set_ylabel('\u0394R\u00b2'); panel_label(ax, 'c')

    save_fig(fig, os.path.join(FDIR, 'fig3_freq_sweep'))


#per-method spurious rate, detection rate and exponent RMSE after correction
def fig4_spurious_reduction():
    set_style()
    cp = os.path.join(RES, 'specparam', 'correction_results.json')
    if not os.path.exists(cp): print('  fig4: no correction_results.json'); return
    data = _json(cp); md = data['methods']
    methods = [m for m in ORDER if m in md and m != 'svd']  #svd is catastrophic here, leave it out
    if not methods: return
    n = len(methods)
    colors = [PAL.get(m, '#999') for m in methods]
    labels = [LABELS.get(m, m) for m in methods]

    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.4))  #figure* in LaTeX
    fig.subplots_adjust(wspace=0.42, bottom=0.28)

    #(a) spurious rate
    ax = axes[0]
    spur = [md[m]['spurious_rate']*100 for m in methods]
    ax.bar(range(n), spur, color=colors, alpha=0.85, width=0.55, edgecolor='none')
    for i, v in enumerate(spur):
        ax.text(i, v + 0.8, f'{v:.1f}%', ha='center', fontsize=6.5, color=PAL['text'])
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=35, ha='right')
    ax.set_ylabel('Spurious rate (%)'); ax.set_ylim(0, max(spur)*1.25)
    panel_label(ax, 'a')

    #(b) detection rate
    ax = axes[1]
    det = [md[m]['detection_rate']*100 for m in methods]
    ax.bar(range(n), det, color=colors, alpha=0.85, width=0.55, edgecolor='none')
    for i, v in enumerate(det):
        ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=6.5, color=PAL['text'])
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=35, ha='right')
    ax.set_ylabel('Detection rate (%)'); ax.set_ylim(0, 110)
    panel_label(ax, 'b')

    #(c) exponent rmse
    ax = axes[2]
    exp_rmse = [md[m]['exponent_rmse'] for m in methods]
    ax.bar(range(n), exp_rmse, color=colors, alpha=0.85, width=0.55, edgecolor='none')
    for i, v in enumerate(exp_rmse):
        ax.text(i, v + 0.001, f'{v:.3f}', ha='center', fontsize=6, color=PAL['text'])
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=35, ha='right')
    ax.set_ylabel('Exponent RMSE'); ax.set_ylim(0, max(exp_rmse)*1.25)
    panel_label(ax, 'c')

    save_fig(fig, os.path.join(FDIR, 'fig4_spurious_reduction'))


#every snapshot from one hemisphere, raw against corrected (Q-PPCA then SG)
def fig5_exemplar():
    set_style()
    from methods import fit_qppca, correct_qppca, correct_sg
    snap_path = os.path.join(RES, 'snapshot_exemplar.npz')
    if os.path.exists(snap_path):
        d = np.load(snap_path)
        freqs = d['freqs']
        raw_db = d['raw_db']
        corr_db = d['corr_db']
    else:
        data_dir = os.environ.get('RECAST_DATA_DIR', '')
        if not data_dir or not os.path.isdir(data_dir):
            print('  fig5: no snapshot_exemplar.npz and RECAST_DATA_DIR not set')
            print('    Run: RECAST_DATA_DIR=/path/to/RawData python make_figures.py --only fig5')
            return
        from load_raw import find_jsons, load_snapshots
        patient_dirs = sorted([p for p in os.listdir(data_dir)
                               if os.path.isdir(os.path.join(data_dir, p))])
        best_mat, best_freqs, best_pid, best_hemi = None, None, None, None
        for pd in patient_dirs:
            pdir = os.path.join(data_dir, pd)
            jsons = find_jsons(pdir)
            all_snaps = {'left': [], 'right': []}
            for jp in jsons:
                for s in load_snapshots(jp):
                    h = s['hemisphere']
                    if h in all_snaps and len(s['fftbin']) > 50:
                        all_snaps[h].append(s)
            for hemi in ('left', 'right'):
                snaps = all_snaps[hemi]
                if len(snaps) < 50:
                    continue
                F0 = len(snaps[0]['fftbin'])
                freq_full = snaps[0]['frequency']
                good_bins = freq_full < 45
                freqs_cut = freq_full[good_bins]
                rows = []
                freq_full = []

                for s in snaps:
                    if len(s['fftbin']) != F0:
                        continue
                    row = s['fftbin'][good_bins].copy()
                    row[row <= 0] = 0.11
                    rows.append(row)
                mat = np.array(rows)
                if best_mat is None or mat.shape[0] > best_mat.shape[0]:
                    best_mat = mat
                    best_freqs = freqs_cut
                    best_pid = pd
                    best_hemi = hemi

        if best_mat is None:
            print('  fig5: no suitable snapshots found')
            return

        mat = best_mat
        freqs = best_freqs
        print(f'    {best_pid} {best_hemi}: {mat.shape[0]} snapshots, {mat.shape[1]} bins')

        vals = np.sort(np.unique(mat[mat > 0.05]))
        diffs = np.diff(vals)
        diffs = diffs[(diffs > 0.001) & (diffs < 0.5)]
        q_est = float(np.median(diffs[:max(len(diffs)//4, 1)])) if len(diffs) > 0 else 0.11
        print(f'    q = {q_est:.4f}')

        K = min(5, mat.shape[0] - 2)
        W, mu, sig2 = fit_qppca(mat, q_est, K=K, max_iter=30)

        #q-ppca first, then full SG on top — SG smooths everything, no plateau detection
        corrected = np.zeros_like(mat)
        for si in range(mat.shape[0]):
            qppca_out = correct_qppca(freqs, mat[si], q_est,
                                      W=W, mu_qppca=mu, sigma2=sig2)
            corrected[si] = correct_sg(freqs, qppca_out, q_est)

        raw_db = to_db(np.maximum(mat, 1e-10))
        corr_db = to_db(np.maximum(corrected, 1e-10))

        np.savez_compressed(snap_path, freqs=freqs, raw_db=raw_db, corr_db=corr_db)
        print(f'    Cached to {snap_path}')

    N = raw_db.shape[0]
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.6))  #figure* in LaTeX
    fig.subplots_adjust(wspace=0.25)

    for ax, data, title, mean_col in [
        (axes[0], raw_db, 'Snapshot (quantized)', PAL['qppca']),
        (axes[1], corr_db, 'Corrected', PAL['qppca']),
    ]:
        for i in range(N):
            ax.plot(freqs, data[i], color=PAL['neutral'], lw=0.15, alpha=0.25)
        ax.plot(freqs, np.mean(data, axis=0), color=mean_col, lw=1.2, alpha=0.9)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_title(title, fontsize=9)
        ax.set_xlim(freqs[0], freqs[-1])

    axes[0].set_ylabel('Amplitude (dB re \u00b5Vp)')

    ylo = min(raw_db.min(), corr_db.min()) - 1
    yhi = max(raw_db.max(), corr_db.max()) + 1
    for ax in axes:
        ax.set_ylim(ylo, yhi)

    panel_label(axes[0], 'a')
    panel_label(axes[1], 'b')

    save_fig(fig, os.path.join(FDIR, 'fig5_exemplar'))


#supplementary

#linear RMSE, grouped by regime
def figS1_method_rmse():
    set_style()
    ep = os.path.join(RES, 'eval', 'evaluation_results.json')
    if not os.path.exists(ep): print('  figS1: no eval results'); return
    summary = _json(ep)['cohort']; methods = _mlist(summary)
    if not methods: return
    regimes = list(REGIMES.keys()); nM = len(methods); nR = len(regimes)
    fig, ax = plt.subplots(figsize=(COL2, 2.8))
    x = np.arange(nR); bw = 0.75 / nM
    for mi, mk in enumerate(methods):
        s = summary.get(mk)
        if s is None: continue
        vals = [s['regime_rmse'].get(rn) or 0 for rn in regimes]
        #clip svd or it blows the axis
        vals = [min(v, 0.3) for v in vals]
        ax.bar(x + (mi - nM/2 + 0.5)*bw, vals, width=bw*0.88,
               color=PAL.get(mk, '#999'), alpha=0.85, label=LABELS.get(mk, mk), edgecolor='none')
    qr = Q / np.sqrt(12)
    ax.axhline(qr, color=PAL['neutral'], ls=':', lw=0.6, zorder=0)
    ax.text(nR - 0.5, qr + 0.002, 'q/\u221a12', fontsize=7, color=PAL['neutral'], ha='right')
    rlabels = [f'{rn.replace("_"," ").title()}\n({REGIMES[rn][0]:.0f}\u2013{REGIMES[rn][1]:.0f} Hz)' for rn in regimes]
    ax.set_xticks(x); ax.set_xticklabels(rlabels)
    ax.set_ylabel('RMSE (\u00b5Vp)'); ax.set_ylim(0, 0.32)
    ax.legend(fontsize=6, ncol=min(nM, 4), loc='upper center',
              bbox_to_anchor=(0.5, 1.22), columnspacing=0.8, handlelength=1.2)
    fig.subplots_adjust(top=0.78)
    save_fig(fig, os.path.join(FDIR, 'figS1_method_rmse'))


#SSIM + consistency + plateaus
def figS2_summary():
    set_style()
    ep = os.path.join(RES, 'eval', 'evaluation_results.json')
    if not os.path.exists(ep): print('  figS2: no eval results'); return
    summary = _json(ep)['cohort']; methods = _mlist(summary)
    if not methods: return
    n = len(methods)
    colors = [PAL.get(m, '#999') for m in methods]
    labels = [LABELS.get(m, m) for m in methods]
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.2))
    fig.subplots_adjust(wspace=0.45, bottom=0.30)

    ax = axes[0]
    sv = [summary[m].get('ssim_mean', 0) or 0 for m in methods]
    ax.bar(range(n), sv, color=colors, alpha=0.85, width=0.55, edgecolor='none')
    for i, v in enumerate(sv): ax.text(i, v + 0.012, f'{v:.3f}', ha='center', fontsize=6, color=PAL['text'])
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=40, ha='right')
    ax.set_ylabel('Spectral SSIM'); ax.set_ylim(0, 1.08); panel_label(ax, 'a')

    ax = axes[1]
    cv = [summary[m].get('consistency', 0) or 0 for m in methods]
    ax.bar(range(n), [v*100 for v in cv], color=colors, alpha=0.85, width=0.55, edgecolor='none')
    for i, v in enumerate(cv): ax.text(i, v*100 + 1.5, f'{v:.0%}', ha='center', fontsize=6, color=PAL['text'])
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=40, ha='right')
    ax.set_ylabel('Consistency (%)'); ax.set_ylim(0, 110); panel_label(ax, 'b')

    ax = axes[2]
    pv = [summary[m].get('plateaus_median', 0) or 0 for m in methods]
    ax.bar(range(n), pv, color=colors, alpha=0.85, width=0.55, edgecolor='none')
    for i, v in enumerate(pv): ax.text(i, v + 0.3, f'{v:.0f}', ha='center', fontsize=6, color=PAL['text'])
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=40, ha='right')
    ax.set_ylabel('Plateaus (median)'); ax.set_ylim(bottom=0); panel_label(ax, 'c')

    save_fig(fig, os.path.join(FDIR, 'figS2_summary'))


def figS3_db_rmse():
    set_style()
    ep = os.path.join(RES, 'eval', 'evaluation_results.json')
    if not os.path.exists(ep): print('  figS3: no eval results'); return
    summary = _json(ep)['cohort']; methods = _mlist(summary)
    if not methods: return
    regimes = list(REGIMES.keys()); nM = len(methods); nR = len(regimes)
    fig, ax = plt.subplots(figsize=(COL2, 2.8))
    x = np.arange(nR); bw = 0.75 / nM
    for mi, mk in enumerate(methods):
        s = summary.get(mk)
        if s is None: continue
        rmse_db = s.get('regime_rmse_db', {})
        vals = [min(rmse_db.get(rn) or 0, 2.0) for rn in regimes]
        ax.bar(x + (mi - nM/2 + 0.5)*bw, vals, width=bw*0.88,
               color=PAL.get(mk, '#999'), alpha=0.85, label=LABELS.get(mk, mk), edgecolor='none')
    rlabels = [f'{rn.replace("_"," ").title()}\n({REGIMES[rn][0]:.0f}\u2013{REGIMES[rn][1]:.0f} Hz)' for rn in regimes]
    ax.set_xticks(x); ax.set_xticklabels(rlabels)
    ax.set_ylabel('RMSE (dB)'); ax.set_ylim(bottom=0)
    ax.legend(fontsize=6, ncol=min(nM, 4), loc='upper center',
              bbox_to_anchor=(0.5, 1.22), columnspacing=0.8, handlelength=1.2)
    fig.subplots_adjust(top=0.78)
    save_fig(fig, os.path.join(FDIR, 'figS3_db_rmse'))


#LOO next to LOPO, one axis per regime
def figS4_lopo():
    set_style()
    lp = os.path.join(RES, 'lopo', 'lopo_results.json')
    if not os.path.exists(lp): print('  figS4: no lopo_results.json'); return
    cohort_lopo = _json(lp)['cohort']
    ep = os.path.join(RES, 'eval', 'evaluation_results.json')
    cohort_loo = _json(ep)['cohort'] if os.path.exists(ep) else {}
    methods = [m for m in ORDER if m in cohort_lopo]
    if not methods: return
    regimes = list(REGIMES.keys()); nR = len(regimes)
    fig, axes = plt.subplots(1, nR, figsize=(COL2, 2.4))
    fig.subplots_adjust(wspace=0.35, bottom=0.30)
    for ri, rn in enumerate(regimes):
        ax = axes[ri]; n = len(methods); w = 0.32
        for i, mk in enumerate(methods):
            loo_v = min(cohort_loo.get(mk, {}).get('regime_rmse', {}).get(rn, 0) or 0, 0.3)
            lopo_v = min(cohort_lopo.get(mk, {}).get('regime_rmse', {}).get(rn, 0) or 0, 0.3)
            ax.bar(i - w/2, loo_v, width=w, color=PAL.get(mk, '#999'), alpha=0.45, edgecolor='none')
            ax.bar(i + w/2, lopo_v, width=w, color=PAL.get(mk, '#999'), alpha=0.9, edgecolor='none')
        flo, fhi = REGIMES[rn]
        ax.set_title(f'{rn.replace("_"," ").title()}\n({flo:.0f}\u2013{fhi:.0f} Hz)', fontsize=8)
        ax.set_xticks(range(n))
        ax.set_xticklabels([LABELS.get(m, m) for m in methods], rotation=40, ha='right', fontsize=6)
        if ri == 0: ax.set_ylabel('RMSE (\u00b5Vp)')
        ax.set_ylim(bottom=0)
        if ri == 0:
            from matplotlib.patches import Patch
            ax.legend([Patch(fc='gray', alpha=0.45), Patch(fc='gray', alpha=0.9)],
                      ['LOO', 'LOPO'], fontsize=6, loc='upper left')
    save_fig(fig, os.path.join(FDIR, 'figS4_lopo'))


ALL_FIGS = {
    'fig2': fig2_specparam_impact,
    'fig3': fig3_freq_sweep,
    'fig4': fig4_spurious_reduction,
    'fig5': fig5_exemplar,
    'figS1': figS1_method_rmse,
    'figS2': figS2_summary,
    'figS3': figS3_db_rmse,
    'figS4': figS4_lopo,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', nargs='*', default=None)
    args = parser.parse_args()
    os.makedirs(FDIR, exist_ok=True)
    figs = args.only or sorted(ALL_FIGS.keys())
    for name in figs:
        fn = ALL_FIGS.get(name)
        if fn is None: print(f'  Unknown: {name}'); continue
        print(f'  {name}...')
        try: fn()
        except Exception as e: print(f'    FAILED: {e}')
    print(f'\nFigures in {FDIR}/')

if __name__ == '__main__': main()
