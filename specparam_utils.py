"""
Shared specparam fitting and peak matching
"""
import warnings
import numpy as np

try:
    from fooof import FOOOF; _BACKEND = 'fooof'
except ImportError:
    try:
        from specparam import SpectralModel as FOOOF; _BACKEND = 'specparam'
    except ImportError:
        FOOOF = None; _BACKEND = None

SP_FREQ_RANGE = [2, 45]
SP_PEAK_WIDTH = [2, 8]
SP_MAX_PEAKS = 4
SP_MIN_PEAK_HEIGHT = 0.1
SP_APERIODIC_MODE = 'fixed'


def fit_fooof(freqs, power, freq_range):
    if FOOOF is None:
        return None
    ok = np.isfinite(freqs) & np.isfinite(power) & (power > 0)
    if ok.sum() < 10:
        return None
    try:
        fm = FOOOF(peak_width_limits=SP_PEAK_WIDTH, max_n_peaks=SP_MAX_PEAKS,
                   min_peak_height=SP_MIN_PEAK_HEIGHT,
                   aperiodic_mode=SP_APERIODIC_MODE, verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fm.fit(freqs[ok], power[ok], freq_range=freq_range)
        if not fm.has_model:
            return None
        ap = fm.aperiodic_params_
        if ap is None or len(ap) < 2:
            return None
        peaks = fm.peak_params_
        if peaks is None or len(peaks) == 0:
            peaks = np.empty((0, 3))
        return {
            'exponent': float(ap[1]),
            'offset': float(ap[0]),
            'r_squared': float(fm.r_squared_),
            'peaks': np.atleast_2d(peaks),
            'n_peaks': int(fm.n_peaks_),
        }
    except Exception:
        return None


def match_peaks(truth_peaks, test_peaks, tol=2.0):
    tp, qp = np.atleast_2d(truth_peaks), np.atleast_2d(test_peaks)
    M, N = len(tp), len(qp)
    used_t, used_q = set(), set()
    cf_errors, pw_errors = [], []

    if M > 0 and N > 0 and tp.shape[1] >= 2 and qp.shape[1] >= 2:
        dist = np.abs(tp[:, 0:1] - qp[:, 0:1].T)
        for idx in np.argsort(dist.ravel()):
            ti, qi = idx // N, idx % N
            if ti in used_t or qi in used_q:
                continue
            if dist[ti, qi] > tol:
                break
            used_t.add(ti)
            used_q.add(qi)
            cf_errors.append(float(qp[qi, 0] - tp[ti, 0]))
            pw_errors.append(float(qp[qi, 1] - tp[ti, 1]))

    spur_pw = [float(qp[qi, 1]) for qi in range(N) if qi not in used_q]
    return {
        'n_matched': len(used_t),
        'n_missed': M - len(used_t),
        'n_spurious': N - len(used_q),
        'cf_errors': np.array(cf_errors),
        'power_errors': np.array(pw_errors),
        'spurious_powers': np.array(spur_pw),
    }
