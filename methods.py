import numpy as np
import warnings
from scipy.signal import savgol_filter
from scipy.stats import norm
from forward_model import quantize, quantization_interval, to_db, from_db


def correct_raw(freqs, quantized_amp, q, **kw):
    return np.array(quantized_amp, dtype=np.float64)


#savgol in dB space, zeros are interpolated over and then put back
def correct_sg(freqs, quantized_amp, q, **kw):
    a = np.array(quantized_amp, dtype=np.float64)
    pos = a > 0
    if pos.sum() < 5: return a
    db = np.full_like(a, np.nan); db[pos] = to_db(a[pos])
    v = np.where(pos)[0]
    filled = np.interp(np.arange(len(db)), v, db[v])
    sm = savgol_filter(filled, window_length=5, polyorder=2)

    out = a.copy(); out[pos] = from_db(sm[pos])
    return np.maximum(out, 0.0)


#runs of 2+ bins stuck at the same dB value, i.e. the quantization plateaus
def _plateaus(db, tol=0.001):
    n = len(db); mask = np.zeros(n, bool); i = 0
    while i < n:
        j = i + 1
        while j < n and abs(db[j] - db[i]) < tol: j += 1
        if j - i >= 2: mask[i:j] = True
        i = j
    return mask


def correct_sg_sel(freqs, quantized_amp, q, **kw):
    #same smoothing as correct_sg but only bins inside a plateau get replaced
    a = np.array(quantized_amp, dtype=np.float64)
    pos = a > 0
    if pos.sum() < 5: return a
    db = np.full_like(a, np.nan); db[pos] = to_db(a[pos])
    v = np.where(pos)[0]
    filled = np.interp(np.arange(len(db)), v, db[v])
    sm = savgol_filter(filled, window_length=5, polyorder=2)

    plat = _plateaus(filled)
    out = a.copy(); replace = pos & plat
    out[replace] = from_db(sm[replace])
    return np.maximum(out, 0.0)


#needs specparam installed, otherwise falls through to the quantized input
def correct_specparam(freqs, quantized_amp, q, **kw):
    try:
        from specparam import SpectralModel
    except ImportError:
        return np.array(quantized_amp, dtype=np.float64)
    a = np.array(quantized_amp, dtype=np.float64)
    fr = kw.get('freq_range', [2, 30])

    #specparam wants power, we carry amplitude everywhere else
    power = np.maximum(a, 1e-10) ** 2
    sm = SpectralModel(peak_width_limits=[2,8], max_n_peaks=4,
                       min_peak_height=0.1, aperiodic_mode='fixed', verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try: sm.fit(freqs, power, freq_range=fr)
        except: return a

    out = a.copy()
    mask = (freqs >= fr[0]) & (freqs <= fr[1])
    if hasattr(sm, 'fooofed_spectrum_') and sm.fooofed_spectrum_ is not None:
        mp = 10 ** sm.fooofed_spectrum_
        n = min(len(mp), mask.sum())
        out[np.where(mask)[0][:n]] = np.sqrt(np.maximum(mp[:n], 0.0))
    return np.maximum(out, 0.0)


#z-scored pca, returns mean, scale, top-K components and their variances
def build_basis(X, K=5):
    mu = X.mean(0); sig = np.maximum(X.std(0), 1e-6)
    U, S, Vt = np.linalg.svd((X - mu) / sig, full_matrices=False)
    K = min(K, len(S))
    return mu, sig, Vt[:K], (S[:K]**2) / max(X.shape[0]-1, 1)


def correct_svd(freqs, quantized_amp, q, **kw):
    mu, sig, B = kw['mu'], kw['sigma'], kw['basis']
    a = np.array(quantized_amp, dtype=np.float64)
    F = min(len(a), len(mu)); K = B.shape[0]
    z = (a[:F] - mu[:F]) / sig[:F]
    c = B[:,:F] @ z
    r = mu[:F] + sig[:F] * (B[:,:F].T @ c)
    out = a.copy(); out[:F] = np.maximum(r, 0.0)
    return out


def correct_sccd(freqs, quantized_amp, q, **kw):
    import cvxpy as cp
    mu, sig, B, ev = kw['mu'], kw['sigma'], kw['basis'], kw['eigenvalues']
    a = np.array(quantized_amp, dtype=np.float64)
    F = min(len(a), len(mu)); K = B.shape[0]

    #only constrain bins we trust: in band and above the floor
    fmask = (freqs[:F] >= 1.0) & (freqs[:F] <= 42.0) & (a[:F] > 0)
    ci = np.where(fmask)[0]
    if len(ci) < K: return a

    #cheapest coefficients that still land the recon inside every quantization bin
    lo, hi = quantization_interval(a[:F], q)
    c = cp.Variable(K)
    recon = mu[:F] + cp.multiply(sig[:F], B[:,:F].T @ c)
    prob = cp.Problem(cp.Minimize(cp.sum(cp.multiply(1.0/np.maximum(ev,1e-8), cp.square(c)))),
                      [recon[ci] >= lo[ci], recon[ci] <= hi[ci]])
    try: prob.solve(solver=cp.OSQP, max_iter=10000, eps_abs=1e-6, eps_rel=1e-6)
    except: pass
    if prob.status in ('optimal','optimal_inaccurate') and c.value is not None:
        r = mu[:F] + sig[:F] * (B[:,:F].T @ c.value)
        out = a.copy(); out[:F] = np.maximum(r, 0.0); return out

    #infeasible, so let the bins slip with a heavy penalty
    sl = cp.Variable(len(ci), nonneg=True)
    prob2 = cp.Problem(
        cp.Minimize(cp.sum(cp.multiply(1.0/np.maximum(ev,1e-8), cp.square(c))) + 1000*cp.sum(sl)),
        [recon[ci] >= lo[ci]-sl, recon[ci] <= hi[ci]+sl])
    try: prob2.solve(solver=cp.OSQP, max_iter=10000)
    except: return a
    if c.value is not None:
        r = mu[:F] + sig[:F] * (B[:,:F].T @ c.value)
        out = a.copy(); out[:F] = np.maximum(r, 0.0); return out
    return a


#mean and variance of a gaussian truncated to [lo, hi], elementwise
def _trunc_moments(mu, sd, lo, hi):
    sd = np.maximum(sd, 1e-12)
    a, b = (lo-mu)/sd, (hi-mu)/sd
    pa, pb, ca, cb = norm.pdf(a), norm.pdf(b), norm.cdf(a), norm.cdf(b)
    d = np.maximum(cb-ca, 1e-15)
    lam = (pa-pb)/d; delta = (a*pa - b*pb)/d
    m = mu + sd*lam; v = sd**2 * np.maximum(1+delta-lam**2, 0)

    #degenerate interval, fall back to the clipped mean
    bad = d < 1e-12; m[bad] = np.clip(mu[bad], lo[bad], hi[bad]); v[bad] = 0
    return m, v


#em for ppca where every observation is only known to lie in its quantization bin
def fit_qppca(Y_q, q, K=5, max_iter=50, tol=1e-4):
    Y = np.array(Y_q, dtype=np.float64); N, F = Y.shape
    K = min(K, N-1, F-1)
    Lo, Hi = quantization_interval(Y, q)
    mu = Y.mean(0)
    U_, S_, Vt_ = np.linalg.svd(Y-mu, full_matrices=False)
    raw_var = S_**2 / max(N-1,1)
    sig2 = max(float(np.mean(raw_var[K:])) if K<len(raw_var) else 0.01, 1e-6)
    W = Vt_[:K].T * np.sqrt(np.maximum(raw_var[:K]-sig2, 1e-8))
    for _ in range(max_iter):
        W_old = W.copy()
        M_inv = np.linalg.inv(W.T@W + sig2*np.eye(K))
        cov_f = sig2 * M_inv
        msd = np.sqrt(np.maximum(np.sum((W@cov_f)*W, 1) + sig2, 1e-10))

        #e-step, one truncated normal per segment
        Sh, S2h = np.zeros_like(Y), np.zeros_like(Y)
        for i in range(N):
            pred = W @ (M_inv @ (W.T @ (Y[i]-mu))) + mu
            m, v = _trunc_moments(pred, msd, Lo[i], Hi[i])
            Sh[i], S2h[i] = m, v + m**2

        mu = Sh.mean(0); Sc = Sh - mu
        C = Sc.T@Sc/N + np.diag(np.mean(S2h - Sh**2, 0))
        ev, ec = np.linalg.eigh(C); idx = np.argsort(ev)[::-1]
        sig2 = max(float(np.mean(ev[idx[K:]])) if K<F else 1e-6, 1e-6)
        W = ec[:,idx[:K]] * np.sqrt(np.maximum(ev[idx[:K]]-sig2, 1e-10))
        if np.linalg.norm(W-W_old)/(np.linalg.norm(W)+1e-10) < tol: break
    return W, mu, sig2


def correct_qppca(freqs, quantized_amp, q, **kw):
    W, mu_m, sig2 = kw['W'], kw['mu_qppca'], kw.get('sigma2', 0.01)
    a = np.array(quantized_amp, dtype=np.float64)
    F = min(len(a), len(mu_m), W.shape[0]); K = W.shape[1]
    lo, hi = quantization_interval(a[:F], q)
    M_inv = np.linalg.inv(W[:F].T@W[:F] + sig2*np.eye(K))
    pred = W[:F] @ (M_inv @ (W[:F].T @ (a[:F]-mu_m[:F]))) + mu_m[:F]
    msd = np.sqrt(np.maximum(np.sum((W[:F]@(sig2*M_inv))*W[:F],1)+sig2, 1e-10))
    m, _ = _trunc_moments(pred, msd, lo, hi)
    out = a.copy(); out[:F] = m; return np.maximum(out, 0.0)


#adam on the interval likelihood, betas are hardcoded 0.9/0.999 inline
def fit_qmf(Y_q, q, K=5, lr=0.001, max_iter=300, tol=1e-5):
    Y = np.array(Y_q, dtype=np.float64); N, F = Y.shape
    K = min(K, N-1, F-1); Lo, Hi = quantization_interval(Y, q)
    sigma = q/2.0; mu = Y.mean(0)
    U_, S_, Vt_ = np.linalg.svd(Y-mu, full_matrices=False)
    sc = np.sqrt(S_[:K]); U = U_[:,:K]*sc; V = Vt_[:K].T*sc
    mU,vU = np.zeros_like(U), np.zeros_like(U)
    mV,vV = np.zeros_like(V), np.zeros_like(V)
    prev_ll = -np.inf
    for it in range(max_iter):
        M = U@V.T + mu
        a_z, b_z = (Lo-M)/sigma, (Hi-M)/sigma
        Pa, Pb, pa, pb = norm.cdf(a_z), norm.cdf(b_z), norm.pdf(a_z), norm.pdf(b_z)
        dc = np.maximum(Pb-Pa, 1e-15); ll = np.sum(np.log(dc))
        if it>0 and abs(ll-prev_ll) < tol*max(1,abs(ll)): break
        prev_ll = ll

        #clip the score, it blows up when a bin gets very unlikely
        dM = np.clip((pa-pb)/dc/sigma, -5, 5)
        gU, gV = dM@V, dM.T@U; t = it+1
        for m_,v_,g in [(mU,vU,gU),(mV,vV,gV)]:
            m_[:] = 0.9*m_ + 0.1*g; v_[:] = 0.999*v_ + 0.001*g**2
        U += lr*(mU/(1-0.9**t))/(np.sqrt(vU/(1-0.999**t))+1e-8)
        V += lr*(mV/(1-0.9**t))/(np.sqrt(vV/(1-0.999**t))+1e-8)
        mu = (Y - U@V.T).mean(0)
    return U, V, mu, sigma


def correct_qmf(freqs, quantized_amp, q, **kw):
    V, mu_m, sig = kw['V_qmf'], kw['mu_qmf'], kw.get('sigma_qmf', 0.055)
    a = np.array(quantized_amp, dtype=np.float64)
    F = min(len(a), V.shape[0])
    u = np.linalg.lstsq(V[:F], a[:F]-mu_m[:F], rcond=None)[0]
    pred = V[:F]@u + mu_m[:F]

    lo, hi = quantization_interval(a[:F], q)
    out = a.copy(); out[:F] = np.clip(pred, lo, hi)
    return np.maximum(out, 0.0)
