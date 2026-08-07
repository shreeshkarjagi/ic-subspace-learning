#quantization forward model for percept fftbindata
import numpy as np
from scipy.signal import welch


def quantize(amplitude, q):
    return np.round(np.asarray(amplitude, dtype=np.float64) / q) * q


def quantization_interval(y, q):
    #[lo, hi] for each quantized value, lo clipped at 0
    y = np.asarray(y, dtype=np.float64)
    return np.maximum(y - q / 2.0, 0.0), y + q / 2.0


def to_db(amp):
    return 10.0 * np.log10(np.maximum(np.asarray(amp, dtype=np.float64), 1e-10))


def from_db(db):
    return 10.0 ** (np.asarray(db, dtype=np.float64) / 10.0)


#psd density -> per-bin amplitude, same units the device reports
def stream_to_amp(timeseries, fs, nperseg=256, noverlap=128):
    f, pxx = welch(np.asarray(timeseries, dtype=np.float64),
                   fs=fs, nperseg=nperseg, noverlap=noverlap,
                   window='hann', scaling='density', return_onesided=True)
    df = f[1] - f[0] if len(f) > 1 else 1.0
    return f, np.sqrt(2.0 * np.maximum(pxx, 0.0) * df)


def band_power_db(amp, freqs, flo, fhi):
    mask = (freqs >= flo) & (freqs < fhi)
    if mask.sum() == 0: return np.nan
    return float(to_db(np.mean(amp[mask] ** 2)))


#hz ranges grouped by how hard quantization bites
REGIMES = {
    'well_resolved': (1.0, 8.0),
    'correctable':   (8.0, 30.0),
    'noise_floor':   (30.0, 42.0),
}
BANDS = {'theta': (4,8), 'alpha': (8,13), 'beta': (13,30), 'low_gamma': (30,50)}
