"""
Panels for the MLSP schematic figure
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ── synthetic spectrum ────────────────────────────────────────────────────

def make_spectrum(freqs, aperiodic=(1.5, 1.8), peak_cf=10.0, peak_pw=0.6, peak_bw=2.5):
    """1/f aperiodic + Gaussian peak, returns amplitude in µVp."""
    offset, exponent = aperiodic
    log_power = offset - exponent * np.log10(freqs)
    log_power += peak_pw * np.exp(-0.5 * ((freqs - peak_cf) / peak_bw) ** 2)
    amp = 10 ** (log_power / 2.0)
    return amp


def quantize_linear(amp, q):
    """Uniform scalar quantization in linear µVp, floor at 0."""
    n = np.round(amp / q)
    n = np.maximum(n, 0)
    return n * q


def to_db(amp, floor=1e-10):
    return 10.0 * np.log10(np.maximum(amp, floor))


# ── parameters ────────────────────────────────────────────────────────────

q = 0.109
freqs = np.arange(1, 42.5, 1.0)

np.random.seed(42)
amp_true = make_spectrum(freqs)
amp_true *= (1 + 0.02 * np.random.randn(len(freqs)))
amp_true = np.maximum(amp_true, 0)

amp_quant = quantize_linear(amp_true, q)
db_true = to_db(amp_true)
db_quant = to_db(amp_quant)


# ── colors ────────────────────────────────────────────────────────────────

C_TRUE = '#4F8FCC'
C_QUANT = '#C45B4A'
C_STORED = '#2D3142'
C_GRID = '#C45B4A'
C_BG_LOW = '#E8B4B4'
C_FIT = '#E69F00'
C_APERIODIC = '#999999'
C_REAL = '#5B9E6F'
C_SPUR = '#C45B4A'
C_DATA = '#2D3142'

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Liberation Serif', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'dejavuserif',
    'font.size': 10, 'axes.labelsize': 11,
    'axes.titlesize': 12, 'axes.titleweight': 'bold',
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 9, 'legend.frameon': False,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'lines.linewidth': 1.2,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.dpi': 600, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
})


# ── Panel (a): True signal in dB ─────────────────────────────────────────

fig_a, ax_a = plt.subplots(figsize=(3.5, 2.8))
ax_a.plot(freqs, db_true, color=C_TRUE, lw=1.8)
ax_a.set_xlabel('Frequency (Hz)')
ax_a.set_ylabel('dB')
ax_a.set_xlim(0, 43)
fig_a.savefig('panel_a_true_signal.svg', facecolor='white')
plt.close(fig_a)
print('Saved panel_a_true_signal.svg')


# ── Panel (b): Snap to grid in µVp ───────────────────────────────────────

fig_b, ax_b = plt.subplots(figsize=(3.5, 2.8))

crossover_freq = 13
ax_b.axvspan(0, crossover_freq, alpha=0.08, color='#4F8FCC', zorder=0)
ax_b.axvspan(crossover_freq, 43, alpha=0.12, color=C_BG_LOW, zorder=0)

y_max = amp_true.max() * 1.15
levels = np.arange(0, y_max, q)
for lev in levels:
    ax_b.axhline(lev, color=C_GRID, lw=0.2, alpha=0.35)

ax_b.plot(freqs, amp_true, color=C_TRUE, lw=1.2, alpha=0.45)
ax_b.plot(freqs, amp_quant, color=C_QUANT, lw=0.6, marker='.', ms=5, zorder=3)

ax_b.set_xlabel('Frequency (Hz)')
ax_b.set_ylabel('Amplitude (µVp)')
ax_b.set_xlim(0, 43)
ax_b.set_ylim(0, y_max)
fig_b.savefig('panel_b_snap_to_grid.svg', facecolor='white')
plt.close(fig_b)
print('Saved panel_b_snap_to_grid.svg')


# ── Panel (c): What Percept stores in dB ─────────────────────────────────

fig_c, ax_c = plt.subplots(figsize=(3.5, 2.8))
ax_c.plot(freqs, db_true, color=C_TRUE, lw=1.0, alpha=0.45, label='True')
ax_c.step(freqs, db_quant, where='mid', color=C_STORED, lw=1.2, label='Stored')
ax_c.set_xlabel('Frequency (Hz)')
ax_c.set_ylabel('Amplitude (µVp)')
ax_c.set_xlim(0, 43)
ax_c.legend(loc='upper right')
fig_c.savefig('panel_c_percept_stores.svg', facecolor='white')
plt.close(fig_c)
print('Saved panel_c_percept_stores.svg')


# ── Panel (d): FOOOF on quantized spectrum ────────────────────────────────
# Shows how staircase bumps trigger false peak detections.

from fooof import FOOOF

# Use steeper aperiodic for more visible plateaus
amp_true_d = make_spectrum(freqs, aperiodic=(1.5, 2.0))
amp_true_d *= (1 + 0.02 * np.random.randn(len(freqs)))
amp_true_d = np.maximum(amp_true_d, 0)
amp_quant_d = quantize_linear(amp_true_d, q)
db_quant_d = to_db(amp_quant_d)

fm = FOOOF(peak_width_limits=[1.0, 8.0], max_n_peaks=8,
           min_peak_height=0.02, peak_threshold=2.0)
fm.fit(freqs, np.maximum(amp_quant_d, 1e-10), [2, 42])

f_fit = fm.freqs
model_fit = 10.0 * fm.fooofed_spectrum_
aperiodic_fit = 10.0 * fm._ap_fit
peaks = fm.peak_params_

TRUE_CF = 10.0
real_mask = np.array([abs(p[0] - TRUE_CF) < 3.0 for p in peaks])
n_real = real_mask.sum()
n_spurious = len(peaks) - n_real

fig, ax = plt.subplots(figsize=(3.5, 2.8))

mask_plot = (freqs >= 2) & (freqs <= 42)
ax.step(freqs[mask_plot], db_quant_d[mask_plot], where='mid', color=C_DATA, lw=1.0,
        label='Quantized Spectrum', zorder=2)
ax.plot(f_fit, model_fit, color=C_FIT, lw=1.3, label='FOOOF fit', zorder=3)
ax.plot(f_fit, aperiodic_fit, color=C_APERIODIC, lw=0.8, ls='--',
        label='Aperiodic', zorder=1)

ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('dB')
ax.set_xlim(2, 42)
ax.legend(loc='upper right', fontsize=8)

fig.savefig('panel_d_fooof_spurious.png', facecolor='white')
plt.close(fig)
print('Saved panel_d_fooof_spurious.png')
print(f'Peaks: {len(peaks)} total, {n_real} real, {n_spurious} spurious')
for i, p in enumerate(peaks):
    tag = "REAL" if real_mask[i] else "SPUR"
    print(f'  {tag} CF={p[0]:.1f} Hz  PW={p[1]:.3f}  BW={p[2]:.1f}')
