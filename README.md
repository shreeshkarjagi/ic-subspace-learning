# ic-subspace-learning

Code for "Subspace Learning with Interval-Censored Likelihoods for Dequantizing
Percept™ PC LFP Snapshots", IEEE MLSP 2026 (Atlanta, Sep 28 to Oct 1).

The Percept PC rounds every stored spectral amplitude to a multiple of
q ≈ 0.11 µVp. Adjacent bins whose true amplitudes differ by less than q collapse
onto the same value, flattening the spectrum into plateaus. FOOOF reads those
plateaus as oscillations (20.6% of the peaks it finds at 2 to 45 Hz are not in
the clean spectrum). We treat the correction as interval-censored subspace
estimation & compare 7 corrections against it. Q-PPCA is the only one that
helps, cutting the spurious rate to 18.3% while peak detection barely moves
(81.4% to 81.1%). That is a 2.24 pp reduction, lower in 14 of 14 hemispheres,
Wilcoxon signed-rank p = 1.2e-4 two-sided, 95% bootstrap CI [1.86, 2.60] pp over
9,438 paired spectra.

## Data

Raw recordings are PHI and cannot be shared. 7 patients (14 hemispheres) with
subcallosal cingulate DBS, consented under an IRB-approved protocol.

To run on other Percept exports: each immediate subdirectory of the data
directory is treated as one patient, and every JSON below it is picked up at any
depth. Ours are laid out as

```
RawData/PATIENT_ID/0_LFP/SESSION_FOLDER/*.json
```

The loader keeps IndefiniteStreaming blocks (stim off, all six channels) of at
least 30 s.

Ground truth is synthetic, because the device discards the time domain after
computing each snapshot. Instead we segment the in-clinic recordings into 30 s
segments (9,438 in total), compute clean Welch spectra as the target, and
quantize those to make the input.

## Setup

```
pip install -r requirements.txt
```

`requirements.txt` pins fooof 1.1.0, the version behind the paper's numbers.
Steps 3 to 5 fall back to specparam if fooof is missing, but
`graphical_abstract.py` needs fooof outright.

## Running

```
python run_pipeline.py /path/to/RawData
python run_pipeline.py /path/to/RawData --steps 1 2 --workers 16
```

Step 1 takes the data directory; every later step reads from `results/`.

1. [prep_data.py](prep_data.py) load JSONs, segment to 30 s, one npz per hemisphere
2. [run_loo.py](run_loo.py) all methods, leave one recording out
3. [specparam_correction.py](specparam_correction.py) FOOOF on corrected spectra
4. [specparam_bias.py](specparam_bias.py) FOOOF on uncorrected spectra, the 20.6% baseline
5. [freq_sweep.py](freq_sweep.py) spurious rate against FOOOF fit range
6. [rank_sweep.py](rank_sweep.py) K in {3, 5, 7} on the largest hemisphere
7. [run_lopo.py](run_lopo.py) leave one patient out
8. [make_figures.py](make_figures.py) paper figures
9. [run_dae.py](run_dae.py) DAE across seeds 42, 43, 44

[spurious_paired_test.py](spurious_paired_test.py) runs the paired Wilcoxon and
bootstrap CI on the 20.6% to 18.3% difference. It reads `results/eval`, so run
step 2 first. [graphical_abstract.py](graphical_abstract.py) draws the Figure 1
schematic from a synthetic spectrum and writes its panels to the working
directory. Neither is part of `run_pipeline.py`.

Figure 5 needs the raw data the first time, then caches
`results/snapshot_exemplar.npz` and reuses it:

```
RECAST_DATA_DIR=/path/to/RawData python make_figures.py --only fig5
```

## Methods

These keys suffix the `corrected_*` arrays in `results/eval/*.npz` and name the
fields in the summary JSONs.

| Key | Method | Where |
|---|---|---|
| `raw` | uncorrected reference | [methods.py:10](methods.py#L10) |
| `sg` | Savitzky-Golay in dB, 5-bin window | [methods.py:14](methods.py#L14) |
| `sg_sel` | same filter, plateau bins only | [methods.py:35](methods.py#L35) |
| `svd` | projection onto the training PCA basis | [methods.py:78](methods.py#L78) |
| `sccd` | MAP scores under hard interval bounds, CVXPY and OSQP | [methods.py:88](methods.py#L88) |
| `qppca` | quantized PPCA, truncated normal E-step | [methods.py:129](methods.py#L129) |
| `qmf` | quantized matrix factorization | [methods.py:168](methods.py#L168) |
| `dae` | denoising autoencoder | [method_dae.py](method_dae.py) |

`raw`, `sg` and `sg_sel` need no training data. `qppca` and `qmf` see only
quantized training spectra. `svd`, `sccd` and `dae` are given the unquantized
ones, which the DAE quantizes internally to form its inputs. K = 5 by default.

RMSE covers the bands set in [forward_model.py](forward_model.py#L33): well
resolved (1 to 8 Hz), correctable (8 to 30 Hz), noise floor (30 to 42 Hz).

## Output

```
results/data        segmented spectra, one npz per hemisphere
results/eval        corrected spectra and RMSE tables      (step 2)
results/specparam   FOOOF metrics, frequency sweep, paired test
results/k_sweep     K sensitivity                          (step 6)
results/lopo        leave-one-patient-out                  (step 7)
results/dae         DAE seed runs and console logs         (step 9)
figures/            paper figures
```

The committed results match the paper so all reported results can be checked without the raw data. The
per-segment arrays in `results/data` and `results/eval` are not tracked: 60 MB of patient derived spectra, held back due PHI restrictions under the study IRB.

`results/eval/*.npz` holds `freqs`, `truth` and one `corrected_*` array per
method, in µVp amplitude, one row per segment. A NaN row means that method
failed on that fold. The LOPO summary covers `raw`, `sg_sel`, `svd`, `sccd`,
`qppca` and `qmf` only. SG needs no training, so its LOPO figures equal its LOO
figures; the DAE's cross-validation is in `results/dae`.

Steps 2 and 9 are slow: `sccd` solves a QP per spectrum, and 9 retrains the DAE
for every fold and seed.

## Key numbers

Uncorrected, at 2 to 45 Hz (`results/specparam/bias_results.json`): 20.6%
spurious detections (6,342 of 30,823), 81.4% of true peaks recovered, exponent
RMSE 0.037 against a cohort spread of 0.32, FOOOF R² 0.991 to 0.987. Band power
RMSE in dB is 0.048 theta, 0.058 alpha, 0.062 beta, 0.176 low gamma.

Widening the fit range makes it worse (`freq_sweep_results.json`): spurious
15.5%, 20.6%, 44.2% and detection 86.9%, 81.4%, 55.3% at 2 to 30, 2 to 45 and
2 to 80 Hz.

Per method at 2 to 45 Hz, detection / spurious / exponent RMSE
(`correction_results.json`): raw 81.4 / 20.6 / 0.037, sg 76.2 / 20.0 / 0.039,
sg_sel 78.4 / 20.5 / 0.037, svd 40.6 / 45.8 / 8.95, sccd 51.6 / 32.5 / 0.080,
qppca 81.1 / 18.3 / 0.033, qmf 73.0 / 19.7 / 0.053, dae 52.0 / 32.2 / 0.074.
`svd` is the only method with failed fits (24, so n = 9,414).

Cohort (`data_summary.json`): 7 patients, 14 hemispheres, 1,116 recordings
(45 to 114 each), 9,438 segments (366 to 936 each), 129 bins at 0.977 Hz.

## Citation

If you would like to use our code, please cite us with the
following.

```bibtex
@inproceedings{karjagi2026subspace,
  title     = {Subspace Learning with Interval-Censored Likelihoods for
               Dequantizing {Percept} {PC} {LFP} Snapshots},
  author    = {Karjagi, Shreesh and Fitoz, Elif Ceren and Khalid, Maryam and
               Nauvel, Tanya and Sarikhani, Parisa and Mayberg, Helen S. and
               Rozell, Christopher J. and Alagapan, Sankaraleengam},
  booktitle = {IEEE International Workshop on Machine Learning for Signal
               Processing (MLSP)},
  address   = {Atlanta, GA, USA},
  month     = sep,
  year      = {2026}
}
```
