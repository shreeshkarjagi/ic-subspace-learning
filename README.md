# RECAST

Code for "Subspace Learning with Interval-Censored Likelihoods for
Dequantizing Percept PC LFP Snapshots" (IEEE MLSP 2026).

## Data

Raw clinical data cannot be shared due to PHI restrictions under the
study IRB. De-identified summary statistics available on request.

## Setup

```
pip install -r requirements.txt
```

## Usage

```bash
python run_pipeline.py /path/to/RawData
python run_pipeline.py /path/to/RawData --steps 1 2 3
```

Or run individual scripts (see `run_pipeline.py` for the order).
Results in `results/`, figures in `figures/`.
