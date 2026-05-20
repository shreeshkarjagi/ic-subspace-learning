"""
Denoising autoencoder baseline (Section 3.6)
"""
import numpy as np
import torch
import torch.nn as nn
from forward_model import quantize


class _DAE(nn.Module):
    def __init__(self, F):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(F, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, F),
        )
    def forward(self, x):
        return self.net(x)


def fit_dae(train_amps, q, seed=42, max_epochs=100, patience=10, batch_size=64,
            lr=1e-3, weight_decay=1e-4, val_frac=0.1):
    """Train DAE on clean amps, using quantized version as input."""
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    clean = np.array(train_amps, dtype=np.float64)
    N, F = clean.shape
    quant = np.array([quantize(clean[i], q) for i in range(N)])

    # standardize using clean statistics
    mu = clean.mean(0)
    sig = np.maximum(clean.std(0), 1e-6)
    clean_z = (clean - mu) / sig
    quant_z = (quant - mu) / sig

    # train/val split
    idx = rng.permutation(N)
    n_val = max(int(N * val_frac), 1)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_tr = torch.tensor(quant_z[train_idx], dtype=torch.float32)
    Y_tr = torch.tensor(clean_z[train_idx], dtype=torch.float32)
    X_va = torch.tensor(quant_z[val_idx], dtype=torch.float32)
    Y_va = torch.tensor(clean_z[val_idx], dtype=torch.float32)

    model = _DAE(F)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val = np.inf
    best_state = None
    wait = 0

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(len(X_tr))
        for i in range(0, len(X_tr), batch_size):
            batch = perm[i:i+batch_size]
            opt.zero_grad()
            loss_fn(model(X_tr[batch]), Y_tr[batch]).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_va), Y_va).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        'model': model,
        'mu': mu,
        'sig': sig,
        'val_mse': float(best_val),
        'epochs_trained': epoch + 1,
    }


def correct_dae(freqs, quantized_amp, q, **kw):
    sd = kw['dae_state']
    model, mu, sig = sd['model'], sd['mu'], sd['sig']
    a = np.array(quantized_amp, dtype=np.float64)
    F = min(len(a), len(mu))
    z = (a[:F] - mu[:F]) / sig[:F]
    model.eval()
    with torch.no_grad():
        out_z = model(torch.tensor(z, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
    out = a.copy()
    out[:F] = out_z * sig[:F] + mu[:F]
    return np.maximum(out, 0.0)
