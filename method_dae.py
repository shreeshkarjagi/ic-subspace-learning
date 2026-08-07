#denoising autoencoder baseline, section 3.6
import numpy as np

try:
    import torch
    import torch.nn as nn
    _HAVE_TORCH = True
except ImportError:
    _HAVE_TORCH = False

from forward_model import quantize


class _MLP(nn.Module if _HAVE_TORCH else object):
    def __init__(self, F, hidden=256, dropout=0.1):
        if not _HAVE_TORCH:
            raise ImportError('torch required')
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(F, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, F),
        )

    def forward(self, x):
        return self.net(x)


#fit on clean amps, quantized version goes in as input; comes back as plain numpy weights
def fit_dae(train_amps, q, seed=42, hidden=256, dropout=0.1,
            epochs=100, batch=64, lr=1e-3, wd=1e-4,
            val_frac=0.1, patience=10, verbose=False):
    if not _HAVE_TORCH:
        raise ImportError('torch required for DAE training')
    torch.manual_seed(seed)
    np.random.seed(seed)

    N, F = train_amps.shape
    quant = quantize(train_amps, q)
    mu = train_amps.mean(axis=0)
    sig = np.maximum(train_amps.std(axis=0), 1e-6)
    X = ((quant - mu) / sig).astype(np.float32)
    Y = ((train_amps - mu) / sig).astype(np.float32)

    idx = np.random.permutation(N)
    n_val = max(1, int(N * val_frac))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    Xt = torch.from_numpy(X[tr_idx])
    Yt = torch.from_numpy(Y[tr_idx])
    Xv = torch.from_numpy(X[val_idx])
    Yv = torch.from_numpy(Y[val_idx])

    torch.set_num_threads(1)
    model = _MLP(F, hidden=hidden, dropout=dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()

    best_val = float('inf')
    best_state = None
    bad = 0
    n_tr = len(tr_idx)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_tr)
        for i in range(0, n_tr, batch):
            sl = perm[i:i + batch]
            opt.zero_grad()
            loss_fn(model(Xt[sl]), Yt[sl]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), Yv).item()
        if not np.isfinite(val_loss):
            raise RuntimeError(f'DAE val loss non-finite at epoch {ep}')
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().cpu().numpy().copy()
                          for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if verbose:
        print(f'    DAE seed={seed}: {ep+1} ep, val_mse={best_val:.5f}')

    #layer indices track the Sequential above, they shift if the net changes
    return {
        'W1': best_state['net.0.weight'],
        'b1': best_state['net.0.bias'],
        'W2': best_state['net.3.weight'],
        'b2': best_state['net.3.bias'],
        'W3': best_state['net.6.weight'],
        'b3': best_state['net.6.bias'],
        'mu': mu, 'sig': sig,
        'val_mse': float(best_val),
        'epochs_trained': ep + 1,
    }


#forward pass in numpy so inference needs no torch
def correct_dae(freqs, quantized_amp, q, **kw):
    state = kw['dae_state']
    a = np.asarray(quantized_amp, dtype=np.float64)
    F = min(len(a), len(state['mu']))
    x = (a[:F] - state['mu'][:F]) / state['sig'][:F]
    h1 = np.maximum(x @ state['W1'][:, :F].T + state['b1'], 0.0)
    h2 = np.maximum(h1 @ state['W2'].T + state['b2'], 0.0)
    y_n = h2 @ state['W3'][:F, :].T + state['b3'][:F]
    y = y_n * state['sig'][:F] + state['mu'][:F]
    out = a.copy()
    out[:F] = np.maximum(y, 0.0)
    return out
