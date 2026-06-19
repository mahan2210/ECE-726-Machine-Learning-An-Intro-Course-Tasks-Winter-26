"""
ECE 726 — Assignment 3: BONUS EXPERIMENTS  (Fast Version)
McMaster University, Winter 2026
Student: Mahan Choudhury  |  ID: 400648097  |  Seed: 8097

Run this file AFTER the main assignment3_neural_networks.py has finished.
It is completely independent — no imports from the main file.

Speed optimizations vs. the main experiment
-------------------------------------------
  1. Minibatch only (B=128) — stochastic (B=1) proved to fail/be extremely
     slow in the main experiment; no point repeating it here.
  2. Eval on a 4096-sample training subset each epoch (not 48k).
     Final reported errors still use the full dataset.
  3. float32 weights throughout — halves memory bandwidth.
  4. Early stopping patience = 10 (tighter than main's 15).
  5. Max epochs = 150 (sufficient for minibatch models).

Expected runtime: ~5-15 minutes total on a modern laptop.

Experiment overview (all use 3 hidden layers, H=80, lambda=0.001232, B=128)
-----------------------------------------------------------------------
  Baseline   : Adam-Minibatch lr=0.01, ReLU   (reproduces main result)
  Exp 1      : Adam-Minibatch lr=0.001         (corrected learning rate)
  Exp 2a     : Adam-Minibatch lr=0.001, Leaky ReLU
  Exp 2b     : Adam-Minibatch lr=0.001, ELU
  Exp 3      : Adam-Minibatch lr=0.003, Batch Normalization
  Exp 4      : SGD-Minibatch, Cosine Annealing (lr_0=0.05 to lr_min=1e-4)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os, time, warnings
warnings.filterwarnings('ignore')

# Seed
SEED = 8097
np.random.seed(SEED)

try:
    import torch
    from torchvision import datasets
    torch.manual_seed(SEED)
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

os.makedirs('./bonus_results', exist_ok=True)

# ================================================================
# 1. DATA  (identical pipeline to main experiment)
# ================================================================

def one_hot(labels, n=10):
    return np.eye(n, dtype='float32')[labels]


def load_data(data_dir='./data'):
    """Load Fashion MNIST, split 80/20, standardize -- same as main."""
    if TORCH_AVAILABLE:
        tr = datasets.FashionMNIST(root=data_dir, train=True,  download=True)
        te = datasets.FashionMNIST(root=data_dir, train=False, download=True)
        X_full = tr.data.numpy().reshape(-1, 784).astype('float32') / 255.
        Y_full = tr.targets.numpy().astype(np.int64)
        X_te   = te.data.numpy().reshape(-1, 784).astype('float32') / 255.
        Y_te   = te.targets.numpy().astype(np.int64)
    else:
        import struct, gzip, urllib.request
        base  = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
        names = ['train-images-idx3-ubyte', 'train-labels-idx1-ubyte',
                 't10k-images-idx3-ubyte',  't10k-labels-idx1-ubyte']
        os.makedirs(data_dir, exist_ok=True)
        def fetch(fname):
            p = os.path.join(data_dir, fname)
            if not os.path.exists(p):
                gz = p + '.gz'
                urllib.request.urlretrieve(base + fname + '.gz', gz)
                with gzip.open(gz, 'rb') as fi, open(p, 'wb') as fo:
                    fo.write(fi.read())
                os.remove(gz)
            return p
        def read_img(p):
            with open(p, 'rb') as f:
                _, n, r, c = struct.unpack('>IIII', f.read(16))
                return (np.frombuffer(f.read(), np.uint8)
                          .reshape(n, r * c).astype('float32') / 255.)
        def read_lbl(p):
            with open(p, 'rb') as f:
                _, n = struct.unpack('>II', f.read(8))
                return np.frombuffer(f.read(), np.uint8).astype(np.int64)
        X_full, Y_full = read_img(fetch(names[0])), read_lbl(fetch(names[1]))
        X_te,   Y_te   = read_img(fetch(names[2])), read_lbl(fetch(names[3]))

    # 80/20 split -- same seed/shuffle as main experiment
    N = X_full.shape[0]
    np.random.seed(SEED)
    idx = np.random.permutation(N)
    v   = int(0.2 * N)
    X_v,  Y_v_o  = X_full[idx[:v]], Y_full[idx[:v]]
    X_tr, Y_tr_o = X_full[idx[v:]], Y_full[idx[v:]]

    # Standardize using training-set statistics only
    mu  = X_tr.mean(0);  sd = X_tr.std(0);  sd[sd == 0] = 1.
    X_tr = (X_tr - mu) / sd
    X_v  = (X_v  - mu) / sd
    X_te = (X_te - mu) / sd

    print(f'  Train: {X_tr.shape}  Val: {X_v.shape}  Test: {X_te.shape}')
    return (X_tr, one_hot(Y_tr_o), Y_tr_o,
            X_v,  one_hot(Y_v_o),  Y_v_o,
            X_te, one_hot(Y_te),   Y_te)


# ================================================================
# 2. ACTIVATIONS  (all from scratch, NumPy only)
# ================================================================

def softmax(Z):
    e = np.exp(Z - Z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


# Each entry: (forward_fn, derivative_fn)
ACTIVATIONS = {
    'relu': (
        lambda Z: np.maximum(0., Z),
        lambda Z: (Z > 0).astype('float32'),
    ),
    'leaky_relu': (
        # f(z) = z if z > 0, else 0.01 * z
        lambda Z: np.where(Z > 0, Z, 0.01 * Z),
        # f'(z) = 1 if z > 0, else 0.01
        lambda Z: np.where(Z > 0, 1., 0.01).astype('float32'),
    ),
    'elu': (
        # f(z) = z if z > 0, else exp(z) - 1   (alpha=1)
        lambda Z: np.where(Z > 0, Z, np.expm1(np.clip(Z, -50, 0))),
        # f'(z) = 1 if z > 0, else f(z) + 1
        lambda Z: np.where(Z > 0, 1.,
                           np.expm1(np.clip(Z, -50, 0)) + 1.).astype('float32'),
    ),
}


# ================================================================
# 3. NEURAL NETWORK  (NumPy only, with optional Batch Normalization)
# ================================================================

class BonusNet:
    """
    Fully-connected feedforward network.

    Supports three hidden activations: 'relu', 'leaky_relu', 'elu'.

    Batch Normalization (Ioffe & Szegedy, 2015)
    -------------------------------------------
    After computing Z = A_{l-1} @ W + b, before the activation:

        Z_hat   = (Z - mu_B) / sqrt(var_B + eps)     [normalize over batch]
        Z_tilde = gamma * Z_hat + beta                [learnable scale/shift]

    During training:  mu_B, var_B are batch statistics.
    During inference: use running averages (updated with momentum=0.9).

    Backward gradient for the normalization step (Eq. 6, Ioffe & Szegedy):

        dZ = (1/N) * (1/sqrt(var+eps)) * [
                N * dZ_hat
              - sum(dZ_hat, axis=0)
              - Z_hat * sum(dZ_hat * Z_hat, axis=0)
             ]
    """

    def __init__(self, layer_sizes, weight_decay=0.001232,
                 activation='relu', use_bn=False, seed=SEED):
        self.sizes    = layer_sizes
        self.L        = len(layer_sizes) - 1
        self.wd       = weight_decay
        self.act_fn, self.dact_fn = ACTIVATIONS[activation]
        self.use_bn   = use_bn
        self.training = True

        # Weight and bias parameters
        self.params = {}
        # Batch norm learnable parameters: gamma (scale) and beta (shift)
        self.bn     = {}
        # Running statistics for inference
        self.bn_run = {}

        rng = np.random.default_rng(seed)
        for l in range(1, len(layer_sizes)):
            fi, fo = layer_sizes[l - 1], layer_sizes[l]
            # He initialization for hidden layers (works well with ReLU variants)
            # Xavier initialization for the output layer
            sc = (np.sqrt(2. / fi) if l < len(layer_sizes) - 1
                  else np.sqrt(1. / fi))
            self.params[f'W{l}'] = rng.normal(0, sc, (fi, fo)).astype('float32')
            self.params[f'b{l}'] = np.zeros((1, fo), dtype='float32')
            if use_bn and l < len(layer_sizes) - 1:
                self.bn[f'g{l}']       = np.ones( (1, fo), dtype='float32')
                self.bn[f'b{l}']       = np.zeros((1, fo), dtype='float32')
                self.bn_run[f'mu{l}']  = np.zeros((1, fo), dtype='float32')
                self.bn_run[f'var{l}'] = np.ones( (1, fo), dtype='float32')

    def forward(self, X):
        """Forward pass, caching all intermediates needed for backprop."""
        cache = {'A0': X}
        A = X
        for l in range(1, self.L + 1):
            Z = A @ self.params[f'W{l}'] + self.params[f'b{l}']
            cache[f'Zpre{l}'] = Z          # pre-BN pre-activation

            if l < self.L:
                # Batch Normalization
                if self.use_bn:
                    if self.training:
                        mu  = Z.mean(0, keepdims=True)
                        var = Z.var(0,  keepdims=True)
                        # Update running statistics (momentum = 0.9)
                        self.bn_run[f'mu{l}']  = (0.9 * self.bn_run[f'mu{l}']
                                                   + 0.1 * mu)
                        self.bn_run[f'var{l}'] = (0.9 * self.bn_run[f'var{l}']
                                                   + 0.1 * var)
                    else:
                        mu  = self.bn_run[f'mu{l}']
                        var = self.bn_run[f'var{l}']
                    Zh = (Z - mu) / np.sqrt(var + 1e-8)
                    cache[f'mu{l}']  = mu
                    cache[f'var{l}'] = var
                    cache[f'Zh{l}']  = Zh
                    Z = self.bn[f'g{l}'] * Zh + self.bn[f'b{l}']

                cache[f'Z{l}'] = Z
                # Activation
                A = self.act_fn(Z)
                cache[f'dA{l}'] = self.dact_fn(Z)   # element-wise derivative
            else:
                cache[f'Z{l}'] = Z
                A = softmax(Z)

            cache[f'A{l}'] = A
        return cache

    def loss(self, Y_pred, Y_true):
        """Cross-entropy loss + L2 weight decay."""
        N  = Y_true.shape[0]
        ce = -np.sum(Y_true * np.log(np.clip(Y_pred, 1e-12, 1.))) / N
        ce += (0.5 * self.wd
               * sum(np.sum(self.params[f'W{l}'] ** 2)
                     for l in range(1, self.L + 1)))
        return float(ce)

    def backward(self, cache, Y_true):
        """
        Backpropagation through the network.

        Returns
        -------
        grads    : dict  gradients for W and b at each layer
        bn_grads : dict  gradients for gamma and beta (empty if use_bn=False)
        """
        N, grads, bn_grads = Y_true.shape[0], {}, {}

        # Combined Softmax + cross-entropy gradient at the output
        dZ = (cache[f'A{self.L}'] - Y_true) / N

        for l in range(self.L, 0, -1):
            Ap = cache[f'A{l - 1}']
            grads[f'W{l}'] = Ap.T @ dZ + self.wd * self.params[f'W{l}']
            grads[f'b{l}'] = dZ.sum(0, keepdims=True)

            if l > 1:
                dA = dZ @ self.params[f'W{l}'].T
                dZ = dA * cache[f'dA{l - 1}']       # through activation

                if self.use_bn:
                    # Gradients for gamma and beta
                    bn_grads[f'g{l-1}'] = (dZ * cache[f'Zh{l-1}']).sum(0, keepdims=True)
                    bn_grads[f'b{l-1}'] = dZ.sum(0, keepdims=True)

                    # Gradient through the normalization (Ioffe & Szegedy Eq. 6)
                    dZh     = dZ * self.bn[f'g{l-1}']
                    inv_std = 1. / np.sqrt(cache[f'var{l-1}'] + 1e-8)
                    Zh      = cache[f'Zh{l-1}']
                    dZ = inv_std / N * (
                        N * dZh
                        - dZh.sum(0, keepdims=True)
                        - Zh * (dZh * Zh).sum(0, keepdims=True)
                    )

        return grads, bn_grads

    def predict(self, X):
        """Integer class predictions (inference mode)."""
        was = self.training
        self.training = False
        c = self.forward(X)
        self.training = was
        return np.argmax(c[f'A{self.L}'], 1)

    def error(self, X, Y_int):
        """Misclassification error (fraction)."""
        return float(np.mean(self.predict(X) != Y_int))

    # Snapshot / restore for early stopping
    def snapshot(self):
        return ({k: v.copy() for k, v in self.params.items()},
                {k: v.copy() for k, v in self.bn.items()},
                {k: v.copy() for k, v in self.bn_run.items()})

    def restore(self, snap):
        p, b, r = snap
        self.params = {k: v.copy() for k, v in p.items()}
        self.bn     = {k: v.copy() for k, v in b.items()}
        self.bn_run = {k: v.copy() for k, v in r.items()}


# ================================================================
# 4. OPTIMIZERS
# ================================================================

class Adam:
    """
    Adam optimizer (Kingma & Ba, 2015).
    Default lr=0.001 matches the paper's recommended value.
    """
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.b1, self.b2, self.eps = beta1, beta2, eps
        self._m = {};  self._v = {};  self._t = 0

    def reset(self):
        self._m, self._v, self._t = {}, {}, 0

    def step(self, params, grads):
        if not self._m:
            for k in params:
                self._m[k] = np.zeros_like(params[k])
                self._v[k] = np.zeros_like(params[k])
        self._t += 1
        bc1 = 1. - self.b1 ** self._t
        bc2 = 1. - self.b2 ** self._t
        for k in params:
            g = grads.get(k)
            if g is None:
                continue
            self._m[k] = self.b1 * self._m[k] + (1. - self.b1) * g
            self._v[k] = self.b2 * self._v[k] + (1. - self.b2) * g * g
            params[k] -= self.lr * (self._m[k] / bc1) / (
                np.sqrt(self._v[k] / bc2) + self.eps)
        return params


class SGD:
    """
    SGD-Minibatch with optional learning rate schedule.

    schedule='cosine': lr(t) = lr_min + 0.5*(lr0-lr_min)*(1+cos(pi*t/T))
                       Starts high, decays smoothly to lr_min.
    schedule=None    : fixed learning rate lr0.
    """
    def __init__(self, lr=0.05, lr_min=1e-4,
                 schedule='cosine', max_epochs=150):
        self.lr0, self.lr_min = lr, lr_min
        self.schedule   = schedule
        self.max_epochs = max_epochs
        self._epoch     = 0

    def epoch_step(self):
        """Call once per epoch to advance the schedule."""
        self._epoch += 1

    @property
    def current_lr(self):
        if self.schedule == 'cosine':
            t, T = self._epoch, self.max_epochs
            return self.lr_min + 0.5 * (self.lr0 - self.lr_min) * (
                1. + np.cos(np.pi * t / T))
        return self.lr0

    def step(self, params, grads):
        lr = self.current_lr
        for k in params:
            if k in grads:
                params[k] -= lr * grads[k]
        return params


# ================================================================
# 5. TRAINING LOOP
# ================================================================

def train(net, w_opt, bn_opt=None,
          X_tr=None, Y_tr=None, Y_tr_o=None,
          X_v=None,  Y_v=None,  Y_v_o=None,
          max_epochs=150, patience=10, batch_size=128,
          eval_size=4096, verbose=True, label=''):
    """
    Train BonusNet with early stopping on validation loss.

    Speed notes
    -----------
    - eval_size: training-set metrics are estimated on a fixed random
      subset of eval_size examples each epoch instead of the full 48k.
      This cuts per-epoch eval time by ~12x with minimal accuracy loss.
    - Full-dataset errors are computed ONLY after training completes.
    - bn_opt: if provided, BN params use a separate Adam instance.
      This is recommended because BN gamma/beta typically need a
      different learning rate than the weights.
    """
    hist = {'train_loss': [], 'val_loss': [], 'lr': []}
    best_val  = np.inf
    patience_c = 0
    N          = X_tr.shape[0]
    best_snap  = net.snapshot()

    if hasattr(w_opt, 'reset'):  w_opt.reset()
    if bn_opt and hasattr(bn_opt, 'reset'): bn_opt.reset()

    # Fixed random subset for training-loss estimation
    sub_idx = np.random.choice(N, min(eval_size, N), replace=False)
    X_sub   = X_tr[sub_idx]
    Y_sub   = Y_tr[sub_idx]

    for ep in range(1, max_epochs + 1):
        # Mini-batch parameter updates
        net.training = True
        perm  = np.random.permutation(N)
        Xs, Ys = X_tr[perm], Y_tr[perm]

        for s in range(0, N, batch_size):
            Xb, Yb       = Xs[s:s + batch_size], Ys[s:s + batch_size]
            cache        = net.forward(Xb)
            g, bg        = net.backward(cache, Yb)
            net.params   = w_opt.step(net.params, g)
            if net.use_bn:
                tgt = bn_opt if bn_opt else w_opt
                net.bn = tgt.step(net.bn, bg)

        # Advance LR schedule (SGD cosine annealing)
        if hasattr(w_opt, 'epoch_step'):
            w_opt.epoch_step()

        # Epoch-level metrics
        net.training = False
        c_sub = net.forward(X_sub)
        tr_l  = net.loss(c_sub[f'A{net.L}'], Y_sub)

        c_v   = net.forward(X_v)
        val_l = net.loss(c_v[f'A{net.L}'], Y_v)
        val_e = float(np.mean(np.argmax(c_v[f'A{net.L}'], 1) != Y_v_o))
        net.training = True

        hist['train_loss'].append(tr_l)
        hist['val_loss'].append(val_l)
        lr_now = getattr(w_opt, 'current_lr', getattr(w_opt, 'lr', 0.))
        hist['lr'].append(float(lr_now))

        # Early stopping
        if val_l < best_val - 1e-6:
            best_val    = val_l
            best_snap   = net.snapshot()
            patience_c  = 0
        else:
            patience_c += 1

        if verbose and ep % 20 == 0:
            print(f'  [{label}] Epoch {ep:4d} | '
                  f'Train loss: {tr_l:.4f} | '
                  f'Val loss: {val_l:.4f} ({val_e * 100:.1f}%)')

        if patience_c >= patience:
            if verbose:
                print(f'  Early stop @ epoch {ep}  (best val = {best_val:.4f})')
            break

    net.restore(best_snap)
    return hist


# ================================================================
# 6. EXPERIMENTS
# ================================================================

def run_all(data_dir='./data'):
    """
    Run all 6 bonus experiments.

    Returns
    -------
    results   : dict  label -> (train_err%, test_err%, n_epochs)
    histories : dict  label -> hist dict
    """
    print('Loading data...')
    (X_tr, Y_tr, Y_tr_o,
     X_v,  Y_v,  Y_v_o,
     X_te, Y_te, Y_te_o) = load_data(data_dir)

    SIZES = [784, 80, 80, 80, 10]   # 3 hidden layers, H=80
    WD    = 0.001232
    B     = 128
    EP    = 150
    PAT   = 10

    results   = {}
    histories = {}

    # Each experiment: (label, activation, use_bn, optimizer_factory)
    # optimizer_factory() -> (weight_optimizer, bn_optimizer_or_None)
    experiments = [
        (
            'Baseline\nAdam lr=0.01',
            'relu', False,
            lambda: (Adam(lr=0.01), None)
        ),
        (
            'Exp1\nAdam lr=0.001',
            'relu', False,
            lambda: (Adam(lr=0.001), None)
        ),
        (
            'Exp2a\nAdam lr=0.001\nLeaky ReLU',
            'leaky_relu', False,
            lambda: (Adam(lr=0.001), None)
        ),
        (
            'Exp2b\nAdam lr=0.001\nELU',
            'elu', False,
            lambda: (Adam(lr=0.001), None)
        ),
        (
            'Exp3\nAdam lr=0.003\n+ Batch Norm',
            'relu', True,
            lambda: (Adam(lr=0.003), Adam(lr=0.003))
        ),
        (
            'Exp4\nSGD + Cosine\nAnnealing',
            'relu', False,
            lambda: (SGD(lr=0.05, lr_min=1e-4, schedule='cosine', max_epochs=EP), None)
        ),
    ]

    for label, act, use_bn, opt_fn in experiments:
        clean = label.replace('\n', ' | ')
        print(f'\n{"─" * 58}')
        print(f'  {clean}')
        print(f'{"─" * 58}')

        net          = BonusNet(SIZES, WD, activation=act,
                                use_bn=use_bn, seed=SEED)
        w_opt, bn_opt = opt_fn()

        t0   = time.time()
        hist = train(net, w_opt, bn_opt,
                     X_tr=X_tr, Y_tr=Y_tr, Y_tr_o=Y_tr_o,
                     X_v=X_v,   Y_v=Y_v,   Y_v_o=Y_v_o,
                     max_epochs=EP, patience=PAT,
                     batch_size=B, eval_size=4096,
                     verbose=True, label=clean[:22])

        # Final evaluation on the FULL datasets (exact numbers for report)
        net.training = False
        tr_e = net.error(X_tr, Y_tr_o)
        te_e = net.error(X_te, Y_te_o)
        elapsed = time.time() - t0

        print(f'\n  Train: {tr_e * 100:.2f}%  |  Test: {te_e * 100:.2f}%  |  '
              f'Epochs: {len(hist["train_loss"])}  |  Time: {elapsed:.0f}s')

        results[label]   = (tr_e * 100, te_e * 100, len(hist['train_loss']))
        histories[label] = hist

    return results, histories


# ================================================================
# 7. FIGURES
# ================================================================

COLORS = ['#888888', '#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#ff7f0e']


def plot_curves(histories, results,
                out='./bonus_results/bonus_figA_curves.png'):
    """One subplot per experiment showing train and validation loss."""
    keys = list(histories.keys())
    n    = len(keys);  cols = 3;  rows = (n + cols - 1) // cols

    # Extra top margin via subplot_adjust so suptitle never touches subplot titles
    fig, axes = plt.subplots(rows, cols, figsize=(16, 5.2 * rows))
    fig.subplots_adjust(top=0.88, hspace=0.52, wspace=0.32)
    axes = axes.flatten()

    fig.suptitle(
        'Bonus Experiments — Learning Curves\n'
        r'3 Hidden Layers $|$ H=80 $|$ $\lambda$=0.001232 $|$ B=128 $|$ seed=8097',
        fontsize=12, fontweight='bold',
    )

    for i, (key, col) in enumerate(zip(keys, COLORS)):
        ax = axes[i]
        h  = histories[key]
        ep = np.arange(1, len(h['train_loss']) + 1)

        l_tr, = ax.plot(ep, h['train_loss'], color=col, lw=1.8, label='Train')
        l_val, = ax.plot(ep, h['val_loss'],  color=col, lw=1.8, ls='--', label='Val')

        # Legend in upper-right (just Train / Val handles — no text clutter)
        ax.legend(handles=[l_tr, l_val], labels=['Train', 'Val'],
                  fontsize=8, loc='upper right',
                  framealpha=0.80, handlelength=1.6,
                  borderpad=0.4, labelspacing=0.2)

        tr_e, te_e, _ = results[key]
        # Subplot title on its own line (no overlap with suptitle)
        ax.set_title(key.replace('\n', ' | '), fontsize=8.5,
                     fontweight='bold', pad=6)
        ax.set_xlabel('Epoch', fontsize=8)
        ax.set_ylabel('Loss',  fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)

        # Test-error annotation in BOTTOM-LEFT to avoid legend clash
        ax.text(0.03, 0.04, f'test {te_e:.2f}%',
                transform=ax.transAxes, ha='left', va='bottom',
                fontsize=8.5, fontweight='bold', color=col,
                bbox=dict(boxstyle='round,pad=0.25', fc='white',
                          alpha=0.88, ec=col, lw=0.6))

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')


def plot_bar(results,
             out='./bonus_results/bonus_figB_comparison.png'):
    """Bar chart comparing test and train errors against main-experiment refs."""
    keys    = list(results.keys())
    x       = np.arange(len(keys))
    te_vals = [results[k][1] for k in keys]
    tr_vals = [results[k][0] for k in keys]
    w       = 0.35

    # Taller figure; top margin reserved for title
    fig, ax = plt.subplots(figsize=(14, 6.2))
    fig.suptitle('Bonus Experiments — Error Comparison\n'
                 '(3 Hidden Layers, H=80, Fashion MNIST)',
                 fontsize=11, fontweight='bold', y=0.98)
    fig.subplots_adjust(top=0.86)

    bars = ax.bar(x - w / 2, te_vals, w, color=COLORS[:len(keys)],
                  label='Test Error', edgecolor='white', lw=0.4)
    ax.bar(x + w / 2, tr_vals, w, color=COLORS[:len(keys)],
           alpha=0.4, label='Train Error', edgecolor='white', lw=0.4)

    # Value labels on test-error bars
    for bar, v in zip(bars, te_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.18,
                f'{v:.2f}%', ha='center', va='bottom',
                fontsize=8.5, fontweight='bold')

    # Reference lines
    l1, = ax.plot([], [], color='#1f77b4', ls=':',  lw=1.8)
    l2, = ax.plot([], [], color='#ff7f0e', ls='--', lw=1.5)
    l3, = ax.plot([], [], color='#d62728', ls='-.', lw=1.5)
    ax.axhline(11.47, color='#1f77b4', ls=':',  lw=1.8, zorder=3)
    ax.axhline(11.88, color='#ff7f0e', ls='--', lw=1.5, zorder=3)
    ax.axhline(15.31, color='#d62728', ls='-.', lw=1.5, zorder=3)

    ax.set_xticks(x)
    xlabels = ['\n'.join(k.split('\n')) for k in keys]
    ax.set_xticklabels(xlabels, fontsize=8.5, ha='center')
    ax.set_ylabel('Misclassification Error (%)', fontsize=10)

    # Legend inside the axes at upper-left — that area has no tall bars
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor='#555', label='Test Error (dark bar)'),
        Patch(facecolor='#aaa', label='Train Error (light bar)'),
        l1, l2, l3,
    ]
    legend_labels = [
        'Test Error (dark bar)',
        'Train Error (light bar)',
        'Main best: 1L SGD-Mini (11.47%)',
        'Main 3L SGD-Mini (11.88%)',
        'Main 3L Adam-Mini lr=0.01 (15.31%)',
    ]
    ax.legend(legend_handles, legend_labels,
              fontsize=8.2, framealpha=0.93, ncol=1,
              loc='upper left', borderpad=0.6)

    ax.set_ylim(0, max(te_vals) * 1.25)
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')


def plot_lr_schedule(out='./bonus_results/bonus_figC_lr_schedule.png'):
    """Visualise the cosine annealing LR schedule used in Exp 4."""
    T  = 150
    ep = np.arange(0, T + 1)
    lr_cos  = 1e-4 + 0.5 * (0.05 - 1e-4) * (1. + np.cos(np.pi * ep / T))
    lr_step = [max(0.01 * (0.5 ** (e // 50)), 1e-4) for e in ep]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(ep, lr_cos,  '#ff7f0e', lw=2.2,
            label=r'Cosine annealing ($\alpha_0$=0.05 $\to$ $\alpha_{\min}$=1e-4)')
    ax.plot(ep, lr_step, '#9467bd', lw=2.0, ls='--',
            label=r'Step decay ($\alpha_0$=0.01, halved every 50 ep)')
    ax.axhline(0.01, color='grey', ls=':', lw=1.4,
               label=r'Fixed $\alpha$=0.01 (main baseline)')

    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Learning Rate', fontsize=10)
    ax.set_title('Learning Rate Schedules — Exp 4 (Cosine Annealing)',
                 fontsize=10, fontweight='bold')

    # Legend in lower-left: cosine curve is low there, no overlap
    ax.legend(fontsize=9, framealpha=0.92, loc='lower left',
              borderpad=0.6, handlelength=2.0)
    ax.grid(True, alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()

    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')


# ================================================================
# 8. SUMMARY TABLE
# ================================================================

def print_table(results):
    W = 38
    sep = '=' * (W + 30)
    print('\n' + sep)
    print('BONUS EXPERIMENTS — RESULTS SUMMARY')
    print('(Architecture: 3 Hidden Layers | H=80 | B=128)')
    print(sep)
    print(f"  {'Model':<{W}} {'Train%':>8} {'Test%':>8} {'Epochs':>7}")
    print('  ' + '-' * (W + 26))

    # Main experiment references
    print(f"  {'[Main] 1L SGD-Minibatch (best baseline)':<{W}} "
          f"{'5.85':>8} {'11.47':>8} {'134':>7}")
    print(f"  {'[Main] 3L SGD-Minibatch':<{W}} "
          f"{'4.10':>8} {'11.88':>8} {'94':>7}")
    print(f"  {'[Main] 3L Adam-Mini lr=0.01':<{W}} "
          f"{'13.49':>8} {'15.31':>8} {'57':>7}")
    print('  ' + '-' * (W + 26))

    for label, (tr_e, te_e, ep) in results.items():
        clean = label.replace('\n', ' | ')
        print(f"  {clean:<{W}} {tr_e:>8.2f} {te_e:>8.2f} {ep:>7}")
    print(sep)


# ================================================================
# 9. ENTRY POINT
# ================================================================

if __name__ == '__main__':
    print('ECE 726 — Assignment 3: Bonus Experiments')
    print(f'Seed: {SEED} | float32 | B=128 | max_epochs=150 | patience=10')
    print('Architecture: 3 hidden layers, H=80 (best from main experiment)')
    print()

    results, histories = run_all(data_dir='./data')

    print('\nGenerating figures...')
    plot_curves(histories, results)
    plot_bar(results)
    plot_lr_schedule()

    print_table(results)
    print('\nDone. Figures and results saved to ./bonus_results/')