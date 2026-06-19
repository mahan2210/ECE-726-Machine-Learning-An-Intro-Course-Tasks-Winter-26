"""
ECE 726 - Assignment 3: Neural Networks
Winter 2026
Author: Mahan Choudhury

Description:
    Implements and trains neural networks with 1, 2, 3, and 4 hidden layers
    on the Fashion MNIST dataset using four training algorithms:
        1) Stochastic Gradient Descent (SGD)
        2) Minibatch Gradient Descent
        3) Stochastic Adam
        4) Minibatch Adam
    All neural network operations are implemented from scratch using NumPy.
    PyTorch/torchvision are used ONLY for data loading and preprocessing.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import struct
import gzip
import urllib.request
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 0. REPRODUCIBILITY SEEDS
#    Last 4 digits of student ID: 8097
# ============================================================
SEED = 8097
np.random.seed(SEED)

# Try to set PyTorch seed if available (used only for data loading)
try:
    import torch
    torch.manual_seed(SEED)
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ============================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================

def load_fashion_mnist_idx(data_dir='./data'):
    """
    Load Fashion MNIST from local IDX binary files.
    Downloads if not present.
    """
    os.makedirs(data_dir, exist_ok=True)
    base_url = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
    files = {
        'train-images': 'train-images-idx3-ubyte',
        'train-labels': 'train-labels-idx1-ubyte',
        'test-images':  't10k-images-idx3-ubyte',
        'test-labels':  't10k-labels-idx1-ubyte',
    }

    def download_and_extract(fname):
        gz_path = os.path.join(data_dir, fname + '.gz')
        bin_path = os.path.join(data_dir, fname)
        if not os.path.exists(bin_path):
            print(f'Downloading {fname}...')
            urllib.request.urlretrieve(base_url + fname + '.gz', gz_path)
            with gzip.open(gz_path, 'rb') as f_in:
                with open(bin_path, 'wb') as f_out:
                    f_out.write(f_in.read())
            os.remove(gz_path)
        return bin_path

    def read_images(path):
        with open(path, 'rb') as f:
            magic, n, rows, cols = struct.unpack('>IIII', f.read(16))
            data = np.frombuffer(f.read(), dtype=np.uint8)
            return data.reshape(n, rows * cols).astype('float32') / 255.0

    def read_labels(path):
        with open(path, 'rb') as f:
            magic, n = struct.unpack('>II', f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)

    try:
        X_train_raw = read_images(download_and_extract(files['train-images']))
        Y_train_raw = read_labels(download_and_extract(files['train-labels']))
        X_test_raw  = read_images(download_and_extract(files['test-images']))
        Y_test_raw  = read_labels(download_and_extract(files['test-labels']))
        return X_train_raw, Y_train_raw, X_test_raw, Y_test_raw
    except Exception as e:
        print(f'IDX loading failed: {e}')
        return None


def load_fashion_mnist_torch(data_dir='./data'):
    """Load Fashion MNIST using torchvision (preferred if available)."""
    from torchvision import datasets
    train_ds = datasets.FashionMNIST(root=data_dir, train=True,  download=True)
    test_ds  = datasets.FashionMNIST(root=data_dir, train=False, download=True)
    X_train = train_ds.data.numpy().reshape(-1, 784).astype('float32') / 255.0
    Y_train = train_ds.targets.numpy().astype(np.int64)
    X_test  = test_ds.data.numpy().reshape(-1, 784).astype('float32') / 255.0
    Y_test  = test_ds.targets.numpy().astype(np.int64)
    return X_train, Y_train, X_test, Y_test


def one_hot_encode(labels, num_classes=10):
    """Convert integer labels to one-hot encoded matrix."""
    return np.eye(num_classes)[labels]   # shape: (N, C)


def preprocess_data(data_dir='./data'):
    """
    Load, split, standardize, and one-hot encode the Fashion MNIST dataset.

    Returns
    -------
    X_train, Y_train, Y_train_orig : training features, one-hot labels, integer labels
    X_val,   Y_val,   Y_val_orig   : validation features, one-hot labels, integer labels
    X_test,  Y_test,  Y_test_orig  : test features, one-hot labels, integer labels
    """
    # --- Load data ---
    if TORCH_AVAILABLE:
        X_full, Y_full, X_test_raw, Y_test_raw = load_fashion_mnist_torch(data_dir)
    else:
        result = load_fashion_mnist_idx(data_dir)
        if result is None:
            raise RuntimeError(
                "Could not load Fashion MNIST. Please download manually and place "
                "the IDX files in the ./data directory, or install torchvision."
            )
        X_full, Y_full, X_test_raw, Y_test_raw = result

    print(f'Loaded: X_train={X_full.shape}, X_test={X_test_raw.shape}')

    # --- 80/20 train/validation split ---
    N = X_full.shape[0]
    val_size = int(0.2 * N)                   # 12,000 samples
    np.random.seed(SEED)                       # ensure reproducibility

    idx = np.random.permutation(N)            # shuffle indices
    val_idx   = idx[:val_size]
    train_idx = idx[val_size:]

    X_val,   Y_val_orig   = X_full[val_idx],   Y_full[val_idx]
    X_train, Y_train_orig = X_full[train_idx], Y_full[train_idx]
    X_test,  Y_test_orig  = X_test_raw,         Y_test_raw

    # --- Standardize using training set statistics ---
    X_mean = X_train.mean(axis=0)
    X_std  = X_train.std(axis=0)
    X_std[X_std == 0] = 1.0                   # avoid division by zero

    X_train = (X_train - X_mean) / X_std
    X_val   = (X_val   - X_mean) / X_std
    X_test  = (X_test  - X_mean) / X_std

    # --- One-hot encode labels ---
    Y_train = one_hot_encode(Y_train_orig)
    Y_val   = one_hot_encode(Y_val_orig)
    Y_test  = one_hot_encode(Y_test_orig)

    print(f'Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}')
    return (X_train, Y_train, Y_train_orig,
            X_val,   Y_val,   Y_val_orig,
            X_test,  Y_test,  Y_test_orig)


# ============================================================
# 2. NEURAL NETWORK CLASS (NumPy only)
# ============================================================

class NeuralNetwork:
    """
    Fully-connected feedforward neural network implemented from scratch.

    Architecture:
        Input (784) → [Hidden (H units, ReLU)] x num_hidden_layers → Output (10, Softmax)

    Parameters
    ----------
    layer_sizes : list of int
        Full architecture including input and output dimensions.
        E.g., [784, 80, 80, 10] for 2 hidden layers with 80 units each.
    weight_decay : float
        L2 regularization coefficient (lambda).
    seed : int
        Random seed for weight initialization.
    """

    def __init__(self, layer_sizes, weight_decay=0.001232, seed=SEED):
        self.layer_sizes  = layer_sizes
        self.weight_decay = weight_decay
        self.num_layers   = len(layer_sizes) - 1   # number of weight matrices
        self.params       = {}                      # weights and biases

        rng = np.random.default_rng(seed)

        # initialization for ReLU hidden layers; and output layer
        for l in range(1, len(layer_sizes)):
            fan_in  = layer_sizes[l - 1]
            fan_out = layer_sizes[l]
            if l < len(layer_sizes) - 1:
                # He initialization: std = sqrt(2 / fan_in)
                scale = np.sqrt(2.0 / fan_in)
            else:
                # Xavier/Glorot for output layer
                scale = np.sqrt(1.0 / fan_in)
            self.params[f'W{l}'] = rng.normal(0, scale, (fan_in, fan_out)).astype('float32')
            self.params[f'b{l}'] = np.zeros((1, fan_out), dtype='float32')

    # ----------------------------------------------------------
    # Activation functions
    # ----------------------------------------------------------

    @staticmethod
    def relu(Z):
        """ReLU activation: max(0, Z)."""
        return np.maximum(0.0, Z)

    @staticmethod
    def relu_derivative(Z):
        """Derivative of ReLU: 1 if Z > 0 else 0."""
        return (Z > 0).astype('float64')

    @staticmethod
    def softmax(Z):
        """
        Numerically stable softmax along axis=1.
        Each row is an independent probability distribution.
        """
        Z_shifted = Z - Z.max(axis=1, keepdims=True)   # subtract max for stability
        exp_Z = np.exp(Z_shifted)
        return exp_Z / exp_Z.sum(axis=1, keepdims=True)

    # ----------------------------------------------------------
    # Forward pass
    # ----------------------------------------------------------

    def forward(self, X):
        """
        Perform forward pass through the network.

        Parameters
        ----------
        X : ndarray, shape (N, 784)

        Returns
        -------
        cache : dict containing pre-activations Z, activations A, and input X
        """
        cache = {'A0': X}
        A = X
        for l in range(1, self.num_layers + 1):
            Z = A @ self.params[f'W{l}'] + self.params[f'b{l}']
            cache[f'Z{l}'] = Z
            if l < self.num_layers:
                A = self.relu(Z)              # hidden layers: ReLU
            else:
                A = self.softmax(Z)           # output layer: Softmax
            cache[f'A{l}'] = A
        return cache

    # ----------------------------------------------------------
    # Loss function
    # ----------------------------------------------------------

    def cross_entropy_loss(self, Y_pred, Y_true, include_reg=True):
        """
        Compute cross-entropy loss with optional L2 regularization.

        Parameters
        ----------
        Y_pred : ndarray, shape (N, 10) - softmax probabilities
        Y_true : ndarray, shape (N, 10) - one-hot labels
        include_reg : bool - whether to add L2 penalty

        Returns
        -------
        loss : float
        """
        N = Y_true.shape[0]
        # Clip predictions for numerical stability
        Y_pred_clipped = np.clip(Y_pred, 1e-12, 1.0)
        ce_loss = -np.sum(Y_true * np.log(Y_pred_clipped)) / N

        if include_reg and self.weight_decay > 0:
            l2_penalty = 0.0
            for l in range(1, self.num_layers + 1):
                l2_penalty += np.sum(self.params[f'W{l}'] ** 2)
            ce_loss += 0.5 * self.weight_decay * l2_penalty

        return ce_loss

    # ----------------------------------------------------------
    # Backward pass (backpropagation)
    # ----------------------------------------------------------

    def backward(self, cache, Y_true):
        """
        Compute gradients via backpropagation.

        Parameters
        ----------
        cache : dict from forward()
        Y_true : ndarray, shape (N, 10) - one-hot labels

        Returns
        -------
        grads : dict of gradients for W and b at each layer
        """
        N = Y_true.shape[0]
        grads = {}

        # --- Output layer: derivative of cross-entropy + softmax combined ---
        # dL/dZ_L = (A_L - Y) / N
        dZ = (cache[f'A{self.num_layers}'] - Y_true) / N

        for l in range(self.num_layers, 0, -1):
            A_prev = cache[f'A{l-1}']
            # Gradient w.r.t. weights (with L2 regularization)
            grads[f'W{l}'] = A_prev.T @ dZ + self.weight_decay * self.params[f'W{l}']
            # Gradient w.r.t. biases
            grads[f'b{l}'] = dZ.sum(axis=0, keepdims=True)

            if l > 1:
                # Backpropagate through the ReLU activation
                dA_prev = dZ @ self.params[f'W{l}'].T
                dZ = dA_prev * self.relu_derivative(cache[f'Z{l-1}'])

        return grads

    # ----------------------------------------------------------
    # Prediction utilities
    # ----------------------------------------------------------

    def predict(self, X):
        """Return predicted class labels for input X."""
        cache = self.forward(X)
        return np.argmax(cache[f'A{self.num_layers}'], axis=1)

    def misclassification_error(self, X, Y_orig):
        """
        Compute misclassification error (fraction of incorrect predictions).

        Parameters
        ----------
        X      : ndarray, shape (N, 784)
        Y_orig : ndarray, shape (N,) - integer labels

        Returns
        -------
        error : float in [0, 1]
        """
        preds = self.predict(X)
        return np.mean(preds != Y_orig)

    def get_params_copy(self):
        """Return a deep copy of the current parameters."""
        return {k: v.copy() for k, v in self.params.items()}

    def set_params(self, params_dict):
        """Load parameters from a dictionary."""
        for k, v in params_dict.items():
            self.params[k] = v.copy()


# ============================================================
# 3. OPTIMIZERS
# ============================================================

class SGDOptimizer:
    """
    Stochastic / Minibatch Gradient Descent optimizer.

    Parameters
    ----------
    lr : float - learning rate
    batch_size : int - 1 for stochastic, >1 for minibatch
    """
    def __init__(self, lr=0.01, batch_size=128):
        self.lr         = lr
        self.batch_size = batch_size

    def update(self, params, grads):
        """Apply gradient descent update to parameters."""
        for key in params:
            params[key] -= self.lr * grads[key]
        return params


class AdamOptimizer:
    """
    Adam optimizer (Kingma & Ba, 2015).

    Parameters
    ----------
    lr         : float - learning rate (alpha in paper)
    batch_size : int   - 1 for stochastic, >1 for minibatch
    beta1      : float - exponential decay rate for 1st moment (default 0.9)
    beta2      : float - exponential decay rate for 2nd moment (default 0.999)
    epsilon    : float - small constant for numerical stability (default 1e-8)
    """
    def __init__(self, lr=0.01, batch_size=128, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr         = lr
        self.batch_size = batch_size
        self.beta1      = beta1
        self.beta2      = beta2
        self.epsilon    = epsilon
        self.m          = {}    # 1st moment estimates
        self.v          = {}    # 2nd moment estimates
        self.t          = 0     # time step

    def initialize(self, params):
        """Initialize moment vectors to zero, matching parameter shapes."""
        for key in params:
            self.m[key] = np.zeros_like(params[key])
            self.v[key] = np.zeros_like(params[key])

    def update(self, params, grads):
        """
        Perform one Adam update step.

        Equations (Kingma & Ba 2015):
            m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
            v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
            m_hat = m_t / (1 - beta1^t)
            v_hat = v_t / (1 - beta2^t)
            theta_t = theta_{t-1} - alpha * m_hat / (sqrt(v_hat) + epsilon)
        """
        if not self.m:
            self.initialize(params)

        self.t += 1
        for key in params:
            g = grads[key]
            # Update biased first and second moment estimates
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * g ** 2
            # Bias-corrected moment estimates
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            # Parameter update
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return params


# ============================================================
# 4. TRAINING FUNCTION
# ============================================================

def train_model(net, optimizer, X_train, Y_train, Y_train_orig,
                X_val, Y_val, Y_val_orig,
                max_epochs=200, patience=15, batch_size=128,
                eval_every=1, eval_train_size=4096,
                verbose=True):
    """
    Train a NeuralNetwork using early stopping on validation loss.

    Performance notes
    -----------------
    * For stochastic (batch_size=1) and small-batch variants, metrics are
      evaluated only every `eval_every` epochs (default 1) to avoid the
      dominant cost of 3 full forward passes per epoch on 48 k samples.
    * Training-set metrics are computed on a random subset of `eval_train_size`
      examples (default 4096) for speed; validation loss uses the full val set
      for accurate early stopping.
    * Predictions are derived from the forward pass already computed for the
      loss — no second forward pass is needed.

    Parameters
    ----------
    net              : NeuralNetwork instance
    optimizer        : SGDOptimizer or AdamOptimizer instance
    X_train          : ndarray, shape (N_tr, 784)
    Y_train          : ndarray, shape (N_tr, 10)  - one-hot
    Y_train_orig     : ndarray, shape (N_tr,)     - integer labels
    X_val            : ndarray, shape (N_v, 784)
    Y_val            : ndarray, shape (N_v, 10)   - one-hot
    Y_val_orig       : ndarray, shape (N_v,)      - integer labels
    max_epochs       : int  - maximum training epochs
    patience         : int  - early stopping patience (in eval checkpoints)
    batch_size       : int  - mini-batch size (1 = stochastic)
    eval_every       : int  - evaluate metrics every N epochs (use 5 for stochastic)
    eval_train_size  : int  - number of training samples for train metric estimation
    verbose          : bool - print progress

    Returns
    -------
    history : dict with keys 'train_loss', 'val_loss', 'train_err', 'val_err'
    best_params : dict - model parameters at best validation loss
    """
    history = {
        'train_loss': [], 'val_loss': [],
        'train_err':  [], 'val_err':  []
    }

    best_val_loss    = np.inf
    best_params      = net.get_params_copy()
    patience_counter = 0
    N_train          = X_train.shape[0]
    L                = net.num_layers

    # Reset Adam moment vectors if applicable
    if isinstance(optimizer, AdamOptimizer):
        optimizer.m = {}
        optimizer.v = {}
        optimizer.t = 0

    # Fixed random subset indices for training-set metric estimation
    eval_train_size = min(eval_train_size, N_train)
    rng_idx = np.random.choice(N_train, eval_train_size, replace=False)
    X_tr_sub  = X_train[rng_idx]
    Y_tr_sub  = Y_train[rng_idx]
    Y_tr_orig_sub = Y_train_orig[rng_idx]

    for epoch in range(1, max_epochs + 1):
        # ---- Shuffle training data ----
        perm       = np.random.permutation(N_train)
        X_shuffled = X_train[perm]
        Y_shuffled = Y_train[perm]

        # ---- Mini-batch parameter updates ----
        for start in range(0, N_train, batch_size):
            end  = min(start + batch_size, N_train)
            X_mb = X_shuffled[start:end]
            Y_mb = Y_shuffled[start:end]

            cache      = net.forward(X_mb)
            grads      = net.backward(cache, Y_mb)
            net.params = optimizer.update(net.params, grads)

        # ---- Epoch-level metrics (every eval_every epochs) ----
        if epoch % eval_every == 0:
            # Training subset — single forward pass, reuse for both loss & error
            tr_cache = net.forward(X_tr_sub)
            tr_pred  = tr_cache[f'A{L}']
            tr_loss  = net.cross_entropy_loss(tr_pred, Y_tr_sub, include_reg=True)
            tr_err   = float(np.mean(np.argmax(tr_pred, axis=1) != Y_tr_orig_sub))

            # Validation — full set, single forward pass
            val_cache = net.forward(X_val)
            val_pred  = val_cache[f'A{L}']
            val_loss  = net.cross_entropy_loss(val_pred, Y_val, include_reg=True)
            val_err   = float(np.mean(np.argmax(val_pred, axis=1) != Y_val_orig))

            history['train_loss'].append(tr_loss)
            history['val_loss'].append(val_loss)
            history['train_err'].append(tr_err)
            history['val_err'].append(val_err)

            # ---- Early stopping ----
            if val_loss < best_val_loss - 1e-6:
                best_val_loss    = val_loss
                best_params      = net.get_params_copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and (epoch % (eval_every * 10) == 0 or patience_counter == 0):
                print(f'  Epoch {epoch:4d}/{max_epochs} | '
                      f'Train Loss: {tr_loss:.4f} ({tr_err*100:.1f}%) | '
                      f'Val Loss: {val_loss:.4f} ({val_err*100:.1f}%)')

            if patience_counter >= patience:
                if verbose:
                    print(f'  Early stopping at epoch {epoch} '
                          f'(best val loss: {best_val_loss:.4f})')
                break

    # Restore best parameters
    net.set_params(best_params)
    return history, best_params


# ============================================================
# 5. MAIN EXPERIMENT
# ============================================================

def run_experiments(data_dir='./data', results_dir='./results',
                    max_epochs=200, patience=15):
    """
    Run all 16 training experiments:
        4 architectures (1, 2, 3, 4 hidden layers)  ×
        4 optimizers   (SGD-stoch, SGD-minibatch, Adam-stoch, Adam-minibatch)

    Hyperparameters (same for all 16 models):
        H       = 80    (units per hidden layer)
        lr      = 0.01  (learning rate)
        batch   = 128   (minibatch size)
        λ       = 0.001232 (L2 weight decay)
        stoch batch_size = 1 (single example)
    """
    os.makedirs(results_dir, exist_ok=True)

    # ---- Hyperparameters ----
    H            = 80
    LR           = 0.01
    BATCH_SIZE   = 128
    WEIGHT_DECAY = 0.001232
    INPUT_DIM    = 784
    NUM_CLASSES  = 10

    # ---- Load and preprocess data ----
    print('='*60)
    print('Loading and preprocessing Fashion MNIST...')
    print('='*60)
    (X_train, Y_train, Y_train_orig,
     X_val,   Y_val,   Y_val_orig,
     X_test,  Y_test,  Y_test_orig) = preprocess_data(data_dir)

    # ---- Define experiment grid ----
    num_hidden_layers_list = [1, 2, 3, 4]
    optimizer_configs = [
        ('SGD-Stochastic',  'sgd',  1),
        ('SGD-Minibatch',   'sgd',  BATCH_SIZE),
        ('Adam-Stochastic', 'adam', 1),
        ('Adam-Minibatch',  'adam', BATCH_SIZE),
    ]

    # ---- Storage for results ----
    all_histories   = {}    # {(n_layers, opt_name): history}
    results_table   = []    # rows: [arch, opt_name, tr_err, test_err]
    initial_params  = {}    # {n_layers: initial_params_dict}
                            # — same init across all optimizers for same architecture

    print(f'\nHyperparameters: H={H}, lr={LR}, batch={BATCH_SIZE}, λ={WEIGHT_DECAY}')
    print(f'Early stopping patience: {patience}, Max epochs: {max_epochs}')
    print(f'Random seed: {SEED}\n')

    # ---- Run all 16 experiments ----
    for n_layers in num_hidden_layers_list:
        layer_sizes = [INPUT_DIM] + [H] * n_layers + [NUM_CLASSES]
        arch_name   = f'{n_layers}-Hidden-Layer(s)'
        print(f'\n{"="*60}')
        print(f'Architecture: {arch_name}  |  Sizes: {layer_sizes}')
        print(f'{"="*60}')

        # Create ONE network to get the initial parameters, then reuse
        # This ensures ALL 4 optimizers start from the same initial weights
        ref_net = NeuralNetwork(layer_sizes,
                                weight_decay=WEIGHT_DECAY,
                                seed=SEED + n_layers)   # unique seed per arch
        initial_params[n_layers] = ref_net.get_params_copy()

        for opt_name, opt_type, bs in optimizer_configs:
            print(f'\n  Optimizer: {opt_name}  (batch_size={bs})')

            # Build a fresh network with the same initial parameters
            net = NeuralNetwork(layer_sizes,
                                weight_decay=WEIGHT_DECAY,
                                seed=SEED + n_layers)
            net.set_params(initial_params[n_layers])   # same init for all opts

            # Build the optimizer
            if opt_type == 'sgd':
                optimizer = SGDOptimizer(lr=LR, batch_size=bs)
            else:
                optimizer = AdamOptimizer(lr=LR, batch_size=bs)

            # eval_every: evaluate loss/error every N epochs.
            # For stochastic (bs=1), one epoch = 48,000 gradient steps —
            # already far more updates than a full minibatch run. We check
            # metrics every 5 stochastic epochs to reduce overhead, but
            # max_epochs stays the same for all 16 models (assignment rule).
            eval_every = 5 if bs == 1 else 1

            # Train
            start_time = time.time()
            history, best_params = train_model(
                net, optimizer,
                X_train, Y_train, Y_train_orig,
                X_val,   Y_val,   Y_val_orig,
                max_epochs=max_epochs, patience=patience,
                batch_size=bs, eval_every=eval_every,
                eval_train_size=4096, verbose=True
            )
            elapsed = time.time() - start_time

            # Final evaluation uses the FULL training and test sets
            # (eval_train_size only affects the per-epoch training metric
            # logged during training; these final numbers are exact)
            net.set_params(best_params)
            tr_err   = net.misclassification_error(X_train, Y_train_orig)
            test_err = net.misclassification_error(X_test,  Y_test_orig)

            print(f'  → Train Error: {tr_err*100:.2f}%  |  Test Error: {test_err*100:.2f}%'
                  f'  |  Time: {elapsed:.1f}s')

            key = (n_layers, opt_name)
            all_histories[key] = history
            results_table.append({
                'Architecture':    arch_name,
                'Optimizer':       opt_name,
                'Train Error (%)': round(tr_err   * 100, 2),
                'Test Error (%)':  round(test_err * 100, 2),
                'Epochs Trained':  len(history['train_loss']),
                'Best Val Loss':   round(min(history['val_loss']), 4),
            })

    # ---- Save and display results ----
    plot_learning_curves(all_histories, results_dir)
    print_results_table(results_table)
    save_results_table(results_table, results_dir)

    return all_histories, results_table


# ============================================================
# 6. PLOTTING UTILITIES
# ============================================================

def plot_learning_curves(all_histories, results_dir):
    """
    Plot learning curves (cross-entropy loss vs. epoch) for all 16 models.
    Organized as a 4×4 grid: rows = architectures, cols = optimizers.
    """
    num_hidden_layers_list = [1, 2, 3, 4]
    optimizer_names = [
        'SGD-Stochastic', 'SGD-Minibatch',
        'Adam-Stochastic', 'Adam-Minibatch'
    ]

    fig, axes = plt.subplots(4, 4, figsize=(22, 18))
    fig.suptitle(
        'Learning Curves: Cross-Entropy Loss vs. Epoch\n'
        'Fashion MNIST Neural Network (H=80 units/layer, λ=0.001232)',
        fontsize=14, fontweight='bold', y=1.01
    )

    colors = {'train': '#1f77b4', 'val': '#d62728'}

    for row, n_layers in enumerate(num_hidden_layers_list):
        for col, opt_name in enumerate(optimizer_names):
            ax  = axes[row][col]
            key = (n_layers, opt_name)

            if key not in all_histories:
                ax.set_visible(False)
                continue

            hist     = all_histories[key]
            epochs   = np.arange(1, len(hist['train_loss']) + 1)

            ax.plot(epochs, hist['train_loss'],
                    color=colors['train'], linewidth=1.5, label='Training')
            ax.plot(epochs, hist['val_loss'],
                    color=colors['val'], linewidth=1.5, linestyle='--', label='Validation')

            ax.set_title(f'{n_layers}L | {opt_name}', fontsize=9, fontweight='bold')
            ax.set_xlabel('Epoch', fontsize=8)
            ax.set_ylabel('Loss', fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=7, loc='upper right')
            ax.grid(True, alpha=0.3)

            # Annotate best val loss
            best_val = min(hist['val_loss'])
            ax.axhline(best_val, color='gray', linestyle=':', linewidth=0.8)

    plt.tight_layout()
    out_path = os.path.join(results_dir, 'learning_curves_all.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nLearning curves saved: {out_path}')


def plot_learning_curves_per_arch(all_histories, results_dir):
    """
    Additional plots: one figure per architecture showing all 4 optimizers.
    """
    num_hidden_layers_list = [1, 2, 3, 4]
    optimizer_names = [
        'SGD-Stochastic', 'SGD-Minibatch',
        'Adam-Stochastic', 'Adam-Minibatch'
    ]
    opt_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for n_layers in num_hidden_layers_list:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f'Learning Curves — {n_layers} Hidden Layer(s), H=80',
            fontsize=13, fontweight='bold'
        )

        for i, opt_name in enumerate(optimizer_names):
            key  = (n_layers, opt_name)
            if key not in all_histories:
                continue
            hist   = all_histories[key]
            epochs = np.arange(1, len(hist['train_loss']) + 1)
            c      = opt_colors[i]
            axes[0].plot(epochs, hist['train_loss'], color=c, label=opt_name, linewidth=1.5)
            axes[1].plot(epochs, hist['val_loss'],   color=c, label=opt_name,
                         linewidth=1.5, linestyle='--')

        for ax, title in zip(axes, ['Training Loss', 'Validation Loss']):
            ax.set_title(title, fontsize=11)
            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel('Cross-Entropy Loss', fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = os.path.join(results_dir, f'learning_curves_{n_layers}layer.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Per-architecture plot saved: {out_path}')


# ============================================================
# 7. RESULTS TABLE
# ============================================================

def print_results_table(results_table):
    """Print a formatted results table to stdout."""
    print('\n' + '='*75)
    print('RESULTS SUMMARY TABLE')
    print('='*75)
    header = f"{'Architecture':<26} {'Optimizer':<20} {'Train Err%':>10} {'Test Err%':>10} {'Epochs':>8}"
    print(header)
    print('-'*75)

    prev_arch = ''
    for row in results_table:
        arch = row['Architecture']
        if arch != prev_arch:
            if prev_arch:
                print('-'*75)
            prev_arch = arch
        print(f"{arch:<26} {row['Optimizer']:<20} "
              f"{row['Train Error (%)']:>10.2f} "
              f"{row['Test Error (%)']:>10.2f} "
              f"{row['Epochs Trained']:>8}")
    print('='*75)


def save_results_table(results_table, results_dir):
    """Save results table as a CSV file."""
    import csv
    path = os.path.join(results_dir, 'results_table.csv')
    fieldnames = ['Architecture', 'Optimizer', 'Train Error (%)',
                  'Test Error (%)', 'Epochs Trained', 'Best Val Loss']
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_table)
    print(f'Results table saved: {path}')


# ============================================================
# 8. ENTRY POINT
# ============================================================

if __name__ == '__main__':
    print('ECE 726 Assignment 3 - Neural Networks from Scratch')
    print(f'Student ID seed: {SEED}')
    print(f'NumPy version: {np.__version__}')

    # Run all experiments
    # Adjust max_epochs / patience as desired:
    #   max_epochs=200, patience=15 is a good balance of speed and quality.
    #   Increase max_epochs to 500+ for better convergence.
    all_histories, results = run_experiments(
        data_dir    = './data',
        results_dir = './results',
        max_epochs  = 200,    # increase for better results
        patience    = 15,     # early stopping patience
    )

    # Additional per-architecture plots
    plot_learning_curves_per_arch(all_histories, results_dir='./results')

    print('\nDone! Report has been saved to ./results directory for plots and CSV table.')