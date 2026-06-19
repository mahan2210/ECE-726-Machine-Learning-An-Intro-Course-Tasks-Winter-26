# -*- coding: utf-8 -*-


"""
ECE 726 - Assignment 1: Trade-off between Overfitting and Underfitting
=======================================================================

Mahan Choudhury
Student ID: 400648097

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

plt.style.use('seaborn-v0_8-whitegrid')

#==============================================================================
# INSTRUCTION 1: Understanding the Problem
#==============================================================================
"""
FROM ASSIGNMENT:
"The target t is a noisy measurement of the function f_true(x) = sin(2πx)"
"t = sin(2πx) + ε, where ε is random noise with Gaussian distribution,
 mean 0 and variance 0.04"

"""

#==============================================================================
# INSTRUCTION 2: Generate Training, Validation, and Test Data
#==============================================================================
"""
FROM ASSIGNMENT:
"Construct a training set consisting of only N = 9 data samples"
"x values uniformly spaced in [0, 1], with x(1) = 0, x(N) = 1"
"Construct a validation set consisting of 100 data samples"
"Similarly, construct a test set with 100 data samples"
"Use a four-digit number containing the last 4 digits of your student ID
 as the seed for the pseudo number generator"
"""

def generate_data(N_train, N_valid, N_test, seed):
    """
    Generate synthetic data for our experiment.

    Parameters:
    -----------
    N_train : int - Number of training samples (9 in our case)
    N_valid : int - Number of validation samples (100)
    N_test : int - Number of test samples (100)
    seed : int - Random seed for reproducibility (8097 = last 4 digits of student ID)

    Returns:
    --------
    X_train, t_train : Training inputs and targets
    X_valid, t_valid : Validation inputs and targets
    X_test, t_test : Test inputs and targets
    """
    np.random.seed(seed)

    # Create uniformly spaced x values
    X_train = np.linspace(0., 1., N_train)
    X_valid = np.linspace(0., 1., N_valid)
    X_test = np.linspace(0., 1., N_test)

    # Generate targets: t = sin(2πx) + noise
    # np.random.randn(N) generates N random numbers from standard normal (mean=0, std=1)
    # We multiply by 0.2 to get std=0.2 (variance=0.04 as specified)
    t_train = np.sin(2 * np.pi * X_train) + 0.2 * np.random.randn(N_train)
    t_valid = np.sin(2 * np.pi * X_valid) + 0.2 * np.random.randn(N_valid)
    t_test = np.sin(2 * np.pi * X_test) + 0.2 * np.random.randn(N_test)

    return X_train, t_train, X_valid, t_valid, X_test, t_test


#==============================================================================
# INSTRUCTION 3: Create Polynomial Features (Design Matrix)
#==============================================================================
"""
FROM ASSIGNMENT:
"Train N-1 regression models of increasing capacity"
"f_M(x) = w_0 + w_1*x + w_2*x^2 + ... + w_M*x^M"
"Think of it as a linear regression model, where the feature vector x
 has dimension M and is equal to (x, x^2, x^3, ..., x^M)^T"
"""

def create_polynomial_features(X, M):
    """
    Create the design matrix Φ (Phi) for polynomial regression.

    Parameters:
    -----------
    X : array of shape (N,) - Input values
    M : int - Polynomial degree

    Returns:
    --------
    Phi : array of shape (N, M+1) - Design matrix

    Example:
    --------
    X = [0, 0.5, 1], M = 2
    Returns:
    [[1, 0,   0   ],
     [1, 0.5, 0.25],
     [1, 1,   1   ]]
    """
    N = len(X)

    # Create array of powers: [0, 1, 2, ..., M]
    powers = np.arange(M + 1)

    # VECTORIZATION:
    Phi = X[:, np.newaxis] ** powers

    return Phi


#==============================================================================
# INSTRUCTION 4: Train the Model to Find Optimal Weight
#==============================================================================
"""
FROM ASSIGNMENT:
"Train the model by minimizing the average squared error on the training set"
"Apply the formula given in class to obtain the vector of parameters w"
The FORMULA Equation is:
    w* = (Φ^T Φ)^(-1) Φ^T t

WHERE:
- Φ^T means "transpose of Φ" (flip rows and columns)
- (...)^(-1) means "inverse of matrix"
- This formula gives us the weights that minimize the squared error

"""

def train_polynomial_regression(Phi, t):
    """
    Train polynomial regression using the closed-form (Normal Equation) solution.

    Parameters:
    -----------
    Phi : array of shape (N, M+1) - Design matrix
    t : array of shape (N,) - Target values

    Returns:
    --------
    w : array of shape (M+1,) - Optimal weight vector

    The formula is: w = (Φ^T Φ)^(-1) Φ^T t
    """

    # Using np.linalg.lstsq to find w that minimizes ||Φw - t||^2 which is numerically stable

    w, residuals, rank, singular_values = np.linalg.lstsq(Phi, t, rcond=None)

    return w


#==============================================================================
# INSTRUCTION 5: Compute RMSE (Root Mean Squared Error)
#==============================================================================
"""
FROM ASSIGNMENT:
"Record the training and validation root mean squared errors (RMSE)"

RMSE = sqrt( (1/N) * Σ(t_i - prediction_i)^2 )

"""

def compute_rmse(Phi, t, w):
    """
    Compute Root Mean Squared Error using VECTORIZATION.

    Parameters:
    -----------
    Phi : array of shape (N, M+1) - matrix
    t : array of shape (N,) - True target values
    w : array of shape (M+1,) - Weight vector

    Returns:
    --------
    rmse : float - Root mean squared error

    VECTORIZATION:
    - We compute predictions for ALL points at once using matrix multiplication
    """

    # Step 1: Compute predictions for all points at once
    # Phi @ w is matrix-vector multiplication
    predictions = Phi @ w  # vectorization

    # Step 2: Compute errors (vectorized subtraction)
    errors = t - predictions  # Element-wise subtraction

    # Step 3: Compute squared errors (vectorized)
    squared_errors = errors ** 2  # Element-wise squaring

    # Step 4: Compute mean (vectorized)
    mse = np.mean(squared_errors)

    # Step 5: Take square root
    rmse = np.sqrt(mse)

    return rmse


def compute_optimal_rmse(X, t):
    """
    Compute RMSE for the optimal predictor f_opt(x) = sin(2πx).

    """
    optimal_predictions = np.sin(2 * np.pi * X)
    errors = t - optimal_predictions
    rmse = np.sqrt(np.mean(errors ** 2))
    return rmse


#==============================================================================
# INSTRUCTION 6: L2 Regularization
#==============================================================================
"""
FROM ASSIGNMENT:
"For M = N-1 you have to train the model with L2 regularization"
"Consider λ = 10^i for all integers i between -14 and 2 inclusive"
"Features have to be standardized first"
 FORMULA:
    w* = (Φ^T Φ + λI)^(-1) Φ^T t
"""

def train_with_regularization(Phi, t, lambda_reg):
    """
    Train polynomial regression with L2 regularization (Ridge Regression).

    Parameters:
    -----------
    Phi : array of shape (N, M+1) - Design matrix (should be standardized!)
    t : array of shape (N,) - Target values
    lambda_reg : float - Regularization strength

    Returns:
    --------
    w : array of shape (M+1,) - Optimal weight vector

    Formula: w = (Φ^T Φ + λI)^(-1) Φ^T t
    """

    num_features = Phi.shape[1]  # M + 1

    # Create identity matrix
    I = np.eye(num_features)

    # Apply the regularized equation

    w = np.linalg.inv(Phi.T @ Phi + lambda_reg * I) @ Phi.T @ t

    return w


#==============================================================================
# INSTRUCTION 7: Plotting Functions
#==============================================================================
"""
FROM ASSIGNMENT:
"Plot the predictor function f_M(x) and the curve f_opt(x) for x ∈ [0,1]"
"Include all points in training set and validation set"
"Use different colours for training examples, validation examples,
 the trained prediction function and the optimal predictor"
"Have separate figures for different values of M"
"In a separate figure, plot training and validation RMSEs versus M"
"""

def plot_polynomial_fit(X_train, t_train, X_valid, t_valid, w, M, save_path):
    """
    Plot the polynomial fit along with training and validation data.
    """
    plt.figure(figsize=(10, 6))

    # Create dense x values for smooth curves
    X_plot = np.linspace(0, 1, 200)

    # Compute optimal predictor (true function without noise)
    y_optimal = np.sin(2 * np.pi * X_plot)

    # Compute our trained predictor
    Phi_plot = create_polynomial_features(X_plot, M)
    y_predicted = Phi_plot @ w

    # Plot everything
    plt.plot(X_plot, y_optimal, 'g-', linewidth=2,
             label=r'$f_{opt}(x) = \sin(2\pi x)$')
    plt.plot(X_plot, y_predicted, 'r-', linewidth=2,
             label=f'$f_{{{M}}}(x)$ (Trained, M={M})')
    plt.scatter(X_train, t_train, c='blue', s=80, marker='o',
                edgecolors='black', label='Training data', zorder=5)
    plt.scatter(X_valid, t_valid, c='orange', s=30, marker='x',
                alpha=0.7, label='Validation data', zorder=4)

    plt.xlabel('x', fontsize=12)
    plt.ylabel('t', fontsize=12)
    plt.title(f'Polynomial Regression with M = {M}', fontsize=14)
    plt.legend(loc='upper right')
    plt.xlim(-0.05, 1.05)
    plt.ylim(-1.5, 1.5)
    plt.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_regularized_fit(X_train, t_train, X_valid, t_valid, w, lambda_val,
                         scaler, M, save_path):
    """
    Plot the regularized polynomial fit.
    Note: We need the scaler because weights were trained on standardized features.
    """
    plt.figure(figsize=(10, 6))

    X_plot = np.linspace(0, 1, 200)
    y_optimal = np.sin(2 * np.pi * X_plot)

    # For prediction, we need to standardize the features the same way
    Phi_plot = create_polynomial_features(X_plot, M)
    Phi_plot_scaled = scaler.transform(Phi_plot)
    y_predicted = Phi_plot_scaled @ w

    plt.plot(X_plot, y_optimal, 'g-', linewidth=2,
             label=r'$f_{true}(x) = \sin(2\pi x)$')
    plt.plot(X_plot, y_predicted, 'r-', linewidth=2,
             label=f'Regularized fit ($\\lambda = 10^{{{int(np.log10(lambda_val))}}}$)')
    plt.scatter(X_train, t_train, c='blue', s=80, marker='o',
                edgecolors='black', label='Training data', zorder=5)
    plt.scatter(X_valid, t_valid, c='orange', s=30, marker='x',
                alpha=0.7, label='Validation data', zorder=4)

    plt.xlabel('x', fontsize=12)
    plt.ylabel('t', fontsize=12)
    plt.title(f'Regularized Polynomial (M=8, $\\lambda = 10^{{{int(np.log10(lambda_val))}}}$)', fontsize=14)
    plt.legend(loc='upper right')
    plt.xlim(-0.05, 1.05)
    plt.ylim(-1.5, 1.5)
    plt.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_rmse_vs_M(M_values, train_rmse, valid_rmse, optimal_rmse, save_path):
    """
    Plot training and validation RMSE vs polynomial degree M.
    """
    plt.figure(figsize=(10, 6))

    plt.plot(M_values, train_rmse, 'b-o', linewidth=2, markersize=8,
             label='Training RMSE')
    plt.plot(M_values, valid_rmse, 'r-s', linewidth=2, markersize=8,
             label='Validation RMSE')
    plt.axhline(y=optimal_rmse, color='g', linestyle='--', linewidth=2,
                label=f'Optimal Predictor RMSE = {optimal_rmse:.4f}')

    plt.xlabel('Polynomial Degree M', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.title('Training and Validation RMSE vs Polynomial Degree', fontsize=14)
    plt.legend()
    plt.xticks(M_values)
    plt.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_rmse_vs_lambda(log_lambdas, train_rmse, valid_rmse, optimal_rmse, save_path):
    """
    Plot training and validation RMSE vs log10(lambda).
    """
    plt.figure(figsize=(12, 6))

    plt.plot(log_lambdas, train_rmse, 'b-o', linewidth=2, markersize=6,
             label='Training RMSE')
    plt.plot(log_lambdas, valid_rmse, 'r-s', linewidth=2, markersize=6,
             label='Validation RMSE')
    plt.axhline(y=optimal_rmse, color='g', linestyle='--', linewidth=2,
                label=f'Optimal Predictor RMSE = {optimal_rmse:.4f}')

    plt.xlabel(r'$\log_{10}(\lambda)$', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.title(r'Training and Validation RMSE vs $\log_{10}(\lambda)$ (M=8)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


#==============================================================================
# MAIN EXPERIMENT Function
#==============================================================================

def run_experiment():
    """
    Run the complete experiment as specified in the assignment.
    """

    print("=" * 70)
    print("ECE 726 Assignment 1: Trade-off between Overfitting and Underfitting")
    print("=" * 70)

    #--------------------------------------------------------------------------
    # STEP 1: Generate Data
    #--------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 1: GENERATE DATA")
    print("=" * 70)
    print("""
    What we're doing:
    - Creating synthetic data where we KNOW the true function
    - True function: sin(2πx)
    - Adding Gaussian noise with variance 0.04 (std = 0.2)
    - Using seed 8097 for reproducibility
    """)

    SEED = 8097
    N_train = 9
    N_valid = 100
    N_test = 100

    X_train, t_train, X_valid, t_valid, X_test, t_test = generate_data(
        N_train, N_valid, N_test, SEED
    )

    print(f"  Training set: {N_train} samples")
    print(f"  Validation set: {N_valid} samples")
    print(f"  Test set: {N_test} samples")
    print(f"  Random seed: {SEED}")

    # Compute optimal predictor RMSE
    optimal_rmse_valid = compute_optimal_rmse(X_valid, t_valid)
    optimal_rmse_test = compute_optimal_rmse(X_test, t_test)

    print(f"\n  Optimal predictor RMSE (validation): {optimal_rmse_valid:.4f}")
    print(f"  Optimal predictor RMSE (test): {optimal_rmse_test:.4f}")
    print(f"  (This is the irreducible error due to noise ≈ 0.2)")

    #--------------------------------------------------------------
    # STEP 2: Train Polynomial Models for M = 1 to 8
    #--------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2: TRAIN POLYNOMIAL MODELS (M = 1 to 8)")
    print("=" * 70)
    print("""
    Steps:
    - Training 8 models with increasing complexity (M = 1, 2, ..., 8)
    - M = 1: Linear
    - M = 2: Quadratic
    - M = 8: 8th degree polynomial

    For each M:
    1. Create polynomial features (design matrix Φ)
    2. Solve for optimal weights using normal equation
    3. Compute training and validation RMSE
    """)

    M_values = list(range(1, N_train))  # M = 1, 2, ..., 8
    train_rmse_list = []
    valid_rmse_list = []
    weights_list = []

    print(f"\n  {'M':<4} {'Train RMSE':<15} {'Valid RMSE':<15}")
    print("  " + "-" * 34)

    for M in M_values:
        # Create polynomial features
        Phi_train = create_polynomial_features(X_train, M)
        Phi_valid = create_polynomial_features(X_valid, M)

        # Train model (find optimal weights)
        w = train_polynomial_regression(Phi_train, t_train)
        weights_list.append(w)

        # Compute RMSE
        train_rmse = compute_rmse(Phi_train, t_train, w)
        valid_rmse = compute_rmse(Phi_valid, t_valid, w)

        train_rmse_list.append(train_rmse)
        valid_rmse_list.append(valid_rmse)

        print(f"  {M:<4} {train_rmse:<15.6f} {valid_rmse:<15.6f}")

        # Save plot for this M
        plot_polynomial_fit(X_train, t_train, X_valid, t_valid, w, M,
                           f'polynomial_fit_M{M}.png')

    # Plot RMSE vs M
    print("\n  Creating RMSE vs M plot...")
    plot_rmse_vs_M(M_values, train_rmse_list, valid_rmse_list,
                   optimal_rmse_valid, 'rmse_vs_M.png')

    #--------------------------------------------------------------------
    # STEP 3: Analysis - Identify Regions
    #---------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3: ANALYSIS - IDENTIFYING REGIONS")
    print("=" * 70)

    # Find best M
    best_M_idx = np.argmin(valid_rmse_list)
    best_M = M_values[best_M_idx]
    print(f"  Best polynomial degree: M = {best_M}")
    print(f"  Best validation RMSE: {valid_rmse_list[best_M_idx]:.6f}")

    #---------------------------------------------------------------------
    # STEP 4: L2 Regularization for M = 8
    #----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4: L2 REGULARIZATION (M = 8)")
    print("=" * 70)
    print("""
    What we're doing:
    - Taking the most complex model (M = 8) which overfits
    - Adding L2 regularization to control overfitting
    - Testing λ = 10^i for i = -14, -13, ..., 2
    """)

    M = 8
    Phi_train = create_polynomial_features(X_train, M)
    Phi_valid = create_polynomial_features(X_valid, M)

    # Standardize features
    scaler = StandardScaler()
    Phi_train_scaled = scaler.fit_transform(Phi_train)  # Fit on training data
    Phi_valid_scaled = scaler.transform(Phi_valid)       # Apply same transform to validation

    # Test different lambda values
    log_lambdas = list(range(-14, 3))  # -14, -13, ..., 2
    lambdas = [10**i for i in log_lambdas]

    reg_train_rmse = []
    reg_valid_rmse = []
    reg_weights = {}

    print(f"\n  {'log10(λ)':<12} {'Train RMSE':<15} {'Valid RMSE':<15}")
    print("  " + "-" * 42)

    for i, lam in enumerate(lambdas):
        w = train_with_regularization(Phi_train_scaled, t_train, lam)
        reg_weights[log_lambdas[i]] = w

        train_rmse = compute_rmse(Phi_train_scaled, t_train, w)
        valid_rmse = compute_rmse(Phi_valid_scaled, t_valid, w)

        reg_train_rmse.append(train_rmse)
        reg_valid_rmse.append(valid_rmse)

        print(f"  {log_lambdas[i]:<12} {train_rmse:<15.6f} {valid_rmse:<15.6f}")

    # Plot RMSE vs lambda
    print("\n  Creating RMSE vs λ plot...")
    plot_rmse_vs_lambda(log_lambdas, reg_train_rmse, reg_valid_rmse,
                        optimal_rmse_valid, 'rmse_vs_lambda.png')

    # Find best lambda
    best_lambda_idx = np.argmin(reg_valid_rmse)
    best_log_lambda = log_lambdas[best_lambda_idx]
    best_lambda = lambdas[best_lambda_idx]

    print(f"\n  Best log10(λ): {best_log_lambda}")
    print(f"  Best λ: {best_lambda:.2e}")
    print(f"  Best validation RMSE: {reg_valid_rmse[best_lambda_idx]:.6f}")


    #--------------------------------------------------------------------------
    # STEP 5: Visualize Three Lambda Values
    #--------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 6: VISUALIZE THREE λ VALUES")
    print("=" * 70)

    # Select three lambda values
    lambda1_log = -14  # Overfitting
    lambda2_log = best_log_lambda  # Optimal
    lambda3_log = 2    # Underfitting

    lambda1 = 10 ** lambda1_log
    lambda2 = 10 ** lambda2_log
    lambda3 = 10 ** lambda3_log

    print(f"\n  λ₁ (Overfitting):   10^{lambda1_log} = {lambda1:.2e}")
    print(f"  λ₂ (Optimal):       10^{lambda2_log} = {lambda2:.2e}")
    print(f"  λ₃ (Underfitting):  10^{lambda3_log} = {lambda3:.2e}")

    # Plot and report weights for each
    for lam_log, lam, label in [(lambda1_log, lambda1, 'overfitting'),
                                 (lambda2_log, lambda2, 'optimal'),
                                 (lambda3_log, lambda3, 'underfitting')]:
        w = train_with_regularization(Phi_train_scaled, t_train, lam)

        print(f"\n  Weight vector for λ = 10^{lam_log} ({label}):")
        print(f"    ||w|| = {np.linalg.norm(w):.4f}")
        print(f"    w = [{', '.join([f'{wi:.4f}' for wi in w])}]")

        plot_regularized_fit(X_train, t_train, X_valid, t_valid, w, lam,
                            scaler, M, f'regularized_fit_{label}.png')

    #------------------------------------------------------------------------
    # STEP 6: Select Best Model and Test
    #--------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 7: SELECT BEST MODEL AND TEST")
    print("=" * 70)

    best_poly_valid_rmse = valid_rmse_list[best_M_idx]
    best_reg_valid_rmse = reg_valid_rmse[best_lambda_idx]

    print(f"\n  Best polynomial (M={best_M}, no reg): Valid RMSE = {best_poly_valid_rmse:.6f}")
    print(f"  Best regularized (M=8, λ=10^{best_log_lambda}): Valid RMSE = {best_reg_valid_rmse:.6f}")

    # Choose the better one
    if best_poly_valid_rmse <= best_reg_valid_rmse:
        print(f"\n Selected: Polynomial M={best_M} (no regularization)")
        w_final = weights_list[best_M_idx]
        Phi_test = create_polynomial_features(X_test, best_M)
        test_rmse = compute_rmse(Phi_test, t_test, w_final)
    else:
        print(f"\n Selected: Regularized polynomial M=8, λ=10^{best_log_lambda}")
        w_final = reg_weights[best_log_lambda]
        Phi_test = create_polynomial_features(X_test, M)
        Phi_test_scaled = scaler.transform(Phi_test)
        test_rmse = compute_rmse(Phi_test_scaled, t_test, w_final)


    print(f"  FINAL TEST SET RMSE: {test_rmse:.6f}      ")

    print(f"\n  Optimal predictor RMSE (test): {optimal_rmse_test:.6f}")
    print(f"  Difference from optimal: {test_rmse - optimal_rmse_test:.6f}")



    return {
        'train_rmse_list': train_rmse_list,
        'valid_rmse_list': valid_rmse_list,
        'reg_train_rmse': reg_train_rmse,
        'reg_valid_rmse': reg_valid_rmse,
        'best_M': best_M,
        'best_lambda': best_lambda,
        'test_rmse': test_rmse
    }
#--------------------------------------------------------------------
# RUN THE EXPERIMENT
#--------------------------------------------------------------------------
if __name__ == "__main__":
    results = run_experiment()