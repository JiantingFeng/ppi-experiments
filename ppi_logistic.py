import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from functools import partial


np.random.seed(42)

n_samples = 100000
n_dims = 10
n_exps = 100
labeled_unlabeled_ratio = 0.3

lr_bias_list = []
ppi_bias_list = []


def log1pexp(x):
    """
    Numerically accurate evaluation of log(1 + exp(x)).

    This function computes log(1 + exp(x)) in a way that is numerically stable
    for both large positive and large negative values of x.

    Args:
    x (np.ndarray): Input array

    Returns:
    np.ndarray: log(1 + exp(x)) computed in a numerically stable way
    """
    # For large positive x, log(1 + exp(x)) ≈ x
    # For x near zero, we use the direct computation
    # For large negative x, we use exp(x) to avoid overflow
    threshold = np.log(np.finfo(x.dtype).max) - 1e-4

    result = np.where(
        x > threshold,
        x,
        np.where(
            x > -threshold, np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0), np.exp(x)
        ),
    )

    return result


def sigmoid(x):
    """
    Numerically stable sigmoid function.

    This function computes the sigmoid of x in a way that is numerically stable
    for both large positive and large negative values of x.

    Args:
    x (np.ndarray): Input array

    Returns:
    np.ndarray: Sigmoid of x computed in a numerically stable way
    """
    # For large negative x, sigmoid(x) ≈ exp(x)
    # For large positive x, sigmoid(x) ≈ 1 - exp(-x)
    # For x near zero, we use the standard formula

    mask = x >= 0
    z = np.exp(-np.abs(x))

    return np.where(mask, 1 / (1 + z), z / (1 + z))


def logistic_loss(X, y, temp, beta):
    # X: [n, d], beta: [d]
    z = X @ beta / temp
    loss = np.mean(-y * z + log1pexp(z))
    return loss


def grad_logistic_loss(X, y, temp, beta):
    """
    Calculate the gradient of logistic loss with respect to beta.

    Args:
    X (np.ndarray): Feature matrix (n_features, n_samples)
    y (np.ndarray): Target vector (n_samples,)
    beta (np.ndarray): Coefficient vector (n_features,)

    Returns:
    np.ndarray: Gradient vector (n_features,)
    """
    z = (X @ beta) / temp
    sigmoid_z = sigmoid(z)
    grad_loss = (X.T @ (sigmoid_z - y)) / len(y)
    return grad_loss


def optimize_logistic_regression(X, y, temp: float = 1):
    """
    Optimize the logistic regression parameters using L-BFGS.

    Args:
    X (np.array): Feature matrix (N x D)
    y (np.array): True binary labels (N,)

    Returns:
    np.array: Optimized parameters
    """
    # Initialize beta (parameters)
    initial_beta = np.zeros(X.shape[1])
    loss_func = partial(logistic_loss, X, y, temp)
    grad_func = partial(grad_logistic_loss, X, y, temp)
    # Optimize using L-BFGS
    result = minimize(
        fun=loss_func,
        x0=initial_beta,
        method="BFGS",
        jac=grad_func,
        options={"disp": False},
    )

    return result.x


def ppi_logistic_loss(X_labeled, y_labeled, X_unlabeled, beta_hat, temp, beta):
    Y_unlabeled_psedo = np.where(X_unlabeled @ beta_hat > 0.5, 1, 0)
    Y_labeled_psedo = np.where(X_labeled @ beta_hat > 0.5, 1, 0)
    # loss_unlabeled = logistic_loss(X_unlabeled, X_unlabeled @ beta_hat, temp, beta)
    # loss_correction = logistic_loss(X_labeled, y_labeled, temp, beta) - logistic_loss(
    #     X_labeled, X_labeled @ beta_hat, temp, beta
    # )
    loss_unlabeled = logistic_loss(X_unlabeled, Y_unlabeled_psedo, temp, beta)
    loss_correction = logistic_loss(X_labeled, y_labeled, temp, beta) - logistic_loss(
        X_labeled, Y_labeled_psedo, temp, beta
    )
    return loss_unlabeled - loss_correction


def grad_ppi_logistic_loss(X_labeled, y_labeled, X_unlabeled, beta_hat, temp, beta):
    Y_unlabeled_psedo = np.where(X_unlabeled @ beta_hat > 0.5, 1, 0)
    Y_labeled_psedo = np.where(X_labeled @ beta_hat > 0.5, 1, 0)
    # grad_unlabeled = grad_logistic_loss(X_unlabeled, X_unlabeled @ beta_hat, temp, beta)
    # grad_correction = grad_logistic_loss(
    #     X_labeled, y_labeled, temp, beta
    # ) - grad_logistic_loss(X_labeled, X_labeled @ beta_hat, temp, beta)
    grad_unlabeled = grad_logistic_loss(X_unlabeled, Y_unlabeled_psedo, temp, beta)
    grad_correction = grad_logistic_loss(
        X_labeled, y_labeled, temp, beta
    ) - grad_logistic_loss(X_labeled, Y_labeled_psedo, temp, beta)
    return grad_unlabeled + grad_correction


def optimize_ppi_logistic_regression(
    X_labeled, y_labeled, X_unlabeled, temp: float = 1
):
    """
    Optimize the logistic regression parameters using L-BFGS.

    Args:
    X_labeled (np.array): Labeled feature matrix
    y_labeled (np.array): Labels for labeled data
    X_unlabeled (np.array): Unlabeled feature matrix
    temp (float): Temperature parameter

    Returns:
    np.array: Optimized parameters
    """
    # Initialize beta (parameters)
    initial_beta = np.zeros(X_labeled.shape[1])

    beta_hat = optimize_logistic_regression(
        X_holdout, y_holdout, temp
    )  # Only use once, then discard this part of data

    loss_func = partial(
        ppi_logistic_loss, X_labeled, y_labeled, X_unlabeled, beta_hat, temp
    )
    grad_func = partial(
        grad_ppi_logistic_loss, X_labeled, y_labeled, X_unlabeled, beta_hat, temp
    )

    # Optimize using BFGS
    result = minimize(
        fun=loss_func,
        x0=initial_beta,
        method="BFGS",
        jac=grad_func,
        options={"disp": False},
    )

    return result.x


def generate_data(n_samples, n_dims):
    X = np.random.randn(n_samples, n_dims)
    beta_true = np.asarray([0] * (n_dims // 2) + [1] * (n_dims - n_dims // 2))

    # Generate target variable
    y = np.random.binomial(1, 1 / (1 + np.exp(-X.dot(beta_true))))

    return X, y


for i in range(n_exps):
    print(f"Experiment no. {i}")
    beta_true = np.asarray([0] * (n_dims // 2) + [1] * (n_dims - n_dims // 2))
    n_labeled = int(n_samples * labeled_unlabeled_ratio)
    n_unlabeled = n_samples - int(n_samples * labeled_unlabeled_ratio)

    X_train_labeled, y_train_labeled = generate_data(n_labeled, n_dims)
    X_train_unlabeled, _ = generate_data(n_unlabeled, n_dims)
    X_holdout, y_holdout = generate_data(n_samples * 10, n_dims)

    beta_hat = optimize_logistic_regression(X_holdout, y_holdout)
    beta_ppi = optimize_ppi_logistic_regression(
        X_train_labeled, y_train_labeled, X_train_unlabeled
    )

    lr_bias_list.append(beta_hat - beta_true)
    ppi_bias_list.append(beta_ppi - beta_true)

lr_bias_array = np.array(lr_bias_list)
ppi_bias_array = np.array(ppi_bias_list)

# Save as NPY file
np.save("lr_bias_array.npy", lr_bias_array)
np.save("ppi_bias_array.npy", ppi_bias_array)

# Combine data for boxplot
data = np.concatenate([lr_bias_array, ppi_bias_array], axis=0)
labels = ["LR"] * lr_bias_array.shape[0] * n_dims + ["PPI"] * ppi_bias_array.shape[
    0
] * n_dims
dimensions = np.tile(
    np.arange(1, n_dims + 1), lr_bias_array.shape[0] + ppi_bias_array.shape[0]
)

# Set up the plot
plt.figure(figsize=(15, 8))
sns.boxplot(x=dimensions, y=data.flatten(), hue=labels, dodge=True, palette="Set2")

# Set title and labels
plt.title(
    "Bias Distribution: Logistic Regression vs PPI Logistic Regression",
    fontsize=16,
    fontweight="bold",
)
plt.xlabel("Dimensions", fontsize=12)
plt.ylabel("Bias", fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Show the plot
plt.tight_layout()
plt.savefig("psedo_label.png", dpi=300)  # Save with higher resolution
