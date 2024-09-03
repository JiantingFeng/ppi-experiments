import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from functools import partial
import argparse
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
)
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import os


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
    if args.pseudo_label:
        Y_unlabeled_pseudo = np.where(X_unlabeled @ beta_hat > 0.5, 1, 0)
        Y_labeled_pseudo = np.where(X_labeled @ beta_hat > 0.5, 1, 0)
        loss_unlabeled = logistic_loss(X_unlabeled, Y_unlabeled_pseudo, temp, beta)
        loss_correction = logistic_loss(
            X_labeled, y_labeled, temp, beta
        ) - logistic_loss(X_labeled, Y_labeled_pseudo, temp, beta)
    else:
        loss_unlabeled = logistic_loss(X_unlabeled, X_unlabeled @ beta_hat, temp, beta)
        loss_correction = logistic_loss(
            X_labeled, y_labeled, temp, beta
        ) - logistic_loss(X_labeled, X_labeled @ beta_hat, temp, beta)
    return loss_unlabeled - loss_correction


def grad_ppi_logistic_loss(X_labeled, y_labeled, X_unlabeled, beta_hat, temp, beta):
    if args.pseudo_label:
        Y_unlabeled_pseudo = np.where(X_unlabeled @ beta_hat > 0.5, 1, 0)
        Y_labeled_pseudo = np.where(X_labeled @ beta_hat > 0.5, 1, 0)
        grad_unlabeled = grad_logistic_loss(X_unlabeled, Y_unlabeled_pseudo, temp, beta)
        grad_correction = grad_logistic_loss(
            X_labeled, y_labeled, temp, beta
        ) - grad_logistic_loss(X_labeled, Y_labeled_pseudo, temp, beta)
    else:
        grad_unlabeled = grad_logistic_loss(
            X_unlabeled, X_unlabeled @ beta_hat, temp, beta
        )
        grad_correction = grad_logistic_loss(
            X_labeled, y_labeled, temp, beta
        ) - grad_logistic_loss(X_labeled, X_labeled @ beta_hat, temp, beta)
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


def bootstrap_sample(X, y, n_samples):
    indices = np.random.choice(len(X), size=n_samples, replace=True)
    return X[indices], y[indices]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPI Logistic Regression Experiment")
    parser.add_argument(
        "--n_samples", type=int, default=10000, help="Number of samples"
    )
    parser.add_argument("--n_dims", type=int, default=10, help="Number of dimensions")
    parser.add_argument(
        "--n_exps", type=int, default=1000, help="Number of experiments"
    )
    parser.add_argument(
        "--labeled_unlabeled_ratio",
        type=float,
        default=0.3,
        help="Ratio of labeled to unlabeled data",
    )
    parser.add_argument(
        "--pseudo_label", type=bool, default=True, help="Whether to use pseudo-labels"
    )
    parser.add_argument("--temp", type=float, default=1.0, help="Temperature parameter")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--save_figure",
        action="store_true",
        help="Save the figure as 'pseudo_label.png'",
    )
    parser.add_argument(
        "--result_file",
        type=str,
        default="result.csv",
        help="Name of the CSV file to save results",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    console = Console()

    console.print(
        Panel.fit(
            "[bold green]Starting PPI Logistic Regression Experiment[/bold green]",
            border_style="green",
        )
    )

    param_table = Table(
        title="Experiment Parameters", show_header=True, header_style="bold magenta"
    )
    param_table.add_column("Parameter", style="cyan", justify="right")
    param_table.add_column("Value", style="yellow")
    param_table.add_row("Number of samples", str(args.n_samples))
    param_table.add_row("Number of dimensions", str(args.n_dims))
    param_table.add_row("Number of experiments", str(args.n_exps))
    param_table.add_row("Labeled/Unlabeled ratio", str(args.labeled_unlabeled_ratio))
    param_table.add_row("Use pseudo-labels", str(args.pseudo_label))
    param_table.add_row("Temperature", str(args.temp))
    param_table.add_row("Random seed", str(args.seed))
    param_table.add_row("Save figure", str(args.save_figure))
    console.print(param_table)

    lr_bias_list = []
    ppi_bias_list = []

    # Generate the full dataset only once
    X_full, y_full = generate_data(
        args.n_samples * 2, args.n_dims
    )  # Generate extra data for holdout
    n_labeled = int(args.n_samples * args.labeled_unlabeled_ratio)
    n_unlabeled = args.n_samples - n_labeled

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    ) as progress:
        main_task = progress.add_task(
            "[green]Running experiments...", total=args.n_exps
        )

        for i in range(args.n_exps):
            progress.update(
                main_task,
                advance=1,
                description=f"[green]Running experiment {i+1}/{args.n_exps}...",
            )

            beta_true = np.asarray(
                [0] * (args.n_dims // 2) + [1] * (args.n_dims - args.n_dims // 2)
            )

            # Bootstrap sample from the full dataset
            X_train, y_train = bootstrap_sample(
                X_full[: args.n_samples], y_full[: args.n_samples], args.n_samples
            )
            X_holdout, y_holdout = bootstrap_sample(
                X_full[args.n_samples :], y_full[args.n_samples :], args.n_samples
            )

            # Split the training data into labeled and unlabeled sets
            X_train_labeled, y_train_labeled = X_train[:n_labeled], y_train[:n_labeled]
            X_train_unlabeled = X_train[n_labeled:]

            beta_hat = optimize_logistic_regression(X_holdout, y_holdout, args.temp)

            beta_ppi = optimize_ppi_logistic_regression(
                X_train_labeled, y_train_labeled, X_train_unlabeled, args.temp
            )

            lr_bias_list.append(beta_hat - beta_true)
            ppi_bias_list.append(beta_ppi - beta_true)

    console.print(
        Panel.fit(
            "[bold green]Experiments completed![/bold green]", border_style="green"
        )
    )

    lr_bias_array = np.array(lr_bias_list)
    ppi_bias_array = np.array(ppi_bias_list)

    console.print("[bold cyan]Generating plot...[/bold cyan]")
    # Combine data for boxplot
    data = np.concatenate([lr_bias_array, ppi_bias_array], axis=0)
    labels = ["LR"] * lr_bias_array.shape[0] * args.n_dims + [
        "LR-PPI"
    ] * ppi_bias_array.shape[0] * args.n_dims
    dimensions = np.tile(
        np.arange(1, args.n_dims + 1), lr_bias_array.shape[0] + ppi_bias_array.shape[0]
    )

    # Set up the plot with new color scheme
    plt.figure(figsize=(15, 8))
    sns.set_style("whitegrid")
    sns.boxplot(
        x=dimensions,
        y=data.flatten(),
        hue=labels,
        dodge=True,
        palette={"LR": "lightblue", "LR-PPI": "gray"},
    )

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

    # Customize legend
    plt.legend(title="Method", loc="upper right", frameon=True)

    # Show the plot
    plt.tight_layout()

    if args.save_figure:
        plt.savefig("pseudo_label.png", dpi=300, bbox_inches="tight")
        console.print("[cyan]Plot saved as 'pseudo_label.png'[/cyan]")
    else:
        plt.show()

    results_table = Table(
        title="Experiment Results", show_header=True, header_style="bold magenta"
    )
    results_table.add_column("Metric", style="cyan", justify="right")
    results_table.add_column("Logistic Regression", style="blue")
    results_table.add_column("PPI Logistic Regression", style="green")
    results_table.add_row(
        "Mean Bias", f"{lr_bias_array.mean():.4f}", f"{ppi_bias_array.mean():.4f}"
    )
    results_table.add_row(
        "Std Bias", f"{lr_bias_array.std():.4f}", f"{ppi_bias_array.std():.4f}"
    )

    results_table.add_row(
        "Max Bias", f"{lr_bias_array.max():.4f}", f"{ppi_bias_array.max():.4f}"
    )
    results_table.add_row(
        "Min Bias", f"{lr_bias_array.min():.4f}", f"{ppi_bias_array.min():.4f}"
    )
    results_table.add_row(
        "Variance", f"{lr_bias_array.var():.4f}", f"{ppi_bias_array.var():.4f}"
    )
    console.print(results_table)

    # Calculate and report average variance across dimensions
    lr_avg_var = np.mean(np.var(lr_bias_array, axis=0))
    ppi_avg_var = np.mean(np.var(ppi_bias_array, axis=0))

    # Create a dictionary with the results
    result_dict = {
        "n_samples": args.n_samples,
        "n_dims": args.n_dims,
        "n_exps": args.n_exps,
        "labeled_unlabeled_ratio": args.labeled_unlabeled_ratio,
        "pseudo_label": args.pseudo_label,
        "temp": args.temp,
        "lr_mean_bias": lr_bias_array.mean(),
        "ppi_mean_bias": ppi_bias_array.mean(),
        "lr_std_bias": lr_bias_array.std(),
        "ppi_std_bias": ppi_bias_array.std(),
        "lr_max_bias": lr_bias_array.max(),
        "ppi_max_bias": ppi_bias_array.max(),
        "lr_min_bias": lr_bias_array.min(),
        "ppi_min_bias": ppi_bias_array.min(),
        "lr_variance": lr_bias_array.var(),
        "ppi_variance": ppi_bias_array.var(),
        "lr_avg_var": lr_avg_var,
        "ppi_avg_var": ppi_avg_var,
    }

    # Create a DataFrame from the dictionary
    result_df = pd.DataFrame([result_dict])

    # Check if the file exists
    if os.path.exists(args.result_file):
        # If the file exists, append without writing the header
        result_df.to_csv(args.result_file, mode="a", header=False, index=False)
        console.print(f"[cyan]Results appended to '{args.result_file}'[/cyan]")
    else:
        # If the file doesn't exist, create it and write the header
        result_df.to_csv(args.result_file, index=False)
        console.print(f"[cyan]Results saved to new file '{args.result_file}'[/cyan]")

    console.print(
        Panel.fit(
            "[bold green]Experiment completed successfully![/bold green]",
            border_style="green",
        )
    )
