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


# Function to compute log(1 + exp(x)) in a numerically stable way
def log1pexp(x):
    """
    Numerically accurate evaluation of log(1 + exp(x)).
    This function computes log(1 + exp(x)) in a way that is numerically stable
    for both large positive and large negative values of x.
    """
    # Define a threshold to avoid overflow in exponential calculations
    threshold = np.log(np.finfo(x.dtype).max) - 1e-4

    # Use numpy's where to handle different ranges of x
    result = np.where(
        x > threshold,
        x,  # For large positive x, return x
        np.where(
            x > -threshold,
            np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0),  # For x near zero
            np.exp(x),  # For large negative x, return exp(x)
        ),
    )

    return result


# Function to compute the sigmoid function in a numerically stable way
def sigmoid(x):
    """
    Numerically stable sigmoid function.
    This function computes the sigmoid of x in a way that is numerically stable
    for both large positive and large negative values of x.
    """
    # Create a mask for values of x that are non-negative
    mask = x >= 0
    z = np.exp(-np.abs(x))  # Compute exp(-|x|)

    # Use numpy's where to return the appropriate sigmoid values
    return np.where(mask, 1 / (1 + z), z / (1 + z))


# Function to compute the logistic loss
def logistic_loss(X, y, temp, beta):
    """
    Calculate the logistic loss given features, labels, temperature, and parameters.
    """
    z = X @ beta / temp  # Compute the linear combination of inputs and parameters
    loss = np.mean(-y * z + log1pexp(z))  # Calculate the mean logistic loss
    return loss


# Function to compute the gradient of the logistic loss
def grad_logistic_loss(X, y, temp, beta):
    """
    Calculate the gradient of logistic loss with respect to beta.
    """
    z = (X @ beta) / temp  # Compute the linear combination of inputs and parameters
    sigmoid_z = sigmoid(z)  # Compute the sigmoid of z
    grad_loss = (X.T @ (sigmoid_z - y)) / len(y)  # Calculate the gradient
    return grad_loss


# Function to optimize logistic regression parameters using L-BFGS
def optimize_logistic_regression(X, y, temp: float = 1):
    """
    Optimize the logistic regression parameters using L-BFGS.
    """
    initial_beta = np.zeros(X.shape[1])  # Initialize beta (parameters) to zeros
    loss_func = partial(logistic_loss, X, y, temp)  # Create a partial function for loss
    grad_func = partial(
        grad_logistic_loss, X, y, temp
    )  # Create a partial function for gradient

    # Optimize using L-BFGS
    result = minimize(
        fun=loss_func,
        x0=initial_beta,
        method="BFGS",
        jac=grad_func,
        options={"disp": False},
    )

    return result.x  # Return the optimized parameters


# Function to compute the PPI logistic loss
def ppi_logistic_loss(X_labeled, y_labeled, X_unlabeled, beta_hat, temp, beta):
    """
    Compute the PPI logistic loss considering pseudo-labeling.
    """
    if args.pseudo_label:  # Check if pseudo-labeling is enabled
        # Generate pseudo-labels for unlabeled and labeled data
        Y_unlabeled_pseudo = np.where(X_unlabeled @ beta_hat > 0.5, 1, 0)
        Y_labeled_pseudo = np.where(X_labeled @ beta_hat > 0.5, 1, 0)

        # Calculate loss for unlabeled data and correction for labeled data
        loss_unlabeled = logistic_loss(X_unlabeled, Y_unlabeled_pseudo, temp, beta)
        loss_correction = logistic_loss(
            X_labeled, y_labeled, temp, beta
        ) - logistic_loss(X_labeled, Y_labeled_pseudo, temp, beta)
    else:
        # If not using pseudo-labels, compute losses directly
        loss_unlabeled = logistic_loss(X_unlabeled, X_unlabeled @ beta_hat, temp, beta)
        loss_correction = logistic_loss(
            X_labeled, y_labeled, temp, beta
        ) - logistic_loss(X_labeled, X_labeled @ beta_hat, temp, beta)

    return loss_unlabeled - loss_correction  # Return the difference in losses


# Function to compute the gradient of the PPI logistic loss
def grad_ppi_logistic_loss(X_labeled, y_labeled, X_unlabeled, beta_hat, temp, beta):
    """
    Calculate the gradient of PPI logistic loss with respect to beta.
    """
    if args.pseudo_label:  # Check if pseudo-labeling is enabled
        # Generate pseudo-labels for unlabeled and labeled data
        Y_unlabeled_pseudo = np.where(X_unlabeled @ beta_hat > 0.5, 1, 0)
        Y_labeled_pseudo = np.where(X_labeled @ beta_hat > 0.5, 1, 0)

        # Calculate gradients for unlabeled and correction for labeled data
        grad_unlabeled = grad_logistic_loss(X_unlabeled, Y_unlabeled_pseudo, temp, beta)
        grad_correction = grad_logistic_loss(
            X_labeled, y_labeled, temp, beta
        ) - grad_logistic_loss(X_labeled, Y_labeled_pseudo, temp, beta)
    else:
        # If not using pseudo-labels, compute gradients directly
        grad_unlabeled = grad_logistic_loss(
            X_unlabeled, X_unlabeled @ beta_hat, temp, beta
        )
        grad_correction = grad_logistic_loss(
            X_labeled, y_labeled, temp, beta
        ) - grad_logistic_loss(X_labeled, X_labeled @ beta_hat, temp, beta)

    return grad_unlabeled + grad_correction  # Return the sum of gradients


# Function to optimize PPI logistic regression parameters
def optimize_ppi_logistic_regression(
    X_labeled, y_labeled, X_unlabeled, temp: float = 1
):
    """
    Optimize the logistic regression parameters using L-BFGS.
    """
    initial_beta = np.zeros(X_labeled.shape[1])  # Initialize beta (parameters) to zeros

    # Optimize logistic regression on holdout data to get beta_hat
    beta_hat = optimize_logistic_regression(X_holdout, y_holdout, temp)

    # Create partial functions for loss and gradient
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

    return result.x  # Return the optimized parameters


# Function to generate synthetic data for testing
def generate_data(n_samples, n_dims):
    """
    Generate synthetic data for logistic regression.
    """
    X = np.random.randn(n_samples, n_dims)  # Generate random features
    beta_true = np.asarray(
        [0] * (n_dims // 2) + [1] * (n_dims - n_dims // 2)
    )  # True coefficients
    noise = np.random.randn(n_samples)  # Generate noise
    # Generate target variable based on linear combination of features and noise
    y = np.where(X @ beta_true + noise > 0, 1, 0)

    return X, y  # Return features and labels


# Function to create a bootstrap sample from the dataset
def bootstrap_sample(X, y, n_samples):
    """
    Create a bootstrap sample from the dataset.
    """
    indices = np.random.choice(
        len(X), size=n_samples, replace=True
    )  # Randomly sample indices
    return X[indices], y[indices]  # Return the sampled features and labels


# Main execution block
if __name__ == "__main__":
    # Set up argument parser for command line options
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
        default="results/result.csv",
        help="Name of the CSV file to save results",
    )
    args = parser.parse_args()  # Parse the command line arguments

    np.random.seed(args.seed)  # Set the random seed for reproducibility

    console = Console()  # Initialize console for rich output

    # Print starting message
    console.print(
        Panel.fit(
            "[bold green]Starting PPI Logistic Regression Experiment[/bold green]",
            border_style="green",
        )
    )

    # Create a table to display experiment parameters
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
    console.print(param_table)  # Display the parameter table

    lr_bias_list = []  # List to store biases for logistic regression
    ppi_bias_list = []  # List to store biases for PPI logistic regression

    # Generate the full dataset only once
    X_full, y_full = generate_data(
        args.n_samples * 2, args.n_dims
    )  # Generate extra data for holdout
    n_labeled = int(
        args.n_samples * args.labeled_unlabeled_ratio
    )  # Calculate number of labeled samples
    n_unlabeled = args.n_samples - n_labeled  # Calculate number of unlabeled samples

    # Progress bar for running experiments
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

        for i in range(args.n_exps):  # Loop over the number of experiments
            progress.update(
                main_task,
                advance=1,
                description=f"[green]Running experiment {i+1}/{args.n_exps}...",
            )

            beta_true = np.asarray(
                [0] * (args.n_dims // 2) + [1] * (args.n_dims - args.n_dims // 2)
            )  # Define the true coefficients for bias calculation

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

            # Optimize logistic regression on holdout data to get beta_hat
            beta_hat = optimize_logistic_regression(X_holdout, y_holdout, args.temp)

            # Optimize PPI logistic regression parameters
            beta_ppi = optimize_ppi_logistic_regression(
                X_train_labeled, y_train_labeled, X_train_unlabeled, args.temp
            )

            # Calculate biases for both methods
            lr_bias_list.append(beta_hat - beta_true)
            ppi_bias_list.append(beta_ppi - beta_true)

    # Print completion message
    console.print(
        Panel.fit(
            "[bold green]Experiments completed![/bold green]", border_style="green"
        )
    )

    lr_bias_array = np.array(lr_bias_list)  # Convert list to numpy array for analysis
    ppi_bias_array = np.array(ppi_bias_list)  # Convert list to numpy array for analysis

    console.print("[bold cyan]Generating plot...[/bold cyan]")
    # Combine data for boxplot
    data = np.concatenate([lr_bias_array, ppi_bias_array], axis=0)  # Combine biases
    labels = ["LR"] * lr_bias_array.shape[0] * args.n_dims + [
        "LR-PPI"
    ] * ppi_bias_array.shape[0] * args.n_dims  # Create labels for plotting
    dimensions = np.tile(
        np.arange(1, args.n_dims + 1), lr_bias_array.shape[0] + ppi_bias_array.shape[0]
    )  # Create dimension labels for each bias

    # Set up the plot with new color scheme
    plt.figure(figsize=(15, 8))
    sns.set_style("whitegrid")  # Set seaborn style for the plot
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
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
    plt.grid(axis="y", linestyle="--", alpha=0.7)  # Add grid lines

    # Customize legend
    plt.legend(title="Method", loc="upper right", frameon=True)

    # Show the plot
    plt.tight_layout()  # Adjust layout to prevent clipping

    if args.save_figure:  # Check if the figure should be saved
        plt.savefig(
            "figures/pseudo_label.png", dpi=300, bbox_inches="tight"
        )  # Save the figure
        console.print("[cyan]Plot saved as 'figures/pseudo_label.png'[/cyan]")
    else:
        plt.show()  # Display the plot

    # Create a table to display experiment results
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
    console.print(results_table)  # Display the results table

    # Calculate and report average variance across dimensions
    lr_avg_var = np.mean(np.var(lr_bias_array, axis=0))  # Average variance for LR
    ppi_avg_var = np.mean(np.var(ppi_bias_array, axis=0))  # Average variance for PPI

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

    # Check if the file exists to save results
    if os.path.exists(args.result_file):
        # If the file exists, append without writing the header
        result_df.to_csv(args.result_file, mode="a", header=False, index=False)
        console.print(f"[cyan]Results appended to '{args.result_file}'[/cyan]")
    else:
        # If the file doesn't exist, create it and write the header
        result_df.to_csv(args.result_file, index=False)
        console.print(f"[cyan]Results saved to new file '{args.result_file}'[/cyan]")

    # Print completion message
    console.print(
        Panel.fit(
            "[bold green]Experiment completed successfully![/bold green]",
            border_style="green",
        )
    )
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
        plt.savefig("figures/pseudo_label.png", dpi=300, bbox_inches="tight")
        console.print("[cyan]Plot saved as 'figures/pseudo_label.png'[/cyan]")
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
