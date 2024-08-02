import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import trange
from scipy.stats import norm
from loguru import logger

import os


class Config(object):
    def __init__(self):
        self.seed = 123
        self.n = 1000  # Number of labeled samples
        self.p = 10  # Number of features
        self.alpha = 0.05  # Significance level
        self.n_iter = 200
        self.savedir = "results"
        self.verbose = True

        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)


save_path = os.path.join(Config().savedir, "results.pdf")


def centralized_orth_mat(n, p):
    X = np.random.randn(n, p)
    X -= X.mean(axis=0)
    X_orth = np.linalg.qr(X)[0]
    assert np.allclose(X_orth.T @ X_orth, np.eye(p))
    assert np.allclose(X_orth.mean(axis=0), 0)
    return X_orth


def generate_data(n1: int, n2: int, m: int, p: int):
    # X_labeled_ols = np.random.randn(n1, p)
    # X_labeled_rest = np.random.randn(n2, p)
    # X_unlabeled = np.random.randn(m, p)
    # X_labeled_ols = np.random.uniform(-1, 1, (n1, p))
    # X_labeled_rest = np.random.uniform(-1, 1, (n2, p))
    # X_unlabeled = np.random.uniform(-1, 1, (m, p))
    X_labeled_ols = np.random.standard_cauchy((n1, p))
    X_labeled_rest = np.random.standard_cauchy((n2, p))
    X_unlabeled = np.random.standard_cauchy((m, p))
    beta = np.random.randn(p)
    assert p >= 3, "p must be greater than or equal to 3"
    # Make first feature irrelevant
    beta[0] = 0
    # The second and third features are correlated with target
    beta[1] = 0.05
    beta[2] = 0.1

    # Fix variance of noise as 1
    y_ols = X_labeled_ols @ beta + np.random.randn(n1)
    y_rest = X_labeled_rest @ beta + np.random.randn(n2)

    return X_labeled_ols, X_labeled_rest, X_unlabeled, y_ols, y_rest, beta


def generate_orthogonal_data(n, m, p):
    # X_orth = centralized_orth_mat(2 * n + m, p)
    # X_labeled_ols_orth = X_orth[:n]
    # X_labeled_rest_orth = X_orth[n : 2 * n]
    # X_unlabeled_orth = X_orth[2 * n :]
    X_labeled_ols_orth = centralized_orth_mat(n, p)
    X_labeled_rest_orth = centralized_orth_mat(n, p)
    X_unlabeled_orth = centralized_orth_mat(m, p)
    # verify orthogonality
    # assert np.allclose(X_labeled_ols_orth.T @ X_labeled_ols_orth, np.eye(p))
    beta = np.random.randn(p)
    assert p >= 3, "p must be greater than or equal to 3"
    # Make first feature irrelevant
    beta[0] = 0
    # The second and third features are correlated with target
    beta[1] = 0.05
    beta[2] = 0.1

    # Fix variance of noise as 1
    y_ols = X_labeled_ols_orth @ beta + np.random.randn(n)
    y_rest = X_labeled_rest_orth @ beta + np.random.randn(n)
    return (
        X_labeled_ols_orth,
        X_labeled_rest_orth,
        X_unlabeled_orth,
        y_ols,
        y_rest,
    )


def fit_ols(X, y):
    beta_ols = np.linalg.inv(X.T @ X) @ X.T @ y
    return beta_ols


def hypothesis_testing(X, y, beta_ols, alpha):
    # Since we fixed the variance as known
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)))
    z_stat = beta_ols / se
    p_values = 2 * (1 - norm.cdf(np.abs(z_stat)))
    return p_values


def run_simulation(config):
    np.random.seed(config.seed)
    n1 = config.n
    n2 = config.n
    p = config.p
    alpha = config.alpha
    r_list = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    m_var = [int(n2 * r) for r in r_list]
    # varying number of unlabeled samples

    results = pd.DataFrame()
    for m in m_var:
        print(f"Running simulation for m = {m}")
        type_I_error_feat_0 = np.zeros(config.n_iter)
        power_feat_1 = np.zeros(config.n_iter)
        power_feat_2 = np.zeros(config.n_iter)
        beta_diff = np.zeros(config.n_iter)
        for i in trange(config.n_iter):
            X_labeled_ols, X_labeled_rest, X_unlabeled, y_ols, y_rest, beta = (
                generate_data(n1, n2, m, p)
            )
            # X_labeled_ols, y_ols for fitting OLS model
            # X_labeled_rest, y_rest combined with X_unlabeled for hypothesis testing

            # Fit OLS model
            beta_ols = fit_ols(X_labeled_ols, y_ols)
            # Predict on unlabeled data
            y_unlabeled_pred = X_unlabeled @ beta_ols

            # Combine labeled and unlabeled data
            X = np.vstack([X_labeled_rest, X_unlabeled])
            y = np.hstack([y_rest, y_unlabeled_pred]).T

            # Fit new model on combined data and perform hypothesis testing
            beta_ols_combined = fit_ols(X, y)
            p_values = hypothesis_testing(X, y, beta_ols_combined, alpha)

            # Test whether the difference between beta_ols and beta is significant
            beta_diff[i] = beta_ols[0] - beta[0]

            # Check if null hypothesis is rejected for first 3 feature
            # First feature is irrelevant, second and third are correlated with target
            type_I_error_feat_0[i] = p_values[0] < alpha
            power_feat_1[i] = p_values[1] < alpha
            power_feat_2[i] = p_values[2] < alpha

        # Save average type I error and power
        type_I_error_feat_0 = np.mean(type_I_error_feat_0)
        power_feat_1 = np.mean(power_feat_1)
        power_feat_2 = np.mean(power_feat_2)
        logger.info(
            f"m = {m}, Type I error (feature 0) = {type_I_error_feat_0}, Power (feature 1) = {power_feat_1}, Power (feature 2) = {power_feat_2}"
        )
        results[m] = {
            "type_I_error_feat_0": type_I_error_feat_0,
            "power_feat_1": power_feat_1,
            "power_feat_2": power_feat_2,
        }
        # hist of beta_diff
        plt.hist(beta_diff, bins=20)
        plt.xlabel("||beta_ols - beta||")
        plt.ylabel("Frequency")
        plt.title(f"m = {m}")
        plt.savefig(f"results/beta_diff_{m}.pdf")
        plt.close()
        break

    results.to_csv(os.path.join(config.savedir, "results.csv"))
    return results


def run_simulation_orth(config):
    np.random.seed(config.seed)
    n = config.n
    p = config.p
    alpha = config.alpha
    m_var = [
        100,
        200,
        300,
        500,
        800,
        1000,
        2000,
        3000,
        4000,
        5000,
    ]  # varying number of unlabeled samples

    results = pd.DataFrame()
    for m in m_var:
        print(f"Running simulation for m = {m}")
        type_I_error_feat_0 = np.zeros(config.n_iter)
        power_feat_1 = np.zeros(config.n_iter)
        power_feat_2 = np.zeros(config.n_iter)
        for i in trange(config.n_iter):
            # X_labeled_ols, y_ols for fitting OLS model
            # X_labeled_rest, y_rest combined with X_unlabeled for hypothesis testing
            X_labeled_ols, X_labeled_rest, X_unlabeled, y_ols, y_rest = (
                generate_orthogonal_data(n, m, p)
            )
            # Fit OLS model
            beta_ols = fit_ols(X_labeled_ols, y_ols)

            # Predict on unlabeled data
            y_unlabeled_pred = X_unlabeled @ beta_ols

            # Combine labeled and unlabeled data
            X = np.vstack([X_labeled_rest, X_unlabeled])
            # TODO: combination of orthogonal matrices is not orthogonal, we need to rescale both X and y simultaneously in order to keep the orthogonality of design matrix X
            y = np.hstack([y_rest, y_unlabeled_pred]).T

            # Fit new model on combined data and perform hypothesis testing
            beta_ols_combined = fit_ols(X, y)
            p_values = hypothesis_testing(X, y, beta_ols_combined, alpha)

            # Check if null hypothesis is rejected for first 3 feature
            # First feature is irrelevant, second and third are correlated with target
            type_I_error_feat_0[i] = p_values[0] < alpha
            power_feat_1[i] = p_values[1] < alpha
            power_feat_2[i] = p_values[2] < alpha

        # Save average type I error and power
        type_I_error_feat_0 = np.mean(type_I_error_feat_0)
        power_feat_1 = np.mean(power_feat_1)
        power_feat_2 = np.mean(power_feat_2)
        logger.info(
            f"m = {m}, Type I error (feature 0) = {type_I_error_feat_0}, Power (feature 1) = {power_feat_1}, Power (feature 2) = {power_feat_2}"
        )
        results[m] = {
            "type_I_error_feat_0": type_I_error_feat_0,
            "power_feat_1": power_feat_1,
            "power_feat_2": power_feat_2,
        }

    results.to_csv(os.path.join(config.savedir, "results_orth.csv"))
    return results


def plot_results(result, save_path):
    """
    Plot type I error and power for different features.

    Args:
        result (pd.DataFrame): DataFrame containing the results.
        save_path (str): Path to save the plot.
    """
    type_I_error_feat_0 = result.loc["type_I_error_feat_0"]
    power_feat_1 = result.loc["power_feat_1"]
    power_feat_2 = result.loc["power_feat_2"]

    # Plot results
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    sns.color_palette("hls", 8)
    plt.plot(result.columns, type_I_error_feat_0, marker="o")
    plt.plot(result.columns, power_feat_1, marker="o")
    plt.plot(result.columns, power_feat_2, marker="o")
    plt.axhline(0.05, color="r", linestyle="--")
    plt.xlabel("Number of unlabeled samples, fixed n=1000")
    plt.ylabel("Probability")
    plt.title("Type I error and power")
    plt.legend(["Type I error (Feature 0)", "Power (Feature 1)", "Power (Feature 2)"])

    # Save the plot
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    result = run_simulation(Config())
    # Plot results
    plot_results(result, save_path)

    # result_orth = run_simulation_orth(Config())

    # save_path_orth = os.path.join(Config().savedir, "results_orth.pdf")

    # plot_results(result_orth, save_path_orth)
