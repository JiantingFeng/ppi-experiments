import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm, trange

import os

# parameters
n = 1000
p = 10
sigma = 1
alpha = 0.05
n_iter = 1000

# set random seed
np.random.seed(0)
os.makedirs("results", exist_ok=True)


# generate data
def generate_data(n1, n2, n3, p, sigma):
    X_1 = np.random.normal(size=(n1, p))
    X_2 = np.random.normal(size=(n2, p))
    X_3 = np.random.normal(size=(n3, p))

    assert p >= 3, "p must be greater than or equal to 3"

    beta = np.zeros(p)
    beta[:3] = [0, 0.01, 0.05]

    y_1 = X_1 @ beta + np.random.normal(scale=sigma, size=n1)
    y_2 = X_2 @ beta + np.random.normal(scale=sigma, size=n2)
    y_3 = X_3 @ beta + np.random.normal(scale=sigma, size=n3)

    return X_1, y_1, X_2, y_2, X_3, y_3


def plot_result(results, x_label, y_label, title, log_scale=False):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    results.T.plot(ax=ax, marker="o")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if log_scale:
        ax.set_xscale("log")
    # plot alpha=0.05
    ax.axhline(y=alpha, color="r", linestyle="--")
    plt.savefig(f"results/{title}.pdf", format="pdf")


# run experiment


def run_experiment(n1, n2, n3, p=10, sigma=1, alpha=0.05, n_iter=1000):
    type_I_errors = np.zeros(n_iter)
    power_1, power_2 = np.zeros(n_iter), np.zeros(n_iter)
    for i in trange(n_iter):
        X_1, y_1, X_2, _, X_3, y_3 = generate_data(n1, n2, n3, p, sigma)
        # Y_2 is treated as unlabeled data
        # Fit model on first dataset
        model = sm.OLS(y_1, X_1)
        results = model.fit()

        # Predict on second dataset
        y_2_hat = X_2 @ results.params

        # combine D2 and D3
        X = np.vstack([X_2, X_3])
        y = np.concatenate([y_2_hat, y_3])

        # Fit model on combined dataset
        model_comb = sm.OLS(y, X)
        results_comb = model_comb.fit()

        # Type I error
        pvalues = results_comb.pvalues
        type_I_errors[i] = pvalues[0] < alpha

        # Power
        power_1[i] = results_comb.pvalues[1] < alpha
        power_2[i] = results_comb.pvalues[2] < alpha

    return type_I_errors, power_1, power_2


if __name__ == "__main__":
    print("Running experiment")
    # Varing n1 and n3, fixing n2
    n_list = np.array([1000, 5000, 10000, 25000, 50000, 100000])

    results_n = pd.DataFrame()

    print("Running experiment for varying n")

    for n_var in tqdm(n_list):
        type_I_errors, power_1, power_2 = run_experiment(
            n_var, n, n_var, p, sigma, alpha, n_iter
        )
        results_n[n_var] = {
            "Type I error": type_I_errors.mean(),
            "Power 1": power_1.mean(),
            "Power 2": power_2.mean(),
        }

    # # varying p
    # p_list = np.array([10, 20, 50, 100, 200])

    # results_p = pd.DataFrame()

    # print("Running experiment for varying p")

    # for p_var in tqdm(p_list):
    #     type_I_errors, power_1, power_2 = run_experiment(
    #         10000, n, 10000, p_var, sigma, alpha, n_iter
    #     )
    #     results_p[p_var] = {
    #         "Type I error": type_I_errors.mean(),
    #         "Power 1": power_1.mean(),
    #         "Power 2": power_2.mean(),
    #     }

    results_n.to_csv("results/results_n.csv")
    # results_p.to_csv("results/results_p.csv")

    plot_result(
        results_n,
        "n",
        "Type I error / Power",
        "Type I error and Power vs n",
        log_scale=True,
    )
    # plot_result(
    #     results_p,
    #     "p",
    #     "Type I error / Power",
    #     "Type I error and Power vs p",
    #     log_scale=False,
    # )

    print("Experiment completed")
