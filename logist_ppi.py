import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ppi_py
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# Generating synthetic data for logistic regression, training, validation and testing

np.random.seed(0)
n = 1000
p = 10
n_exp = 100

# Hold all residuals
residuals = np.zeros((n_exp, p))

# Check the coverage of the confidence intervals
coverage_count = np.zeros(p)

for i in tqdm(range(n_exp)):
    X = np.random.randn(n, p)
    beta = np.random.randn(p)
    y = np.random.binomial(1, 1 / (1 + np.exp(-X @ beta)))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.5)
    X_train_labeled, X_train_unlabeled, y_train_labeled, y_train_unlabeled = (
        train_test_split(X_train, y_train, test_size=0.5)
    )

    # Validation data is used for hyperparameter tuning
    # Labeled training data is used for training the model
    # Model is used to predict on the unlabeled training data

    # Training a logistic regression model on the labeled data
    model = LogisticRegression()
    model.fit(X_train_labeled, y_train_labeled)

    # Predicting on the unlabeled data
    y_hat_train_labeled = model.predict(X_train_labeled)
    Y_hat_train_unlabeled = model.predict(X_train_unlabeled)

    # Use ppi_logistic_pointestimate to estimate the parameters of the logistic regression model
    beta_hat = ppi_py.ppi_logistic_pointestimate(
        X_train_labeled,
        y_train_labeled,
        y_hat_train_labeled,
        X_train_unlabeled,
        Y_hat_train_unlabeled,
    )

    # Use ppi_logistic_ci to estimate the confidence intervals of the parameters of the logistic regression model
    ci = ppi_py.ppi_logistic_ci(
        X_train_labeled,
        y_train_labeled,
        y_hat_train_labeled,
        X_train_unlabeled,
        Y_hat_train_unlabeled,
    )

    # Residuals of beta
    residuals[i] = beta - beta_hat

    # Check if the true beta is within the confidence interval
    for j in range(p):
        if beta[j] >= ci[0][j] and beta[j] <= ci[1][j]:
            coverage_count[j] += 1

# Plotting the residuals
plt.figure(figsize=(10, 5))
# Violin plot of the residuals
sns.violinplot(data=residuals)
plt.xlabel("Features")
plt.ylabel("Residuals")
plt.title("Residuals of the estimated coefficients")
plt.savefig("residuals.png")

# Calculate the coverage of the confidence intervals
coverage = coverage_count / n_exp

# Plotting the coverage of the confidence intervals
plt.figure(figsize=(10, 5))
plt.bar(range(p), coverage)
# Add a horizontal line at 0.90
plt.axhline(0.90, color="red", linestyle="--")
plt.xlabel("Features")
plt.ylabel("Coverage")
plt.title("Coverage of the confidence intervals")
plt.savefig("coverage.png")
