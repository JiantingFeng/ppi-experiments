import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


def plot_variance(result_file):
    print("Plotting Variance Graph")

    # Read the CSV file
    df = pd.read_csv(result_file)

    # Create the plot
    plt.figure(figsize=(15, 8))
    sns.set_style("whitegrid")

    # Plot LR and PPI variances
    sns.lineplot(
        x="labeled_unlabeled_ratio",
        y="lr_variance",
        data=df,
        label="LR Variance",
        marker="o",
        color="lightblue",
    )

    sns.lineplot(
        x="labeled_unlabeled_ratio",
        y="ppi_variance",
        data=df,
        label="LR-PPI Variance",
        marker="o",
        color="gray",
    )
    # Set labels and title
    plt.title(
        "Variance of LR and PPI Methods vs Labeled/Unlabeled Ratio",
        fontsize=16,
        fontweight="bold",
    )
    plt.xlabel("Labeled/Unlabeled Ratio", fontsize=12)
    plt.ylabel("Variance", fontsize=12)
    plt.xticks(df["labeled_unlabeled_ratio"], rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Customize legend
    plt.legend(title="Method", loc="upper right", frameon=True)

    # Show the plot
    plt.tight_layout()

    # Save the plot
    plt.savefig("figures/variance_vs_ratio.png", dpi=300, bbox_inches="tight")
    print("Variance graph saved as 'figures/variance_vs_ratio.png'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot variance graph from results")
    parser.add_argument(
        "--result_file", type=str, required=True, help="Path to the CSV result file"
    )
    args = parser.parse_args()

    plot_variance(args.result_file)

    print("Variance graph created successfully!")
