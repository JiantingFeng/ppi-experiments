#!/bin/bash

# Set the base parameters
N_SAMPLES=10000
N_DIMS=10
N_EXPS=1000
TEMP=1.0
SEED=42
RESULT_FILE="vary_ul_ratio_results.csv"

# Ensure the result file doesn't exist before starting
rm -f $RESULT_FILE

# Loop through different labeled_unlabeled_ratio values
for ratio in $(seq 0.1 0.1 0.9); do
    echo "Running experiment with labeled_unlabeled_ratio = $ratio"
    python ppi_logistic.py \
        --n_samples $N_SAMPLES \
        --n_dims $N_DIMS \
        --n_exps $N_EXPS \
        --labeled_unlabeled_ratio $ratio \
        --temp $TEMP \
        --seed $SEED \
        --result_file $RESULT_FILE
done

echo "All experiments completed. Results saved in $RESULT_FILE"

# Draw the variance graph
python plot_variance.py --result_file $RESULT_FILE

echo "Variance graph has been created."