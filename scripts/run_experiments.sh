#!/bin/bash

# Set the base parameters
N_SAMPLES=10000
N_DIMS=10
N_EXPS=1000
TEMP=1.0
SEED=2024
RESULT_FILE="data/results/vary_ul_ratio_results.csv"

# Ensure the directory for the result file exists
mkdir -p $(dirname "$RESULT_FILE")

# If the result file doesn't exist, create it with a header
if [ ! -f "$RESULT_FILE" ]; then
    echo "Creating new result file: $RESULT_FILE"
    echo "labeled_unlabeled_ratio,lr_variance,ppi_variance" > "$RESULT_FILE"
else
    echo "Appending to existing result file: $RESULT_FILE"
    # Optionally, remove the existing file if you want to start fresh each time
    # rm -f "$RESULT_FILE"
fi

# Record start time
start_time=$(date +%s)

# Loop through different labeled_unlabeled_ratio values
for ratio in $(seq 0.1 0.1 0.9); do
    echo "Running experiment with labeled_unlabeled_ratio = $ratio"
    experiment_start_time=$(date +%s)
    
    python src/models/ppi_logistic.py \
        --n_samples $N_SAMPLES \
        --n_dims $N_DIMS \
        --n_exps $N_EXPS \
        --labeled_unlabeled_ratio $ratio \
        --temp $TEMP \
        --seed $SEED \
        --result_file $RESULT_FILE
    
    experiment_end_time=$(date +%s)
    experiment_duration=$((experiment_end_time - experiment_start_time))
    echo "Experiment completed in $experiment_duration seconds"
done

echo "All experiments completed. Results saved in $RESULT_FILE"

# Draw the variance graph
python src/visualization/plot_variance.py --result_file $RESULT_FILE

echo "Variance graph has been created."

# Calculate and print total running time
end_time=$(date +%s)
total_duration=$((end_time - start_time))
echo "Total running time: $total_duration seconds"