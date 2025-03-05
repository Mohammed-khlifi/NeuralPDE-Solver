#!/bin/bash

# Define model configurations: "model_type model_name epochs learning_rate PDE config_file"
models=(
    "PINN 1D_PINNmodel 10000 0.01 PDE1 params.yml"
    "PINN 1D_PINNmodel 10000 0.01 PDE1 AWparams.yml"
    "PINN 1D_PINNmodel 10000 0.01 PDE1 ACparams.yml"
    "PINN 1D_PINNmodel 10000 0.01 PDE1 ACAWparams.yml"
)

# Create a results directory if it doesn't exist
mkdir -p benchmark_results

# Start benchmarking
echo "Starting Benchmarking Process..."

for model in "${models[@]}"; do
    # Read values into variables
    set -- $model
    model_type=$1
    model_name=$2
    epochs=$3
    lr=$4
    pde=$5
    config_file=$6

    # Create log file for each model
    log_file="benchmark_results/${model_name}_${config_file}_log.txt"

    echo "---------------------------------------------"
    echo "Running benchmark for: $model_name with $config_file"
    echo "Model Type: $model_type | Epochs: $epochs | LR: $lr | PDE: $pde | Config: $config_file"
    echo "---------------------------------------------"

    # Start timing the execution
    start_time=$(date +%s)

    # Run the Python script with the parameters
    python main.py --model_type "$model_type" --model_name "$model_name" --epochs "$epochs" --lr "$lr" --PDE "$pde" --config "$config_file" > "$log_file" 2>&1

    # Calculate execution time
    end_time=$(date +%s)
    runtime=$((end_time - start_time))

    echo "Finished benchmarking for $model_name with $config_file in $runtime seconds"
    echo "Results saved in $log_file"
    echo "---------------------------------------------"
done

echo "✅ Benchmarking Completed! All results are in the 'benchmark_results' folder."
