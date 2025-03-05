#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Define model configurations:
# "model_type model_name epochs learning_rate PDE num_points update_rate config_file save_model_flag"
models=(
    "PINN 1D_PINNmodel 10000 0.01 PDE1 0 500 Config/params.yml 1"
    "PINN 1D_PINNmodel 10000 0.01 PDE1 0 500 Config/AWparams.yml 1"
    "PINN 1D_PINNmodel 10000 0.01 PDE1 5 500 Config/ACparams.yml 1"
    "PINN 1D_PINNmodel 10000 0.01 PDE1 5 500 Config/ACAWparams.yml 1"
)

# Create a results directory if it doesn't exist
mkdir -p benchmark_results

echo "Starting Benchmarking Process..."

# Loop over each model configuration
for model in "${models[@]}"; do
    # Split the model string into individual parameters
    set -- $model
    model_type=$1
    model_name=$2
    epochs=$3
    lr=$4
    pde=$5
    num_points=$6
    update_rate=$7
    config_file=$8
    save_model=$9

    # Create a log file for each model run
    log_file="benchmark_results/${model_name}_$(basename ${config_file})_log.txt"

    echo "---------------------------------------------"
    echo "Running benchmark for: $model_name with config: $config_file"
    echo "Parameters:"
    echo "   Model Type: $model_type"
    echo "   Epochs: $epochs"
    echo "   Learning Rate: $lr"
    echo "   PDE: $pde"
    echo "   Num Points: $num_points"
    echo "   Update Rate: $update_rate"
    echo "   Save Model Flag: $save_model"
    echo "---------------------------------------------"

    # Start timing the execution
    start_time=$(date +%s)

    # Construct the command
    cmd="python main.py --model_type \"$model_type\" --model_name \"$model_name\" --epochs \"$epochs\" --lr \"$lr\" --PDE \"$pde\" --AC \"$num_points\" --update_rate \"$update_rate\" --config \"$config_file\" --save_model \"$save_model\""
    echo "Executing: $cmd"
    
    # Run the command and redirect all output to the log file
    eval $cmd > "$log_file" 2>&1

    # Check if the log file contains indications of error (optional)
    if grep -qi "error" "$log_file"; then
        echo "Warning: Errors detected during training. Check $log_file for details."
    fi

    # Calculate execution time
    end_time=$(date +%s)
    runtime=$((end_time - start_time))
    echo "Finished benchmarking for $model_name with config: $config_file in $runtime seconds"
    echo "Results saved in $log_file"
    echo "---------------------------------------------"
done

echo "✅ Benchmarking Completed! All results are in the 'benchmark_results' folder."
