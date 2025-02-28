#!/bin/bash

# Define model configurations (modify as needed)
models=(
    "PINN 1D_PINNmodel 10 0.001 PDE1"
    "PINN 1D_PINNmodel 10 0.0005 PDE1"
    "PINN 1D_PINNmodel 10 0.0003 PDE1"
    "PINN 1D_PINNmodel 251000 0.0001 PDE1"
)

# Loop through each model configuration
for model in "${models[@]}"; do
    # Read values into variables
    set -- $model
    model_type=$1
    model_name=$2
    epochs=$3
    lr=$4
    pde=$5

    # Print execution details
    echo "Running model: $model_name"
    echo "Type: $model_type | Epochs: $epochs | Learning Rate: $lr | PDE: $pde"

    # Run the Python script
    python main.py --model_type "$model_type" --model_name "$model_name" --epochs "$epochs" --lr "$lr" --PDE "$pde"

    echo "-----------------------------------"
done

echo "All models have been executed."
