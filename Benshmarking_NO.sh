#!/bin/bash

# Exit on error
set -e

# Define models to test
models=("CNN" "FNO" "TFNO" "UNO")

# Common parameters
EPOCHS=10
LR=0.01
SAVE_VERSION=1
MODEL_TYPE="NO"

# Run each model
for model in "${models[@]}"; do
    echo "Training $model..."
    python main.py \
        --epochs $EPOCHS \
        --model_type $MODEL_TYPE \
        --model_name $model \
        --lr $LR \
        --save_version $SAVE_VERSION \
        || echo "Error training $model"
    
    echo "Completed $model"
    echo "------------------------"
done

echo "All models completed"