#!/bin/bash

# Script to run all model-dataset combinations and store results in out/ folder

# Create output directory if it doesn't exist
mkdir -p out

# Define datasets and models
DATASETS=("DISASTER_TWEET_JAILBREAKING" "AEGIS" "TRUST_AI_RLAB_JAILBREAK")
MODELS=("ARCH_GUARD" "LLAMA_GUARD" "SAMSUNG_JAILBREAK_FILTER" "SHIELD_GEMMA" "CHAIN")

# Function to convert enum name to filename (e.g., ARCH_GUARD -> arch_guard)
to_filename() {
    echo "$1" | tr '[:upper:]' '[:lower:]'
}

# Function to get dataset filename
get_dataset_filename() {
    case "$1" in
        DISASTER_TWEET_JAILBREAKING)
            echo "disaster_jailbreak"
            ;;
        *)
            to_filename "$1"
            ;;
    esac
}

# Function to get model filename (just lowercase the model name)
get_model_filename() {
    to_filename "$1"
}

# Counter for tracking progress
total_combinations=$((${#DATASETS[@]} * ${#MODELS[@]}))
current=0

echo "Running $total_combinations experiments..."
echo "=========================================="
echo ""

# Loop through all combinations
for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        current=$((current + 1))

        # Generate output filename
        dataset_file=$(get_dataset_filename "$dataset")
        model_file=$(get_model_filename "$model")
        output_file="out/${model_file}-${dataset_file}.out"

        echo "[$current/$total_combinations] Running: $model on $dataset"
        echo "Output: $output_file"

        # Run the experiment and append to output file
        uv run -m src.jailbreak --dataset "$dataset" --model "$model" >> "$output_file" 2>&1

        if [ $? -eq 0 ]; then
            echo "✓ Completed: $model on $dataset"
        else
            echo "✗ Failed: $model on $dataset (check $output_file for details)"
        fi

        echo ""
    done
done

echo "=========================================="
echo "All experiments completed!"
echo "Results stored in: out/"
