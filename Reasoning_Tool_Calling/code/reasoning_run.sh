#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
#set -e

# Validate input arguments
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <checkpoint> <base_path> <model>"
    exit 1
fi

CHECKPOINT=$1
BASE_PATH=$2
MODEL=$3

# Validate that the checkpoint is an integer
if ! [[ "$CHECKPOINT" =~ ^[0-9]+$ ]]; then
    echo "Error: Checkpoint must be an integer." >&2
    exit 1
fi

# Change to the script's directory to ensure relative paths work correctly.
cd "$(dirname "$0")"

# Activate the Python virtual environment
if [ -f "LLM_Agent_Tester/bin/activate" ]; then
    source "LLM_Agent_Tester/bin/activate"
else
    echo "Error: Virtual environment not found."
    exit 1
fi

TP_SIZE=1
OUTPUT_DIR="output_reasoner"
DATASETS=("airbnb" "yelp" "scirex" "flight" "coffee")

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Loop through all datasets
for DATASET in "${DATASETS[@]}"; do
    echo "Starting tests for dataset: $DATASET with checkpoint: $CHECKPOINT"

    # Run test for easy hardness
    echo "Running easy test for $DATASET..."
    python Test_Final_Multiple_Agent_Evaluation.py \
        --path "$BASE_PATH" \
        --dataset "$DATASET" \
        --hardness easy \
        --prompt easy \
        --pattern Reasoner \
        --local_model "$MODEL" \
        --checkpoint "$CHECKPOINT" \
        --tensor_parallel_size "$TP_SIZE" > "${OUTPUT_DIR}/${DATASET}_easy.out" 2> "${OUTPUT_DIR}/${DATASET}_easy.err"

    # Run test for hard hardness
    echo "Running hard test for $DATASET..."
    python Test_Final_Multiple_Agent_Evaluation.py \
        --path "$BASE_PATH" \
        --dataset "$DATASET" \
        --hardness hard \
        --prompt hard \
        --pattern Reasoner \
        --local_model "$MODEL" \
        --checkpoint "$CHECKPOINT" \
        --tensor_parallel_size "$TP_SIZE" > "${OUTPUT_DIR}/${DATASET}_hard.out" 2> "${OUTPUT_DIR}/${DATASET}_hard.err"

    echo "Tests for $DATASET completed."
done

echo "All tests completed successfully."

# Deactivate the virtual environment (optional, but good practice)
deactivate