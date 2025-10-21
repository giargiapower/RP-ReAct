#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
#set -e

# Validate input arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 {flight|airbnb|coffee|scirex|yelp} <checkpoint>"
    exit 1
fi

DATASET=$1
CHECKPOINT=$2

# Validate that the checkpoint is an integer
if ! [[ "$CHECKPOINT" =~ ^[0-9]+$ ]]; then
    echo "Error: Checkpoint must be an integer." >&2
    exit 1
fi

# Validate the dataset name
case "$DATASET" in
    flight|airbnb|coffee|scirex|yelp)
        ;; # Valid dataset, continue
    *)
        echo "Error: Invalid dataset name '$DATASET'."
        echo "Please use one of: flight, airbnb, coffee, scirex, yelp"
        exit 1
        ;;
esac

# Change to the script's directory to ensure relative paths work correctly.
cd "$(dirname "$0")"

# Activate the Python virtual environment
if [ -f "LLM_Agent_Tester/bin/activate" ]; then
    source "LLM_Agent_Tester/bin/activate"
else
    echo "Error: Virtual environment not found."
    exit 1
fi

# Define common variables
BASE_PATH="/leonardo_scratch/fast/IscrC_EAGLE-DE/gianni/Agent_design_architectures/ToolQA"
MODEL="DeepSeek-R1-Distill-Qwen-7B"
TP_SIZE=2
OUTPUT_DIR="output_reasoner"


# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run tests for the specified dataset.
echo "Starting tests for dataset: $DATASET with checkpoint: $CHECKPOINT"
#python Test_Final_Multiple_Agent_Evaluation.py --path "$BASE_PATH" --dataset "$DATASET" --hardness easy --prompt easy --pattern Reasoner --local_model "$MODEL" --checkpoint 10 --tensor_parallel_size "$TP_SIZE" > "${OUTPUT_DIR}/${DATASET}_easy.out" 2> "${OUTPUT_DIR}/${DATASET}_easy.err"
python Test_Final_Multiple_Agent_Evaluation.py --path "$BASE_PATH" --dataset "$DATASET" --hardness easy --prompt easy --pattern Reasoner --local_model "$MODEL" --checkpoint "$CHECKPOINT" --tensor_parallel_size "$TP_SIZE" > "${OUTPUT_DIR}/${DATASET}_hard.out" 2> "${OUTPUT_DIR}/${DATASET}_hard.err"
#python Test_Final_Multiple_Agent_Evaluation.py --path "$BASE_PATH" --dataset "$DATASET" --hardness hard --prompt hard --pattern Reasoner --local_model "$MODEL" --checkpoint "$CHECKPOINT" --tensor_parallel_size "$TP_SIZE" > "${OUTPUT_DIR}/${DATASET}_hard.out" 2> "${OUTPUT_DIR}/${DATASET}_hard.err"


echo "Tests for $DATASET completed successfully."

# Deactivate the virtual environment (optional, but good practice)
deactivate