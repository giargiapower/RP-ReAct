#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
#set -e

# Change to the script's directory to ensure relative paths work correctly.
# The cd command in the prompt is the same as the file's location, so we can use this.
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
MODEL="Qwen3-32B"
TP_SIZE=1
OUTPUT_DIR="output_react"
PATTERN="React" # Reflexion 

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run tests sequentially. The shell waits for each command to finish before starting the next.
echo "Starting tests for dataset: flight"
python test_offline.py --path "$BASE_PATH" --dataset flight --hardness easy --prompt easy --pattern "$PATTERN" --local_model "$MODEL" --tensor_parallel_size "$TP_SIZE" > "${OUTPUT_DIR}/${DATASET}_hard.out" 2> "${OUTPUT_DIR}/${DATASET}_hard.err"
python test_offline.py --path "$BASE_PATH" --dataset flight --hardness hard --prompt hard --pattern "$PATTERN" --local_model "$MODEL" --tensor_parallel_size "$TP_SIZE" > "${OUTPUT_DIR}/flight_hard.out" 2> "${OUTPUT_DIR}/flight_hard.err"

echo "Starting tests for dataset: airbnb"
python test_offline.py --path "$BASE_PATH" --dataset airbnb --hardness easy --prompt easy --pattern "$PATTERN" --local_model "$MODEL" --tensor_parallel_size "$TP_SIZE" > "${OUTPUT_DIR}/${DATASET}_hard.out" 2> "${OUTPUT_DIR}/${DATASET}_hard.err"
python test_offline.py --path "$BASE_PATH" --dataset airbnb --hardness hard --prompt hard --pattern "$PATTERN" --local_model "$MODEL" --tensor_parallel_size "$TP_SIZE" > "${OUTPUT_DIR}/airbnb_hard.out" 2> "${OUTPUT_DIR}/airbnb_hard.err"

echo "Starting tests for dataset: coffee"
python test_offline.py --path "$BASE_PATH" --dataset coffee --hardness easy --prompt easy --pattern "$PATTERN" --local_model "$MODEL" --tensor_parallel_size "$TP_SIZE" > "${OUTPUT_DIR}/${DATASET}_hard.out" 2> "${OUTPUT_DIR}/${DATASET}_hard.err"
python test_offline.py --path "$BASE_PATH" --dataset coffee --hardness hard --prompt hard --pattern "$PATTERN" --local_model "$MODEL" --tensor_parallel_size "$TP_SIZE" > "${OUTPUT_DIR}/coffee_hard.out" 2> "${OUTPUT_DIR}/coffee_hard.err"

echo "Starting tests for dataset: scirex"
python test_offline.py --path "$BASE_PATH" --dataset scirex --hardness easy --prompt easy --pattern "$PATTERN" --local_model "$MODEL" --tensor_parallel_size "$TP_SIZE" > "${OUTPUT_DIR}/${DATASET}_hard.out" 2> "${OUTPUT_DIR}/${DATASET}_hard.err"
python test_offline.py --path "$BASE_PATH" --dataset scirex --hardness hard --prompt hard --pattern "$PATTERN" --local_model "$MODEL" --tensor_parallel_size "$TP_SIZE" > "${OUTPUT_DIR}/scirex_hard.out" 2> "${OUTPUT_DIR}/scirex_hard.err"

echo "Starting tests for dataset: yelp"
python test_offline.py --path "$BASE_PATH" --dataset yelp --hardness easy --prompt easy --pattern "$PATTERN" --local_model "$MODEL" --tensor_parallel_size "$TP_SIZE" > "${OUTPUT_DIR}/${DATASET}_hard.out" 2> "${OUTPUT_DIR}/${DATASET}_hard.err"
python test_offline.py --path "$BASE_PATH" --dataset yelp --hardness hard --prompt hard --pattern "$PATTERN" --local_model "$MODEL" --tensor_parallel_size "$TP_SIZE" > "${OUTPUT_DIR}/yelp_hard.out" 2> "${OUTPUT_DIR}/yelp_hard.err"

echo "All tests completed successfully."

# Deactivate the virtual environment (optional, but good practice)
deactivate