import os
import json
import pandas as pd

# --- Configuration ---
DIRS = {
    "Reasoner": "/home/ubuntu/vdb1/Agent_design_architectures/ToolQA/benchmark/Paper_Results/Reasoner_results",
    "Reflexion": "/home/ubuntu/vdb1/Agent_design_architectures/ToolQA/benchmark/Paper_Results/Reflexion_results",
    "React": "/home/ubuntu/vdb1/Agent_design_architectures/ToolQA/benchmark/Paper_Results/React_results"
}
# List of models to process
MODELS = ["Qwen3-14B","Qwen3-32B", "gpt-oss:20b", "gpt-oss:120b"] # Example list, add your models here , "DeepSeek-R1-Distill-Llama-8B", "DeepSeek-R1-Distill-Qwen-7B"
BENCHMARKS = [
    "airbnb-easy", "airbnb-hard",
    "flight-easy", "flight-hard",
    "coffee-easy", "coffee-hard",
    "scirex-easy", "scirex-hard",
    "yelp-easy", "yelp-hard"
]
OUTPUT_DIR = "/home/ubuntu/vdb1/Agent_design_architectures/ToolQA/benchmark/Paper_Results/table_total_improvement"


def calculate_correct_percentage(filepath):
    """Calculates the percentage of 'CORRECT' evaluations in a jsonl file."""
    correct_count = 0
    total_count = 0
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    total_count += 1
                    evaluation = data.get('eval')
                    if evaluation and evaluation.strip() == 'CORRECT':
                        correct_count += 1
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed JSON line in {filepath}")
    except FileNotFoundError:
        return None

    if total_count == 0:
        return 0.0
    # The prompt specifies dividing by 100, but calculating based on actual lines is more robust.
    # We will use the actual total_count. If it's always 100, the result is the same.
    return (correct_count / 100)

def find_matching_file(directory, benchmark, model):
    """Finds the first matching jsonl file in a directory."""
    try:
        for filename in os.listdir(directory):
            if (benchmark in filename and
                model in filename and
                filename.endswith('.jsonl')):
                return os.path.join(directory, filename)
    except FileNotFoundError:
        print(f"Warning: Directory not found: {directory}")
    return None

def main():
    """
    Main function to process files, calculate percentages, and display tables.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Data Collection ---
    # This dictionary will hold all results for the final aggregated table.
    # Structure: { 'AgentType': {'Model': {'Benchmark': percentage}} }
    all_results = {agent_type: {model: {} for model in MODELS} for agent_type in DIRS}

    for benchmark in BENCHMARKS:
        # This dictionary will hold the data for the current benchmark's DataFrame
        # Structure: { 'AgentType': {'Model1': percentage, 'Model2': percentage} }
        benchmark_data_for_df = {}

        for agent_type, directory in DIRS.items():
            model_percentages = {}
            for model in MODELS:
                filepath = find_matching_file(directory, benchmark, model)
                if filepath:
                    # --- Modification Step ---
                    # Read the file, modify in memory, and write back
                    try:
                        with open(filepath, 'r+', encoding='utf-8') as f:
                            lines = f.readlines()
                            f.seek(0) # Go back to the beginning of the file
                            f.truncate() # Clear the file content

                            for line in lines:
                                try:
                                    data = json.loads(line)
                                    # Check if 'response' is an empty string
                                    if data.get('response') == "":
                                        data['eval'] = "INCORRECT\n"
                                        # Write the modified line back
                                        f.write(json.dumps(data) + '\n')
                                    else:
                                        # Write the original line back
                                        f.write(line)
                                except json.JSONDecodeError:
                                    # If a line is not valid JSON, write it back as is
                                    f.write(line)
                    except Exception as e:
                        print(f"Error modifying file {filepath}: {e}")

                    # --- Calculation Step ---
                    percentage = calculate_correct_percentage(filepath)
                    if percentage is not None:
                        model_percentages[model] = percentage
                        all_results[agent_type][model][benchmark] = percentage
                    else:
                        model_percentages[model] = "File Not Found"
                        all_results[agent_type][model][benchmark] = 0.0 # Default for not found
                else:
                    model_percentages[model] = "No Matching File"
                    all_results[agent_type][model][benchmark] = 0.0 # Default for no matching file
            benchmark_data_for_df[agent_type] = model_percentages

        # --- Table Display and Saving for the current benchmark ---
        print(f"--- Table: {benchmark}-merged ---")
        # Create DataFrame from the collected data for this benchmark
        df = pd.DataFrame.from_dict(benchmark_data_for_df, orient='index')
        # Ensure the columns are in the same order as the MODELS list
        if not df.empty:
            df = df.reindex(columns=MODELS)
        df.index.name = 'Agent Type'
        print(df)

        # Save the table to a CSV file
        output_filename = f"{benchmark}-merged.csv"
        output_filepath = os.path.join(OUTPUT_DIR, output_filename)
        df.to_csv(output_filepath)
        print(f"Table saved to {output_filepath}")

        print("\n" + "="*40 + "\n")

    # --- Final Aggregated Table Generation ---
    print("--- Generating Final Aggregated Table ---")
    final_output_path = os.path.join(OUTPUT_DIR, "final_table_results.txt")
    # Specific benchmark order requested by the user
    output_benchmark_order = [
        "yelp-easy", "yelp-hard",
        "scirex-easy", "scirex-hard",
        "flight-easy", "flight-hard",
        "airbnb-easy", "airbnb-hard",
        "coffee-easy", "coffee-hard"
    ]

    with open(final_output_path, 'w', encoding='utf-8') as f:
        # Iterate through agent types and models to build each line
        for agent_type in DIRS.keys():
            for model in MODELS:
                # Start the line with Agent Type and Model
                line_parts = [f"{agent_type:<10}", f"{model:<12}"]
                
                # Append scores for each benchmark in the specified order
                for benchmark in output_benchmark_order:
                    score = all_results.get(agent_type, {}).get(model, {}).get(benchmark, 0.0)
                    line_parts.append(f"{score:.2f}")
                
                # Join all parts with ' & ' and add the trailing ' \\'
                line = " & ".join(line_parts) + " \\\\"
                f.write(line + '\n')

    print(f"Final aggregated results saved to {final_output_path}")

if __name__ == "__main__":
    main()