import json
import pathlib

def analyze_jsonl_files(directory_path):
    """
    Reads all .jsonl files in a directory. For each file, it counts the total
    number of rows and the number of rows where the "eval" value does not
    contain "INCORRECT". It then prints these counts for each file.
    """
    path = pathlib.Path(directory_path)
    if not path.is_dir():
        print(f"Error: Directory not found at '{directory_path}'")
        return

    # Find all files ending with .jsonl in the specified directory
    jsonl_files = list(path.glob('*.jsonl'))

    if not jsonl_files:
        print(f"No .jsonl files found in '{directory_path}'")
        return
    mean_percentage = 0
    total_files = 0
    for file_path in jsonl_files:
        file_total_rows = 0
        file_correct_rows = 0
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                total_files += 1
                for line in f:
                    # Skip empty lines
                    if not line.strip():
                        continue
                    
                    file_total_rows += 1
                    try:
                        data = json.loads(line)
                        # Check if 'eval' key exists and its value is a string
                        eval_value = data.get("eval")
                        if isinstance(eval_value, str) and "INCORRECT" not in eval_value:
                            file_correct_rows += 1
                    except json.JSONDecodeError:
                        print(f"Warning: Could not decode JSON from a line in {file_path.name}")
                    except Exception as e:
                        print(f"An unexpected error occurred while processing a line in {file_path.name}: {e}")
            
            # Print the results for the current file
            print(f"Benchmark: {file_path.name}, Total Rows: {file_total_rows}, Correct Rows: {file_correct_rows}, Total Percentage: {(file_correct_rows/file_total_rows)*100}%, Total Percentage: {file_correct_rows}%")
            mean_percentage += (file_correct_rows)
        except IOError as e:
            print(f"Error reading file {file_path}: {e}")
    print(f"Mean Percentage across all files: {(mean_percentage/total_files)}%")


if __name__ == '__main__':
    # The directory containing the .jsonl files
    target_directory = "/home/ubuntu/vdb1/Agent_design_architectures/ToolQA/benchmark/Reasoning_Tool_Calling/code/React_results_rerun_high_steps"
    analyze_jsonl_files(target_directory)
