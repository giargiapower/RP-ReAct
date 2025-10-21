import os
import json
from collections import defaultdict
import re

def get_model_sort_key(model_name):
    """
    Generates a sort key for a model name.
    It attempts to parse the model family and size (in billions of parameters)
    to group models by family and then sort by size.
    e.g., 'Qwen3-32B' -> ('Qwen3', 32.0)
    Models without a size in 'B' are sorted alphabetically by their name.
    """
    # Regex to find a number (int or float) followed by 'B' at the end of the string.
    match = re.search(r'(\d+(\.\d+)?B)$', model_name)
    if match:
        size_str = match.group(1)
        # The part before the size becomes the family name
        family = model_name[:-len(size_str)].strip('-')
        # Extract the numeric part of the size string
        size_val = float(re.search(r'\d+(\.\d+)?', size_str).group())
        # Sort by family, then by size. A low number ensures these are sorted first.
        return (0, family, size_val)
    
    # Fallback for models that don't match the pattern (e.g., 'gpt-4')
    # They will be sorted alphabetically after the others.
    return (1, model_name, 0)


def analyze_results():
    """
    Analyzes benchmark results, groups them by full benchmark name (e.g., 'yelp-easy'),
    and generates a summary table for each group, comparing different models.
    """
    results_dir = '/home/ubuntu/vdb1/Agent_design_architectures/ToolQA/benchmark/Paper_Results/React_results'
    output_dir = '/home/ubuntu/vdb1/Agent_design_architectures/ToolQA/benchmark/Paper_Results/table_results_react'

    os.makedirs(output_dir, exist_ok=True)

    # Structure: {full_benchmark_name: {model: {stats}}}
    grouped_data = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0, 'true_positives': 0, 'relevant_items': 0}))
    all_models = set()

    if not os.path.isdir(results_dir):
        print(f"Error: Directory '{results_dir}' not found.")
        return

    for filename in os.listdir(results_dir):
        if filename.endswith('.jsonl'):
            try:
                base_name = filename[:-6]  # Remove .jsonl
                parts = base_name.rsplit('-', 1)
                
                # Handle complex model names that may contain hyphens
                # We assume the last part is the model name
                model = parts[-1]
                full_benchmark_name = '-'.join(parts[:-1])

                if not full_benchmark_name:
                    print(f"Warning: Could not parse benchmark and model from '{filename}'. Skipping.")
                    continue

                all_models.add(model)
                
                file_path = os.path.join(results_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            stats = grouped_data[full_benchmark_name][model]
                            stats['total'] += 1
                            
                            is_correct = data.get('eval', '').strip() == 'CORRECT'
                            is_relevant = data.get('is_relevant', False)

                            if is_correct:
                                stats['correct'] += 1
                            
                            if is_relevant:
                                stats['relevant_items'] += 1
                                if is_correct:
                                    stats['true_positives'] += 1

                        except json.JSONDecodeError:
                            print(f"Warning: Could not decode JSON from a line in '{filename}'.")

            except Exception as e:
                print(f"An error occurred processing file {filename}: {e}")

    # Sort models using the custom sort key
    sorted_models = sorted(list(all_models), key=get_model_sort_key)
    
    # Generate a table for each full benchmark name
    for benchmark_name, models_data in grouped_data.items():
        table_path = os.path.join(output_dir, f"{benchmark_name}_results.txt")
        
        with open(table_path, 'w', encoding='utf-8') as f:
            f.write(f"Results for Benchmark: {benchmark_name}\n")
            f.write("=" * 80 + "\n")
            
            # Header
            header = f"{'Model':<35} | {'Accuracy':<20} | {'Recall':<20}"
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")

            # Data rows
            for model in sorted_models:
                if model in models_data:
                    stats = models_data[model]
                    total = stats['total']
                    correct = stats['correct']
                    true_positives = stats['true_positives']
                    relevant_items = stats['relevant_items']
                    
                    # Calculate Accuracy
                    if total > 0:
                        accuracy = (correct / total) * 100
                        acc_cell = f"{accuracy:>6.2f}% ({correct}/{total})"
                    else:
                        acc_cell = "N/A"
                    
                    # Calculate Recall
                    if relevant_items > 0:
                        recall = (true_positives / relevant_items) * 100
                        recall_cell = f"{recall:>6.2f}% ({true_positives}/{relevant_items})"
                    else:
                        recall_cell = "N/A"
                    
                    row = f"{model:<35} | {acc_cell:<20} | {recall_cell:<20}"
                    f.write(row + "\n")
        

        print(f"Generated results table for '{benchmark_name}' at '{table_path}'")

def merge_benchmark_results():
        """
        Merges benchmark result files from the 'table_results' directory.
        It groups files by a common benchmark prefix (e.g., 'yelp-hard')
        and combines their content into a single merged file.
        """
        table_dir = '/home/ubuntu/vdb1/Agent_design_architectures/ToolQA/benchmark/Paper_Results/table_results_reflexion'
        output_dir = '/home/ubuntu/vdb1/Agent_design_architectures/ToolQA/benchmark/Paper_Results/table_results_merged_reflexion'
        os.makedirs(output_dir, exist_ok=True)

        # {benchmark_prefix: [list_of_file_paths]}
        benchmarks_to_merge = defaultdict(list)

        if not os.path.isdir(table_dir):
            print(f"Error: Directory '{table_dir}' not found.")
            return

        for filename in os.listdir(table_dir):
            if filename.endswith('_results.txt'):
                # Extract benchmark prefix, e.g., 'yelp-hard' from 'yelp-hard-some-model_results.txt'
                match = re.match(r'([a-zA-Z0-9_-]+?)(-[A-Z][a-zA-Z0-9_]+.*)?_results\.txt', filename)
                if match:
                    # Heuristic to find the base benchmark name (e.g., yelp-easy, yelp-hard)
                    # This assumes model names start with a capital letter or are complex.
                    # A simpler split would be `filename.split('-')[0]` if benchmarks are simple names.
                    base_benchmark = filename.split('-')[0] + '-' + filename.split('-')[1] if filename.count('-') > 1 else filename.split('-')[0]
                    
                    # A more robust way might be to look for common prefixes.
                    # For this implementation, we'll use a simpler split logic.
                    # Example: 'yelp-hard-Qwen_results.txt' -> 'yelp-hard'
                    parts = filename.rsplit('-', 1)
                    if len(parts) > 1 and 'results' in parts[1]:
                         # This logic is a guess. A more robust way is needed if filenames are complex.
                         # Let's try to find a common prefix among all files.
                         # For now, we assume a simple structure like 'yelp-easy' or 'yelp-hard'.
                         prefix = '-'.join(filename.split('-')[:2]) # e.g., yelp-hard
                         benchmarks_to_merge[prefix].append(os.path.join(table_dir, filename))
                    else: # Fallback for simple names
                         benchmarks_to_merge[filename.replace('_results.txt', '')].append(os.path.join(table_dir, filename))


        # This logic is complex. Let's simplify based on the user's example.
        # 'yelp-hard-DeepSeek..._results.txt' and 'yelp-hard-Qwen3..._results.txt' -> 'yelp-hard'
        benchmarks_to_merge.clear() # Resetting the dictionary
        for filename in os.listdir(table_dir):
            if filename.endswith('_results.txt'):
                # Assuming benchmark names are like 'yelp-easy', 'yelp-hard', 'gsm8k-easy', etc.
                prefix = '-'.join(filename.split('-')[:2])
                benchmarks_to_merge[prefix].append(os.path.join(table_dir, filename))


        for prefix, files in benchmarks_to_merge.items():
            if len(files) <= 1:
                continue # No merging needed for single files

            merged_content = {}
            header = ""
            title = ""

            for file_path in files:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if not title:
                        title = f"Results for Benchmark: {prefix}\n"
                        header = lines[2:4] # Capture header and separator
                    
                    # Read data rows, skipping header
                    for line in lines[4:]:
                        line = line.strip()
                        if line:
                            # Use model name as key to avoid duplicates
                            model_name = line.split('|')[0].strip()
                            if model_name not in merged_content:
                                merged_content[model_name] = line
            
            # Sort models in the merged file
            sorted_models_data = sorted(merged_content.values(), key=lambda row: get_model_sort_key(row.split('|')[0].strip()))

            merged_file_path = os.path.join(output_dir, f"{prefix}_merged_results.txt")
            with open(merged_file_path, 'w', encoding='utf-8') as f:
                f.write(title)
                f.write("=" * 80 + "\n")
                f.writelines(header)
                for row in sorted_models_data:
                    f.write(row + "\n")
            
            print(f"Merged {len(files)} files into '{merged_file_path}'")

        # The above logic assumes a file-per-model structure which `analyze_results` does not create.
        # The user's request might be based on a misunderstanding.
        # If the goal is to combine multiple benchmark tables (e.g., yelp-easy, yelp-hard) into one,
        # the logic would be different.
        # Given the ambiguity, the function is provided but might need adjustment
        # based on the actual file structure in `table_results`.
        # If `analyze_results` works as intended, no files will be merged by the above logic,
        # as it produces one file per benchmark (e.g., 'yelp-hard_results.txt'), not per model.

if __name__ == '__main__':
    analyze_results()
    merge_benchmark_results()




