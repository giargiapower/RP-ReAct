import json
import os
from collections import defaultdict

def find_qids_for_analysis(reasoner_dir, react_dir):
    """
    Analyzes jsonl files present in two directories to find specific question IDs.

    This function identifies common .jsonl files between the reasoner_dir and
    react_dir. For each pair of common files, it extracts 'qid's that meet
    the following criteria:
    1. The 'response' field for that 'qid' in the React file is an empty string.
    2. The 'eval' field for that same 'qid' in the Reasoner file is "CORRECT"
       (stripping any whitespace).

    Args:
        reasoner_dir (str): Path to the directory with Reasoner results.
        react_dir (str): Path to the directory with React results.

    Returns:
        dict: A dictionary where keys are filenames and values are lists of
              qualifying 'qid's.
    """
    try:
        reasoner_files = {f for f in os.listdir(reasoner_dir) if f.endswith('.jsonl')}
        react_files = {f for f in os.listdir(react_dir) if f.endswith('.jsonl')}
    except FileNotFoundError as e:
        print(f"Error: Directory not found - {e}")
        return {}

    common_files = sorted(list(reasoner_files.intersection(react_files)))
    results = defaultdict(list)

    for filename in common_files:
        print(f"Processing file: {filename}")
        react_filepath = os.path.join(react_dir, filename)
        reasoner_filepath = os.path.join(reasoner_dir, filename)

        # Step 1: Find qids from React file with empty responses
        empty_response_qids = set()
        try:
            with open(react_filepath, 'r', encoding='utf-8') as f_react:
                for line in f_react:
                    try:
                        data = json.loads(line)
                        if data.get("response") == "":
                            empty_response_qids.add(data.get("qid"))
                    except (json.JSONDecodeError, AttributeError):
                        continue # Skip malformed lines or lines without expected keys
        except IOError as e:
            print(f"Could not read file {react_filepath}: {e}")
            continue

        if not empty_response_qids:
            continue

        # Step 2: Check Reasoner file for corresponding qids with "CORRECT" eval
        try:
            with open(reasoner_filepath, 'r', encoding='utf-8') as f_reasoner:
                for line in f_reasoner:
                    try:
                        data = json.loads(line)
                        qid = data.get("qid")
                        # Check if this qid is one we're interested in
                        if qid in empty_response_qids:
                            evaluation = data.get("eval") # Get the 'eval' value, or None if it's missing
                            # Keep the qid only if 'eval' exists and is 'CORRECT' after stripping whitespace
                            if evaluation and evaluation.strip() == "CORRECT":
                                results[filename].append(qid)
                    except (json.JSONDecodeError, AttributeError):
                        continue # Skip malformed lines
        except IOError as e:
            print(f"Could not read file {reasoner_filepath}: {e}")
            continue

    return dict(results)

if __name__ == '__main__':
    REASONER_DIR = "/home/ubuntu/vdb1/Agent_design_architectures/ToolQA/benchmark/Paper_Results/Reasoner_results"
    REACT_DIR = "/home/ubuntu/vdb1/Agent_design_architectures/ToolQA/benchmark/Paper_Results/React_results"
    QUESTIONS_BASE_DIR = "/home/ubuntu/vdb1/Agent_design_architectures/ToolQA/data/questions"

    analysis_results = find_qids_for_analysis(REASONER_DIR, REACT_DIR)

    if analysis_results:
        print("\n--- Analysis Complete ---")
        output_dir = "max_step_list_rerun"
        os.makedirs(output_dir, exist_ok=True)
        print(f"Results will be saved in the '{output_dir}' directory.")

        # Pre-load question/answer data to avoid repeated file reads
        benchmark_data_cache = {}

        # Group results by model first
        model_data = defaultdict(list)
        for filename, qids in analysis_results.items():
            if not qids:
                continue

            try:
                parts = filename.split('-', 2)
                if len(parts) < 3:
                    print(f"\nWarning: Could not parse benchmark/model from '{filename}'. Skipping.")
                    continue

                benchmark = f"{parts[0]}-{parts[1]}"
                model = parts[2].replace('.jsonl', '')

                # Load the corresponding question/answer file if not already cached
                if benchmark not in benchmark_data_cache:
                    difficulty = "hard" if "hard" in benchmark else "easy"
                    qa_filepath = os.path.join(QUESTIONS_BASE_DIR, difficulty, f"{benchmark}.jsonl")
                    
                    try:
                        with open(qa_filepath, 'r', encoding='utf-8') as f_qa:
                            # Create a dictionary mapping qid to its data for quick lookup
                            benchmark_data_cache[benchmark] = {json.loads(line)['qid']: json.loads(line) for line in f_qa}
                    except (FileNotFoundError, IOError) as e:
                        print(f"\nWarning: Could not load question data file {qa_filepath}: {e}. Skipping benchmark '{benchmark}'.")
                        benchmark_data_cache[benchmark] = None # Mark as failed to avoid retries
                        continue
                
                # Skip if the data file for this benchmark could not be loaded
                if benchmark_data_cache[benchmark] is None:
                    continue

                for qid in qids:
                    qa_info = benchmark_data_cache[benchmark].get(qid)
                    if not qa_info:
                        print(f"\nWarning: Could not find qid '{qid}' in {benchmark}.jsonl. Skipping record.")
                        continue

                    record = {
                        "benchmark": benchmark,
                        "model": model,
                        "qid": qid,
                        "question": qa_info.get("question"),
                        "answer": qa_info.get("answer")
                    }
                    model_data[model].append(record)

            except (IndexError, json.JSONDecodeError) as e:
                print(f"\nWarning: Could not parse or process '{filename}'. Error: {e}. Skipping.")
                continue

        # Write grouped data to separate files
        if not model_data:
             print("\nNo valid records to write after parsing filenames.")
        else:
            for model, records in model_data.items():
                output_filename = os.path.join(output_dir, f"{model}.jsonl")
                print(f"\nWriting {len(records)} records for model '{model}' to {output_filename}")
                with open(output_filename, 'w', encoding='utf-8') as f_out:
                    for record in records:
                        f_out.write(json.dumps(record) + '\n')

            print(f"\nAll results have been saved.")
    else:
        print("\nNo matching QIDs found based on the specified criteria.")
