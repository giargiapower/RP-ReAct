import os
import json
import argparse
from vllm import LLM, SamplingParams
from prompts import llm_as_a_judge_prompt
import requests

ALERT = True

def generate_online(current_prompt: str, model_name: str):
         #outputs = self.llm.generate([current_prompt], self.sampling_params)
                #chunk = outputs[0].outputs[0].text
                # The URL of the API endpoint
                print("Sending request to LLM API...")
                url = "https://utopia.hpc4ai.unito.it/api/chat/completions"

                # The bearer token, also defined in other cells
                bearer_token = "sk-b1b3ea2ee80b4a2795cad7a3174399f0"

                # The headers for the request
                headers = {
                    "Authorization": f"Bearer {bearer_token}",
                    "Content-Type": "application/json"
                }
                
                # The data payload for the request, incorporating parameters from the build_chat_payload example
                data = {
                    "model": "SLURM."+ model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": current_prompt
                        }
                    ],
                    # Parameters from the more detailed example in cell 11
                    "temperature": 0.0,
                    "top_p": 0.95,
                    "max_tokens": 5,
                    "presence_penalty": 0.0,
                    "frequency_penalty": 0.0,
                    "n": 1,
                    "seed": 42,
                    "stream": False,
                    "ignore_system_prompt": True

                }

                try:
                    # Send the POST request
                    response = requests.post(url, headers=headers, json=data)
                    global ALERT
                    if ALERT:
                        while response is None:
                            print("Request returned None, retrying in 1 second...")
                            time.sleep(1)
                            response = requests.post(url, headers=headers, json=data)
                        # The first successful request will disable the alert for subsequent calls
                        ALERT = False

                    # Raise an exception for bad status codes (4xx or 5xx)
                    response.raise_for_status()
                    json_response = response.json()
                    if json_response is None:
                        print("Bad response (empty JSON)")
                        return ""
                    choices = json_response.get('choices')
                    if not isinstance(choices, list) or not choices:
                        # Optional: log any error payload for debugging
                        err_msg = json_response.get('error') or json_response
                        print(f"Bad response (no choices): {err_msg}")
                        return ""  # or: raise RuntimeError("No choices in response")

                    first = choices[0] or {}
                    message = first.get('message') or {}
                    content = message.get('content') or ""

                    # Trim at tool-level stop token if configured
                    chunk = str(content)
                    return chunk
                except requests.exceptions.HTTPError as http_err:
                    print(f"HTTP error occurred: {http_err}")
                    print(f"Response content: {response.text}")
                    return response.text
                except requests.exceptions.RequestException as err:
                    print(f"An error occurred: {err}")
                    return str(err)


def create_prompt(question, response, correct_answer):
    return  llm_as_a_judge_prompt.format(
            question=question,
            answer1 = response,
            answer2 = correct_answer
        ) 

def main(args):
    # Initialize the LLM from a local HuggingFace-style folder or a remote repo
    if args.mod == "offline":
        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=100000,
            gpu_memory_utilization=0.9,
            dtype="auto",
            enforce_eager=True,
        )

        # Configure sampling parameters
        sampling_params = SamplingParams(
            max_tokens=5,  # "CORRECT" or "INCORRECT" is short
            temperature=0.0, # Deterministic output
            stop=["\n"], # Stop after the first word
        )

        # Process each JSON file in the target directory
        for filename in os.listdir(args.result_dir):
            if filename.endswith(".jsonl"):
                file_path = os.path.join(args.result_dir, filename)
                print(f"Processing file: {file_path}")

                with open(file_path, 'r', encoding='utf-8') as f:
                    data = []
                    for line_num, line in enumerate(f, 1):
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"Error in file {file_path}, line {line_num}: {e}")
                            print(f"Problematic line: {line.strip()}")
                            data = None  # Signal an error
                            break
                    if data is None:
                        continue
                items_to_evaluate = [item for item in data if "eval" not in item]
                
                if not items_to_evaluate:
                    print(f"Skipping {filename}, already fully evaluated.")
                    continue

                print(f"Found {len(items_to_evaluate)} items to evaluate in {filename}.")
                
                # Process each item that needs evaluation
                for item in items_to_evaluate:
                    prompt = create_prompt(
                        item.get("question", ""), 
                        item.get("response", ""), 
                        item.get("correct_answer", "")
                    )
                    # Generate output for the single prompt
                    outputs = llm.generate([prompt], sampling_params)
                    # Get the result and save it
                    eval_result = outputs[0].outputs[0].text.strip()
                    item["eval"] = eval_result

                # Write the updated data back to the same file
                with open(file_path, 'w', encoding='utf-8') as f:
                    for item in data:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                
                print(f"Finished processing and updated {file_path}")
    else:

        # Process each JSON file in the target directory
        for filename in os.listdir(args.result_dir):
            if filename.endswith(".jsonl"): #and "Qwen" in filename
                file_path = os.path.join(args.result_dir, filename)
                print(f"Processing file: {file_path}")

                with open(file_path, 'r', encoding='utf-8') as f:
                    data = []
                    for line_num, line in enumerate(f, 1):
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"Error in file {file_path}, line {line_num}: {e}")
                            print(f"Problematic line: {line.strip()}")
                            data = None  # Signal an error
                            break
                    if data is None:
                        continue
                items_to_evaluate = [item for item in data if "eval" not in item]
                
                if not items_to_evaluate:
                    print(f"Skipping {filename}, already fully evaluated.")
                    continue

                print(f"Found {len(items_to_evaluate)} items to evaluate in {filename}.")
                
                # Process each item that needs evaluation
                for item in items_to_evaluate:
                    prompt = create_prompt(
                        item.get("question", ""), 
                        item.get("response", ""), 
                        item.get("correct_answer", "")
                    )
                    #print(prompt)
                    # Generate output for the single prompt
                    outputs = generate_online(prompt, args.model)
                    #print(f"LLM output: {outputs}")
                    # Get the result and save it
                    eval_result = outputs
                    item["eval"] = eval_result

                # Write the updated data back to the same file
                with open(file_path, 'w', encoding='utf-8') as f:
                    for item in data:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                
                print(f"Finished processing and updated {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model responses using an LLM as a judge.")
    parser.add_argument("--model", type=str, required=True, help="Path to the local HF-style folder or a remote repo for the LLM.")
    parser.add_argument("--result_dir", type=str, default="./React_results/", help="Directory containing the .json files to evaluate.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs for tensor parallelism.")
    parser.add_argument("--mod", type=str, default="offline", help="use offline or online model")
    args = parser.parse_args()
    main(args)
