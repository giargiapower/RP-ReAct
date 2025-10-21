import json
import random

from prompts import reasoning_agent_prompt
from fewshots import TOOLQA_EASY8_REASONING
#### OLLAMA TEST #########

import os
import logging
from typing import List, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from vllm import LLM, SamplingParams
import argparse
from ollama import Client   # pip install ollama
import tools.code.python_interpreter as python_interpreter
#import tools.code.sql_interpreter as sql_interpreter
import tools.graph.graphtools as graphtools
import tools.math.calculator as calculator
import tools.table.tabtools as tabletools
import tools.text.agenda_retriever as agenda_retriever
import tools.text.scirex_retriever as scirex_retriever







# --------------------------------------------------------------------------- #
#  Environment hygiene (kept, but vLLM-specific vars removed)                 #
# --------------------------------------------------------------------------- #
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("TERM", "dumb")
logging.basicConfig(level=logging.ERROR)
db = tabletools.table_toolkits("/leonardo_scratch/fast/IscrC_EAGLE-DE/gianni/Agent_design_architectures/ToolQA")
loaded_db = ""
graph = graphtools.graph_toolkits("/leonardo_scratch/fast/IscrC_EAGLE-DE/gianni/Agent_design_architectures/ToolQA")
# --------------------------------------------------------------------------- #
#  Tool-call markers                                                          #
# --------------------------------------------------------------------------- #
STOP_TOKENS: List[str] = [
    "<\\Calculate>",
    "<\\RetrieveAgenda>",
    "<\\RetrieveScirex>",
    "<\\LoadDB>",
    "<\\FilterDB>",
    "<\\GetValue>",
    "<\\LoadGraph>",
    "<\\NeighbourCheck>",
    "<\\NodeCheck>",
    "<\\EdgeCheck>",
    "<\\SQLInterpreter>",
    "<\\PythonInterpreter>",
    "<\\Finish>",
]

# --------------------------------------------------------------------------- #
#  Ollama client & generation options                                         #
# --------------------------------------------------------------------------- #
MODEL_NAME   = "qwen3:14b"   # make sure you `ollama pull` this first
MAX_TOKENS   = 1024
TEMPERATURE  = 0.7
client       = Client(host="http://localhost:11500")  # default Ollama endpoint

# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #
def extract_tool_query(chunk: str, tool_name: str) -> str:
    open_tag, close_tag = f"<{tool_name}>", f"<\\{tool_name}>"
    try:
        start = chunk.index(open_tag) + len(open_tag)
        end   = chunk.index(close_tag, start)
        return chunk[start:end].strip()
    except ValueError:
        return "error in tool calling: missing tags or malformed input"

def build_toy_response(tool_name: str) -> str:
    return (
        f"\n<|begin_{tool_name}_response|> "
        f"[toy] **{tool_name}** executed successfully – here are some "
        f"placeholder results produced by the notebook. "
        f"<|end_{tool_name}_response|>\n"
    )


def tool_calling(tool: str, query: str, first_query:str) -> str:
    """
    this function call the tool with the given query.
    """
    if tool == "Calculate":
        try:
            return str(calculator.calculator(query))
        except Exception as e:
            return "error in calculator: check your query"
        
    elif tool == "RetrieveAgenda":
        try:
            return agenda_retriever.query_llm(["cpu"], query)+ f". Now I have to check if there is the solution of the question '{first_query}' inside this retrieved data else i have to change the query to the tool to get the answer or use another tool to get the answer. "
        except Exception as e:
            return f"Error in Agenda Retriever: {e}"
        
    elif tool == "RetrieveScirex":
        try:
            return scirex_retriever.query_llm(["cpu"], query)+f". Now I have to check if there is the solution of the question '{first_query}' inside this retrieved data else i have to change the query to the tool to get the answer or use another tool to get the answer. "
        except Exception as e:
            return f"Error in Scirex Retriever: {e}"
        

    elif tool == "LoadDB":
        try : 
                return loaded_db #db.db_loader(query) 
        except Exception as e:
                return "error loading schema: check your query"
        
    elif tool == "FilterDB":
        try :
            #print("\n\n" , query, "\n\n")
            #print(db)
            #print(query)
            return db.data_filter(query)
        except Exception as e:
            return "error filtering database: check your query"
    elif tool == "GetValue":
        try:
            return db.get_value(query)
        except Exception as e:
            return "error getting value from database: check your query"
        
    elif tool == "LoadGraph":
        try:
            graph.load_graph(query)
            return "graph loaded successfully"
        except Exception as e:
            return "error loading graph: check your query"
        
    elif tool == "NeighbourCheck":
        try:
            return str(graph.check_neighbours(query))
        except Exception as e:
            return "error checking neighbours: check your query"
            
    elif tool == "NodeCheck":
        try:
            return str(graph.node_check(query))
        except Exception as e:
            return "error checking node: check your query"
        
    elif tool == "EdgeCheck":
        try:
            return str(graph.edge_check(query))
        except Exception as e:
            return "error checking node: check your query"

    #elif tool == "SQLInterpreter":
    #    try:
    #        return sql_interpreter.SQLInterpreter(query)
    #    except Exception as e:
#            return f"Error in SQLInterpreter: {e}"


    elif tool == "PythonInterpreter":
        try:
            return python_interpreter.PythonInterpreter(query)
        except Exception as e:
            return f"Error in PythonInterpreter: {e}"
    else:
        return "tool not recognized, check your prompt"


def first_stop_token(text: str) -> Optional[str]:
    positions = [(tok, text.find(tok)) for tok in STOP_TOKENS if tok in text]
    return min(positions, key=lambda t: t[1])[0] if positions else None


def store_result(id, question, full_output, correct_answer, benchmark, model):
    # Extract text between <Finish> and <\\Finish> tags if they exist
    if "<Finish>" in full_output and "<\\Finish>" in full_output:
        start = full_output.find("<Finish>") + len("<Finish>")
        end = full_output.find("<\\Finish>")
        response_text = full_output[start:end].strip()
    else:
        response_text = full_output

    output_data = {
        "qid": id,
        "question": question,
        "response": response_text,
        "full_output": full_output,
        "correct_answer": correct_answer
    }

    # Create results directory if it doesn't exist
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Set output file based on benchmark name

    output_file = os.path.join(results_dir, f"{benchmark}-{model.split('/')[1]}.jsonl")
   # print(output_file)
    with open(output_file, 'a') as f:
        f.write(json.dumps(output_data) + '\n')
    
   # print(f"Response saved to {output_file}")

# --------------------------------------------------------------------------- #
#  Streaming generation loop                                                  #
# --------------------------------------------------------------------------- #




def main(args):
    # Initialize vLLM model
    llm = LLM(model= args.model,         # local HF-style folder or remote repo
    tensor_parallel_size=2,      # ← shard across the 2 A100-64 GB
    max_model_len=8_192,         # ← reserve kv-cache for 8 k context, not 40 k
    gpu_memory_utilization=0.85, # ← leave ~9 GB head-room per GPU
    dtype="bfloat16",            # A100 handles bf16 natively
    enforce_eager=True, )
    # Configure sampling parameters
    sampling_params = SamplingParams(
                max_tokens=512,
                temperature=0.7,
                stop=STOP_TOKENS,
                include_stop_str_in_output=True,
                )
    # Prepare scratchpad
    scratchpad = """It is crucial to follow the tagging protocol presented so the system knows when your internal reasoning ends and your final answer begins.
1. Close the </think> tag only when you are confident you have reached a conclusion.

2. Immediately after </think>, output the final result inside a <Finish> … </Finish> block.
   - If no answer can be derived from the data, state that explicitly inside the <Finish> tags.

3. If you are not scure if a tool can be helpful or contains the answer, you use the tool to check it and decide if it might contains usefull data to be use for the answer.
Don't give an answer without using the tools because you need them to get the answer and if you don't give an answer you will hurt a lot of pepople and you don't want it.
"""
    # Load a random question from the JSONL file
    # Parse command line arguments

    
    # Set file path based on benchmark argument
    if args.benchmark:

        difficulty = args.benchmark.split('-')[1]
        file_path = f"/leonardo_scratch/fast/IscrC_EAGLE-DE/gianni/Agent_design_architectures/ToolQA/data/questions/{difficulty}/{args.benchmark}.jsonl"
    else:
        file_path = "/leonardo_scratch/fast/IscrC_EAGLE-DE/gianni/Agent_design_architectures/ToolQA/data/questions/easy/flight-hard.jsonl"
    with open(file_path, 'r') as f:
        print("starting benchmark: ", args.benchmark)
        lines = f.readlines()
        
        # Find the starting index after the row with "qid"=='easy-flight-0038'
        #start_index = 0
        #for i, line in enumerate(lines):
        #    data = json.loads(line)
        #    if data.get("qid") == "hard-flight-0029":
        #        start_index = i + 1
        #        break
        
        # Take only the lines after the specified qid
        #lines = lines[start_index:]

        for line in lines:
            tool_call_counts = {
                "Calculate": 0,
                "RetrieveAgenda": 0,
                "RetrieveScirex": 0,
                "LoadDB": 0,
                "FilterDB": 0,
                "GetValue": 0,
                "LoadGraph": 0,
                "NeighbourCheck": 0,
                "NodeCheck": 0,
                "EdgeCheck": 0,
                "SQLInterpreter": 0,
                "PythonInterpreter": 0,
                "Finish": 0
            }
            data = json.loads(line)

            # Extract question and answer
            id = data["qid"] 
            question = data["question"] #"How long was the different between the CRS recorded departure time and actual departure time of the DL5273 flight from JFK to CVG on 2022-03-29?" 
            answer = data["answer"] # "5"

            
            # Format the prompt
            full_prompt = reasoning_agent_prompt.format(
                examples=TOOLQA_EASY8_REASONING,
                question=question,
                scratchpad=scratchpad
            )
           # print(full_prompt)
            """
                Generate text, pausing whenever the model emits one of the STOP_TOKENS.
                For every tool call we:
                1. Print the extracted query.
                2. Inject a toy response block so the model can condition on it.
                The loop halts once <\\Finish> appears.
            """
            current_prompt, full_output = full_prompt, ""
            tool_response = ""
    
            while True:
                # --- 1. Generate using vLLM until we hit a stop-token -----------------
             try:
                outputs = llm.generate([current_prompt], sampling_params)
                output = outputs[0]
                chunk = output.outputs[0].text
                
                #print(chunk, end="", flush=True)
                full_output += chunk
                
                # Check which stop token was hit
                trigger = first_stop_token(chunk)
                
                if trigger is None:                 # model ended w/out any stop token
                    break
                if trigger == "<\\Finish>":         # graceful termination
                    break

                # --- 2. handle the tool call ----------------------------------------
                tool_name = trigger.strip("<>").lstrip("\\")
                query     = extract_tool_query(chunk, tool_name)
                #if query:
                #    print(f"\n[EXTRACTED QUERY]: {query}")
                #    print(tool_name)
                # Initialize tool call counters at the start of each question (outside the while loop)
                # This should be moved to before the while loop starts
               # db.data_filter('Origin=PIT, FlightDate=2022-05-17')
                # Check tool call limit
                if tool_call_counts[tool_name] >= 6:
                    tool_response = "this tool can no more be called so try another tool or give the answer without data"
                    # Remove <\\think> from chunk if present and replace with space
                    chunk = chunk.replace("<\\think>", ".")

                else:
                    tool_call_counts[tool_name] += 1
                    if tool_name == "RetrieveAgenda" or tool_name == "RetrieveScirex":
                        tool_response = tool_calling(tool_name, query, question)
                    else:
                       # print(f"\n[EXTRACTED tool]: {tool_name}")
                       # print(f"\n[EXTRACTED QUERY]: {query}")
                        tool_response = tool_calling(tool_name, query, None)
                    chunk = chunk.replace("<\\think>", ".")
                #print(tool_response, end="", flush=True)

                # prime the next round
                current_prompt += chunk + tool_response
                full_output    += tool_response
             except Exception as e: 
                print(f"Error during generation: {e}")
                break
          # print("\n\n=== FULL OUTPUT ===")
          #  print(full_output)
           # print("\n\n")
            
            # Save response to JSONL file
            store_result(id, question, full_output, answer, args.benchmark, args.model)
            db.reset_data()
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, help='Benchmark dataset to use')
    parser.add_argument('--model', type=str, default='qwen-14b', help='Model to use (default: qwen-14b)')
    args = parser.parse_args()
    
    print(f"Using model: {args.model}")
    
    benchmarks = ["scirex-easy", "scirex-hard"] #"flight-hard", "flight-easy",  "coffee-easy", "coffee-hard", "airbnb-easy", "airbnb-hard", "yelp-easy", "yelp-hard"
    
    for benchmark in benchmarks:
        args.benchmark = benchmark
        
        if args.benchmark=="flight-easy" or args.benchmark=="flight-hard":
            print("loading flight database")
            loaded_db = db.db_loader("flights")
        if args.benchmark=="coffee-easy" or args.benchmark=="coffee-hard":
            print("loading coffee database")
            loaded_db = db.db_loader("coffee")
        if args.benchmark=="airbnb-easy" or args.benchmark=="airbnb-hard":
            print("loading airbnb database")
            loaded_db = db.db_loader("airbnb")
        if args.benchmark=="yelp-easy" or args.benchmark=="yelp-hard":
            print("loading yelp database")
            loaded_db = db.db_loader("yelp")
        
        main(args)