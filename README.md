# Reasoner, Planner - ReAct (RP-ReAct)

This repository contains the implementation of various agent architectures for tool-calling and reasoning tasks, including:
1.  **ReAct**: A baseline agent using the "Reason and Act" paradigm.
2.  **Reflexion**: An agent that incorporates self-reflection on past failures to improve subsequent attempts.
3.  **RP-ReAct (Reasoner Planner-ReAct)**: A novel two-agent framework that fundamentally decouples strategic planning from low-level execution to achieve superior reliability and efficiency

The agents are benchmarked against the **ToolQA** dataset.

## 1. Setup and Installation

### 1.1. Dataset Preparation

This project uses the [ToolQA dataset](https://github.com/night-chen/ToolQA). Before running any experiments, you must download the dataset and place its contents into the correct directory.

1.  Download the ToolQA dataset.
2.  Unzip the contents.
3.  Move all the data files into the `RP-ReAct/data/` directory within this repository.

After this step, your `RP-ReAct/data/` directory should contain the JSON files and other data associated with the ToolQA benchmark.

### 1.2. Environment Setup

All required Python libraries are listed in `requirements.txt`.

1.  First, ensure you have a Python virtual environment activated.
2.  Install the dependencies using pip:
    ```bash
    pip install -r requirements.txt
    ```

## 2. How to Run Experiments

The experiments are managed through shell scripts. Each script is configured to run a specific agent architecture.

### 2.1. Running ReAct and Reflexion Agents

To run the baseline **ReAct** or the **Reflexion** agent, use the `sota_benchmark.sh` script.

The script takes several command-line arguments to configure the run:

*   `--agent_type`: Specifies the agent architecture.
    *   `react`: For the standard ReAct agent.
    *   `reflexion`: For the Reflexion agent.
*   `--benchmark`: The ToolQA benchmark to run on (e.g., `flight-easy`, `coffee-hard`).
*   `--prompt`: The set of few-shot examples to use.
    *   `easy`: Uses easier examples.
    *   `hard`: Uses more complex examples.
*   `--model_path`: The local path to the language model you want to use.
*   `--max_steps`: (Optional) The maximum number of steps per trajectory. Defaults to 20.
*   `--reflexion_steps`: (Optional, for Reflexion agent only) The number of reflection iterations. Defaults to 2.

**Example Usage:**

*   **To run the ReAct agent on the easy flight benchmark:**
    ```bash
    ./sota_benchmark.sh --agent_type react --benchmark flight-easy --prompt easy --model_path /path/to/your/llm
    ```

*   **To run the Reflexion agent on the hard coffee benchmark:**
    ```bash
    ./sota_benchmark.sh --agent_type reflexion --benchmark coffee-hard --prompt hard --model_path /path/to/your/llm --reflexion_steps 3
    ```

### 2.2. Running the RP-ReAct Agent

To run our proposed **RP-ReAct** agent with its reasoning-polishing loop, use the `reasoning_run.sh` script. This agent corresponds to the `SelfRefineAgentLocal` class in the code.

The script takes similar arguments:

*   `--benchmark`: The ToolQA benchmark to run on.
*   `--prompt`: The set of few-shot examples (`easy` or `hard`).
*   `--model_path`: The local path to the language model.
*   `--max_steps`: (Optional) The maximum number of steps per trajectory. Defaults to 20.
*   `--max_refines`: (Optional) The maximum number of refinement (polishing) iterations per action. Defaults to 2.

**Example Usage:**

*   **To run the RP-ReAct agent:**
    ```bash
    ./reasoning_run.sh --benchmark flight-easy --prompt easy --model_path /path/to/your/llm --max_refines 3
    ```

## 3. Code Structure

*   `RP-ReAct/Reasoning_Tool_Calling/code/`: Contains the core source code.
    *   `agents_offline.py`: Defines the `ReactAgentLocal`, `ReflexionAgentLocal`, and `SelfRefineAgentLocal` (RP-ReAct) classes.
    *   `prompts.py`: Contains all prompt templates used by the agents.
    *   `tools/`: Implementation of the various tools the agents can use (Calculator, Database Queriers, etc.).
    *   `main.py`: The main entry point script called by the shell scripts.
*   `RP-ReAct/data/`: **(User-provided)** This is where the ToolQA dataset files must be placed.
*   `sota_benchmark.sh`: Shell script for running the ReAct and Reflexion baselines.
*   `reasoning_run.sh`: Shell script for running the RP-ReAct agent.
*   `requirements.txt`: A list of Python package dependencies.