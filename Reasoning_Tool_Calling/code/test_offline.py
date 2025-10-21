import os

import joblib

from mocks import DocStoreExplorerMock, LLMMock

import argparse

import jsonlines

from util import summarize_react_trial, log_react_trial, save_agents, remove_fewshot

import datetime

from prompts import reflect_prompt, react_agent_prompt, self_refine_instruction, feedback_generator, refine_generator, react_reflexion, reflexion_evaluator, reflexion_self_reflection

from vllm import LLM, SamplingParams

import json


def store_result(qid, question, full_output, correct_answer, benchmark, model_name, pattern):
        """Saves the results to a JSONL file."""

        output_data = {
        "qid": qid,
        "question": question,
        "response": full_output,
        "correct_answer": correct_answer
        }

        results_dir = pattern+"_results"
        os.makedirs(results_dir, exist_ok=True)
        
        model_short_name = model_name.split('/')[-1]
        output_file = os.path.join(results_dir, f"{benchmark}-{model_short_name}.jsonl")
        
        with open(output_file, 'a') as f:
            f.write(json.dumps(output_data) + '\n')
            print(f"Response for {qid} saved to {output_file}")

def main():
    current_datetime = datetime.datetime.now()
    datetime_string = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

    # root = '{}/benchmark/ReAct/root'

    parser = argparse.ArgumentParser("")
    parser.add_argument("--dataset", type=str, default="flight")
    parser.add_argument("--hardness", type=str, default="easy")
    parser.add_argument("--openai_api_key", type=str, default="<OPENAI_API_KEY>")
    parser.add_argument("--path", type=str, default="<YOUR_OWN_PATH>")
    parser.add_argument("--wolframalpha_api_key", type=str, default="<WOLFALPHA_API_KEY>")
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--debug_id", type=int, default=0)
    parser.add_argument("--gpt", type=str, default="None")
    parser.add_argument("--prompt", type=str, default="easy")
    parser.add_argument("--local_model", type=str, default="Qwen3-14B")
    parser.add_argument("--max_tokens", type=int, default=40000)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--reflexion_steps", type=int, default=2)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--pattern", type=str, default="React")
    args = parser.parse_args()
    #print(args)
    root = '{}/benchmark/ReAct/root'.format(args.path)

    os.environ['OPENAI_API_KEY'] = args.openai_api_key
    if args.gpt == "None" and args.local_model != "None":
        from agents_offline import ReflexionStrategy, ReactAgentLocal, ReflexionAgentLocal, SelfRefineAgentLocal
    elif args.gpt == "gpt3":
        from agents import ReactReflectAgent, ReactAgent, ReflexionStrategy
    elif args.gpt == "chatgpt":
        from agents_chatgpt import ReactReflectAgent, ReactAgent, ReflexionStrategy
    elif args.gpt == "azure":
        from agents_azure import ReactReflectAgent, ReactAgent, ReflexionStrategy
    file_path = "{}/data/questions/{}/{}-{}.jsonl".format(args.path, args.hardness, args.dataset, args.hardness)
    print(file_path)
    with open(file_path, 'r') as f:
        contents = []
        for item in jsonlines.Reader(f):
            contents.append(item)

    if args.debug:
        random_indices = args.debug_id
        test_q = contents[random_indices]['question']
        test_a = contents[random_indices]['answer']
        agent = ReactAgent(args, test_q, test_a)
        agent.run()
        #print(test_q)
        #print(agent._build_agent_prompt())
        print("Ground-Truth: ", test_a)
    else:
        #print(args.path, args.local_model, datetime_string, args.dataset, args.hardness)
        if args.pattern == "React":
            if not os.path.exists('{}/benchmark/ReAct/logs/{}-{}/{}-{}'.format(args.path, args.local_model, datetime_string, args.dataset, args.hardness)):
                os.makedirs('{}/benchmark/ReAct/logs/{}-{}/{}-{}'.format(args.path, args.local_model, datetime_string, args.dataset, args.hardness))
                logs_dir = '{}/benchmark/ReAct/logs/{}-{}/{}-{}'.format(args.path, args.local_model, datetime_string, args.dataset, args.hardness)
            agent_cls = ReactAgentLocal
        elif args.pattern == "Reflexion":
            if not os.path.exists('{}/benchmark/Reflexion/logs/{}-{}/{}-{}'.format(args.path, args.local_model, datetime_string, args.dataset, args.hardness)):
                os.makedirs('{}/benchmark/Reflexion/logs/{}-{}/{}-{}'.format(args.path, args.local_model, datetime_string, args.dataset, args.hardness))
                logs_dir = '{}/benchmark/Reflexion/logs/{}-{}/{}-{}'.format(args.path, args.local_model, datetime_string, args.dataset, args.hardness)
            agent_cls = ReflexionAgentLocal
        else : 
            if not os.path.exists('{}/benchmark/SelfRefine/logs/{}-{}/{}-{}'.format(args.path, args.local_model, datetime_string, args.dataset, args.hardness)):
                os.makedirs('{}/benchmark/SelfRefine/logs/{}-{}/{}-{}'.format(args.path, args.local_model, datetime_string, args.dataset, args.hardness))
                logs_dir = '{}/benchmark/SelfRefine/logs/{}-{}/{}-{}'.format(args.path, args.local_model, datetime_string, args.dataset, args.hardness)
            agent_cls = SelfRefineAgentLocal



        local_llm = LLM(model=args.local_model,
                            pipeline_parallel_size=args.tensor_parallel_size,
                            tensor_parallel_size=1,
                            max_model_len=args.max_tokens,
                            gpu_memory_utilization=0.90,
                            dtype="auto",
                            enforce_eager=True,
                            )
        sampling_params = SamplingParams(
                                max_tokens=args.max_tokens,
                                temperature=0.6,
                                include_stop_str_in_output=True,
                            )
        n = 1
        log = ''
        trial = 0
        unanswered_questions = []
        agents = []
        
        if args.pattern == "React":
            agent = agent_cls(args, local_llm, sampling_params, args.steps, react_agent_prompt, args.dataset, args.dataset+"-"+args.hardness)
        elif args.pattern == "Reflexion":
            agent = agent_cls(args, local_llm, sampling_params,  args.reflexion_steps, args.steps, react_reflexion, reflexion_evaluator, reflexion_self_reflection, args.dataset, args.dataset+"-"+args.hardness)
        else: 
            agent = agent_cls(args, self_refine_instruction, feedback_generator, refine_generator, local_llm, sampling_params, args.reflexion_steps , args.steps,  args.dataset, args.dataset+"-"+args.hardness)
        results_dir = args.pattern + "_results"
        model_short_name = args.local_model.split('/')[-1]
        benchmark_name = args.dataset + "-" + args.hardness
        output_file = os.path.join(results_dir, f"{benchmark_name}-{model_short_name}.jsonl")

        answered_qids = set()
        if os.path.exists(output_file):
            with jsonlines.open(output_file) as reader:
                for obj in reader:
                    if 'qid' in obj:
                        answered_qids.add(obj['qid'])
        
        print(f"Found {len(answered_qids)} already answered questions. Skipping them.")

        for i in range(len(contents)):
            if contents[i]['qid'] in answered_qids:
                continue
            print(contents[i]['qid'])
            agent.next_question(contents[i]['question'], contents[i]['answer'])
            #agent.next_question("What was the departure time of the F9124 flight from JAX to SJU on 2022-02-28?", "11:25")
            try:
                agent.run()
                print(f'Answer: {agent.key}')
                print('---------')
                scratchpad_file = f"{args.pattern}_{args.dataset}.txt"
                with open(scratchpad_file, 'a') as f:
                    f.write(f"--- QID: {contents[i]['qid']} ---\n")
                    f.write(str(agent.scratchpad))
                    f.write("\n\n")
                store_result(contents[i]['qid'], contents[i]['question'], agent.answer, contents[i]['answer'], args.dataset+"-"+args.hardness, args.local_model, args.pattern)
                log = f"""
            ########################################
            BEGIN TRIAL {contents[i]['qid']}
            #######################################
            """
                log += remove_fewshot(agent._build_agent_prompt()) + f'\nCorrect answer: {agent.answer}\n\n'
                with open(os.path.join(logs_dir, str(contents[i]['qid'])+'.txt'), 'w') as f:
                    f.write(log)
            except Exception as e:
                print(e)
                print('Error when computing answer for {}.'.format(contents[i]['qid']))
                print('---------')
                log = f"""
        ########################################
        BEGIN TRIAL {contents[i]['qid']}
        #######################################
        """
                log += remove_fewshot(agent._build_agent_prompt()) + f'\nCorrect answer: {agent.key}\n\n'
                log += 'ERROR!'
                with open(os.path.join(logs_dir, str(contents[i]['qid'])+'.txt'), 'w') as f:
                    f.write(log)
                unanswered_questions.append(contents[i]['qid'])
            agents.append(agent)
        trial += 1
        log += log_react_trial(agents, trial)
        correct, incorrect, halted = summarize_react_trial(agents)
        print(f'Finished Trial {trial}, Correct: {len(correct)}, Incorrect: {len(incorrect)}, Halted: {len(halted)}')
        print('Unanswered questions: {}'.format(unanswered_questions))
        # save_agents(agents, os.path.join(root, 'ReAct', 'agents'))'''



if __name__ == "__main__":
    main()
