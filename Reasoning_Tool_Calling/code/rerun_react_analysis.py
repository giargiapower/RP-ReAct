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
    parser.add_argument("--local_model", type=str, default="gpt-oss-120b")
    parser.add_argument("--repo_reference", type=str, default="openai/gpt-oss-120b")
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--pattern", type=str, default="React")
    parser.add_argument("--key", type=str, default="sk-071b926ebc524e4bb0520c41823996b7")
    args = parser.parse_args()
    #print(args)
    root = '{}/benchmark/ReAct/root'.format(args.path)

    os.environ['OPENAI_API_KEY'] = args.openai_api_key
    if args.gpt == "None" and args.local_model != "None":
        from agents_online import ReflexionStrategy, ReactAgentLocal, ReflexionAgentLocal, SelfRefineAgentLocal
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

        n = 1
        log = ''
        trial = 0
        unanswered_questions = []
        agents = []
        
        if args.pattern == "React":
            agent = agent_cls(args, 20, 40000, 0.6, react_agent_prompt, args.dataset, args.dataset+"-"+args.hardness)
        elif args.pattern == "Reflexion":
            agent = agent_cls(args,  2, 10, 40000, 0.6, react_reflexion, reflexion_evaluator, reflexion_self_reflection, args.dataset, args.dataset+"-"+args.hardness)
        else: 
            agent = agent_cls(args, self_refine_instruction, feedback_generator, refine_generator, 2 , 20,  40000, 0.6, args.dataset, args.dataset+"-"+args.hardness)
        for i in range(len(contents)):
            agent.next_question(contents[i]['question'], contents[i]['answer'])
            try:
                agent.run()
                print(f'Answer: {agent.answer}, Ground-Truth: {contents[i]["answer"]}')
                print('---------')
                store_result(contents[i]['qid'], contents[i]['question'], agent.answer, contents[i]['answer'], args.dataset+"-"+args.hardness, args.local_model, args.pattern)
                log = f"""
    ########################################
    BEGIN TRIAL {contents[i]['qid']}
    #######################################
    """
                log += remove_fewshot(agent._build_agent_prompt()) + f'\nCorrect answer: {agent.answer}\n\n'
                with open(os.path.join(logs_dir, contents[i]['qid']+'.txt'), 'w') as f:
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
                with open(os.path.join(logs_dir, contents[i]['qid']+'.txt'), 'w') as f:
                    f.write(log)
                unanswered_questions.append(contents[i]['qid'])
            agents.append(agent)
        trial += 1
        log += log_react_trial(agents, trial)
        correct, incorrect, halted = summarize_react_trial(agents)
        print(f'Finished Trial {trial}, Correct: {len(correct)}, Incorrect: {len(incorrect)}, Halted: {len(halted)}')
        print('Unanswered questions: {}'.format(unanswered_questions))
        # save_agents(agents, os.path.join(root, 'ReAct', 'agents'))


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



if __name__ == "__main__":
    main()
