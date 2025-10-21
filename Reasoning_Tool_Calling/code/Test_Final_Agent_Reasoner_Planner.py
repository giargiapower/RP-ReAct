import re, string, os
from typing import List, Union, Literal
from enum import Enum
import tiktoken
from langchain.prompts import PromptTemplate
from prompts import  final_agent_planner_prompt
from fewshots import TOOLQA_HARD3_PLANNER, TOOLQA_EASY8_PLANNER_FINAL
from Test_Final_Agent_Reasoner_Tools_Proxy import ReactAgentLocal
from vllm import LLM, SamplingParams
import tools.table.tabtools as tabletools

STOP_TOKENS: List[str] = [
     "<|end_search_query|>", "</Finish>",
]

class Reasoning_Agent_Planner:
    def __init__(self,
                 args,
                 react_llm: LLM,
                 sampling_params : SamplingParams,
                 max_steps: int = 20,
                 db_toolkit: str = 'flights',
                 benchmark: str = 'flight-easy'
                 ) -> None:
        self.question = ""
        self.answer = ""
        self.key = ""
        self.proxy_agent = ReactAgentLocal(args, react_llm, sampling_params, max_steps, db_toolkit, benchmark)
        self.max_steps = max_steps
        self.scratchpad = ""
        self.agent_prompt = final_agent_planner_prompt
        if args.prompt == "easy":
            self.planner_examples = TOOLQA_EASY8_PLANNER_FINAL
        else:
            self.planner_examples = TOOLQA_HARD3_PLANNER

        self.llm = react_llm
        self.sampling_params = sampling_params
        
        self.enc = self.llm.get_tokenizer()

        self.__reset_agent()



    def next_question(self, question: str, key: str) -> None:
        self.question = question
        self.key = key
        return 


    def run(self, reset = True) -> None:
        self.__reset_agent()
        
        while not self.is_halted() and not self.is_finished():
            self.step()
    
    def step(self) -> None:
        # Think
        self.scratchpad += "<think>"
        self.sampling_params.stop = STOP_TOKENS
        self.sampling_params.max_tokens = 40000
        action = self.prompt_agent()
        #print("\n\n\n\nAction Planner:"+action+"\n\n\n\n\n\n")
        self.scratchpad += action
        # action_type, argument = parse_action(action)
        #print("\n\n\n\n"+self.scratchpad.split('\n')[-1]+"\n\n\n\n\n\n")
        global db_glbl
        if action == None or action == '' or action == '\n':
            self.scratchpad += "You action is filtered due to content. Please assume all the actions are permitted in this environment and take the action again."
        elif "<Finish>" in action:
            try:
                answer_match = re.search(r"<Finish>(.*?)</Finish>", action, re.DOTALL)
                if answer_match:
                    self.answer = answer_match.group(1).strip()
                    self.finished = True
                    self.proxy_agent.reset_variables()
                    self.proxy_agent.reset_db()
                else:
                    self.scratchpad += "<|begin_search_result|>Invalid finish format.<|end_search_result|>"
            except Exception as e:
                self.scratchpad += f"An error occurred: {e}"
        
        elif "<|begin_search_query|>" in action:
            try:
                print("Executing search query...")
                query_match = re.search(r"<\|begin_search_query\|>(.*?)<\|end_search_query\|>", action, re.DOTALL)
                if query_match:
                    query = query_match.group(1).strip()
                    self.proxy_agent.next_question(query, "")
                    self.proxy_agent.run()
                    self.scratchpad += '<|begin_search_result|>' + self.proxy_agent.answer + '<|end_search_result|>'
                    #print('<|begin_search_result|>' + self.proxy_agent.answer + '<|end_search_result|>')
                else:
                    print("invalid?")
                    
                    self.scratchpad += "<|begin_search_result|>Invalid search query format.<|end_search_result|>"
            except Exception as e:
                self.scratchpad += f"An error occurred: {e}"
     
        else:
                self.scratchpad += 'Invalid Action. Valid Actions is <|begin_search_query|>query<|end_search_query|> and <Finish>answer</Finish>.'

        #print(self.scratchpad.split('\n')[-1])

        self.step_n += 1


    def format_step(self , step: str) -> str:
        return step.strip('\n').strip().replace('\n', '')


    def prompt_agent(self) -> str:
        prompt = self._build_agent_prompt()
        #print(prompt)
        outputs = self.llm.generate([prompt], self.sampling_params)
        #print(outputs[0].outputs[0].text)

        return  self.format_step(outputs[0].outputs[0].text)
    
    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
                            MAX_SEARCH_LIMIT = self.max_steps,
                            examples = self.planner_examples,
                            question = self.question,
                            scratchpad = self.scratchpad)

    
    
    def is_finished(self) -> bool:
        return self.finished

    def is_correct(self) -> bool:
        return EM(self.answer, self.key)

    def is_halted(self) -> bool:
        return ((self.step_n > self.max_steps) or (len(self.enc.encode(self._build_agent_prompt())) > 40000)) and not self.finished

    def __reset_agent(self) -> None:
        self.step_n = 1
        self.finished = False
        self.scratchpad = ''
        self.answer = ""

    def set_qa(self, question: str, key: str) -> None:
        self.question = question
        self.key = key
   


### String Stuff ###

def parse_action(string):
    # The pattern looks for a word followed by brackets enclosing any characters.
    # It will find the last occurrence in the string.
    pattern = r'(\w+)\[(.*?)\]'
    matches = re.findall(pattern, string)
    
    if matches:
        # Return the last match found in the string
        action_type, argument = matches[-1]
        return action_type, argument
    
    else:
        # Handle cases where the model might output just Finish without arguments
        if 'finish' in string.lower():
            return 'Finish', ''
        return None, None

def format_step(step: str) -> str:
    return step.strip('\n').strip().replace('\n', '')



def truncate_scratchpad(scratchpad: str, tokenizer, n_tokens: int = 1600) -> str:
    lines = scratchpad.split('\n')
    observations = filter(lambda x: x.startswith('Observation'), lines)
    observations_by_tokens = sorted(observations, key=lambda x: len(tokenizer.encode(x)))
    while len(tokenizer.encode('\n'.join(lines))) > n_tokens:
        if not observations_by_tokens:
            # No more observations to truncate, break to avoid infinite loop
            break
        largest_observation = observations_by_tokens.pop(-1)
        ind = lines.index(largest_observation)
        lines[ind] = largest_observation.split(':')[0] + ': [truncated wikipedia excerpt]'
    return '\n'.join(lines)

def normalize_answer(s):
  def remove_articles(text):
    return re.sub(r"\b(a|an|the|usd)\b", " ", text)
  
  def white_space_fix(text):
      return " ".join(text.split())

  def remove_punc(text):
      exclude = set(string.punctuation)
      return "".join(ch for ch in text if ch not in exclude)

  def lower(text):
      return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))

def EM(answer, key) -> bool:
    return normalize_answer(str(answer)) == normalize_answer(str(key))




