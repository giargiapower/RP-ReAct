import re, string, os
from typing import List, Union, Literal
from enum import Enum
import tiktoken
import openai
from langchain.docstore import Wikipedia
from langchain.llms.base import BaseLLM
from langchain.agents.react.base import DocstoreExplorer
from langchain_community.docstore.base import Docstore
from langchain.prompts import PromptTemplate
from prompts import test_react_proxy_prompt
from fewshots import TOOLQA_EASY8_PROXY, TOOLQA_HARD_PROXY
from tools.math import calculator
from tools.text import agenda_retriever, scirex_retriever
from tools.table import tabtools
from tools.graph import graphtools
from tools.code import sql_fake, python_interpreter_mod
from vllm import LLM, SamplingParams
import tools.table.tabtools as tabletools

STOP_TOKENS_THOUGHT = [".", "Action"]
STOP_TOKENS_ACTION = ["]"]
# Determine the base path dynamically to locate the ToolQA directory
# Assumes ToolQA is a sibling directory to RP-ReAct
script_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate up from /.../RP-ReAct/Reasoning_Tool_Calling/code to /.../
base_project_dir = os.path.abspath(os.path.join(script_dir, '..'))
#toolqa_path = os.path.join(base_project_dir, 'ToolQA')

db_glbl = tabletools.table_toolkits(base_project_dir)
db_used = {
    "flights": (None, "", False),
    "coffee": (None, "", False),
    "airbnb": (None, "", False),
    "yelp": (None, "", False),
}

class ReactAgentLocal:
    def __init__(self,
                 args,
                 react_llm: LLM,
                 sampling_params : SamplingParams,
                 max_steps: int = 20,
                 db_toolkit: str = 'flights',
                 benchmark: str = 'flight-easy'
                 ) -> None:
        self.question = ""
        self.previous_actions_finished = []
        self.variable_names_temp = []
        self.answer = ""
        self.key = ""
        self.variable_names = []
        self.max_steps = max_steps
        self.scratchpad = ""
        self.agent_prompt = test_react_proxy_prompt
        self.db_toolkit = db_glbl
        if args.prompt == "easy":
            self.react_examples = TOOLQA_EASY8_PROXY
        else:
            self.react_examples = TOOLQA_HARD_PROXY

        self.llm = react_llm
        self.sampling_params = sampling_params
        
        self.table_toolkits = tabtools.table_toolkits(args.path)
        self.graph_toolkits = graphtools.graph_toolkits(args.path)
        global db_used
        db_name = benchmark.split('-')[0]
        if db_name in ["flight", "coffee", "airbnb", "yelp"]:
            if db_name == "flight":
                db_name = "flights"
            print(f"Loading database: {db_name}")
            db_temp = tabletools.table_toolkits(args.path)
            # Load the database using the global toolkit
            columns = db_temp.db_loader(db_name)
            # Store the loaded data and columns from the global toolkit into the db_used dictionary
            db_used[db_name] = (db_temp, columns, True)
        
        self.enc = self.llm.get_tokenizer()

        self.__reset_agent()


    def _program_extraction(self, action: str):
        """
        Extracts variables and code from a PythonInterpreter action string.
        The format is: PythonInterpreter{var1, var2,...}[python_code]
        """
        variables = {}
        code = ""

        # Regex to find variables in {} and code in []
        # It handles optional variables part {} and multiline code
        match = re.search(r'PythonInterpreter(?:\{(.*?)\})?\[(.*)\]', action, re.DOTALL)

        if not match:
            # Fallback for cases where the format might be slightly different
            # or to extract code if no variables are specified.
            code_match = re.search(r'\[(.*)\]', action, re.DOTALL)
            if code_match:
                code = code_match.group(1)
            return None, code, True

        # Extract variables string (group 1) and code string (group 2)
        vars_str = match.group(1)
        code = match.group(2)

        if vars_str:
            # Split variable names by comma and strip whitespace
            var_names = [name.strip() for name in vars_str.split(',') if name.strip()]
            
            for var_name in var_names:
                # Check if the variable exists in the agent's tracked variable names
                if var_name in self.variable_names:
                    try:
                        # Retrieve the value of the variable from the agent instance
                        value = getattr(self, var_name)
                        variables[var_name] = value
                    except AttributeError:
                        # This case should be rare if self.variable_names is kept consistent
                        print(f"Warning: '{var_name}' in variable_names but not found as an attribute.")
                        return None, None, False
        
        return variables, code, True
        



    def next_question(self, question: str, key: str) -> None:
        self.question = question
        self.key = key
        return 

    def reset_variables(self):
        for var_name in self.variable_names:
            if hasattr(self, var_name):
                delattr(self, var_name)
        self.variable_names = []
        self.__reset_agent()
        return 

    def run(self, reset = True) -> None:
        self.__reset_agent()
        self.variable_names_temp = []
        while not self.is_halted() and not self.is_finished():
            self.step()
    
    def step(self) -> None:
     try:
        # Think
        self.sampling_params.stop = STOP_TOKENS_THOUGHT
        self.sampling_params.max_tokens = 3200
        self.scratchpad += f'\nThought {self.step_n}:'
        self.scratchpad += ' ' + self.prompt_agent()

        #print("\n\n\n\n"+self.scratchpad+"\n\n\n\n\n\n")

        # Act
        self.sampling_params.stop = STOP_TOKENS_ACTION
        self.scratchpad += f'\nAction {self.step_n}:'
        self.sampling_params.max_tokens = 3200
        action = self.prompt_agent()
        self.scratchpad += ' ' + action
        # action_type, argument = parse_action(action)
        #print("\n\n\n\n"+self.scratchpad.split('\n')[-1]+"\n\n\n\n\n\n")
        global db_glbl
        # Observe
        self.scratchpad += f'\nObservation {self.step_n}: '
        if action == None or action == '' or action == '\n':
            self.scratchpad += "You action is filtered due to content. Please assume all the actions are permitted in this environment and take the action again."
        elif "PythonInterpreter" in action:
            try:
                    variables , argument, ok = self._program_extraction(action)
                    
                    if not ok:
                        self.scratchpad += "Invalid PythonInterpreter format. Please use PythonInterpreter{var1, var2,...}[python_code] or PythonInterpreter[python_code]. Ensure variables are previously defined."
                        self.step_n += 1
                        return
                    
                    action_type = "PythonInterpreter"
                    if variables is None:
                        variables = {}

                    observation = python_interpreter_mod.exec(argument, variables)
                    if len(self.enc.encode(str(observation))) > 100:
                            observation = observation[:100]+"...(truncated). The full sql_result is too long to fit in observation so the entire sql_result is stored in variable: "
                            var_name = "result"
                            counter = 0
                            while hasattr(self.env, var_name):
                                var_name = f"result{counter}"
                                counter += 1
                            
                            setattr(self.env, var_name, observation)
                            self.variable_names.append(var_name)
                            self.variable_names_temp.append(var_name)
                            # Correctly append to the observation string
                            observation += f"{var_name} with type {type(observation)}."
                    else : 
                        self.scratchpad += str(observation)
            except Exception as e:
                self.scratchpad += f"An error occurred: {e}"
        elif '], ' in action:
            action_type, argument = parse_action(action)
            # print(self.scratchpad.split('\n')[-1])
            self.scratchpad += "You are sending multiple actions at once. Please send one action at a time."
        else:  
            action_type, argument = parse_action(action)
            #print("Action Type:", action_type)
            #print("Argument:", argument)  
            if action_type == 'Finish':
                self.answer = argument
                if self.variable_names_temp:
                    vars_info = []
                    for var_name in self.variable_names_temp:
                        if hasattr(self, var_name):
                            value = getattr(self, var_name)
                            vars_info.append(f"{var_name}: {type(value).__name__}")
                    if vars_info:
                        self.answer += f" (Variables used: {', '.join(vars_info)})"
                self.previous_actions_finished.append(self.answer)
                if self.is_correct():
                    self.scratchpad += 'Answer is CORRECT'
                else: 
                    self.scratchpad += 'Answer is INCORRECT'
                self.finished = True
                self.step_n += 1
                return

            elif action_type == 'Calculate':
                try:
                    self.scratchpad += str(calculator.WolframAlphaCalculator(argument)).strip('\n').strip()
                except Exception as e:
                    print(e)
                    self.scratchpad += f'Illegal Mathematical Expression. Please try again.'
                        
            elif action_type == 'RetrieveAgenda':
                try:
                    self.scratchpad += agenda_retriever.query_llm([0], argument).strip('\n').strip()
                except:
                    self.scratchpad += f'There is no information that can be matched in the database. Please try another query.'

            elif action_type == 'RetrieveScirex':
                try:
                    self.scratchpad += scirex_retriever.query_llm([0], argument).strip('\n').strip()
                except:
                    self.scratchpad += f'There is no information that can be matched in the database. Please try another query.'
            
            elif action_type == 'LoadDB':
                try:
                    global db_used
                    toolkit, columns_str, loaded = db_used.get(argument, (None, "", False))
                    if toolkit is None:
                        # create a new toolkit and load columns
                        toolkit = tabletools.table_toolkits(args.path)
                        columns_str = toolkit.db_loader(argument)
                        db_used[argument] = (toolkit, columns_str, True)
                    # switch this agent to that toolkit
                    self.db = toolkit
                    
                    self.scratchpad += str(columns_str)
                except Exception as e:
                    self.scratchpad += f"Error loading database: {e}"

            # elif action_type == 'GetColumnNames':
            #     try:
            #         self.scratchpad +=  self.table_toolkits.get_column_names(argument)
            #     except openai.error.RateLimitError:
            #         self.scratchpad += f'OpenAI API Rate Limit Exceeded. Please try again.'
            #     except:
            #         self.scratchpad += f'The database you want to query in not in the list. Please change another database for query.'
            
            elif action_type == 'FilterDB':
                try:
                
                    self.scratchpad += self.db.data_filter(argument)

                except Exception as e:
                    print(e)
                    self.scratchpad += 'There is something wrong with the arguments you send for filtering. Please modify it.'

            elif action_type == 'GetValue':
                try:
                    observation = self.db.get_value(argument)
                    if (len(self.enc.encode(observation)) > 100):
                        observation = observation[:100]+"...(truncated). The full value is too long to fit in observation so the entire value is stored in variable: "
                        var_name = "value"
                        counter = 0
                        # Find the first available variable name like value, value0, value1, ...
                        while hasattr(self, var_name):
                            var_name = f"value{counter}"
                            counter += 1
                        setattr(self, var_name, observation)
                        self.variable_names.append(var_name)
                        self.variable_names_temp.append(var_name)
                        observation += f"{var_name} with type {type(observation)}."
                    else:
                        self.scratchpad += observation
                    
                except Exception as e:
                    print(e)
                    self.scratchpad += 'The value you are querying does not exist. Please modify it.'

            elif action_type == 'LoadGraph':
                try:
                    self.scratchpad += self.graph_toolkits.load_graph(argument)
                except openai.error.RateLimitError:
                    self.scratchpad += f'OpenAI API Rate Limit Exceeded. Please try again.'
                except:
                    self.scratchpad += f'The graph you want to query in not in the list. Please change another graph for query.'

            elif action_type == 'NeighbourCheck':
                try:
                    self.scratchpad += self.graph_toolkits.check_neighbours(argument)
                except openai.error.RateLimitError:
                    self.scratchpad += f'OpenAI API Rate Limit Exceeded. Please try again.'
                except:
                    self.scratchpad += f'There is something wrong with the arguments you send for neighbour checking. Please modify it.'
            
            elif action_type == 'NodeCheck':
                try:
                    self.scratchpad += self.graph_toolkits.check_nodes(argument)
                except openai.error.RateLimitError:
                    self.scratchpad += f'OpenAI API Rate Limit Exceeded. Please try again.'
                except KeyError:
                    self.scratchpad += f'The node does not exist in the graph. Please modify it.'
                except:
                    self.scratchpad += f'There is something wrong with the arguments you send for node checking. Please modify it.'
            
            elif action_type == 'EdgeCheck':
                try:
                    self.scratchpad += self.graph_toolkits.check_edges(argument)
                except openai.error.RateLimitError:
                    self.scratchpad += f'OpenAI API Rate Limit Exceeded. Please try again.'
                except KeyError:
                    self.scratchpad += f'There is no edge between the two nodes. Please modify it.'
                except:
                    self.scratchpad += f'There is something wrong with the arguments you send for edge checking. Please modify it.'

            elif action_type == 'SQLInterpreter':
                try:
                    observation = sql_fake.execute(argument)
                    if (len(self.enc.encode(observation)) > 100):
                        observation = observation[:100]+"...(truncated). The full sql_result is too long to fit in observation so the entire sql_result is stored in variable: "
                        var_name = "sql_result"
                        counter = 0
                        # Find the first available variable name like value, value0, value1, ...
                        while hasattr(self, var_name):
                            var_name = f"sql_result{counter}"
                            counter += 1
                        setattr(self, var_name, observation)
                        self.variable_names.append(var_name)
                        self.variable_names_temp.append(var_name)
                        observation += f"{var_name} with type {type(observation)}."
                    else:
                        self.scratchpad += observation
                    
                except openai.error.RateLimitError:
                    self.scratchpad += f'OpenAI API Rate Limit Exceeded. Please try again.'
                except:
                    self.scratchpad += f'There is something wrong with the SQL command you send. Please modify it.'
            
            elif action_type == 'PythonInterpreter':
                try:

                    if variables is None:
                        variables = []

                    observation = python_interpreter_mod.exec(argument, variables)
                    if isinstance(observation, str) and len(self.enc.encode(observation)) > 100:
                            observation = observation[:100]+"...(truncated). The full sql_result is too long to fit in observation so the entire sql_result is stored in variable: "
                            var_name = "result"
                            counter = 0
                            while hasattr(self.env, var_name):
                                var_name = f"result{counter}"
                                counter += 1
                            
                            setattr(self.env, var_name, observation)
                            self.variable_names.append(var_name)
                            self.variable_names_temp.append(var_name)
                            # Correctly append to the observation string
                            observation += f"{var_name} with type {type(observation)}."
                    else : 
                        self.scratchpad += str(observation)

                except Exception as e:
                    print(f"An error occurred: {e}")
                    
            else:
                self.scratchpad += 'Invalid Action. Valid Actions are Calculate [<Formula>] RetrieveAgenda[<Content>] RetrieveScirex[<Content>] LoadDB[<DBName>] FilterDB[<Condition>, <Condition>, ...] GetValue[<Column>] LoadGraph[<GraphName>] NeighbourCheck[<GraphName>, <Node>] NodeCheck[<GraphName>, <Node>] EdgeCheck[<GraphName>, <Node1>, <Node2>] SQLInterpreter[<SQLCommand>] PythonInterpreter[<PythonCode>] and Finish[<answer>].'

        #print("\n\n\n\n"+self.scratchpad.split('\n')[-1]+"\n\n\n\n\n\n")

        self.step_n += 1
     except Exception as e:
        print(f"An error occurred in step(): {e}")
        self.scratchpad += f"An error occurred in step(): {e}"
        self.step_n += 1
        return

    def format_step(self, step: str) -> str:
        return step.strip('\n').strip().replace('\n', '')

    def prompt_agent(self) -> str:
        prompt = self._build_agent_prompt()
        #print(prompt)
        outputs = self.llm.generate([prompt], self.sampling_params)
        return self.format_step(outputs[0].outputs[0].text)
    
    def _build_agent_prompt(self) -> str:
        print("building agent prompt")
        temp_prev_actions = "- "
        if len(self.previous_actions_finished) > 0: 
            for a in self.previous_actions_finished:
                temp_prev_actions += a + ";\n" + "- "
        else:
            temp_prev_actions += "No previous actions"

        return self.agent_prompt.format(
                            examples = self.react_examples,
                            prev_actions = temp_prev_actions,
                            question = self.question,
                            scratchpad = self.scratchpad)
    
    def is_finished(self) -> bool:
        return self.finished

    def is_correct(self) -> bool:
        return EM(self.answer, self.key)

    def is_halted(self) -> bool:
        return ((self.step_n > self.max_steps)) and not self.finished

    def __reset_agent(self) -> None:
        self.step_n = 1
        self.finished = False
        self.scratchpad = ''
        self.answer = ""

    

    def reset_db(self) -> None:
        try:
            global db_glbl
            db_glbl.reset_data()
            self.db_toolkit.reset_data()
            self.db.reset_data()
            self.previous_actions_finished = []
        except Exception as e:
            print(f"Error during environment reset: {e}")

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
