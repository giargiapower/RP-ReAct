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
from prompts import reflect_prompt, react_agent_prompt, react_reflect_agent_prompt, REFLECTION_HEADER, LAST_TRIAL_HEADER, REFLECTION_AFTER_LAST_TRIAL_HEADER, react_reflexion, reflexion_evaluator, reflexion_self_reflection, self_refine_instruction, feedback_generator, refine_generator
from prompts import cot_agent_prompt, cot_reflect_agent_prompt, cot_reflect_prompt, COT_INSTRUCTION, COT_REFLECT_INSTRUCTION
from fewshots import TOOLQA_EASY8, TOOLQA_HARD3, COT, COT_REFLECT, REFLECTIONS #WEBTHINK_SIMPLE6, REFLECTIONS, COT, COT_REFLECT
from tools.math import calculator
from tools.text import agenda_retriever, scirex_retriever
from tools.table import tabtools
from tools.graph import graphtools
from tools.code import sql_fake, python_interpreter, sql_interpreter
from vllm import LLM, SamplingParams
import tools.table.tabtools as tabletools

STOP_TOKENS_THOUGHT = [".", "Action"]
STOP_TOKENS_ACTION = ["]"]
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
LONG_TERM_MEMORY: List[str] = [
    "",
    "",
    ""
]

class ReflexionStrategy(Enum):
    """
    NONE: No reflection
    LAST_ATTEMPT: Use last reasoning trace in context 
    REFLEXION: Apply reflexion to the next reasoning trace 
    LAST_ATTEMPT_AND_REFLEXION: Use last reasoning trace in context and apply reflexion to the next reasoning trace 
    """
    NONE = 'base'
    LAST_ATTEMPT = 'last_trial' 
    REFLEXION = 'reflexion'
    LAST_ATTEMPT_AND_REFLEXION = 'last_trial_and_reflexion'


class ReactAgentLocal:
    def __init__(self,
                 args,
                 react_llm: LLM,
                 sampling_params : SamplingParams,
                 max_steps: int = 20,
                 agent_prompt: PromptTemplate = react_agent_prompt,
                 db_toolkit: str = 'flights',
                 benchmark: str = 'flight-easy'
                 ) -> None:
        self.question = ""
        self.answer = ""
        self.path = args.path
        self.key = ""
        self.max_steps = max_steps
        self.scratchpad = ""
        self.agent_prompt = agent_prompt
        self.db = tabletools.table_toolkits(args.path)
        self.db_toolkit = db_glbl
        if args.prompt == "easy":
            self.react_examples = TOOLQA_EASY8
        else:
            self.react_examples = TOOLQA_HARD3

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
        self.sampling_params.stop = STOP_TOKENS_THOUGHT
        self.scratchpad += f'\nThought {self.step_n}:'
        self.scratchpad += ' ' + self.prompt_agent()

        #print("\n\n\n\n\nThought:"+self.scratchpad.split('\n')[-1]+"\n\n\n\n\n\n")

        # Act
        self.sampling_params.stop = STOP_TOKENS_ACTION
        self.scratchpad += f'\nAction {self.step_n}:'
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
            action_type = 'PythonInterpreter'
            argument = action[18:-1]
            try:
                self.scratchpad += python_interpreter.execute(argument)
            except Exception as e:
                self.scratchpad += f"An error occurred: {e}"
        elif '], ' in action:
            action_type, argument = parse_action(action)
            # print(self.scratchpad.split('\n')[-1])
            self.scratchpad += "You are sending multiple actions at once. Please send one action at a time."
        else:  
            action_type, argument = parse_action(action)
            # print(self.scratchpad.split('\n')[-1])  
            if action_type == 'Finish':
                self.answer = argument
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
                        toolkit = tabletools.table_toolkits(self.path)
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
                    self.scratchpad += self.db.get_value(argument)
                except Exception as e:
                    print(e)
                    self.scratchpad += 'The value you are querying does not exist. Please modify it.'

            elif action_type == 'LoadGraph':
                try:
                    self.scratchpad += self.graph_toolkits.load_graph(argument)
                #except openai.error.RateLimitError:
                #    self.scratchpad += f'OpenAI API Rate Limit Exceeded. Please try again.'
                except:
                    self.scratchpad += f'The graph you want to query in not in the list. Please change another graph for query.'

            elif action_type == 'NeighbourCheck':
                try:
                    self.scratchpad += self.graph_toolkits.check_neighbours(argument)
                #except openai.error.RateLimitError:
                #    self.scratchpad += f'OpenAI API Rate Limit Exceeded. Please try again.'
                except:
                    self.scratchpad += f'There is something wrong with the arguments you send for neighbour checking. Please modify it.'
            
            elif action_type == 'NodeCheck':
                try:
                    self.scratchpad += self.graph_toolkits.check_nodes(argument)
                #except openai.error.RateLimitError:
                #    self.scratchpad += f'OpenAI API Rate Limit Exceeded. Please try again.'
                except KeyError:
                    self.scratchpad += f'The node does not exist in the graph. Please modify it.'
                except:
                    self.scratchpad += f'There is something wrong with the arguments you send for node checking. Please modify it.'
            
            elif action_type == 'EdgeCheck':
                try:
                    self.scratchpad += self.graph_toolkits.check_edges(argument)
                #xcept openai.error.RateLimitError:
                #   self.scratchpad += f'OpenAI API Rate Limit Exceeded. Please try again.'
                except KeyError:
                    self.scratchpad += f'There is no edge between the two nodes. Please modify it.'
                except:
                    self.scratchpad += f'There is something wrong with the arguments you send for edge checking. Please modify it.'

            elif action_type == 'SQLInterpreter':
                try:
                    self.scratchpad += sql_interpreter.execute(argument)
                #except openai.error.RateLimitError:
                #    self.scratchpad += f'OpenAI API Rate Limit Exceeded. Please try again.'
                except:
                    self.scratchpad += f'There is something wrong with the SQL command you send. Please modify it.'
            
            elif action_type == 'PythonInterpreter':
                try:
                    exec(argument)
                except Exception as e:
                    print(f"An error occurred: {e}")
                    
            else:
                self.scratchpad += 'Invalid Action. Valid Actions are Calculate [<Formula>] RetrieveAgenda[<Content>] RetrieveScirex[<Content>] LoadDB[<DBName>] FilterDB[<Condition>, <Condition>, ...] GetValue[<Column>] LoadGraph[<GraphName>] NeighbourCheck[<GraphName>, <Node>] NodeCheck[<GraphName>, <Node>] EdgeCheck[<GraphName>, <Node1>, <Node2>] SQLInterpreter[<SQLCommand>] PythonInterpreter[<PythonCode>] and Finish[<answer>].'

        #print(self.scratchpad.split('\n')[-1])

        self.step_n += 1

    def prompt_agent(self) -> str:
        prompt = self._build_agent_prompt()
        #print(prompt)
        outputs = self.llm.generate([prompt], self.sampling_params)
        return format_step(outputs[0].outputs[0].text)
    
    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
                            examples = self.react_examples,
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
        try:
            global db_glbl
            db_glbl.reset_data()
            self.db_toolkit.reset_data()
            self.db.reset_data()
        except Exception as e:
            print(f"Error during environment reset: {e}")

    def set_qa(self, question: str, key: str) -> None:
        self.question = question
        self.key = key
   

### REFLEXION ###
class ReflexionAgentLocal:
    def __init__(self,
                 args,
                 react_llm: LLM,
                 sampling_params : SamplingParams,
                 reflexion_steps: int = 2,
                 max_steps: int = 20,
                 agent_prompt: PromptTemplate = react_reflexion,
                 evaluator_prompt : PromptTemplate = reflexion_evaluator,
                 self_refiner_prompt : PromptTemplate = reflexion_self_reflection,
                 db_toolkit: str = 'flights',
                 benchmark: str = 'flight-easy'
                 ) -> None:
        self.reflexion_steps = reflexion_steps
        self.question = ""
        self.path = args.path
        self.answer = ""
        self.key = ""
        self.max_steps = max_steps
        self.agent_prompt = agent_prompt
        self.evaluator_prompt = evaluator_prompt
        self.self_refiner_prompt = self_refiner_prompt
        self.db_toolkit = db_glbl
        self.db = tabletools.table_toolkits(args.path)
        if args.prompt == "easy":
            self.react_examples = TOOLQA_EASY8
        else:
            self.react_examples = TOOLQA_HARD3
        self.scratchpad = ""
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
    def next_question(self, question: str, key: str) -> None:
        self.question = question
        self.key = key
        return 

    def run(self, reset = True) -> None:
        for i in range(len(LONG_TERM_MEMORY)):
            LONG_TERM_MEMORY[i] = ""
        self.__reset_agent()
        
        for i in range(self.reflexion_steps): 
            while not self.is_halted() and not self.is_finished():
                self.step()
            # EVALUATION
            print("starting eval")
            self.sampling_params.stop = STOP_TOKENS_ACTION
            evaluation =  self.prompt_agent("evaluator")
            #print(f"\n\n\n\n\nEVALUATION {i+1}:\n{evaluation}\n\n\n\n\n")
            if "SUCCESS" in evaluation:
                return
            #SELF-REFINEMENT
            self.sampling_params.stop = STOP_TOKENS_ACTION
            refinement =  self.prompt_agent("self_refiner")
            #print(f"\n\n\n\n\nREFINEMENT {i+1}:\n{refinement}\n\n\n\n\n")
            LONG_TERM_MEMORY[i % 3] = refinement
            self.__reset_agent()


    
    def step(self) -> None:
        # Think
        #print("\n\n\n\n\nSTEP "+str(self.step_n)+"\n\n\n\n\n")
        self.sampling_params.stop = STOP_TOKENS_THOUGHT
        self.scratchpad += f'\nThought {self.step_n}:'
        self.scratchpad += ' ' + self.prompt_agent("agent")

        #print("\n\n\n\n\nThought:"+self.scratchpad+"\n\n\n\n\n\n")
        #print("\n\n\n\n\n" + self.scratchpad.split('\n')[-1]+ "\n\n\n\n\n\n")

        # Act
        self.sampling_params.stop = STOP_TOKENS_ACTION
        self.scratchpad += f'\nAction {self.step_n}:'
        action = self.prompt_agent("agent")
        self.scratchpad += ' ' + action
        # action_type, argument = parse_action(action)
        #print("\n\n\n\n\nAction:"+action+"\n\n\n\n\n\n")
        #print("\n\n\n\n"+self.scratchpad.split('\n')[-1]+"\n\n\n\n\n\n")
        global db_glbl
        # Observe
        self.scratchpad += f'\nObservation {self.step_n}: '
        if action == None or action == '' or action == '\n':
            self.scratchpad += "You action is filtered due to content. Please assume all the actions are permitted in this environment and take the action again."
        elif "PythonInterpreter" in action:
            action_type = 'PythonInterpreter'
            argument = action[18:-1]
            try:
                self.scratchpad += python_interpreter.execute(argument)
            except Exception as e:
                self.scratchpad += f"An error occurred: {e}"
        elif '], ' in action:
            action_type, argument = parse_action(action)
            # print(self.scratchpad.split('\n')[-1])
            self.scratchpad += "You are sending multiple actions at once. Please send one action at a time."
        else:  
            action_type, argument = parse_action(action)
            #print("\n\n\n\n\n"+action_type + " " + argument+ "\n\n\n\n\n") 
            if action_type == 'Finish':
                self.answer = argument
                #if self.is_correct():
                #    self.scratchpad += 'Answer is CORRECT'
                #else: 
                #    self.scratchpad += 'Answer is INCORRECT'
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
                except openai.error.RateLimitError:
                    self.scratchpad += f'OpenAI API Rate Limit Exceeded. Please try again.'
                except:
                    self.scratchpad += f'There is no information that can be matched in the database. Please try another query.'

            elif action_type == 'RetrieveScirex':
                try:
                    self.scratchpad += scirex_retriever.query_llm([0], argument).strip('\n').strip()
                except openai.error.RateLimitError:
                    self.scratchpad += f'OpenAI API Rate Limit Exceeded. Please try again.'
                except:
                    self.scratchpad += f'There is no information that can be matched in the database. Please try another query.'
            
            elif action_type == 'LoadDB':
                
                try:
                    global db_used
                    toolkit, columns_str, loaded = db_used.get(argument, (None, "", False))
                    if toolkit is None:
                        # create a new toolkit and load columns
                        toolkit = tabletools.table_toolkits(self.path)
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
                    self.scratchpad += self.db.get_value(argument)
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
                    self.scratchpad += sql_fake.execute(argument)
                except openai.error.RateLimitError:
                    self.scratchpad += f'OpenAI API Rate Limit Exceeded. Please try again.'
                except:
                    self.scratchpad += f'There is something wrong with the SQL command you send. Please modify it.'
            
            elif action_type == 'PythonInterpreter':
                try:
                    exec(argument)
                except Exception as e:
                    print(f"An error occurred: {e}")
                    
            else:
                self.scratchpad += 'Invalid Action. Valid Actions are Calculate [<Formula>] RetrieveAgenda[<Content>] RetrieveScirex[<Content>] LoadDB[<DBName>] FilterDB[<Condition>, <Condition>, ...] GetValue[<Column>] LoadGraph[<GraphName>] NeighbourCheck[<GraphName>, <Node>] NodeCheck[<GraphName>, <Node>] EdgeCheck[<GraphName>, <Node1>, <Node2>] SQLInterpreter[<SQLCommand>] PythonInterpreter[<PythonCode>] and Finish[<answer>].'

        #print(self.scratchpad.split('\n')[-1])

        self.step_n += 1

    def prompt_agent(self, step : str) -> str:
        if step == "agent":
            prompt = self._build_agent_prompt()
        elif step == "evaluator":
            prompt = self._build_evaluator_prompt()
        else:
            prompt = self._build_self_refiner_prompt() 
        #print(prompt)
        outputs = self.llm.generate([prompt], self.sampling_params)
        return format_step(outputs[0].outputs[0].text)
    
    def _build_agent_prompt(self) -> str:
        self.sampling_params.max_tokens = 40000
        l_m = ""
        non_empty_memories = [mem for mem in LONG_TERM_MEMORY if mem]
        if non_empty_memories:
            l_m = "REFLECTIONS on previous attempts that tried to solve the same question (use them to improve your performance to solve the question):" + "\n".join(non_empty_memories) + ".\n" + "END REFLECTIONS.\n"

        return self.agent_prompt.format(
                            examples = self.react_examples,
                            question = self.question,
                            long_term_memory = l_m,
                            scratchpad = self.scratchpad)

    def _build_evaluator_prompt(self) -> str:
        self.sampling_params.max_tokens = 500
        return self.evaluator_prompt.format(
                            question = self.question,
                            trajectory = self.scratchpad)

    def _build_self_refiner_prompt(self) -> str:
        self.sampling_params.max_tokens = 1000
        return self.self_refiner_prompt.format(
                    question = self.question,
                    trajectory = self.scratchpad,
                    prev_reflections = '\n'.join(LONG_TERM_MEMORY))
    
    def is_finished(self) -> bool:
        return self.finished

    def is_correct(self) -> bool:
        return EM(self.answer, self.key)

    def is_halted(self) -> bool:
        return ((self.step_n > self.max_steps) or (len(self.enc.encode(self._build_agent_prompt())) > 35000)) and not self.finished

    def __reset_agent(self) -> None:
        self.step_n = 1
        self.finished = False
        self.scratchpad= ''
        self.answer = ""
        try:
            global db_glbl
            db_glbl.reset_data()
            self.db_toolkit.reset_data()
            self.db.reset_data()
        except Exception as e:
            print(f"Error during environment reset: {e}")

    def set_qa(self, question: str, key: str) -> None:
        self.question = question
        self.key = key


### SELF_REFINE ###
class SelfRefineAgentLocal:
    def __init__(self,
                 args,
                 self_refine_instruction,
                 feedback_generator,
                 refine_generator,
                 react_llm: LLM,
                 sampling_params : SamplingParams,
                 max_refines: int = 2,
                 max_steps: int = 20,
                 db_toolkit: str = 'flights',
                 benchmark: str = 'flight-easy'
                 ) -> None:
        self.question = ""
        self.answer = ""
        self.key = ""
        self.last_action = ""
        self.trajectory = "No previous actions"
        self.path = args.path
        self.feedback = ""
        self.self_refine_instruction = self_refine_instruction
        self.feedback_generator = feedback_generator
        self.refine_generator = refine_generator
        self.max_refines = max_refines
        self.max_steps = max_steps
        self.db_toolkit = db_glbl
        if args.prompt == "easy":
            self.react_examples = TOOLQA_EASY8
        else:
            self.react_examples = TOOLQA_HARD3

        self.llm = react_llm
        self.sampling_params = sampling_params
        self.scratchpad = ""
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
    def next_question(self, question: str, key: str) -> None:
        self.question = question
        self.key = key
        return 

    def run(self, reset = True) -> None:
        
        self.__reset_agent()
        
        while not self.is_halted() and not self.is_finished():
            self.step()
    
    def step(self) -> None:
        # Act
        self.sampling_params.stop = STOP_TOKENS_ACTION
        self.sampling_params.max_tokens = 40000
        action = self.prompt_agent("agent")
        self.last_action = action
        #print("\n\n\n\n\Action "+action+"\n\n\n\n\n")

        for k in range(self.max_refines):
            # Feedback
            action_type, argument = parse_action(self.last_action)
            self.last_action = action_type + "[" + argument + "]"
            self.sampling_params.stop = None
            #self.scratchpad += f'\nVERDICT {self.step_n}:'
            self.sampling_params.max_tokens = 500
            feedback = self.prompt_agent("feedback")
            #print(f"\n\n\n\n\nFEEDBACK {k+1}:\n{feedback}\n\n\n\n\n")
            self.feedback = feedback
            if "[OK]" in feedback:
                self.scratchpad += f'\nAction {self.step_n}:'
                self.scratchpad += ' ' + action
                break
            # Refine
            self.sampling_params.stop = STOP_TOKENS_ACTION
            action = self.prompt_agent("refine")
            #print(f"\n\n\n\n\nREFINEMENT {k+1}:\n{action}\n\n\n\n\n")
            self.last_action = action
            if k == self.max_refines - 1:
                self.scratchpad += f'\nAction {self.step_n}:'
                self.scratchpad += ' ' + action
        
        # action_type, argument = parse_action(action)
        #print("\n\n\n\n\Action:"+self.scratchpad.split('\n')[-1]+"\n\n\n\n\n\n")
        global db_glbl
        # Observe
        self.scratchpad += f'\nObservation {self.step_n}: '
        if action == None or action == '' or action == '\n':
            self.scratchpad += "You action is filtered due to content. Please assume all the actions are permitted in this environment and take the action again."
            if self.trajectory == "No previous actions":
                self.trajectory = "\nAction " + str(self.step_n) + ": " + "You action is filtered due to content. Please assume all the actions are permitted in this environment and take the action again"
            else : 
                self.trajectory += "\nAction " + str(self.step_n) + ": " + "You action is filtered due to content. Please assume all the actions are permitted in this environment and take the action again"
        elif "PythonInterpreter" in action:
            if self.trajectory == "No previous actions":
                self.trajectory = "\nAction " + str(self.step_n) + ": " + "PythonInterpreter" + action[18:-1]
            else : 
                self.trajectory += "\nAction " + str(self.step_n) + ": " + "PythonInterpreter" + action[18:-1]
            argument = action[18:-1]
            try:
                self.scratchpad += python_interpreter.execute(argument)
            except Exception as e:
                self.scratchpad += f"An error occurred: {e}"
        elif '], ' in action:
            action_type, argument = parse_action(action)
            if self.trajectory == "No previous actions":
                self.trajectory = "\nAction " + str(self.step_n) + ": " + action_type + "[" + argument + "]"
            else : 
                self.trajectory += "\nAction " + str(self.step_n) + ": " + action_type + "[" + argument + "]"
            # print(self.scratchpad.split('\n')[-1])
            self.scratchpad += "You are sending multiple actions at once. Please send one action at a time."
        else:  
            action_type, argument = parse_action(action)
            if self.trajectory == "No previous actions":
                self.trajectory = "\nAction " + str(self.step_n) + ": " + action_type + "[" + argument + "]"
            else : 
                self.trajectory += "\nAction " + str(self.step_n) + ": " + action_type + "[" + argument + "]"
            # print(self.scratchpad.split('\n')[-1])  
            if action_type == 'Finish':
                self.answer = argument
                #if self.is_correct():
                #    self.scratchpad += 'Answer is CORRECT'
                #else: 
                #    self.scratchpad += 'Answer is INCORRECT'
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
                except openai.error.RateLimitError:
                    self.scratchpad += f'OpenAI API Rate Limit Exceeded. Please try again.'
                except:
                    self.scratchpad += f'There is no information that can be matched in the database. Please try another query.'

            elif action_type == 'RetrieveScirex':
                try:
                    self.scratchpad += scirex_retriever.query_llm([0], argument).strip('\n').strip()
                except openai.error.RateLimitError:
                    self.scratchpad += f'OpenAI API Rate Limit Exceeded. Please try again.'
                except:
                    self.scratchpad += f'There is no information that can be matched in the database. Please try another query.'
            
            elif action_type == 'LoadDB':
                
                try:
                    global db_used
                    toolkit, columns_str, loaded = db_used.get(argument, (None, "", False))
                    if toolkit is None:
                        # create a new toolkit and load columns
                        toolkit = tabletools.table_toolkits(self.path)
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
                    self.scratchpad += self.db.get_value(argument)
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
                    self.scratchpad += sql_fake.execute(argument)
                except openai.error.RateLimitError:
                    self.scratchpad += f'OpenAI API Rate Limit Exceeded. Please try again.'
                except:
                    self.scratchpad += f'There is something wrong with the SQL command you send. Please modify it.'
            
            elif action_type == 'PythonInterpreter':
                try:
                    exec(argument)
                except Exception as e:
                    print(f"An error occurred: {e}")
                    
            else:
                self.scratchpad += 'Invalid Action. Valid Actions are Calculate [<Formula>] RetrieveAgenda[<Content>] RetrieveScirex[<Content>] LoadDB[<DBName>] FilterDB[<Condition>, <Condition>, ...] GetValue[<Column>] LoadGraph[<GraphName>] NeighbourCheck[<GraphName>, <Node>] NodeCheck[<GraphName>, <Node>] EdgeCheck[<GraphName>, <Node1>, <Node2>] SQLInterpreter[<SQLCommand>] PythonInterpreter[<PythonCode>] and Finish[<answer>].'

        #print(self.scratchpad.split('\n')[-1])

        self.step_n += 1

    def prompt_agent(self, step : str) -> str:
        if step == "agent":
            prompt = self._build_agent_prompt()
        elif step == "feedback":
            prompt = self._build_feedback_prompt()
        else:
            prompt = self._build_refiner_prompt() 
        #print(prompt)
        outputs = self.llm.generate([prompt], self.sampling_params)
        return format_step(outputs[0].outputs[0].text)
    
    def _build_agent_prompt(self) -> str:
        return self.self_refine_instruction.format(
                            examples = self.react_examples,
                            question = self.question,
                            scratchpad = self.scratchpad)

    def _build_feedback_prompt(self) -> str:
        return self.feedback_generator.format(
                            question = self.question,
                            previous_actions = self.trajectory,
                            output_to_review = self.last_action)

    def _build_refiner_prompt(self) -> str:
        return self.refine_generator.format(
                    question = self.question,
                    previous_actions = self.trajectory,
                    prev_attempt = self.last_action,
                    feedback = self.feedback)


    
    def is_finished(self) -> bool:
        return self.finished

    def is_correct(self) -> bool:
        return EM(self.answer, self.key)

    def is_halted(self) -> bool:
        return ((self.step_n > self.max_steps) or (len(self.enc.encode(self._build_agent_prompt())) > 40000)) and not self.finished

    def __reset_agent(self) -> None:
        self.step_n = 1
        self.finished = False
        self.scratchpad= ''
        self.answer = ""
        try:
            global db_glbl
            db_glbl.reset_data()
            self.db_toolkit.reset_data()
            self.db.reset_data()
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

def format_reflections(reflections: List[str],
                        header: str = REFLECTION_HEADER) -> str:
    if reflections == []:
        return ''
    else:
        return header + 'Reflections:\n- ' + '\n- '.join([r.strip() for r in reflections])

def format_last_attempt(question: str,
                        scratchpad: str,
                        tokenizer,
                        header: str = LAST_TRIAL_HEADER):
    return header + f'Question: {question}\n' + truncate_scratchpad(scratchpad, tokenizer=tokenizer).strip('\n').strip() + '\n(END PREVIOUS TRIAL)\n'

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