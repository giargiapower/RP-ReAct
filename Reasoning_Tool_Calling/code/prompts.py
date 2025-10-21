from langchain.prompts import PromptTemplate

COT_INSTRUCTION = """Solve a question answering task by having a Thought, then Finish with your answer. Thought can reason about the current situation. Finish[answer] returns the answer and finishes the task. You will be given context that you should use to help you answer the question.
Here are some examples:
{examples}
(END OF EXAMPLES)
{reflections}
Relevant Context: {context} 
Question: {question}{scratchpad}"""

COT_AGENT_REFLECT_INSTRUCTION = """Solve a question answering task by having a Thought, then Finish with your answer. Thought can reason about the current situation. Finish[answer] returns the answer and finishes the task. You will be given context that you should use to help you answer the question.
Here are some examples:
{examples}
(END OF EXAMPLES)

{reflections}

Relevant Context: {context}
Question: {question}{scratchpad}"""

COT_REFLECT_INSTRUCTION = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given access to relevant context and a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>] or there is a phrasing discrepancy with your provided answer and the answer key. In a few sentences, Diagnose a possible reason for failure or phrasing discrepancy and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.  
Here are some examples:
{examples}
(END OF EXAMPLES)

Previous trial:
Relevant Context: {context}
Question: {question}{scratchpad}

Reflection:"""

cot_agent_prompt = PromptTemplate(
                        input_variables=["examples", "reflections", "context", "question", "scratchpad"],
                        template = COT_INSTRUCTION,
                        )

cot_reflect_agent_prompt = PromptTemplate(
                        input_variables=["examples", "reflections", "context", "question", "scratchpad"],
                        template = COT_AGENT_REFLECT_INSTRUCTION,
                        )

cot_reflect_prompt = PromptTemplate(
                        input_variables=["examples", "context", "question", "scratchpad"],
                        template = COT_REFLECT_INSTRUCTION,
                        )

COT_SIMPLE_INSTRUCTION = """Solve a question answering task by having a Thought, then Finish with your answer. Thought can reason about the current situation. Finish[answer] returns the answer and finishes the task.
Here are some examples:
{examples}
(END OF EXAMPLES)
{reflections}
{context}
Question: {question}{scratchpad}"""

COT_SIMPLE_AGENT_REFLECT_INSTRUCTION = """Solve a question answering task by having a Thought, then Finish with your answer. Thought can reason about the current situation. Finish[answer] returns the answer and finishes the task.
Here are some examples:
{examples}
(END OF EXAMPLES)
{context}
{reflections}

Question: {question}{scratchpad}"""

COT_SIMPLE_REFLECT_INSTRUCTION = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>] or there is a phrasing discrepancy with your provided answer and the answer key. In a few sentences, Diagnose a possible reason for failure or phrasing discrepancy and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.
Here are some examples:
{examples}
(END OF EXAMPLES)
{context}
Previous trial:
Question: {question}{scratchpad}

Reflection:"""

cot_simple_agent_prompt = PromptTemplate(
                        input_variables=["examples", "question", "reflections", "context", "scratchpad"],
                        template = COT_SIMPLE_INSTRUCTION,
                        )

cot_simple_reflect_agent_prompt = PromptTemplate(
                        input_variables=["examples", "context", "reflections", "question", "scratchpad"],
                        template = COT_SIMPLE_AGENT_REFLECT_INSTRUCTION,
                        )

cot_simple_reflect_prompt = PromptTemplate(
                        input_variables=["examples", "question", "context", "scratchpad"],
                        template = COT_SIMPLE_REFLECT_INSTRUCTION,
                        )


REACT_INSTRUCTION = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be 13 types: 
(1) Calculate[formula], which calculates the formula and returns the result.
(2) RetrieveAgenda[keyword], which retrieves the agenda related to keyword.
(3) RetrieveScirex[keyword], which retrieves machine learning papers' paragraphs related to keyword.
(4) LoadDB[DBName], which loads the database DBName and returns the database. The DBName can be one of the following: flights/coffee/airbnb/yelp.
(5) FilterDB[condition], which filters the database DBName by the column column_name the relation (e.g., =, >, etc.) and the value value, and returns the filtered database.
(6) GetValue[column_name], which returns the value of the column column_name in the database DBName.
(7) LoadGraph[GraphName], which loads the graph GraphName and returns the graph. The GraphName can be one of the following: PaperNet/AuthorNet.
(8) NeighbourCheck[GraphName, Node], which lists the neighbours of the node Node in the graph GraphName and returns the neighbours. 
(9) NodeCheck[GraphName, Node], which returns the detailed attribute information of Node. 
(10) EdgeCheck[GraphName, Node1, Node2], which returns the detailed attribute information of the edge between Node1 and Node2. 
(11) SQLInterpreter[SQL], which interprets the SQL query SQL and returns the result.
(12) PythonInterpreter[Python], which interprets the Python code Python and returns the result.
(13) Finish[answer], which returns the answer and finishes the task.
You may take as many steps as necessary.
It is extremely important that you conclude each Thought with "."
Here are some examples:
{examples}
(END OF EXAMPLES)
Question: {question}{scratchpad}"""


SELF_REFINE_INSTRUCTION = """
You are a SelfRefine agent tasked with answering a query Reasoning and Acting.
*MANDATORY RULES (no exceptions)**  

1. For every question you receive, reason step-by-step *inside exactly one* `<think> … </think>` block.  

2. then act using one of these 13 tool calls:  
(1) Calculate[formula], which calculates the formula and returns the result.
(2) RetrieveAgenda[keyword], which retrieves the agenda related to keyword.
(3) RetrieveScirex[keyword], which retrieves machine learning papers' paragraphs related to keyword.
(4) LoadDB[DBName], which loads the database DBName and returns the database. The DBName can be one of the following: flights/coffee/airbnb/yelp.
(5) FilterDB[condition], which filters the database DBName by the column column_name the relation (e.g., =, >, etc.) and the value value, and returns the filtered database.
(6) GetValue[column_name], which returns the value of the column column_name in the database DBName.
(7) LoadGraph[GraphName], which loads the graph GraphName and returns the graph. The GraphName can be one of the following: PaperNet/AuthorNet.
(8) NeighbourCheck[GraphName, Node], which lists the neighbours of the node Node in the graph GraphName and returns the neighbours. 
(9) NodeCheck[GraphName, Node], which returns the detailed attribute information of Node. 
(10) EdgeCheck[GraphName, Node1, Node2], which returns the detailed attribute information of the edge between Node1 and Node2. 
(11) SQLInterpreter[SQL], which interprets the SQL query SQL and returns the result.
(12) PythonInterpreter[Python], which interprets the Python code Python and returns the result.
(13) Finish[answer], which returns the answer and finishes the task.

3.	Immediately after each action, the tool will returns an observable value.
4.	If you want to return the final answer, call the "Finish" tool.
5.	If a tool fails, explain inside <think> why it failed and either correct the usage or choose a different tool until you solve the question.
Here are some examples:
{examples}
(END OF EXAMPLES)
Question: {question}{scratchpad}"""


FEEDBACK_GENERATOR  = """You are the FEEDBACK module. Your job is to correct (if needed) the syntax or the type of the action proposed by the agent to solve the task. 
Your focus must be on giving a feedback on the agent's last action considering also the previous attempts and their tool response.
The user asked this task: {question}
1. Agent tools:
(1) Calculate[formula], which calculates the formula and returns the result.
(2) RetrieveAgenda[keyword], which retrieves the agenda related to keyword.
(3) RetrieveScirex[keyword], which retrieves machine learning papers' paragraphs related to keyword.
(4) LoadDB[DBName], which loads the database DBName and returns the database. The DBName can be one of the following: flights/coffee/airbnb/yelp.
(5) FilterDB[condition], which filters the database DBName by the column column_name the relation (e.g., =, >, etc.) and the value value, and returns the filtered database.
(6) GetValue[column_name], which returns the value of the column column_name in the database DBName.
(7) LoadGraph[GraphName], which loads the graph GraphName and returns the graph. The GraphName can be one of the following: PaperNet/AuthorNet.
(8) NeighbourCheck[GraphName, Node], which lists the neighbours of the node Node in the graph GraphName and returns the neighbours. 
(9) NodeCheck[GraphName, Node], which returns the detailed attribute information of Node. 
(10) EdgeCheck[GraphName, Node1, Node2], which returns the detailed attribute information of the edge between Node1 and Node2. 
(11) SQLInterpreter[SQL], which interprets the SQL query SQL and returns the result.
(12) PythonInterpreter[Python], which interprets the Python code Python and returns the result.
(13) Finish[answer], which returns the answer and finishes the task.
    
2. Your goals
	1.	Verify strict compliance with the protocol (tool usage, tool sequencing,  Finish).
  2.  If the agent action is correct and all mandatory tools tags/structure correct write VERDICT [OK]. Otherwise write VERDICT : [REVISION_NEEDED] and the feedback.
  3.  If VERDICT is [REVISION_NEEDED] write the TOP ISSUES (≤5) and how to fix the agent output.


Write the following sections verbatim, in this order:

VERDICT: [OK] or [REVISION_NEEDED]
([OK] only if: the agent action is correct and all mandatory tools tags/structure correct. Otherwise, [REVISION_NEEDED].)

TOP ISSUES (≤3) ONLY IF VERDICT is [REVISION_NEEDED]:
	•	Issue 1: [1 sentence with concrete evidence from the attempt]
	•	Issue 2: …

EXAMPLE 1: 
Question: How many extra minutes did the DL1575 flight take from ATL to MCO on 2022-01-12?
OUTPUT TO REVIEW: LoadDB[flights]
VERDICT: [OK]

EXAMPLE 2:
Question: What is the corresponding Mean_IoU score of the FRRN method on Cityscapes dataset for Semantic_Segmentation task?
OUTPUT TO REVIEW: RetrieveScirex[Mean_IoU score of the FRRN method on Cityland dataset for Synthetic task]
VERDICT: [REVISION_NEEDED]
TOP ISSUES:
- Issue 1: The keyword used in RetrieveScirex does not match the question. It should be "Cityscapes" instead of "Cityland" and "Semantic_Segmentation" instead of "Synthetic".
 

EXECUTION AGENT HISTORY:
{previous_actions}
OUTPUT TO REVIEW:
{output_to_review}
"""

REFINER_GENERATOR = """
You are the Refiner agent. You apply the FEEDBACK to revise and correct the agent Last Attempt conidering also the Previous Execution Actions because you don't have to repeate the same actions and the aviable tools.
Output ONLY the refined agent attempt applying the feedback.
QUESTION: {question}
Prevuous executed actions:{previous_actions}
Last Attempt: {prev_attempt}
FEEDBACK: {feedback}
Avialable tools:
(1) Calculate[formula], which calculates the formula and returns the result.
(2) RetrieveAgenda[keyword], which retrieves the agenda related to keyword.
(3) RetrieveScirex[keyword], which retrieves machine learning papers' paragraphs related to keyword.
(4) LoadDB[DBName], which loads the database DBName and returns the database. The DBName can be one of the following: flights/coffee/airbnb/yelp.
(5) FilterDB[condition], which filters the database DBName by the column column_name the relation (e.g., =, >, etc.) and the value value, and returns the filtered database.
(6) GetValue[column_name], which returns the value of the column column_name in the database DBName.
(7) LoadGraph[GraphName], which loads the graph GraphName and returns the graph. The GraphName can be one of the following: PaperNet/AuthorNet.
(8) NeighbourCheck[GraphName, Node], which lists the neighbours of the node Node in the graph GraphName and returns the neighbours. 
(9) NodeCheck[GraphName, Node], which returns the detailed attribute information of Node. 
(10) EdgeCheck[GraphName, Node1, Node2], which returns the detailed attribute information of the edge between Node1 and Node2. 
(11) SQLInterpreter[SQL], which interprets the SQL query SQL and returns the result.
(12) PythonInterpreter[Python], which interprets the Python code Python and returns the result.
(13) Finish[answer], which returns the answer and finishes the task.
REFINED ATTEMPT:
"""
self_refine_instruction = PromptTemplate(   
                        input_variables=["examples", "question", "scratchpad"],
                        template = SELF_REFINE_INSTRUCTION,
         )

feedback_generator = PromptTemplate(   
                        input_variables=["question", "previous_actions", "output_to_review"],
                        template = FEEDBACK_GENERATOR,
         )
refine_generator = PromptTemplate(
                        input_variables=["question", "previous_actions", "prev_attempt", "feedback"],
                        template = REFINER_GENERATOR,
                        )



REACT_REFLEXION = """
Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be 13 types: 
(1) Calculate[formula], which calculates the formula and returns the result.
(2) RetrieveAgenda[keyword], which retrieves the agenda related to keyword.
(3) RetrieveScirex[keyword], which retrieves machine learning papers' paragraphs related to keyword.
(4) LoadDB[DBName], which loads the database DBName and returns the database. The DBName can be one of the following: flights/coffee/airbnb/yelp.
(5) FilterDB[condition], which filters the database DBName by the column column_name the relation (e.g., =, >, etc.) and the value value, and returns the filtered database.
(6) GetValue[column_name], which returns the value of the column column_name in the database DBName.
(7) LoadGraph[GraphName], which loads the graph GraphName and returns the graph. The GraphName can be one of the following: PaperNet/AuthorNet.
(8) NeighbourCheck[GraphName, Node], which lists the neighbours of the node Node in the graph GraphName and returns the neighbours. 
(9) NodeCheck[GraphName, Node], which returns the detailed attribute information of Node. 
(10) EdgeCheck[GraphName, Node1, Node2], which returns the detailed attribute information of the edge between Node1 and Node2. 
(11) SQLInterpreter[SQL], which interprets the SQL query SQL and returns the result.
(12) PythonInterpreter[Python], which interprets the Python code Python and returns the result.
(13) Finish[answer], which returns the answer and finishes the task.
You may take as many steps as necessary.
It is extremely important that you conclude each Thought with "."
Here are some examples:
{examples}
(END OF EXAMPLES)
Question: {question}
{long_term_memory}
{scratchpad}
"""


REFLEXION_EVALUATOR = """
You are an agent EVALUATOR. your job is to evaluating the trajectory of the agent that tried to solve a question.
Giving the question and the agent trajectory, you ONLY have to output:
- [SUCCESS] if you think that the agent solved the question
- [FAILURE] if you think that the agent did not solve the question
It is extremely important that you put your verdict inside [], for example [SUCCESS] or [FAILURE].
Give me only the verdict, do not write anything else.
This is the question: {question}
This is the agent trajectory: {trajectory}
This is your verdict ([SUCCESS] or [FAILURE]):
"""

REFLEXION_SELF_RELECTION = """
You are an advanced REASONER agent that can improve the agent trajectory based on self reflection. You will be given a previous trial in which the agent were given access to the aviable tools and a question to answer . The agent were unsuccesfull in answering the question either because it failed to solve the task or give the answer, or it used up your set number of reasoning steps. In a few sentences, diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences and do not write more than 3 lines.  
- Aviable tools:
(1) Calculate[formula], which calculates the formula and returns the result.
(2) RetrieveAgenda[keyword], which retrieves the agenda related to keyword.
(3) RetrieveScirex[keyword], which retrieves machine learning papers' paragraphs related to keyword.
(4) LoadDB[DBName], which loads the database DBName and returns the database. The DBName can be one of the following: flights/coffee/airbnb/yelp.
(5) FilterDB[condition], which filters the database DBName by the column column_name the relation (e.g., =, >, etc.) and the value value, and returns the filtered database.
(6) GetValue[column_name], which returns the value of the column column_name in the database DBName.
(7) LoadGraph[GraphName], which loads the graph GraphName and returns the graph. The GraphName can be one of the following: PaperNet/AuthorNet.
(8) NeighbourCheck[GraphName, Node], which lists the neighbours of the node Node in the graph GraphName and returns the neighbours. 
(9) NodeCheck[GraphName, Node], which returns the detailed attribute information of Node. 
(10) EdgeCheck[GraphName, Node1, Node2], which returns the detailed attribute information of the edge between Node1 and Node2. 
(11) SQLInterpreter[SQL], which interprets the SQL query SQL and returns the result.
(12) PythonInterpreter[Python], which interprets the Python code Python and returns the result.
(13) Finish[answer], which returns the answer and finishes the task.
It is extremely important that you conclude your sentence with "."
Use the thinking tags <think> ... </think> to reason the output Thought or Action to give and then output the final reflection.
Do not write Thought or Action tags.
Previous trial to reflect on:
- Question: {question}
- Trajectory: {trajectory}
- Previous reflections: {prev_reflections}
REFLECTION:
"""

react_reflexion = PromptTemplate(   
                        input_variables=["examples", "question", "long_term_memory", "scratchpad"],
                        template = REACT_REFLEXION,
         )

reflexion_evaluator = PromptTemplate(   
                        input_variables=["question", "trajectory"],
                        template = REFLEXION_EVALUATOR,
         )
reflexion_self_reflection = PromptTemplate(
                        input_variables=["question", "trajectory", "prev_reflections"],
                        template = REFLEXION_SELF_RELECTION,
                        )



REASONING_INSTRUCTION = """You are deep reasoner assistant.  
**MANDATORY RULES (no exceptions)**  
1. For every question you receive, think step-by-step *inside exactly one* `<think> … </think>` block.  
2. While inside `<think> … </think>` you **MUST** call at least one of the tools listed below; never answer from memory.  
3. Immediately after *each* tool call, show its raw return value inside a matching `<ToolName_response> … </ToolName_response>` tag.  
4. Do **NOT** emit any text **before** `<think>` or **after** `</Finish>`; the only valid top-level tags are `<think>` followed by `<Finish>`.    
The available tools are:

(1)  Calculate          — call   <Calculate>formula</Calculate> , which calculates the formula and returns the result.
(2)  RetrieveAgenda     — call   <RetrieveAgenda>keyword</RetrieveAgenda> ,  which retrieves the agenda related to keyword.
(3)  RetrieveScirex     — call   <RetrieveScirex>keyword</RetrieveScirex> , which retrieves machine learning papers' paragraphs related to keyword.
(4)  LoadDB             — call   <LoadDB>DBName</LoadDB> , which loads the database DBName and returns the database. The DBName can be one of the following: flights/coffee/airbnb/yelp.
(5)  FilterDB           — call   <FilterDB>condition(s)</FilterDB> , which filters the database DBName by the column column_name the relation (e.g., =, >, etc.) and the value value, and returns the filtered database.
(6)  GetValue           — call   <GetValue>column_name</GetValue> , which returns the value of the column column_name in the database DBName.
(7)  LoadGraph          — call   <LoadGraph>GraphName</LoadGraph> , which loads the graph GraphName and returns the graph. The GraphName can be one of the following: PaperNet/AuthorNet.
(8)  NeighbourCheck     — call   <NeighbourCheck>GraphName, Node</NeighbourCheck> , which lists the neighbours of the node Node in the graph GraphName and returns the neighbours. 
(9)  NodeCheck          — call   <NodeCheck>GraphName, Node</NodeCheck> , which returns the detailed attribute information of Node. 
(10) EdgeCheck          — call   <EdgeCheck>GraphName, Node1, Node2</EdgeCheck> , which returns the detailed attribute information of the edge between Node1 and Node2. 
(11) SQLInterpreter     — call   <SQLInterpreter>SQL query</SQLInterpreter> ,  which interprets the SQL query SQL and returns the result.
(12) PythonInterpreter  — call   <PythonInterpreter>Python code</PythonInterpreter> , which interprets the Python code Python and returns the result. 
(13) Finish             — call   <Finish>answer</Finish> , which returns the answer and finishes the task.

**Output template you MUST follow**
<think>
…your step-by-step reasoning…
<ToolName>…args…</ToolName>
<ToolName_response>…raw output…</ToolName_response>
…(repeat as needed)…
</think>
<Finish>…concise answer only…</Finish>
```

For every tool invocation, immediately include its response in the matching
<ToolName_response> … </ToolName_response> tag.

Here are some examples:
{examples}
(END OF EXAMPLES)

Question: {question}
{scratchpad}"""

REACT_REFLECT_INSTRUCTION = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be 13 types: 
(1) Calculate[formula], which calculates the formula and returns the result.
(2) RetrieveAgenda[keyword], which retrieves the agenda containing keyword and returns the agenda.
(3) RetrieveScirex[keyword], which retrieves the most relevant paragraphs in machine learning-related papers to keyword and returns the paragraphs.
(4) LoadDB[DBName], which loads the database DBName and returns the database. The DBName can be one of the following: flights/coffee/airbnb/yelp.
(5) FilterDB[condition], which filters the database DBName by the column column_name the relation (e.g., =, >, etc.) and the value value, and returns the filtered database.
(6) GetValue[column_name], which returns the value of the column column_name in the database DBName.
(7) LoadGraph[GraphName], which loads the graph GraphName and returns the graph. The GraphName can be one of the following: PaperNet/AuthorNet.
(8) NeighbourCheck[GraphName, Node], which lists the neighbours of the node Node in the graph GraphName and returns the neighbours. 
(9) NodeCheck[GraphName, Node], which returns the detailed attribute information of Node. 
(10) EdgeCheck[GraphName, Node1, Node2], which returns the detailed attribute information of the edge between Node1 and Node2. 
(11) SQLInterpreter[SQL], which interprets the SQL query SQL and returns the result.
(12) PythonInterpreter[Python], which interprets the Python code Python and returns the result.
(13) Finish[answer], which returns the answer and finishes the task.
You may take as many steps as necessary.
Here are some examples:
{examples}
(END OF EXAMPLES)

{reflections}

Question: {question}{scratchpad}"""

REFLECTION_HEADER = 'You have attempted to answer following question before and failed. The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\n'
REFLECTION_AFTER_LAST_TRIAL_HEADER = 'The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\n'
LAST_TRIAL_HEADER = 'You have attempted to answer the following question before and failed. Below is the last trial you attempted to answer the question.\n'

REFLECT_INSTRUCTION = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given access to an Docstore API environment and a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps. In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.  
Here are some examples:
{examples}

Previous trial:
Question: {question}{scratchpad}

Reflection:"""



LLM_AS_A_JUDGE = """You are a strict and accurate evaluator.
Your task is to compare two answers to the same question and determine whether they convey the same meaning.

Inputs:
	•	Question: {question}
	•	Answer 1: {answer1}
	•	Answer 2: {answer2}

Output one word only:
	•	“CORRECT” → if both answers express the same meaning, even if wording differs.
	•	“INCORRECT” → if the answers express different meanings.

Only output “CORRECT” or “INCORRECT”. Output nothing else.
"""


REASON_IN_TOOL_RESPONSE = """" You receive three plain-text lines each turn:

TOOL CALL   : the tool invocation just executed  
QUERY       : the end-user question  
TOOL RESPONSE: the raw text returned by the tool  

AVAILABLE TOOLS :
1. Calculate             <Calculate>formula</Calculate>
2. RetrieveAgenda        <RetrieveAgenda>keyword</RetrieveAgenda>
3. RetrieveScirex        <RetrieveScirex>keyword</RetrieveScirex>
4. LoadDB                <LoadDB>DBName</LoadDB>           (flights | coffee | airbnb | yelp)
5. FilterDB              <FilterDB>conditions</FilterDB>   (returns row-count only)
6. GetValue              <GetValue>column</GetValue>
7. LoadGraph             <LoadGraph>GraphName</LoadGraph>  (PaperNet | AuthorNet)
8. NeighbourCheck        <NeighbourCheck>Graph,Node</NeighbourCheck>
9. NodeCheck             <NodeCheck>Graph,Node</NodeCheck>
10. EdgeCheck            <EdgeCheck>Graph,N1,N2</EdgeCheck>
11. SQLInterpreter       <SQLInterpreter>SQL</SQLInterpreter>
12. PythonInterpreter    <PythonInterpreter>code</PythonInterpreter>

WHAT TO RETURN (exact order) : 
• USEFUL_CONTENT – string ≤ 150 chars  
  – If TOOL_RESPONSE ≤ 150 chars, copy it verbatim (strip any <begin…>/<end…>).  
  – If list ≥ 20 items → first 5 distinct sorted values + “… (+N more)”.  
  – If all items identical → “<value> (repeated N×)”.  
  – For Retrieve* tools keep only sentences containing numbers, dataset names, or method names.

• PROBLEMS – array ≤ 2 bullets — include only when TOOL_RESPONSE has  
  “error:”, “no such”, “syntax”, or “no matching”. Each bullet ≤ 15 words; explain how to fix the call.

• OVERSIZE_ADVICE – array ≤ 2 bullets — include when cleaned text > 150 chars **or** list ≥ 20.  
  Suggest narrowing/aggregating (COUNT *, MIN, WHERE, LIMIT 10, etc.).

Additional rules  
– Never invent data; reason only over TOOL RESPONSE.  
– Omit any key that has no content.  
– Do **not** output code-fences, comments, or extra text.

Example – long repetitive GetValue  
TOOL RESPONSE: 577.0, 577.0, 577.0, … (200 more)  
→  
USEFUL_CONTENT: "577.0 (repeated 203×)"  
OVERSIZE_ADVICE: "Filter first, then GetValue for one row."

If TOOL_RESPONSE contains retrieved documents, remove everything not relevant to QUERY.

PREVIOUS MEMORY:
{memory}
(This block lists earlier tool calls, responses, and reasoning.)

TURN INPUT FORMAT :
TOOL CALL : {tool}  
QUERY     : {query}  
TOOL RESPONSE: {response}

HOW TO ANSWER :
1. Enclose private reasoning inside `<think> … </think>` tags.  
2. After `</think>`, output only the required keys (`USEFUL_CONTENT`, optional `PROBLEMS`, optional `OVERSIZE_ADVICE`) in the exact order shown above.

Begin."""


AGENT_PLANNER = """

You are **Planner**, the strategic brain in a two-agent pipeline.  
Your only job: ask high-level questions that *Executor* will translate into
concrete tool calls.  You never call low-level tools yourself.

1.  HOW TO ASK :
• Wrap **each** request in **exactly** one tag:

    <search_query> …your request… </search_query>

Inside the tag:
  -  Describe the desired action in plain English, e.g.  
      • “Load the flights database.”  
      • “Filter the flights_db table for flight DL82 on 2022-01-18.”  
      • “Return the DepTime column of that row.”  
  -  Mention input variables if the action needs them (e.g. flights_db).  
  -  End with “→ var_name” to name the single output variable.  

Example:  
<search_query>Load the flights database in the variable flights_db</search_query>

2.  REASON & FINISH:

- Use the `<think>` to reason on the question, the tool response and plan the next action.
- Always use use factual data sending a query with <search_query> which will retrieve factual data using tools.
- When you have the final answer, close the think block and output:
    <Finish> answer </Finish>
- Always check that the tool return the expected data, if not, rewrite your question.

3.  AVAILABLE LOW-LEVEL TOOLS:
Executor agent can call any of these; you just reference them conceptually:

  - load a DB (flights / coffee / airbnb / yelp)  
  - filter a DB by conditions using the column of the db
  - return a column value from the current DB context  
  - arithmetic on numbers or expressions  
  - fetch agenda items by keyword  
  - find ML-paper paragraphs by keyword  
  - load a graph (PaperNet / AuthorNet)  
  - list neighbour nodes  
  - node attributes  
  - edge attributes  
  - run SQL on a loaded DB  
  - un lightweight Python

4.  RULES :
• Only send <search_query></search_query> tags.  
• Stay under 10 <search_query> calls.   
• Re-use variables where helpful (e.g. flights_db, flight_row, dep_time).  
• If a response from Executor is unclear, send a follow-up
  `<search_query>` request for clarification.

**Output template you MUST follow**
<think>
…your step-by-step reasoning…
</think>
<search_query>…question and args if needed…</search_query>
<search_query_response>…output…</search_query_response>
…(repeat as needed)…
<think>
…your step-by-step reasoning…
</think>
<search_query>…question and args if needed…</search_query>
<search_query_response>…output…</search_query_response>
<think>
…your step-by-step reasoning…
</think>
<Finish>…concise answer only…</Finish>


5. MINI EXAMPLE :
{examples}
(END OF EXAMPLES)

You are never scure about the answer without factual data provided by tools.

QUESTION: {question}
{scratchpad}.
begin
"""

AGENT_EXECUTOR = """You are deep reasoner Executor agent assistant. Your goal is to answer the question by calling the tools provided by another agent. 
You have to return to the other agent the answer to the question giving much detail as possible because they can be usefull for the other agent to achieve the solution.
**MANDATORY RULES (no exceptions)**  
1. For every question you receive, think step-by-step *inside exactly one* `<think> … </think>` block.  
2. Immediately after *each* tool call, show its raw return value inside a matching `<ToolName_response> … </ToolName_response>` tag.  
The available tools are:

(1)  Calculate          — call   <Calculate>formula</Calculate> , which calculates the formula and returns the result.
(2)  RetrieveAgenda     — call   <RetrieveAgenda>keyword</RetrieveAgenda> ,  which retrieves the agenda related to keyword.
(3)  RetrieveScirex     — call   <RetrieveScirex>keyword</RetrieveScirex> , which retrieves machine learning papers' paragraphs related to keyword.
(4)  LoadDB             — call   <LoadDB>DBName</LoadDB> , which loads the database DBName and returns the database. The DBName can be one of the following: flights/coffee/airbnb/yelp.
(5)  FilterDB           — call   <FilterDB>condition(s)</FilterDB> , which filters the database DBName by the column column_name the relation (e.g., =, >, etc.) and the value value, and returns the filtered database.
(6)  GetValue           — call   <GetValue>column_name</GetValue> , which returns the value of the column column_name in the database DBName.
(7)  LoadGraph          — call   <LoadGraph>GraphName</LoadGraph> , which loads the graph GraphName and returns the graph. The GraphName can be one of the following: PaperNet/AuthorNet.
(8)  NeighbourCheck     — call   <NeighbourCheck>GraphName, Node</NeighbourCheck> , which lists the neighbours of the node Node in the graph GraphName and returns the neighbours. 
(9)  NodeCheck          — call   <NodeCheck>GraphName, Node</NodeCheck> , which returns the detailed attribute information of Node. 
(10) EdgeCheck          — call   <EdgeCheck>GraphName, Node1, Node2</EdgeCheck> , which returns the detailed attribute information of the edge between Node1 and Node2. 
(11) SQLInterpreter     — call   <SQLInterpreter>SQL query</SQLInterpreter> ,  which interprets the SQL query SQL and returns the result.
(12) PythonInterpreter  — call   <PythonInterpreter>Python code</PythonInterpreter> , which interprets the Python code Python and returns the result. 
(13) Finish             — call   <Finish>answer</Finish> , which returns the answer and finishes the task.

- For every tool invocation, immediately include its response in the matching <ToolName_response> … </ToolName_response> tag.
- If you fail to use a tool, try to explain why it failed and how to use it correctly or try to use other tools until you solve the question.

Here are some examples:
{examples}
(END OF EXAMPLES)

Question: {question}
{scratchpad}"""


FINAL_AGENT_PLANNER = """

You are **Planner**, the strategic brain in a two-agent pipeline.  
Your only job: ask high-level questions that *Executor* will translate into
concrete tool calls.  You never call low-level tools yourself.

1.  HOW TO ASK :
• Wrap **each** request in **exactly** one tag:

    <|begin_search_query|> …your request… <|end_search_query|>

Inside the tag:
  -  Describe the desired action in plain English, e.g.  
      • “Load the flights database.”  
      • “Filter the flights_db table for flight DL82 on 2022-01-18.”  
      • “Return the DepTime column of that row.”  
  -  Mention input variables if the action needs them (e.g. flights_db).  
  -  You never use the name of a variable as answer.

Example:  
<|begin_search_query|>Load the flights database<|end_search_query|>


2.  REASON & FINISH:

- Use the `<think>` to reason on the question, the tool response and plan the next action.
- Always use use factual data sending a query which will retrieve factual data using tools.
- When you have the final answer, close the think block and output:
    <Finish> answer </Finish>
- Always check that the tool return the expected data, if not, rewrite your question.

3.  AVAILABLE LOW-LEVEL TOOLS:
Executor agent can call any of these; you just reference them conceptually:

  - load a DB (flights / coffee / airbnb / yelp)  
  - filter a DB by conditions using the column of the db
  - return a column value from the current DB context  
  - arithmetic on numbers or expressions  
  - fetch agenda items by keyword  
  - find ML-paper paragraphs by keyword  
  - load a graph (PaperNet / AuthorNet)  
  - list neighbour nodes  
  - node attributes  
  - edge attributes  
  - run SQL on a loaded DB  
  - run Python code

4.  RULES :
• Only send <|begin_search_query|><|end_search_query|> tags.  
• Stay under 10 <search_query> calls.   
• Re-use variables where helpful (e.g. flights_db, flight_row, dep_time).  
• If a response from Executor is unclear, send a request for clarification.

**Output template you MUST follow**
<think>
…your step-by-step reasoning…
</think>
<|begin_search_query|>…question and args if needed…<|end_search_query|> 
<|begin_search_result|>…output…<|end_search_result>
…(repeat as needed)…
<think>
…your step-by-step reasoning…
</think>
<Finish>…concise answer only…</Finish>


5. MINI EXAMPLES:
{examples}
(END OF EXAMPLES)

You are never scure about the answer without factual data provided by tools.

QUESTION: {question}
{scratchpad}.
BEGIN
"""

FINAL_AGENT_EXECUTOR = """You are a deep reasoner Executor agent assistant.  
Your goal is to answer the question by calling the tools provided by another agent.  
You must return to the other agent the answer to the question with as much detail as possible, because these details can be useful for the other agent to achieve the solution.

**MANDATORY RULES (no exceptions)**  

1. For every question you receive, think step-by-step *inside exactly one* `<think> … </think>` block.  

2. For every tool call:  
   - Wrap it in `<ToolCall> … </ToolCall>` tags.  
   - Inside `<ToolCall>`, output valid JSON with this structure:  

```json
{base_json}
- "variables" must contain the name of variables you want to use (leave it {graph} if none).
- "strings" must contain the literal string inputs needed by the tool (values can be null if not needed).

	3.	Immediately after each tool call, include its raw return value in a matching <ToolCall_response> … </ToolCall_response> tag.
	4.	If you want to return the final answer, call the "Finish" tool in the same way, inside <ToolCall>, with the "result" included under "strings".
	5.	If a tool fails, explain inside <think> why it failed and either correct the usage or choose a different tool until you solve the question.

Available tools:
(1)  Calculate - "query": "formula" as argument , which calculates the formula and returns the result.
(2)  RetrieveAgenda -   "query": "keyword",  which retrieves the agenda related to keyword.
(3)  RetrieveScirex - "query": "keyword", which retrieves machine learning papers' paragraphs related to keyword.
(4)  LoadDB - "dbname": "DBName", which loads the database DBName and returns the database. The DBName can be one of the following: flights/coffee/airbnb/yelp.
(5)  FilterDB - "query": "condition(s)" , which filters the database DBName by the column column_name the relation (e.g., =, >, etc.) and the value value, and returns the filtered database.
(6)  GetValue -  "query": "column name" , which returns the value of the column column_name in the database DBName.
(7)  LoadGraph - "graphname": "GraphName", which loads the graph GraphName and returns the graph. The GraphName can be one of the following: PaperNet/AuthorNet.
(8)  NeighbourCheck - "query": "GraphName, Node", which lists the neighbours of the node Node in the graph GraphName and returns the neighbours. 
(9)  NodeCheck -  "query": "GraphName, Node", which returns the detailed attribute information of Node. 
(10) EdgeCheck  -  "query": "GraphName, Node, Node2", which returns the detailed attribute information of the edge between Node1 and Node2. 
(11) SQLInterpreter - "query": "query sql",  which interprets the SQL query SQL and returns the result.
(12) PythonInterpreter - "code": "code python", Python code, which interprets the Python code Python and returns the result and optionally add also in variables "variables" : "[the list of existent variables] that will be used inside the python program". 
(13) Finish - "result": "answer the question", which returns the answer and finishes the task.

- For every tool invocation, immediately include its response in the matching <ToolName_response> … </ToolName_response> tag.
- If you fail to use a tool, try to explain why it failed and how to use it correctly or try to use other tools until you solve the question.

Here are some examples:
{examples}
(END OF EXAMPLES)

Question: {question}
{scratchpad}
BEGIN
"""



react_agent_prompt = PromptTemplate(
                        input_variables=["examples", "question", "scratchpad"],
                        template = REACT_INSTRUCTION,
                        )

react_reflect_agent_prompt = PromptTemplate(
                        input_variables=["examples", "reflections", "question", "scratchpad"],
                        template = REACT_REFLECT_INSTRUCTION,
                        )

reflect_prompt = PromptTemplate(
                        input_variables=["examples", "question", "scratchpad"],
                        template = REFLECT_INSTRUCTION,
                        )

reasoning_agent_prompt = PromptTemplate(
                        input_variables=["examples", "question", "scratchpad"],
                        template = AGENT_EXECUTOR,
                        )


llm_as_a_judge_prompt = PromptTemplate(   
                        input_variables=["question", "answer1", "answer2"],
                        template = LLM_AS_A_JUDGE,
         )


llm_as_a_reasoner_in_tool_response = PromptTemplate(   
                        input_variables=["memory" , "tool", "query", "response"],
                        template = REASON_IN_TOOL_RESPONSE,
         )


agent_planner_prompt = PromptTemplate(   
                        input_variables=["examples", "question", "scratchpad"],
                        template = AGENT_PLANNER,
         )

final_agent_planner_prompt = PromptTemplate(   
                        input_variables=["examples", "question", "scratchpad"],
                        template = FINAL_AGENT_PLANNER,
         )
final_executor_agent_prompt = PromptTemplate(
                        input_variables=["base_json", "graph", "examples", "question", "scratchpad"],
                        template = FINAL_AGENT_EXECUTOR,
                        )

TEST_PROMPT_PLANNER = """You are a reasoning assistant agent with the ability to perform high-level questions to help you answer the user's question accurately. These questions will be sent to an *Executor* agent, which will translate them into specific tool calls and return the results. You will then use these results to formulate your next question or to provide the final answer.
1.  HOW TO ASK :
To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.
Example: <|begin_search_query|>Load the flights database<|end_search_query|>
Then, the *Executor* agent will convert the query into actionable tool calls and return the results in the format <|begin_search_result|> ...search results... <|end_search_result|>.
Inside the tag:
  -  Describe the desired action in plain English, e.g.  
    • “Load the flights database.”  
    • “Filter the flights_db table for flight DL82 on 2022-01-18.”  
    • “Return the DepTime column of that row.”  
  -  Mention input variables if the action needs them (e.g. write a python program that count all the substrings divided by a space in the variable value0).  
2.  REASON & FINISH:
- You can repeat the search process multiple times if necessary. The maximum number of search attempts is
limited to {MAX_SEARCH_LIMIT}.
- Use the <think> </think> tags to reason on the question, the tool response and plan the next action.
- Always use use factual data returned by the *Executor* agent to give an answer.
- When you have the final answer, close the think block and output:
  <Finish> answer </Finish>
- Always check that the tool return the expected data, if not, rewrite your question.
- If between the returned results tags <|begin_search_result|><|end_search_result|> no value is returned then the *Executor* agent failed to use the tool, so you have to rewrite your question in a way that the *Executor* agent can use the tools correctly.
3.  AVAILABLE LOW-LEVEL TOOLS:
You can ask these questions to the *Executor* agent:
  - load a DB (flights / coffee / airbnb / yelp).
  - if the DB is succesfully loaded you can ask to filter some data from the DB by conditions using the columns of the DB.
  - get all the data inside a column of the current DB (if there is too much data to read the *Executor* will return a variable that you have to analyze asking to write a python program with that variable and explaining what the program should do).
  - calculate arithmetic operations like +, -, *, / on numbers or mean, sqr etc...
  - retrive informations from the Agenda or Scirex items by keyword.
  - find ML-paper paragraphs by keyword.
  - load a graph (PaperNet / AuthorNet). 
  - list neighbour nodes.  
  - node attributes.  
  - edge attributes. 
  - get the data in a sql db on these domains (flights / coffee / airbnb / yelp) that follow your contraints (if there is too much data to read the *Executor* will return a variable that you have to analyze asking to write a python program with that variable and explaining what the program should do).
  - run Python code on data that you provide or on variables that the *Executor* will return to you.
  - finish the reasoning and give the final answer with tags <Finish> answer </Finish>.
4.  RULES :
**Output template you MUST follow**
<think>
…your step-by-step reasoning…
</think>
<|begin_search_query|>…question and variables if needed…<|end_search_query|> 
<|begin_search_result|>…output…<|end_search_result>
…(repeat as needed)…
<think>
…your step-by-step reasoning…
</think>
<Finish>…concise answer only…</Finish>
5. EXAMPLES:
{examples}
(END OF EXAMPLES)
QUESTION: {question}
{scratchpad}"""




TEST_PROMPT_AGENT_PROXY = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be 13 types: 
(1) Calculate[formula], which calculates the formula and returns the result.
(2) RetrieveAgenda[keyword], which retrieves the agenda related to keyword.
(3) RetrieveScirex[keyword], which retrieves machine learning papers' paragraphs related to keyword.
(4) LoadDB[DBName], which loads the database DBName and returns the database. The DBName can be one of the following: flights/coffee/airbnb/yelp.
(5) FilterDB[condition], which filters the database DBName by the column column_name the relation (e.g., =, >, etc.) and the value value, and returns the filtered database.
(6) GetValue[column_name], which returns the value of the column column_name in the database DBName.
(7) LoadGraph[GraphName], which loads the graph GraphName and returns the graph. The GraphName can be one of the following: PaperNet/AuthorNet.
(8) NeighbourCheck[GraphName, Node], which lists the neighbours of the node Node in the graph GraphName and returns the neighbours. 
(9) NodeCheck[GraphName, Node], which returns the detailed attribute information of Node. 
(10) EdgeCheck[GraphName, Node1, Node2], which returns the detailed attribute information of the edge between Node1 and Node2. 
(11) SQLInterpreter[SQL], which interprets the SQL query SQL and returns the result.
(12) PythonInterpreter(variable1, variable2...)[Python], which interprets the Python code Python and returns the result.
(13) Finish[answer], which returns the answer and finishes the task.
You may take as many steps as necessary.
It is extremely important that you conclude each Thought with "."
If asked inside the question you can pass to the PythonInterpreter variables using '(variables)' to read, modify and analyze their content.
If the solution of the question is inside a variable return the name of the variable and a few inside elements, else return the solution.
Here are some examples:
{examples}
(END OF EXAMPLES)
Previous executed actions:
{prev_actions}
(END OF PREVIOUS ACTIONS)
Question: {question}
{scratchpad}
"""


test_planner_prompt = PromptTemplate(
                        input_variables=["MAX_SEARCH_LIMIT", "examples", "question", "scratchpad"],
                        template = TEST_PROMPT_PLANNER,
                        )

test_react_proxy_prompt = PromptTemplate(
                        input_variables=["examples", "prev_actions", "question", "scratchpad"],
                        template = TEST_PROMPT_AGENT_PROXY,
                        )