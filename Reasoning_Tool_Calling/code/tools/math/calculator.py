'''
input: formula strings
output: the answer of the mathematical formula
'''
import os
import re
from operator import pow, truediv, mul, add, sub
import wolframalpha
query = '1+2*3'

def calculator(query: str):
    operators = {
        '+': add,
        '-': sub,
        '*': mul,
        '/': truediv,
    }
    query = re.sub(r'\s+', '', query)
    if query.isdigit() or (query[0] == '-' and query[1:].isdigit()):
        return float(query)
    
    # Handle parentheses, brackets, and braces
    bracket_pairs = {'(': ')', '[': ']', '{': '}'}
    for open_bracket, close_bracket in bracket_pairs.items():
        while open_bracket in query:
            # Find the innermost bracket pair
            start = query.rfind(open_bracket)
            if start == -1:
                break
            # Find the corresponding closing bracket
            end = query.find(close_bracket, start)
            if end == -1:
                return "Invalid input: mismatched brackets"
            # Calculate the expression inside the brackets
            inner_result = calculator(query[start+1:end])
            if isinstance(inner_result, str):  # Error case
                return inner_result
            # Replace the bracketed expression with its result
            query = query[:start] + str(inner_result) + query[end+1:]
    
    # Check for each operator from left to right
    for i in range(len(query)):
        if query[i] in operators:
            try:
                left = calculator(query[:i])
                right = calculator(query[i+1:])
                return (round(operators[query[i]](left, right), 2))
            except:
                continue
                
    # Handle decimal numbers
    try:
        final_result = float(query)
        return final_result
    except:
        return "Invalid input"

def WolframAlphaCalculator(input_query: str):
    wolfram_alpha_appid = "YOUR_WOLFRAMALPHA_APPID"
    wolfram_client = wolframalpha.Client(wolfram_alpha_appid)
    res = wolfram_client.query(input_query)
    assumption = next(res.pods).text
    answer = next(res.results).text
    # return f"Assumption: {assumption} \nAnswer: {answer}"
    return answer

if __name__ == "__main__":
    #query = 'mean(247.0, 253.0, 230.0, 264.0, 254.0, 275.0, 227.0, 258.0, 245.0, 253.0, 242.0, 229.0, 259.0, 253.0)'
    #print(WolframAlphaCalculator(query))
    result = calculator('1.23-3.2')
    print(result)
