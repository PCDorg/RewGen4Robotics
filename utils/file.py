import re 
import ast


def file_to_string(filename):
    with open(filename, 'r') as file:
        return file.read()
    
def extract_code(response_cur):
    # Regex patterns to extract python code enclosed in GPT response
                patterns = [
                    r'```python(.*?)```',
                    r'```(.*?)```',
                    r'"""(.*?)"""',
                    r'""(.*?)""',
                    r'"(.*?)"',
                ]
                for pattern in patterns:
                    code_string = re.search(pattern, response_cur, re.DOTALL)
                    if code_string is not None:
                        code_string = code_string.group(1).strip()
                        break
                code_string = response_cur if not code_string else code_string
                return code_string

# example
gpt_completion = """
Here is your Python code:

```python
def hello(self,text,text1):
    print("Hello, world!")```"""
code_text = extract_code(gpt_completion)
print(code_text)



def get_function_signature(code_string):
    # Parse the code string into an AST
    module = ast.parse(code_string)

    # Find the function definitions
    function_defs = [node for node in module.body if isinstance(node, ast.FunctionDef)]

    # If there are no function definitions, return None
    if not function_defs:
        return None

    # For simplicity, we'll just return the signature of the first function definition
    function_def = function_defs[0]

    input_lst = []
    # Construct the function signature (within object class)
    signature = function_def.name + '(self' + ',self.'.join( arg.arg for arg in function_def.args.args if arg.arg != "self") + ')'
    #signature = function_def.name + '(self, x_velocity,action)'
    for arg in function_def.args.args:
        input_lst.append(arg.arg)
    return signature, input_lst
def get_function_signature2(code_string):
    module = ast.parse(code_string)
    function_defs = []

    # Traverse AST to find functions (including nested ones)
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef):
            function_defs.append(node)

    if not function_defs:
        return None

    function_def = function_defs[0]
    input_lst = []

    # Extract arguments correctly
    args = function_def.args
    params = [arg.arg for arg in args.args]
    
    # Handle self parameter for methods
    if params and params[0] == "self":
        signature = f"{function_def.name}(self"
        params = params[1:]  # exclude self from parameter list
    else:
        signature = f"{function_def.name}("

    # Add other parameters
    if params:
        signature += ", " + ", ".join(params)
    signature += ")"

    return signature, params


#print(Walker2dEnv().__dict__)