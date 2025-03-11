import re

def extract_code_from_response(response_text):
    """
    Extracts Python code from the OpenAI LLM response.
    Assumes the code is wrapped in triple backticks (```python ... ```).
    """
    match = re.search(r"```python\n(.*?)\n```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()  # Extract only the code
    else:
        raise ValueError("No valid Python code block found in the response.")
