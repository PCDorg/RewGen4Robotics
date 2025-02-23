def extract_content(content, start_marker, end_marker): 
    """
    Function to extract content between two markers.
    """
    start = content.find(start_marker)
    if start == -1:
        return None  

    start += len(start_marker)  
    end = content.find(end_marker, start)  
    if end == -1:
        return None  
    code=content[start:end].strip()  
    explication=content[end:]
    return code , explication 


def extract_code_explication(content): 
    """
    Function to extract Python code from content.
    """
    start_marker = "```python"
    end_marker = "```"
    return extract_content(content, start_marker, end_marker) 

def save_content(content, code_filename , text_filename):
    """
    Extracts Python code and explication from the content and saves it to a file.
    """
    code , explication = extract_code_explication(content)
    if code is not None :
        with open(code_filename, "w") as f:
            f.write(code)
        print(f"Code saved to {code_filename}")
    else:
        print("No Python code found!")
    if explication is not None :
        with open(text_filename, "w") as f:
            f.write(explication)
        print(f"text saved to {text_filename}")
    else:
        print("No Explication found!")    






#file_path = "C:/Users/YourUsername/Documents/myfile.txt"  # Windows (use double backslashes `\\` or raw string `r""`)
# file_path = "/home/user/Documents/myfile.txt"  # Linux/Mac

#with open(file_path, "w") as file:
#   file.write("Hello, this is a test file.")

     
