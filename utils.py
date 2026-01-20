
import ast 
import json 


from typing import List, Optional, Any
from langchain_core.messages import BaseMessage, HumanMessage




def save_python_dict_string_as_json(python_dict_str: str, file_path: str):
    """
    Converts a Python-dict-like string into valid JSON
    and saves it to a file.
    
    Args:
        python_dict_str (str): The input string that looks like a Python dict.
        file_path (str): The full output file path, e.g. "output.json".
    """

    try:
        # Step 1: Convert Python dict string → real Python dict safely
        parsed_dict = ast.literal_eval(python_dict_str)

        # Step 2: Create JSON string
        json_str = json.dumps(parsed_dict, indent=2, ensure_ascii=False)

        # Step 3: Save JSON to file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(json_str)

        print(f"JSON saved successfully to: {file_path}")

    except Exception as e:
        raise ValueError(f"Failed to convert/save JSON: {e}")





def load_chapters(file_path: str) -> dict:
    """
    Loads a JSON file containing chapter mappings
    and returns it as a Python dictionary.

    Args:
        file_path (str): Path to the JSON file, e.g. "chapters.json"

    Returns:
        dict: Parsed chapter data
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in file: {e}")

    except Exception as e:
        raise RuntimeError(f"Unexpected error loading chapters: {e}")
    





def to_openai_messages(lc_messages):
    role_map = {
        "human": "user",
        "ai": "assistant",
        "system": "system",
        "tool": "tool"
    }
    out = []
    for m in lc_messages:
        # LangChain message types expose .type and .content
        role = role_map.get(getattr(m, "type", None))
        if not role:
            raise ValueError(f"Unknown message type: {type(m)}")
        out.append({"role": role, "content": m.content})
    return out









def get_latest_human_message(messages: List[BaseMessage]) -> Optional[HumanMessage]:
    """
    Returns the latest HumanMessage from a list of BaseMessage objects.
    
    Args:
        messages (List[BaseMessage]): List of LangChain messages
    
    Returns:
        Optional[HumanMessage]: The most recent HumanMessage, or None if not found
    """
    # Loop from the LAST message backwards
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg

    return None








def parse_python_like_string(data_str: str) -> Any:
    """
    Safely parses a Python-like list/dict string into a real Python object.
    
    Example input:
        "[{'question': '...', 'options': [...]}]"
    
    Returns:
        A valid Python object (list or dict).
    """
    try:
        return ast.literal_eval(data_str)
    except Exception as e:
        raise ValueError(f"Failed to parse input: {e}")




