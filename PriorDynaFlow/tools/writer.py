import json
from typing import List, Any


def write_jsonl(file_path: str, data: List[Any]) -> None:
    """
    Write a list of data to a JSONL (JSON Lines) file, with one JSON object per line.

    Args:
        file_path: Path to the output JSONL file
        data: List of Python objects to be serialized and written. Each item
              will be converted to a JSON string and placed on a separate line.
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                # Serialize each item to JSON string
                json_str = json.dumps(item, ensure_ascii=False)
                # Write with newline separator
                f.write(json_str + '\n')
    except Exception as e:
        print(f"Unexpected error: {e}")