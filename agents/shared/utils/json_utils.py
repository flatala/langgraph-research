import json
import re
from typing import Any, Dict, List, Union

def clean_and_parse_json(text: str) -> Union[Dict[str, Any], List[Any]]:
    """Cleans and parses a JSON string that may be wrapped in code fences or contain common formatting issues."""
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    text = text.replace("\0", "")
    
    try:
        fixed_text = re.sub(r'\\(?![\\"/bfnrtu])', r'\\\\', text)
        return json.loads(fixed_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON. Error: {e}. Content: {text[:500]}...") from e
