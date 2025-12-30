import json
import re
from typing import Any, Dict, List, Union

def clean_and_parse_json(text: str) -> Union[Dict[str, Any], List[Any]]:
    """Robust JSON parser with multiple fallback strategies.

    Handles:
    - Markdown code fences
    - Invalid escape sequences (e.g., \\e from LaTeX)
    - JSON embedded in mixed text content
    """
    # strip markdown fences and clean
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()
    text = text.replace("\0", "")

    # try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # try with escape sanitization
    try:
        fixed_text = re.sub(r'\\(?![\\"/bfnrtu])', r'\\\\', text)
        return json.loads(fixed_text)
    except json.JSONDecodeError:
        pass

    # try extracting JSON block from mixed content
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        extracted = json_match.group()
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            pass
        # try with escape sanitization on extracted
        try:
            fixed_extracted = re.sub(r'\\(?![\\"/bfnrtu])', r'\\\\', extracted)
            return json.loads(fixed_extracted)
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Failed to parse JSON after all fallbacks. Content: {text[:500]}...")
