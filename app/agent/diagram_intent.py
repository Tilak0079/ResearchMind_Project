"""
Lightweight Diagram Intent Detection module.
"""

import re

DIAGRAM_KEYWORDS = [
    r"\bdiagram\b",
    r"\bflowchart\b",
    r"\barchitecture\b",
    r"\bvisualize\b",
    r"\bdraw\b",
    r"\bschematic\b",
    r"\bgraph\b",
    r"\bplot\b"
]

def detect_diagram_intent(query: str) -> bool:
    """
    Returns True if the query explicitly or implicitly requires a visual diagram.
    This is a lightweight rule-based check to avoid an extra LLM call.
    Can be upgraded to a fast LLM classifier later if needed.
    """
    query_lower = query.lower()
    
    for pattern in DIAGRAM_KEYWORDS:
        if re.search(pattern, query_lower):
            return True
            
    return False
