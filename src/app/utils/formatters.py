# src/app/utils/formatters.py
from typing import Dict, List

def format_sources(sources: Dict[str, List[str]]) -> str:
    """
    Formats source documents for display in a readable format.
    
    This function takes the raw source documents (both text and tables) and 
    formats them with appropriate markdown styling for display in the Streamlit
    interface.
    
    Args:
        sources: Dictionary containing text and table sources
            - texts: List of text excerpts used as sources
            - tables: List of table data used as sources
        
    Returns:
        str: Formatted string with markdown formatting for display
    """
    formatted_text = ""
    
    if sources['texts']:
        formatted_text += "### Text Sources:\n"
        for source in sources['texts']:
            formatted_text += f"- {source}\n\n"
    
    if sources['tables']:
        formatted_text += "### Table Sources:\n"
        for table in sources['tables']:
            formatted_text += f"```\n{table}\n```\n\n"
            
    return formatted_text