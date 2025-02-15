# src/rag/prompts/summarization.py

from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, Any

def get_summarization_prompt() -> ChatPromptTemplate:
    """
    Creates a prompt template for summarizing financial documents and tables.
    This prompt is designed to generate detailed summaries that are optimized
    for semantic retrieval while preserving the key information from both
    text and tabular data.
    
    The prompt instructs the model to:
    1. Create detailed summaries optimized for retrieval
    2. Include table descriptions for tabular data
    3. Preserve the table in markdown format when relevant
    
    Returns:
        ChatPromptTemplate: A configured prompt template for document summarization
    """
    template = """You are an assistant tasked with summarizing tables and text from financial documents for semantic retrieval.
    These summaries will be embedded and used to retrieve the raw text or table elements.
    Give a detailed summary of the table or text below that is well optimized for retrieval.
    For any tables also add in a description of what the table is about besides the summary.
    Then, include the table in markdown format. Do not add additional words like Summary: etc.

    Table or text chunk:
    {element}
    """
    
    return ChatPromptTemplate.from_template(template)

def format_summary_input(content: str) -> Dict[str, Any]:
    """
    Formats the input content for the summarization prompt.
    
    Args:
        content: The text or table content to be summarized
        
    Returns:
        Dict containing the formatted input for the prompt template
    """
    return {"element": content}