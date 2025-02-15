# src/rag/prompts/__init__.py

"""
The prompts package provides templated prompts and related utilities for the RAG system.
This module centralizes prompt management for both document summarization and question answering,
ensuring consistent interaction patterns throughout the application.

The package includes two main components:
1. Summarization prompts for processing documents and tables
2. Question-answering prompts for handling user queries

Example usage:
    from src.rag.prompts import get_summarization_prompt, create_qa_prompt
    
    # Create a summarization prompt
    summary_prompt = get_summarization_prompt()
    
    # Create a QA prompt
    qa_prompt = create_qa_prompt()
"""

from src.rag.prompts.templates import (
    get_summarization_prompt,
    format_summary_input
)

from src.rag.prompts.qa import (
    create_qa_prompt,
    format_context,
    create_qa_message
)

__all__ = [
    'get_summarization_prompt',
    'format_summary_input',
    'create_qa_prompt',
    'format_context',
    'create_qa_message'
]