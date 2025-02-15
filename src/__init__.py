"""
Financial QA RAG System
======================

A question-answering system that combines natural language processing with 
retrieval augmented generation (RAG) to analyze financial documents.

The system consists of two main components:
1. A RAG backend that processes queries and retrieves relevant information
2. A Streamlit web interface for user interaction

Example usage:
    from src.rag import initialize_rag, rag_qa
    
    # Initialize the system
    rag_system, retriever = initialize_rag()
    
    # Process a query
    response = rag_qa(rag_system, "What was the revenue in 2023?")
"""

from src.rag.core import initialize_rag, rag_qa
from src.app.main import main

__version__ = "0.1.0"

__all__ = [
    'initialize_rag',
    'rag_qa',
    'main'
]