
"""
RAG (Retrieval Augmented Generation) module for financial document analysis.

This module provides the core functionality for our question-answering system,
combining document retrieval with language model generation to provide accurate
answers to questions about financial documents.

The main components exposed by this module are:
1. initialize_rag: Function to create and initialize a new RAG system
2. rag_qa: Function to process queries through an initialized system
3. RAGSystem: The main class implementing the RAG functionality
"""

from src.rag.core import (
    initialize_rag,
    rag_qa,
    process_documents,
    create_multi_vector_retriever
)

__all__ = [
    'initialize_rag',
    'rag_qa',
    'process_documents',
    'create_multi_vector_retriever'
]