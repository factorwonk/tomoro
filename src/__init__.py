# src/rag/__init__.py
"""
RAG system implementation module.
Contains the core logic for document processing and question answering.
"""

from src.rag.rag_system import (
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