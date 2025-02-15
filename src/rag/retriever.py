# src/rag/retriever.py

import uuid
from typing import List, Any
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from langchain_chroma import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore

class RAGRetriever:
    """
    Handles the creation and management of the retrieval system,
    including vector storage and multi-vector retrieval functionality.
    """
    
    def __init__(self, embedding_model: Embeddings):
        """
        Initialize the retriever with an embedding model.
        
        Args:
            embedding_model: Model to use for generating embeddings
        """
        self.embedding_model = embedding_model
        self.vectorstore = Chroma(
            collection_name="text_table_rag",
            embedding_function=embedding_model,
            persist_directory="./chroma_db"
        )
        self.docstore = InMemoryStore()
        self.id_key = "doc_id"

    def _add_documents(self, doc_summaries: List[str], doc_contents: List[Document]) -> None:
        """
        Add documents and their summaries to the retrieval system.
        
        Args:
            doc_summaries: List of document summaries
            doc_contents: List of original documents
        """
        doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
        
        # Create summary documents with IDs
        summary_docs = [
            Document(page_content=s, metadata={self.id_key: doc_ids[i]})
            for i, s in enumerate(doc_summaries)
        ]
        
        # Add to vector store and document store
        self.vectorstore.add_documents(summary_docs)
        self.docstore.mset(list(zip(doc_ids, doc_contents)))

    def create_retriever(self) -> MultiVectorRetriever:
        """
        Create a multi-vector retriever that uses summaries for retrieval
        but returns original documents.
        
        Returns:
            Configured MultiVectorRetriever
        """
        return MultiVectorRetriever(
            vectorstore=self.vectorstore,
            docstore=self.docstore,
            id_key=self.id_key,
        )

def create_multi_vector_retriever(
    embedding_model: Embeddings,
    text_summaries: List[str],
    texts: List[Document],
    table_summaries: List[str],
    tables: List[Document]
) -> MultiVectorRetriever:
    """
    Create and initialize a multi-vector retriever with both text and table documents.
    This is the main entry point for setting up the retrieval system.
    
    Args:
        embedding_model: Model to use for generating embeddings
        text_summaries: Summaries of text documents
        texts: Original text documents
        table_summaries: Summaries of table documents
        tables: Original table documents
        
    Returns:
        Configured MultiVectorRetriever
    """
    retriever_system = RAGRetriever(embedding_model)
    
    # Add text documents if present
    if text_summaries:
        retriever_system._add_documents(text_summaries, texts)
        
    # Add table documents if present
    if table_summaries:
        retriever_system._add_documents(table_summaries, tables)
    
    return retriever_system.create_retriever()