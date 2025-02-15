# src/rag/rag_system.py

from typing import Dict, Any, Tuple
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.rag.document_processor import process_documents
from src.rag.retriever import create_multi_vector_retriever
from src.rag.prompts.qa import create_qa_prompt
from src.rag.utils.text_utils import load_json_data

class RAGSystem:
    """
    Main RAG system that orchestrates document processing, retrieval, and question answering.
    This class integrates all components of the RAG system into a cohesive pipeline.
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0):
        """
        Initialize the RAG system with specified model parameters.
        
        Args:
            model_name: Name of the OpenAI model to use
            temperature: Temperature parameter for text generation
        """
        self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        self.chat_model = ChatOpenAI(model=model_name, temperature=temperature)
        self.retriever = None
        self.qa_chain = None

    def initialize(self, data_path: str) -> None:
        """
        Initialize the RAG system by processing documents and setting up the retrieval pipeline.
        
        Args:
            data_path: Path to the JSON data file containing documents
        """
        # Load and process documents
        json_data = load_json_data('./data/convfinqatrain.json', sample_size=20)
        text_docs, text_summaries, table_docs, table_summaries = process_documents(json_data)
        
        # Create retriever
        self.retriever = create_multi_vector_retriever(
            self.embedding_model,
            text_summaries,
            text_docs,
            table_summaries,
            table_docs
        )
        
        # Set up the QA chain
        self._setup_qa_chain()

    def _setup_qa_chain(self) -> None:
        """
        Set up the question-answering chain that combines retrieval and generation.
        """
        # Create the prompt for question answering
        qa_prompt = create_qa_prompt()
        
        # Create the QA chain
        self.qa_chain = (
            RunnablePassthrough.assign(context=self.retriever)
            | qa_prompt
            | self.chat_model
            | StrOutputParser()
        )

    def query(self, question: str) -> Dict[str, Any]:
        """
        Process a question through the RAG system.
        
        Args:
            question: The user's question about the financial data
            
        Returns:
            Dictionary containing the answer and retrieved context
        """
        if not self.qa_chain:
            raise ValueError("RAG system not initialized. Call initialize() first.")
            
        try:
            # Get retrieved context and generate answer
            response = self.qa_chain.invoke({'question': question})
            
            # Get the retrieved documents for context
            context = self.retriever.invoke(question)
            
            return {
                'answer': response,
                'context': context
            }
        except Exception as e:
            raise Exception(f"Error processing query: {str(e)}")

def initialize_rag() -> Tuple[RAGSystem, Any]:
    """
    Initialize the RAG system and return the system and retriever.
    This is the main entry point for setting up the RAG system.
    
    Returns:
        Tuple containing the initialized RAG system and its retriever
    """
    rag_system = RAGSystem()
    rag_system.initialize('./data/convfinqatrain.json')
    return rag_system, rag_system.retriever

def rag_qa(system: RAGSystem, query: str) -> Dict[str, Any]:
    """
    Process a query through the RAG system.
    This is the main entry point for querying the system.
    
    Args:
        system: Initialized RAG system
        query: User's question
        
    Returns:
        Dictionary containing the answer and context
    """
    return system.query(query)