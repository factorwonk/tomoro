# src/rag/document_processor.py

from typing import List, Dict, Any, Tuple
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from src.rag.prompts.summarization import get_summarization_prompt
from src.rag.utils.text_utils import table_to_string

class DocumentProcessor:
    """
    Handles the processing of documents, including text splitting, 
    table processing, and generating summaries for the RAG system.
    """
    
    def __init__(self, chunk_size: int = 4000, chunk_overlap: int = 500):
        """
        Initialize the document processor with specified chunking parameters.
        
        Args:
            chunk_size: Maximum size of text chunks
            chunk_overlap: Amount of overlap between chunks
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Set up summarization chain
        self.summarization_prompt = get_summarization_prompt()
        self.chatgpt = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.summarization_chain = (
            self.summarization_prompt
            | self.chatgpt
            | StrOutputParser()
        )

    def process_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Document]:
        """
        Process text content into chunks.
        
        Args:
            text: Text content to process
            metadata: Additional metadata for the documents
            
        Returns:
            List of Document objects containing the chunked text
        """
        chunks = self.text_splitter.split_text(text)
        return [
            Document(page_content=chunk, metadata=metadata or {})
            for chunk in chunks
        ]

    def process_table(self, table: List[List[str]], metadata: Dict[str, Any] = None) -> Document:
        """
        Process a table into a document.
        
        Args:
            table: Table data as a list of lists
            metadata: Additional metadata for the document
            
        Returns:
            Document object containing the formatted table
        """
        table_str = table_to_string(table)
        return Document(
            page_content=table_str,
            metadata={**(metadata or {}), "is_table": True}
        )

    def generate_summaries(self, documents: List[Document]) -> List[str]:
        """
        Generate summaries for a list of documents.
        
        Args:
            documents: List of documents to summarize
            
        Returns:
            List of summary strings
        """
        return self.summarization_chain.batch(
            [doc.page_content for doc in documents],
            {"max_concurrency": 5}
        )

def process_documents(json_data: List[Dict]) -> Tuple[List[Document], List[str], List[Document], List[str]]:
    """
    Process JSON data into documents and their summaries.
    This is the main entry point for document processing.
    
    Args:
        json_data: List of dictionaries containing document data
        
    Returns:
        Tuple containing:
        - List of text documents
        - List of text summaries
        - List of table documents
        - List of table summaries
    """
    processor = DocumentProcessor()
    
    text_docs = []
    table_docs = []
    
    for entry in json_data:
        # Skip entries without QA data
        if not entry.get("qa"):
            continue
            
        # Process text content
        text_parts = []
        if entry.get("pre_text"):
            text_parts.append(" ".join(entry["pre_text"]))
        if entry.get("post_text"):
            text_parts.append(" ".join(entry["post_text"]))
            
        if text_parts:
            full_text = "\n\n".join(text_parts)
            text_docs.extend(processor.process_text(
                full_text,
                {"parent_id": entry.get("id")}
            ))
            
        # Process table content
        if table := entry.get("table_ori"):
            table_docs.append(processor.process_table(
                table,
                {"parent_id": entry.get("id")}
            ))
    
    # Generate summaries
    text_summaries = processor.generate_summaries(text_docs)
    table_summaries = processor.generate_summaries(table_docs)
    
    return text_docs, text_summaries, table_docs, table_summaries