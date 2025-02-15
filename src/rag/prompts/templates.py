# src/rag/prompts/templates.py

"""
Prompt templates for the RAG system.
This module contains all the prompts used in the system, centralized for easy modification.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

def create_summarization_prompt() -> ChatPromptTemplate:
    """Creates the prompt template for document summarization."""
    template = """
    You are an assistant tasked with summarizing tables and text from financial documents for semantic retrieval.
    These summaries will be embedded and used to retrieve the raw text or table elements.
    Give a detailed summary of the table or text below that is well optimized for retrieval.
    For any tables also add in a description of what the table is about besides the summary.
    Then, include the table in markdown format. Do not add additional words like Summary: etc.

    Table or text chunk:
    {element}
    """
    
    return ChatPromptTemplate.from_template(template)

def create_qa_prompt() -> ChatPromptTemplate:
    """Creates the prompt template for question answering."""
    template = """You are an analyst tasked with understanding detailed information from text documents and data tables.
    Use the provided context information to answer the user's question.
    Do not make up answers, use only the provided context documents below.

    User question:
    {question}

    Text context:
    {text_context}

    Table context:
    {table_context}

    Answer:
    """
    
    return ChatPromptTemplate.from_template(template)

def create_qa_message(question: str, context: dict) -> HumanMessage:
    """Creates a formatted message for the QA system."""
    formatted_texts = "\n".join(context["texts"])
    formatted_tables = "\n".join(context["tables"])
    
    prompt_text = f"""You are an analyst tasked with understanding detailed information from text documents and data tables.
    Use the provided context information to answer the user's question.
    Do not make up answers, use only the provided context documents below.

    User question:
    {question}

    Text context:
    {formatted_texts}

    Table context:
    {formatted_tables}

    Answer:
    """
    
    return HumanMessage(content=prompt_text)