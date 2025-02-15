# src/rag/prompts/qa.py

from typing import Dict, List, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

def create_qa_prompt() -> ChatPromptTemplate:
    """
    Creates a prompt template for the question-answering system.
    This prompt is designed to help the model analyze both text and tabular data
    to provide accurate answers while maintaining context awareness.
    
    The prompt instructs the model to:
    1. Consider both text and table context
    2. Use only provided context for answers
    3. Format responses clearly and consistently
    
    Returns:
        ChatPromptTemplate: A configured prompt template for question answering
    """
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

def format_context(context: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Formats the retrieved context for the QA prompt.
    
    Args:
        context: Dictionary containing lists of text and table contexts
        
    Returns:
        Dict containing formatted context strings for the prompt template
    """
    return {
        "text_context": "\n".join(context.get("texts", [])),
        "table_context": "\n".join(context.get("tables", []))
    }

def create_qa_message(question: str, context: Dict[str, List[str]]) -> HumanMessage:
    """
    Creates a formatted message for the QA system combining the question and context.
    
    Args:
        question: The user's question
        context: Dictionary containing the retrieved context
        
    Returns:
        HumanMessage: A formatted message ready for the language model
    """
    formatted_context = format_context(context)
    prompt = create_qa_prompt()
    
    return HumanMessage(content=prompt.format(
        question=question,
        **formatted_context
    ))