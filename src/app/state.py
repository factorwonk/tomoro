# src/app/state.py
import streamlit as st
from src.rag import initialize_rag, rag_qa

def initialize_session_state():
    """
    Initializes and manages the Streamlit session state. This function handles:
    1. Loading and initializing the RAG model
    2. Setting up conversation history
    3. Managing the current question state
    
    Returns:
        bool: True if initialization successful, False otherwise
    """
    # Initialize RAG system if not already done
    if 'rag_chain' not in st.session_state:
        with st.spinner('Loading RAG model...'):
            try:
                st.session_state.rag_chain, st.session_state.retriever = initialize_rag()
                st.success("RAG model loaded successfully!")
            except Exception as e:
                st.error(f"Failed to initialize RAG model: {str(e)}")
                return False
    
    # Initialize conversation history if not present
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # Initialize question state if not present    
    if 'question' not in st.session_state:
        st.session_state.question = ''
    
    return True

def handle_query():
    """
    Processes user queries and updates the conversation history.
    This function:
    1. Gets the current question from session state
    2. Processes it through the RAG system
    3. Updates the conversation history
    4. Handles any errors that occur during processing
    """
    query = st.session_state.question
    
    if not query:
        return
        
    with st.spinner('Processing question...'):
        try:
            # Send query to RAG system and get response
            response = rag_qa(st.session_state.rag_chain, query)
            
            # Add Q&A pair to history
            st.session_state.history.append({
                "question": query,
                "answer": response['answer'],
                "sources": response['context']
            })
            
            # Reset the question in session state
            st.session_state.question = ""
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")