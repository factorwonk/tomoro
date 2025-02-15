# src/app/ui/sidebar.py
import streamlit as st

def render_sidebar():
    """
    Renders the application sidebar containing:
    - About section
    - Feature list
    - Sample questions
    - Usage tips
    """
    with st.sidebar:
        st.header("About")
        st.markdown("""
            This application allows you to query financial data using natural language.
            The system combines text and table information to provide accurate answers.
            
            ### Features:
            - Natural language queries
            - Combined text and table analysis
            - Source document tracking
            
            ### Sample Questions:
            - What was the percentage change in net cash from operating activities (2008-2009)?
            - What was the revenue growth from 2007 to 2008?
            - How did operating expenses change between 2008 and 2009?
            
            ### Tips:
            - Be specific with dates and metrics
            - Use complete sentences
            - Check source documents for context
        """)