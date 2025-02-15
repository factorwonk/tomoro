# src/app/ui/components.py
import streamlit as st
from typing import Callable
from src.app.utils.formatters import format_sources

def render_query_form(callback: Callable) -> None:
    """
    Renders the query input form with a text field and submit button.
    
    Args:
        callback: Function to be called when the form is submitted
    """
    with st.form(key="query_form"):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            st.text_input(
                "Enter your question about the financial data:",
                key="question",
                placeholder="e.g., What was the net change in sales from 2007 to 2008?"
            )
        
        with col2:
            st.form_submit_button(
                "Submit", 
                use_container_width=True,
                on_click=callback
            )

def render_history():
    """
    Renders the conversation history showing:
    - Questions asked
    - Answers received
    - Source documents used
    
    The most recent interaction is expanded by default.
    """
    if st.session_state.history:
        st.write("---")
        st.write("### Question History")
        
        for i, item in enumerate(reversed(st.session_state.history)):
            with st.expander(f"Q: {item['question']}", expanded=(i == 0)):
                st.write("#### Answer:")
                st.markdown(item['answer'])
                
                st.write("#### Sources:")
                st.markdown(format_sources(item['sources']))