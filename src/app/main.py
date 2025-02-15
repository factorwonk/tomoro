# src/app/main.py
import streamlit as st
from src.config import load_environment
from src.app.state import initialize_session_state, handle_query
from src.app.ui.layout import set_page_config
from src.app.ui.sidebar import render_sidebar
from src.app.ui.components import render_query_form, render_history

def main():
    """
    Main application entry point. This function orchestrates the overall flow of the 
    Streamlit application by initializing the state and rendering all UI components 
    in the correct order.
    """
    try:
        # Load environment variables before anything else
        load_environment()
        
        # Set up the page configuration
        set_page_config()
        
        # Display the main title
        st.title("📊 Financial Document QA System")
        
        # Initialize the application state (RAG model, history, etc.)
        if not initialize_session_state():
            st.stop()
        
        # Render all UI components in the desired order
        render_sidebar()
        render_query_form(handle_query)
        render_history()
        
        # Add helpful tips at the bottom
        st.markdown("---")
        st.markdown(
            "💡 Tip: For best results, be specific in your questions and include relevant time periods."
        )
        
    except ValueError as e:
        st.error(f"Configuration Error: {str(e)}")
        st.info("Please make sure your .env file is properly configured with your OpenAI API key.")
        st.stop()

if __name__ == "__main__":
    main()