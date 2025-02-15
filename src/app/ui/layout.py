# src/app/ui/layout.py
import streamlit as st

def set_page_config():
    """
    Configures the Streamlit page settings. This includes:
    - Page title and icon
    - Layout settings
    - Other global configurations
    """
    st.set_page_config(
        page_title="Financial Data QA System",
        page_icon="💰",
        layout="wide"
    )