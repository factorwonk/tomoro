# src/app/components/__init__.py

"""
UI Components for the Financial QA System
=======================================

This module provides reusable UI components that make up the application interface.
Each component is responsible for rendering and managing a specific part of the UI.
"""

from src.app.ui.sidebar import render_sidebar
from src.app.ui.layout import set_page_config
from src.app.ui.components import render_history, render_query_form

__all__ = [
    'render_sidebar',
    'set_page_config',
    'render_query_form',
    'render_history',
    'format_sources'
]