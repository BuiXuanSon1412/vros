# app.py - Main Application Entry Point

import streamlit as st
from ui.layout import setup_page_config, render_header, render_footer
from ui.styles import apply_custom_styles
from ui.problem_tabs import render_problem_tabs
from utils.session_state import initialize_session_state


def main():
    """Main application entry point"""
    # Page configuration
    setup_page_config()

    # Apply custom styles
    apply_custom_styles()

    # Initialize session state
    initialize_session_state()

    # Render header
    render_header()

    # Render problem tabs
    render_problem_tabs()

    # Render footer
    render_footer()


if __name__ == "__main__":
    main()
