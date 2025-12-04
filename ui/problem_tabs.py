# ui/problem_tabs.py - Problem Tab Manager with Updated Names

import streamlit as st
from ui.problem_view import render_problem_view


def render_problem_tabs():
    """Render tabs for all three problems with updated names"""
    problem_tabs = st.tabs(["PTDS-DDSS", "MSSVTDE", "VRP-MRDR"])

    for tab_idx, tab in enumerate(problem_tabs):
        problem_type = tab_idx + 1
        with tab:
            render_problem_view(problem_type)
