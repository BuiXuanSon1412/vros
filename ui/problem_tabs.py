# ui/problem_tabs.py - Problem Tab Manager

import streamlit as st
from ui.problem_view import render_problem_view


def render_problem_tabs():
    """Render tabs for all three problems"""
    problem_tabs = st.tabs(
        ["Problem 1: Min-Timespan", "Problem 2: Bi-Objective", "Problem 3: Resupply"]
    )

    for tab_idx, tab in enumerate(problem_tabs):
        problem_type = tab_idx + 1
        with tab:
            render_problem_view(problem_type)
