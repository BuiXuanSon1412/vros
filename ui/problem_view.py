# ui/problem_view.py - Main Problem View

import streamlit as st
from ui.config_panel import render_config_panel
from ui.problem_info_panel import render_problem_info
from ui.visualization_panel import render_visualization_panel


def render_problem_view(problem_type):
    """Render the complete view for a specific problem"""
    render_problem_info(problem_type)
    # Layout: Left (65%) - Visualization | Right (35%) - Config
    col_left, col_right = st.columns([65, 35])

    with col_right:
        render_config_panel(problem_type)

    with col_left:
        render_visualization_panel(problem_type)
