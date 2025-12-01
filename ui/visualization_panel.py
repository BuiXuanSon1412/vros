# ui/visualization_panel.py - Visualization Panel

import streamlit as st
from ui.visualization.map_view import render_map_view
from ui.visualization.metrics_view import render_metrics_view
from ui.visualization.convergence_view import render_convergence_view
from ui.visualization.timeline_view import render_timeline_view
from ui.visualization.comparison_view import render_comparison_view


def render_visualization_panel(problem_type):
    """Render the complete visualization panel with tabs"""
    st.markdown("### Visualization")

    # Check if data exists
    has_data = st.session_state.get(f"customers_{problem_type}") is not None

    if not has_data:
        st.info("Generate or upload data to see visualizations")
        return

    # Visualization tabs
    viz_tabs = st.tabs(["Map", "Metrics", "Convergence", "Timeline", "Comparison"])

    # TAB 1: MAP VIEW
    with viz_tabs[0]:
        render_map_view(problem_type)

    # TAB 2: METRICS VIEW
    with viz_tabs[1]:
        render_metrics_view(problem_type)

    # TAB 3: CONVERGENCE VIEW
    with viz_tabs[2]:
        render_convergence_view(problem_type)

    # TAB 4: TIMELINE VIEW
    with viz_tabs[3]:
        render_timeline_view(problem_type)

    # TAB 5: COMPARISON VIEW
    with viz_tabs[4]:
        render_comparison_view(problem_type)
