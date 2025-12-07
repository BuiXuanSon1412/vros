# ui/visualization/map_view.py - Updated to pass problem_type

import streamlit as st
from utils.visualizer import Visualizer


@st.cache_resource
def get_visualizer():
    return Visualizer()


def render_map_view(problem_type):
    """Render map view with routes"""
    customers = st.session_state.get(f"customers_{problem_type}")
    depot = st.session_state.get(f"depot_{problem_type}")
    solution = st.session_state.get(f"solution_{problem_type}")
    chart_counter = st.session_state.get(f"chart_counter_{problem_type}", 0)

    if customers is None or depot is None:
        st.info("Generate or upload data to see the map")
        return

    viz = get_visualizer()

    if solution is not None and solution.get("routes"):
        fig = viz.plot_routes_2d(
            customers,
            depot,
            solution["routes"],
            title=f"Routes - {solution['algorithm']}",
            problem_type=problem_type,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key=f"map_{problem_type}_{chart_counter}",
        )

    else:
        # Determine title based on problem type
        if problem_type in [1, 2]:
            title = "Sample Collection Locations"
        else:
            title = "Customer Locations"

        fig = viz.plot_routes_2d(
            customers,
            depot,
            {},
            title=title,
            problem_type=problem_type,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key=f"map_empty_{problem_type}",
        )
