# ui/visualization/timeline_view.py - Timeline/Gantt Charts

import streamlit as st
from utils.visualizer import Visualizer


@st.cache_resource
def get_visualizer():
    return Visualizer()


def render_timeline_view(problem_type):
    """Render timeline/Gantt chart view"""
    solution = st.session_state.get(f"solution_{problem_type}")
    chart_counter = st.session_state.get(f"chart_counter_{problem_type}", 0)

    if solution is not None and solution.get("schedule"):
        _render_gantt_chart(solution, problem_type, chart_counter)
    else:
        st.info("Run algorithm to see timeline")


def _render_gantt_chart(solution, problem_type, chart_counter):
    """Render Gantt chart for schedule"""
    viz = get_visualizer()

    fig_gantt = viz.plot_gantt_chart(
        solution["schedule"],
        title="Schedule Timeline",
    )
    st.plotly_chart(
        fig_gantt,
        use_container_width=True,
        key=f"timeline_{problem_type}_{chart_counter}",
    )
