# ui/visualization/convergence_view.py - Convergence Charts

import streamlit as st
from utils.visualizer import Visualizer


@st.cache_resource
def get_visualizer():
    return Visualizer()


def render_convergence_view(problem_type):
    """Render convergence charts"""
    solution = st.session_state.get(f"solution_{problem_type}")
    chart_counter = st.session_state.get(f"chart_counter_{problem_type}", 0)

    if solution is not None and solution.get("convergence_history"):
        _render_convergence_chart(solution, problem_type, chart_counter)

        # Render Pareto front for bi-objective problems
        if problem_type == 2 and solution.get("pareto_front"):
            st.markdown("---")
            _render_pareto_front(solution, problem_type, chart_counter)
    else:
        st.info("Run algorithm to see convergence")


def _render_convergence_chart(solution, problem_type, chart_counter):
    """Render convergence history chart"""
    viz = get_visualizer()

    iterations, fitness = zip(*solution["convergence_history"])
    fig_conv = viz.plot_convergence(
        list(iterations),
        list(fitness),
        title=f"Convergence - {solution['algorithm']}",
    )
    st.plotly_chart(
        fig_conv,
        use_container_width=True,
        key=f"convergence_{problem_type}_{chart_counter}",
    )


def _render_pareto_front(solution, problem_type, chart_counter):
    """Render Pareto front for bi-objective optimization"""
    viz = get_visualizer()

    fig_pareto = viz.plot_pareto_front(
        solution["pareto_front"],
        title="Pareto Front",
    )
    st.plotly_chart(
        fig_pareto,
        use_container_width=True,
        key=f"pareto_{problem_type}_{chart_counter}",
    )
