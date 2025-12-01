# ui/visualization/comparison_view.py - Algorithm Comparison

import streamlit as st
from utils.visualizer import Visualizer
from utils.solver import AlgorithmRunner


@st.cache_resource
def get_visualizer():
    return Visualizer()


def render_comparison_view(problem_type):
    """Render algorithm comparison view"""
    results = st.session_state.get(f"results_{problem_type}", {})
    chart_counter = st.session_state.get(f"chart_counter_{problem_type}", 0)

    if len(results) > 1:
        _render_comparison_table(problem_type, results)

        st.markdown("---")

        _render_comparison_charts(problem_type, results, chart_counter)
    else:
        st.info("Run multiple algorithms to see comparison")


def _render_comparison_table(problem_type, results):
    """Render comparison summary table"""
    runner = AlgorithmRunner(problem_type)
    runner.results = results
    comparison_df = runner.get_comparison_summary()

    st.markdown("**Comparison Table:**")
    st.dataframe(comparison_df, use_container_width=True)


def _render_comparison_charts(problem_type, results, chart_counter):
    """Render comparison bar charts"""
    viz = get_visualizer()
    runner = AlgorithmRunner(problem_type)
    runner.results = results
    comparison_df = runner.get_comparison_summary()

    col1, col2 = st.columns(2)

    with col1:
        fig_makespan = viz.plot_metrics_comparison(comparison_df, "Makespan")
        st.plotly_chart(
            fig_makespan,
            use_container_width=True,
            key=f"comp_makespan_{problem_type}_{chart_counter}",
        )

    with col2:
        fig_cost = viz.plot_metrics_comparison(comparison_df, "Cost")
        st.plotly_chart(
            fig_cost,
            use_container_width=True,
            key=f"comp_cost_{problem_type}_{chart_counter}",
        )
