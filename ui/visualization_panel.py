# ui/visualization_panel.py - UPDATED with Comparison Tab

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

    # Check number of results for adaptive tabs
    results = st.session_state.get(f"results_{problem_type}", {})
    num_results = len(results)

    # Adaptive tab configuration
    if num_results > 1:
        # Multiple algorithms - show comparison prominently
        viz_tabs = st.tabs(["Comparison", "Map", "Metrics", "Convergence", "Timeline"])
        # viz_tabs = st.tabs(["Map", "Metrics", "Convergence", "Timeline"])
        # TAB 1: COMPARISON (prioritized for multi-algorithm)
        with viz_tabs[0]:
            render_comparison_view(problem_type)

        # TAB 2: MAP VIEW
        with viz_tabs[1]:
            _render_map_with_selector(problem_type)

        # TAB 3: METRICS VIEW
        with viz_tabs[2]:
            _render_metrics_with_selector(problem_type)

        # TAB 4: CONVERGENCE VIEW
        with viz_tabs[3]:
            _render_convergence_with_selector(problem_type)

        # TAB 5: TIMELINE VIEW
        with viz_tabs[4]:
            _render_timeline_with_selector(problem_type)
    else:
        # Single algorithm - standard view
        viz_tabs = st.tabs(["Map", "Metrics", "Convergence", "Timeline"])

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


def _render_map_with_selector(problem_type):
    """Render map view with algorithm selector"""
    results = st.session_state.get(f"results_{problem_type}", {})

    if not results:
        st.info("Run algorithms to see results")
        return

    # Algorithm selector
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_algo = st.selectbox(
            "Select algorithm to visualize",
            options=list(results.keys()),
            key=f"map_algo_selector_{problem_type}",
        )

    # Update current solution
    if selected_algo:
        st.session_state[f"solution_{problem_type}"] = results[selected_algo]
        render_map_view(problem_type)


def _render_metrics_with_selector(problem_type):
    """Render metrics view with algorithm selector"""
    results = st.session_state.get(f"results_{problem_type}", {})

    if not results:
        st.info("Run algorithms to see results")
        return

    # Algorithm selector
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_algo = st.selectbox(
            "Select algorithm to view details",
            options=list(results.keys()),
            key=f"metrics_algo_selector_{problem_type}",
        )

    # Update current solution
    if selected_algo:
        st.session_state[f"solution_{problem_type}"] = results[selected_algo]
        render_metrics_view(problem_type)


def _render_convergence_with_selector(problem_type):
    """Render convergence view with algorithm selector"""
    results = st.session_state.get(f"results_{problem_type}", {})

    if not results:
        st.info("Run algorithms to see results")
        return

    # Algorithm selector with multi-select for comparison
    selected_algos = st.multiselect(
        "Select algorithms to compare convergence",
        options=list(results.keys()),
        default=[list(results.keys())[0]] if results else [],
        key=f"convergence_algo_selector_{problem_type}",
    )

    if not selected_algos:
        st.warning("Please select at least one algorithm")
        return

    # Render convergence for selected algorithms
    _render_convergence_comparison(problem_type, selected_algos, results)


def _render_convergence_comparison(problem_type, selected_algos, results):
    """Render convergence comparison for multiple algorithms"""
    import plotly.graph_objects as go

    fig = go.Figure()

    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c"]

    for idx, algo in enumerate(selected_algos):
        solution = results[algo]
        convergence = solution.get("convergence_history", [])

        if convergence:
            iterations, fitness = zip(*convergence)

            fig.add_trace(
                go.Scatter(
                    x=list(iterations),
                    y=list(fitness),
                    mode="lines+markers",
                    name=algo,
                    line=dict(color=colors[idx % len(colors)], width=2),
                    marker=dict(size=4),
                )
            )

    fig.update_layout(
        title="Convergence Comparison",
        xaxis_title="Iteration",
        yaxis_title="Fitness Value (Makespan)",
        height=500,
        template="plotly_white",
        hovermode="x unified",
    )

    st.plotly_chart(fig, width="stretch")

    # Show statistics
    st.markdown("**Convergence Statistics**")

    stats_data = []
    for algo in selected_algos:
        solution = results[algo]
        convergence = solution.get("convergence_history", [])

        if convergence:
            _, fitness = zip(*convergence)
            initial = fitness[0]
            final = fitness[-1]
            improvement = ((initial - final) / initial * 100) if initial > 0 else 0

            stats_data.append(
                {
                    "Algorithm": algo,
                    "Initial": f"{initial:.2f}",
                    "Final": f"{final:.2f}",
                    "Improvement": f"{improvement:.2f}%",
                }
            )

    if stats_data:
        import pandas as pd

        df = pd.DataFrame(stats_data)
        st.dataframe(df, width="stretch", hide_index=True)


def _render_timeline_with_selector(problem_type):
    """Render timeline view with algorithm selector"""
    results = st.session_state.get(f"results_{problem_type}", {})

    if not results:
        st.info("Run algorithms to see results")
        return

    # Algorithm selector
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_algo = st.selectbox(
            "Select algorithm to view schedule",
            options=list(results.keys()),
            key=f"timeline_algo_selector_{problem_type}",
        )

    # Update current solution
    if selected_algo:
        st.session_state[f"solution_{problem_type}"] = results[selected_algo]
        render_timeline_view(problem_type)
