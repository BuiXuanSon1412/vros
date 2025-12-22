# ui/visualization/convergence_view.py - Updated without Pareto visualization

import streamlit as st
from utils.visualizer import Visualizer
import numpy as np
import pandas as pd


@st.cache_resource
def get_visualizer():
    return Visualizer()


def render_convergence_view(problem_type):
    """Render convergence charts with statistics"""
    solution = st.session_state.get(f"solution_{problem_type}")
    chart_counter = st.session_state.get(f"chart_counter_{problem_type}", 0)

    if solution is not None and solution.get("convergence_history"):
        # Convergence statistics
        _render_convergence_statistics(solution)

        st.markdown("---")

        # Main convergence chart
        _render_convergence_chart(solution, problem_type, chart_counter)

        st.markdown("---")

        # Convergence analysis
        with st.expander("📊 Detailed Convergence Analysis", expanded=False):
            _render_convergence_analysis(solution)
    else:
        st.info("Run algorithm to see convergence")


def _render_convergence_statistics(solution):
    """Render convergence statistics"""
    st.markdown("**Convergence Statistics**")

    convergence_history = solution.get("convergence_history", [])

    if not convergence_history:
        return

    iterations, fitness_values = zip(*convergence_history)
    fitness_values = list(fitness_values)

    # Calculate statistics
    initial_fitness = fitness_values[0]
    final_fitness = fitness_values[-1]
    best_fitness = min(fitness_values)
    improvement = (initial_fitness - final_fitness) / initial_fitness * 100

    # Find convergence point
    convergence_iter = len(iterations)
    for i in range(1, len(fitness_values)):
        if (
            abs(fitness_values[i] - fitness_values[i - 1]) / fitness_values[i - 1]
            < 0.001
        ):
            convergence_iter = iterations[i]
            break

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Initial Fitness", f"{initial_fitness:.2f}", help="Fitness value at start"
        )

    with col2:
        st.metric(
            "Final Fitness",
            f"{final_fitness:.2f}",
            delta=f"-{improvement:.1f}%",
            delta_color="inverse",
            help="Fitness value at end",
        )

    with col3:
        st.metric("Best Fitness", f"{best_fitness:.2f}", help="Best fitness achieved")

    with col4:
        st.metric(
            "Converged at",
            f"Iter {convergence_iter}",
            help="Iteration where convergence occurred",
        )


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
        width="stretch",
        key=f"convergence_{problem_type}_{chart_counter}",
    )


def _render_convergence_analysis(solution):
    """Render detailed convergence analysis"""
    convergence_history = solution.get("convergence_history", [])

    if not convergence_history:
        st.info("No convergence data available")
        return

    iterations, fitness_values = zip(*convergence_history)
    fitness_values = list(fitness_values)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Convergence Metrics**")

        # Calculate improvement rate
        initial = fitness_values[0]
        final = fitness_values[-1]
        total_improvement = initial - final
        improvement_pct = (total_improvement / initial * 100) if initial > 0 else 0

        # Calculate average improvement per iteration
        num_iters = len(iterations)
        avg_improvement_per_iter = total_improvement / num_iters if num_iters > 0 else 0

        metrics_data = {
            "Metric": [
                "Total Improvement",
                "Improvement %",
                "Avg per Iteration",
                "Total Iterations",
                "Best Iteration",
            ],
            "Value": [
                f"{total_improvement:.2f}",
                f"{improvement_pct:.2f}%",
                f"{avg_improvement_per_iter:.4f}",
                f"{num_iters}",
                f"{iterations[fitness_values.index(min(fitness_values))]}",
            ],
        }

        df_metrics = pd.DataFrame(metrics_data)
        st.dataframe(df_metrics, width="stretch", hide_index=True)

    with col2:
        st.markdown("**Convergence Phases**")

        # Divide convergence into phases
        phase_size = max(1, len(fitness_values) // 4)

        phases_data = []
        for i in range(4):
            start_idx = i * phase_size
            end_idx = min((i + 1) * phase_size, len(fitness_values))

            if start_idx >= len(fitness_values):
                break

            phase_values = fitness_values[start_idx:end_idx]
            phase_improvement = phase_values[0] - phase_values[-1]

            phases_data.append(
                {
                    "Phase": f"Phase {i + 1}",
                    "Iterations": f"{iterations[start_idx]}-{iterations[end_idx - 1]}",
                    "Improvement": f"{phase_improvement:.2f}",
                    "Rate": f"{phase_improvement / len(phase_values):.4f}/iter"
                    if len(phase_values) > 0
                    else "0",
                }
            )

        df_phases = pd.DataFrame(phases_data)
        st.dataframe(df_phases, width="stretch", hide_index=True)

    # Export convergence data
    st.markdown("**Export Convergence Data**")

    convergence_df = pd.DataFrame(
        {"Iteration": list(iterations), "Fitness": fitness_values}
    )

    csv_data = convergence_df.to_csv(index=False)

    col1, col2, col3 = st.columns([2, 1, 1])
    with col3:
        st.download_button(
            label="Export CSV",
            data=csv_data,
            file_name=f"convergence_{solution.get('algorithm', 'solution')}.csv",
            mime="text/csv",
            width="stretch",
        )
