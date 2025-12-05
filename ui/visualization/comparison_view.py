# ui/visualization/comparison_view.py - Enhanced Algorithm Comparison

import streamlit as st
from utils.visualizer import Visualizer
from utils.solver import AlgorithmRunner
import pandas as pd


@st.cache_resource
def get_visualizer():
    return Visualizer()


def render_comparison_view(problem_type):
    """Render enhanced algorithm comparison view"""
    results = st.session_state.get(f"results_{problem_type}", {})
    chart_counter = st.session_state.get(f"chart_counter_{problem_type}", 0)

    if len(results) == 0:
        st.info("Run at least one algorithm to see results")
        return

    if len(results) == 1:
        st.info("Run multiple algorithms to see comparison")
        _render_single_algorithm_summary(problem_type, results)
        return

    # Multiple algorithms comparison
    st.markdown("**Algorithm Comparison**")

    # Comparison table
    _render_comparison_table(problem_type, results)

    st.markdown("---")

    # Comparison charts
    _render_comparison_charts(problem_type, results, chart_counter)

    st.markdown("---")

    # Best algorithm recommendation
    _render_best_algorithm_recommendation(results)

    st.markdown("---")

    # Detailed comparison
    _render_detailed_comparison(results)


def _render_single_algorithm_summary(problem_type, results):
    """Show summary when only one algorithm has been run"""
    algo_name = list(results.keys())[0]
    result = results[algo_name]

    st.markdown(f"**Current Algorithm: {algo_name}**")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Makespan", f"{result['makespan']:.1f} min")
    with col2:
        st.metric("Cost", f"${result['cost']:,.0f}")
    with col3:
        st.metric("Distance", f"{result['total_distance']:.1f} km")
    with col4:
        st.metric("Time", f"{result['computation_time']:.2f}s")

    st.info("💡 Run other algorithms to enable comparison features")


def _render_comparison_table(problem_type, results):
    """Render comparison summary table with ranking"""
    runner = AlgorithmRunner(problem_type)
    runner.results = results
    comparison_df = runner.get_comparison_summary()

    # Add rankings
    comparison_df["Makespan Rank"] = comparison_df["Makespan"].rank().astype(int)
    comparison_df["Cost Rank"] = comparison_df["Cost"].rank().astype(int)
    comparison_df["Distance Rank"] = comparison_df["Total Distance"].rank().astype(int)

    # Calculate overall score (lower is better)
    comparison_df["Overall Score"] = (
        comparison_df["Makespan Rank"]
        + comparison_df["Cost Rank"]
        + comparison_df["Distance Rank"]
    )
    comparison_df["Overall Rank"] = comparison_df["Overall Score"].rank().astype(int)

    # Reorder columns
    columns_order = [
        "Algorithm",
        "Overall Rank",
        "Makespan",
        "Makespan Rank",
        "Cost",
        "Cost Rank",
        "Total Distance",
        "Distance Rank",
        "Computation Time",
    ]

    comparison_df = comparison_df[columns_order]

    # Style the dataframe
    def highlight_best(s):
        if s.name in ["Makespan Rank", "Cost Rank", "Distance Rank", "Overall Rank"]:
            return ["background-color: #d4edda" if v == 1 else "" for v in s]
        return ["" for _ in s]

    styled_df = comparison_df.style.apply(highlight_best)

    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # Export comparison
    col1, col2, col3 = st.columns([2, 1, 1])
    with col3:
        csv_data = comparison_df.to_csv(index=False)
        st.download_button(
            label="📥 Export Table",
            data=csv_data,
            file_name=f"comparison_p{problem_type}.csv",
            mime="text/csv",
            use_container_width=True,
        )


def _render_comparison_charts(problem_type, results, chart_counter):
    """Render comparison bar charts"""
    viz = get_visualizer()
    runner = AlgorithmRunner(problem_type)
    runner.results = results
    comparison_df = runner.get_comparison_summary()

    # Two rows of charts
    st.markdown("**Performance Comparison Charts**")

    # Row 1: Makespan and Cost
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

    # Row 2: Distance and Computation Time
    col3, col4 = st.columns(2)

    with col3:
        fig_distance = viz.plot_metrics_comparison(comparison_df, "Total Distance")
        st.plotly_chart(
            fig_distance,
            use_container_width=True,
            key=f"comp_distance_{problem_type}_{chart_counter}",
        )

    with col4:
        fig_time = viz.plot_metrics_comparison(comparison_df, "Computation Time")
        st.plotly_chart(
            fig_time,
            use_container_width=True,
            key=f"comp_time_{problem_type}_{chart_counter}",
        )


def _render_best_algorithm_recommendation(results):
    """Render recommendation for best algorithm"""
    st.markdown("**🏆 Algorithm Recommendation**")

    # Calculate scores for each algorithm
    scores = {}

    for algo_name, result in results.items():
        scores[algo_name] = {
            "makespan": result["makespan"],
            "cost": result["cost"],
            "distance": result["total_distance"],
            "time": result["computation_time"],
        }

    # Find best in each category
    best_makespan = min(scores.items(), key=lambda x: x[1]["makespan"])
    best_cost = min(scores.items(), key=lambda x: x[1]["cost"])
    best_distance = min(scores.items(), key=lambda x: x[1]["distance"])
    fastest_computation = min(scores.items(), key=lambda x: x[1]["time"])

    col1, col2 = st.columns(2)

    with col1:
        st.success(
            f"**⚡ Fastest Completion:** {best_makespan[0]}\n\n"
            f"Makespan: {best_makespan[1]['makespan']:.2f} min"
        )

        st.success(
            f"**💰 Most Cost-Effective:** {best_cost[0]}\n\n"
            f"Cost: ${best_cost[1]['cost']:,.2f}"
        )

    with col2:
        st.success(
            f"**📏 Shortest Distance:** {best_distance[0]}\n\n"
            f"Distance: {best_distance[1]['distance']:.2f} km"
        )

        st.success(
            f"**⚙️ Fastest Computation:** {fastest_computation[0]}\n\n"
            f"Time: {fastest_computation[1]['time']:.2f}s"
        )

    # Overall recommendation based on weighted score
    st.markdown("**Overall Recommendation:**")

    # Normalize and calculate weighted score
    makespan_values = [s["makespan"] for s in scores.values()]
    cost_values = [s["cost"] for s in scores.values()]
    distance_values = [s["distance"] for s in scores.values()]

    max_makespan = max(makespan_values)
    max_cost = max(cost_values)
    max_distance = max(distance_values)

    weighted_scores = {}
    for algo_name, score in scores.items():
        weighted = (
            0.4 * (score["makespan"] / max_makespan)
            + 0.4 * (score["cost"] / max_cost)
            + 0.2 * (score["distance"] / max_distance)
        )
        weighted_scores[algo_name] = weighted

    best_overall = min(weighted_scores.items(), key=lambda x: x[1])

    st.info(
        f"**Recommended: {best_overall[0]}**\n\n"
        f"Best overall balance between makespan, cost, and distance.\n\n"
        f"Score: {best_overall[1]:.3f} (lower is better)"
    )


def _render_detailed_comparison(results):
    """Render detailed comparison with convergence overlay"""
    st.markdown("**Convergence Comparison**")

    viz = get_visualizer()

    # Check if all algorithms have convergence data
    all_have_convergence = all(
        result.get("convergence_history") for result in results.values()
    )

    if not all_have_convergence:
        st.warning("Some algorithms don't have convergence data")
        return

    # Create overlay convergence plot
    import plotly.graph_objects as go

    fig = go.Figure()

    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c"]

    for idx, (algo_name, result) in enumerate(results.items()):
        history = result.get("convergence_history", [])
        if history:
            iterations, fitness = zip(*history)

            fig.add_trace(
                go.Scatter(
                    x=list(iterations),
                    y=list(fitness),
                    mode="lines",
                    name=algo_name,
                    line=dict(color=colors[idx % len(colors)], width=2),
                    hovertemplate=f"<b>{algo_name}</b><br>Iteration: %{{x}}<br>Fitness: %{{y:.2f}}<extra></extra>",
                )
            )

    fig.update_layout(
        title="Convergence Comparison - All Algorithms",
        xaxis_title="Iteration",
        yaxis_title="Fitness Value (Makespan)",
        height=500,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Convergence statistics comparison
    with st.expander("📊 Convergence Statistics", expanded=False):
        _render_convergence_statistics_comparison(results)


def _render_convergence_statistics_comparison(results):
    """Compare convergence statistics across algorithms"""

    stats_data = []

    for algo_name, result in results.items():
        history = result.get("convergence_history", [])
        if history:
            iterations, fitness_values = zip(*history)
            fitness_values = list(fitness_values)

            initial = fitness_values[0]
            final = fitness_values[-1]
            improvement = ((initial - final) / initial * 100) if initial > 0 else 0

            # Find convergence iteration
            convergence_iter = len(iterations)
            for i in range(1, len(fitness_values)):
                if (
                    abs(fitness_values[i] - fitness_values[i - 1])
                    / fitness_values[i - 1]
                    < 0.001
                ):
                    convergence_iter = iterations[i]
                    break

            stats_data.append(
                {
                    "Algorithm": algo_name,
                    "Initial Fitness": f"{initial:.2f}",
                    "Final Fitness": f"{final:.2f}",
                    "Improvement": f"{improvement:.2f}%",
                    "Converged At": f"Iter {convergence_iter}",
                }
            )

    if stats_data:
        df_stats = pd.DataFrame(stats_data)
        st.dataframe(df_stats, use_container_width=True, hide_index=True)
