# ui/visualization/convergence_view.py - Updated with enhanced Pareto visualization

import streamlit as st
from utils.visualizer import Visualizer
import numpy as np
import pandas as pd


@st.cache_resource
def get_visualizer():
    return Visualizer()


def render_convergence_view(problem_type):
    """Render convergence charts with statistics and Pareto front"""
    solution = st.session_state.get(f"solution_{problem_type}")
    chart_counter = st.session_state.get(f"chart_counter_{problem_type}", 0)

    if solution is not None and solution.get("convergence_history"):
        # Convergence statistics
        _render_convergence_statistics(solution)

        st.markdown("---")

        # Main convergence chart
        _render_convergence_chart(solution, problem_type, chart_counter)

        # Render Pareto front for bi-objective problems (Problem 2)
        if problem_type == 2:
            st.markdown("---")
            _render_pareto_section(solution, problem_type, chart_counter)
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
        use_container_width=True,
        key=f"convergence_{problem_type}_{chart_counter}",
    )


def _render_pareto_section(solution, problem_type, chart_counter):
    """Render Pareto front section with enhanced visualization"""
    st.markdown("### 🎯 Pareto Front Visualization")

    pareto_front = solution.get("pareto_front", [])

    if not pareto_front:
        st.warning(
            "⚠️ No Pareto front data available. The algorithm needs to return Pareto solutions."
        )
        st.info("""
        **How to generate Pareto front:**
        - NSGA-II algorithm should populate `solution['pareto_front']`
        - Format: `[(makespan1, cost1), (makespan2, cost2), ...]`
        - Each tuple represents one non-dominated solution
        """)
        return

    # Pareto statistics
    _render_pareto_statistics(pareto_front, solution)

    st.markdown("")

    # Main Pareto front plot
    viz = get_visualizer()

    # Get current solution point if available
    current_solution = None
    if "makespan" in solution and "cost" in solution:
        current_solution = (solution["makespan"], solution["cost"])

    fig_pareto = viz.plot_pareto_front(
        pareto_front,
        title="Pareto Front - Trade-off between Makespan and Cost",
        current_solution=current_solution,
    )

    st.plotly_chart(
        fig_pareto,
        use_container_width=True,
        key=f"pareto_{problem_type}_{chart_counter}",
    )

    # Interactive solution selector
    st.markdown("---")
    _render_solution_selector(pareto_front, problem_type)

    # Detailed analysis
    with st.expander("📊 Detailed Pareto Analysis", expanded=False):
        _render_pareto_analysis(pareto_front)


def _render_pareto_statistics(pareto_front, solution):
    """Render Pareto front statistics"""
    st.markdown("**Pareto Front Statistics**")

    objectives_1, objectives_2 = zip(*pareto_front)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Solutions Found",
            len(pareto_front),
            help="Number of non-dominated solutions",
        )

    with col2:
        range_obj1 = max(objectives_1) - min(objectives_1)
        st.metric(
            "Makespan Range",
            f"{range_obj1:.1f} min",
            help=f"Min: {min(objectives_1):.1f}, Max: {max(objectives_1):.1f}",
        )

    with col3:
        range_obj2 = max(objectives_2) - min(objectives_2)
        st.metric(
            "Cost Range",
            f"${range_obj2:,.0f}",
            help=f"Min: ${min(objectives_2):,.0f}, Max: ${max(objectives_2):,.0f}",
        )

    with col4:
        # Calculate spread quality
        spread_quality = (range_obj1 / max(objectives_1)) * (
            range_obj2 / max(objectives_2)
        )
        quality_label = (
            "Excellent"
            if spread_quality > 0.5
            else "Good"
            if spread_quality > 0.3
            else "Limited"
        )
        st.metric(
            "Spread Quality",
            quality_label,
            help="Diversity of solutions in the Pareto front",
        )


def _render_solution_selector(pareto_front, problem_type):
    """Interactive solution selector from Pareto front"""
    st.markdown("**🎛️ Solution Selector**")
    st.caption("Choose a solution from the Pareto front based on your priority")

    # Create solution options
    objectives_1, objectives_2 = zip(*pareto_front)

    col1, col2 = st.columns([2, 1])

    with col1:
        # Slider to select solution
        solution_idx = st.slider(
            "Select Solution",
            min_value=0,
            max_value=len(pareto_front) - 1,
            value=len(pareto_front) // 2,
            help="Move slider to explore different trade-offs",
            key=f"pareto_selector_{problem_type}",
        )

        # Show trade-off position
        position = (
            solution_idx / (len(pareto_front) - 1) if len(pareto_front) > 1 else 0.5
        )

        if position < 0.33:
            category = "🔵 Time-Focused"
            description = "Prioritizes fast completion over cost savings"
        elif position > 0.67:
            category = "🟢 Cost-Focused"
            description = "Prioritizes low cost over speed"
        else:
            category = "🟡 Balanced"
            description = "Good balance between time and cost"

        st.info(f"**{category}**: {description}")

    with col2:
        # Display selected solution
        selected_makespan = objectives_1[solution_idx]
        selected_cost = objectives_2[solution_idx]

        st.markdown("**Selected Solution:**")
        st.metric("Makespan", f"{selected_makespan:.1f} min")
        st.metric("Cost", f"${selected_cost:,.0f}")

        # Compare with extremes
        min_makespan = min(objectives_1)
        min_cost = min(objectives_2)

        time_penalty = (selected_makespan - min_makespan) / min_makespan * 100
        cost_penalty = (selected_cost - min_cost) / min_cost * 100

        st.caption(f"⏱️ +{time_penalty:.1f}% vs fastest")
        st.caption(f"💰 +{cost_penalty:.1f}% vs cheapest")


def _render_pareto_analysis(pareto_front):
    """Render detailed Pareto front analysis"""

    # Create dataframe with all solutions
    pareto_data = []
    objectives_1, objectives_2 = zip(*pareto_front)

    for idx, (obj1, obj2) in enumerate(pareto_front):
        # Calculate normalized scores
        norm_obj1 = (
            (obj1 - min(objectives_1)) / (max(objectives_1) - min(objectives_1))
            if max(objectives_1) > min(objectives_1)
            else 0
        )
        norm_obj2 = (
            (obj2 - min(objectives_2)) / (max(objectives_2) - min(objectives_2))
            if max(objectives_2) > min(objectives_2)
            else 0
        )

        # Combined score (equal weight)
        combined_score = (norm_obj1 + norm_obj2) / 2

        pareto_data.append(
            {
                "Solution": f"S{idx + 1}",
                "Makespan (min)": f"{obj1:.2f}",
                "Cost ($)": f"{obj2:,.2f}",
                "Category": _get_tradeoff_category(idx, len(pareto_front)),
                "Score": f"{combined_score:.3f}",
            }
        )

    df_pareto = pd.DataFrame(pareto_data)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("**All Pareto Solutions**")
        st.dataframe(df_pareto, use_container_width=True, hide_index=True, height=350)

    with col2:
        st.markdown("**Recommendations**")

        # Find extreme and balanced solutions
        min_makespan_idx = objectives_1.index(min(objectives_1))
        min_cost_idx = objectives_2.index(min(objectives_2))
        balanced_idx = len(pareto_front) // 2

        recommendations = []

        recommendations.append(
            {
                "Type": "⚡ Fastest",
                "Solution": f"S{min_makespan_idx + 1}",
                "Makespan": f"{objectives_1[min_makespan_idx]:.1f} min",
                "Cost": f"${objectives_2[min_makespan_idx]:,.0f}",
                "Why": "Minimum makespan",
            }
        )

        recommendations.append(
            {
                "Type": "💰 Cheapest",
                "Solution": f"S{min_cost_idx + 1}",
                "Makespan": f"{objectives_1[min_cost_idx]:.1f} min",
                "Cost": f"${objectives_2[min_cost_idx]:,.0f}",
                "Why": "Minimum cost",
            }
        )

        recommendations.append(
            {
                "Type": "⚖️ Balanced",
                "Solution": f"S{balanced_idx + 1}",
                "Makespan": f"{objectives_1[balanced_idx]:.1f} min",
                "Cost": f"${objectives_2[balanced_idx]:,.0f}",
                "Why": "Best trade-off",
            }
        )

        df_recommend = pd.DataFrame(recommendations)
        st.dataframe(df_recommend, use_container_width=True, hide_index=True)

        # Export Pareto front
        st.markdown("**Export**")
        csv_data = df_pareto.to_csv(index=False)
        st.download_button(
            label="📥 Export Pareto Front",
            data=csv_data,
            file_name="pareto_front.csv",
            mime="text/csv",
            use_container_width=True,
        )


def _get_tradeoff_category(index, total):
    """Categorize solution based on position in Pareto front"""
    position = index / (total - 1) if total > 1 else 0.5

    if position < 0.33:
        return "🔵 Time-focused"
    elif position > 0.67:
        return "🟢 Cost-focused"
    else:
        return "🟡 Balanced"
