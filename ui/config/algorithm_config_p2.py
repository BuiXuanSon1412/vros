# ui/config/algorithm_config_p2.py - UPDATED with Multiple Algorithm Selection

import streamlit as st
from config.default_config import PROBLEM2_CONFIG


def render_algorithm_config_p2():
    """Render Problem 2 algorithm configuration with multiple selection"""

    st.markdown("**Algorithm Selection**")

    # Available algorithms for Problem 2
    available_algorithms = ["Tabu Search", "MOEA/D", "NSGAII", "HNSGAII-TS"]

    selected_algorithms = st.multiselect(
        "Select algorithms to run",
        options=available_algorithms,
        default=["HNSGAII-TS"],
        key="p2_selected_algorithms",
        help="You can select multiple algorithms to compare Pareto fronts",
    )

    if not selected_algorithms:
        st.warning("⚠️ Please select at least one algorithm")
        return None

    st.markdown("**Algorithm Parameters**")
    st.caption("These parameters apply to all selected algorithms")

    st.markdown("**Genetic Algorithm Parameters**")
    col1, col2 = st.columns(2)
    with col1:
        crossover_rate = st.text_input(
            "Crossover rate",
            value=PROBLEM2_CONFIG["algorithm"]["crossover_rate"],
            key="p2_crossover_rate",
        )
        num_generations = st.text_input(
            "Number of generations",
            value=PROBLEM2_CONFIG["algorithm"]["num_generations"],
            key="p2_num_generations",
        )

    with col2:
        mutation_rate = st.text_input(
            "Mutation rate",
            value=PROBLEM2_CONFIG["algorithm"]["mutation_rate"],
            key="p2_mutation_rate",
        )
        population_size = st.text_input(
            "Population size",
            value=PROBLEM2_CONFIG["algorithm"]["population_size"],
            key="p2_population_size",
        )

    st.markdown("**Tabu Search Parameters**")

    # [10, 20, 30, 40, 50]
    tabu_iterations = st.text_input(
        "Number of iterations",
        value=PROBLEM2_CONFIG["algorithm"]["tabu_search_iterations"],
        key="p2_tabu_iterations",
    )

    return {
        "algorithms": selected_algorithms,
        "crossover_rate": crossover_rate,
        "mutation_rate": mutation_rate,
        "num_generations": num_generations,
        "population_size": population_size,
        "tabu_search_iterations": tabu_iterations,
    }
