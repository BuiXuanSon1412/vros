# ui/config/algorithm_config_p2.py - Problem 2: HNSGAII-TS Algorithm

import streamlit as st
from config.default_config import PROBLEM2_CONFIG


def render_algorithm_config_p2():
    """Render Problem 2 algorithm configuration (HNSGAII-TS only)"""

    st.markdown("**Algorithm:** HNSGAII-TS")
    # st.markdown("---")

    st.markdown("**Genetic Algorithm Parameters**")
    col1, col2 = st.columns(2)
    with col1:
        crossover_rate = st.number_input(
            "Crossover rate",
            min_value=0.0,
            max_value=1.0,
            value=PROBLEM2_CONFIG["algorithm"]["crossover_rate"],
            step=0.05,
            key="p2_crossover_rate",
        )
        num_generations = st.number_input(
            "Number of generations",
            min_value=50,
            max_value=1000,
            value=PROBLEM2_CONFIG["algorithm"]["num_generations"],
            step=10,
            key="p2_num_generations",
        )

    with col2:
        mutation_rate = st.number_input(
            "Mutation rate",
            min_value=0.0,
            max_value=1.0,
            value=PROBLEM2_CONFIG["algorithm"]["mutation_rate"],
            step=0.01,
            key="p2_mutation_rate",
        )
        population_size = st.number_input(
            "Population size",
            min_value=20,
            max_value=500,
            value=PROBLEM2_CONFIG["algorithm"]["population_size"],
            step=10,
            key="p2_population_size",
        )

    # st.markdown("---")
    st.markdown("**Tabu Search Parameters**")

    tabu_iterations = st.number_input(
        "Number of iterations",
        min_value=10,
        max_value=200,
        value=PROBLEM2_CONFIG["algorithm"]["tabu_search_iterations"],
        step=10,
        key="p2_tabu_iterations",
    )

    return {
        "algorithm": "HNSGAII-TS",
        "crossover_rate": crossover_rate,
        "mutation_rate": mutation_rate,
        "num_generations": num_generations,
        "population_size": population_size,
        "tabu_search_iterations": tabu_iterations,
    }
