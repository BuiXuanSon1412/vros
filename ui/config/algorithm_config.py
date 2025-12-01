# ui/config/algorithm_config.py - Algorithm Configuration

import streamlit as st
from config.default_config import ALGORITHMS


def render_algorithm_config(problem_type):
    """Render algorithm configuration UI and return selected algorithm and parameters"""
    available_algorithms = ALGORITHMS[problem_type]

    selected_algorithm = st.selectbox(
        "Algorithm", available_algorithms, key=f"algorithm_{problem_type}"
    )

    max_iterations = st.number_input(
        "Max iterations",
        min_value=10,
        max_value=10000,
        value=1000,
        step=10,
        key=f"iterations_{problem_type}",
    )

    # Algorithm-specific parameters
    algorithm_params = _get_algorithm_params(
        selected_algorithm, max_iterations, problem_type
    )

    return selected_algorithm, algorithm_params


def _get_algorithm_params(selected_algorithm, max_iterations, problem_type):
    """Get parameters based on algorithm type"""
    if "Tabu" in selected_algorithm:
        return _render_tabu_params(max_iterations, problem_type)
    elif "NSGA" in selected_algorithm or "MOEA" in selected_algorithm:
        return _render_genetic_params(max_iterations, problem_type)
    else:
        return {"max_iterations": max_iterations}


def _render_tabu_params(max_iterations, problem_type):
    """Render Tabu Search parameters"""
    col1, col2 = st.columns(2)

    with col1:
        tabu_tenure = st.number_input(
            "Tabu tenure",
            min_value=5,
            max_value=50,
            value=10,
            key=f"tabu_tenure_{problem_type}",
        )

    with col2:
        neighborhood_size = st.number_input(
            "Neighborhood",
            min_value=10,
            max_value=100,
            value=20,
            key=f"neighborhood_{problem_type}",
        )

    return {
        "max_iterations": max_iterations,
        "tabu_tenure": tabu_tenure,
        "neighborhood_size": neighborhood_size,
    }


def _render_genetic_params(max_iterations, problem_type):
    """Render Genetic Algorithm parameters"""
    population_size = st.number_input(
        "Population size",
        min_value=20,
        max_value=500,
        value=100,
        step=10,
        key=f"population_{problem_type}",
    )

    col1, col2 = st.columns(2)

    with col1:
        crossover_prob = st.number_input(
            "Crossover prob",
            min_value=0.0,
            max_value=1.0,
            value=0.9,
            step=0.05,
            key=f"crossover_{problem_type}",
        )

    with col2:
        mutation_prob = st.number_input(
            "Mutation prob",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05,
            key=f"mutation_{problem_type}",
        )

    return {
        "max_iterations": max_iterations,
        "population_size": population_size,
        "crossover_prob": crossover_prob,
        "mutation_prob": mutation_prob,
    }
