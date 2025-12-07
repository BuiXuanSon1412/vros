# ui/config/algorithm_config_p1.py - UPDATED with Multiple Algorithm Selection

import streamlit as st
from config.default_config import PROBLEM1_CONFIG


def render_algorithm_config_p1():
    """Render Problem 1 algorithm configuration with multiple selection"""

    st.markdown("**Algorithm Selection**")

    # Available algorithms for Problem 1
    available_algorithms = [
        "Tabu Search",
        "Tabu Search Multi-Level",
        "Simulated Annealing",
        "Genetic Algorithm",
    ]

    selected_algorithms = st.multiselect(
        "Select algorithms to run",
        options=available_algorithms,
        default=["Tabu Search Multi-Level"],
        key="p1_selected_algorithms",
        help="You can select multiple algorithms to compare results",
    )

    if not selected_algorithms:
        st.warning("⚠️ Please select at least one algorithm")
        return None

    st.markdown("---")
    st.markdown("**Algorithm Parameters**")
    st.caption("These parameters apply to all selected algorithms")

    st.markdown("**Iteration Parameters**")
    col1, col2 = st.columns(2)
    with col1:
        max_iteration = st.number_input(
            "Maximum iteration",
            min_value=100,
            max_value=10000,
            value=PROBLEM1_CONFIG["algorithm"]["max_iteration"],
            step=100,
            key="p1_max_iteration",
        )

    with col2:
        max_iter_no_improve = st.number_input(
            "Max iteration w/o improvement",
            min_value=10,
            max_value=1000,
            value=PROBLEM1_CONFIG["algorithm"]["max_iteration_no_improve"],
            step=10,
            key="p1_max_iter_no_improve",
        )

    st.markdown("**Penalty Parameters**")

    col1, col2, col3 = st.columns(3)
    with col1:
        alpha1 = st.number_input(
            "Flight endurance penalty",
            min_value=0.1,
            max_value=10.0,
            value=PROBLEM1_CONFIG["algorithm"]["alpha1"],
            step=0.1,
            key="p1_alpha1",
        )

    with col2:
        alpha2 = st.number_input(
            "Waiting time penalty",
            min_value=0.1,
            max_value=10.0,
            value=PROBLEM1_CONFIG["algorithm"]["alpha2"],
            step=0.1,
            key="p1_alpha2",
        )

    with col3:
        beta = st.number_input(
            "Penalty factor",
            min_value=1.0,
            max_value=5.0,
            value=PROBLEM1_CONFIG["algorithm"]["beta"],
            step=0.1,
            key="p1_beta",
        )

    return {
        "algorithms": selected_algorithms,
        "max_iteration": max_iteration,
        "max_iteration_no_improve": max_iter_no_improve,
        "alpha1": alpha1,
        "alpha2": alpha2,
        "beta": beta,
    }
