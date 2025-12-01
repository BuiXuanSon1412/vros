# ui/handlers/run_handler.py - Run Algorithm Handler

import streamlit as st
from utils.solver import DummySolver


def handle_run_button(
    problem_type, selected_algorithm, vehicle_config, algorithm_params
):
    """Handle run algorithm button click"""
    # Validate data exists
    if not _validate_data_exists(problem_type):
        st.error("Please generate or upload data first!")
        return

    # Run algorithm
    with st.spinner(f"Running {selected_algorithm}..."):
        solution = _run_algorithm(
            problem_type, selected_algorithm, vehicle_config, algorithm_params
        )

        # Store results
        _store_results(problem_type, selected_algorithm, solution)

        st.success("Completed!")


def _validate_data_exists(problem_type):
    """Validate that required data exists"""
    return (
        st.session_state.get(f"customers_{problem_type}") is not None
        and st.session_state.get(f"depot_{problem_type}") is not None
        and st.session_state.get(f"distance_matrix_{problem_type}") is not None
    )


def _run_algorithm(problem_type, selected_algorithm, vehicle_config, algorithm_params):
    """Execute the selected algorithm"""
    solver = DummySolver(problem_type, selected_algorithm)

    solution = solver.solve(
        st.session_state[f"customers_{problem_type}"],
        st.session_state[f"depot_{problem_type}"],
        st.session_state[f"distance_matrix_{problem_type}"],
        vehicle_config,
        algorithm_params,
    )

    return solution


def _store_results(problem_type, selected_algorithm, solution):
    """Store solution in session state"""
    st.session_state[f"solution_{problem_type}"] = solution
    st.session_state[f"results_{problem_type}"][selected_algorithm] = solution
    st.session_state[f"chart_counter_{problem_type}"] += 1
