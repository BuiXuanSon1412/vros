# ui/handlers/run_handler.py - UPDATED for Multiple Algorithm Support

import streamlit as st
import time
from utils.solver import Solver


def handle_run_button_multi(
    problem_type, selected_algorithms, system_config, algorithm_params
):
    """Handle run button for multiple algorithms"""
    # Validate data exists
    if not _validate_data_exists(problem_type):
        st.error("Please upload data first!")
        return

    # Progress tracking
    total_algorithms = len(selected_algorithms)
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Run each algorithm
    for idx, algorithm in enumerate(selected_algorithms):
        status_text.markdown(
            f"**Running {algorithm}... ({idx + 1}/{total_algorithms})**"
        )

        with st.spinner(f"Running {algorithm}..."):
            result = _run_algorithm(
                problem_type, algorithm, system_config, algorithm_params
            )

            # Store results
            _store_results(problem_type, algorithm, result)

            # Small delay for UI update
            time.sleep(0.3)

        # Update progress
        progress_bar.progress((idx + 1) / total_algorithms)

    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()

    # Success message
    if total_algorithms == 1:
        st.success(f"{selected_algorithms[0]} completed!")
    else:
        st.success(
            f"All {total_algorithms} algorithms completed! Check the Comparison tab to see results."
        )


def _validate_data_exists(problem_type):
    """Validate that required data exists"""
    return (
        st.session_state.get(f"customers_{problem_type}") is not None
        and st.session_state.get(f"depot_{problem_type}") is not None
    )


def _run_algorithm(problem_type, selected_algorithm, vehicle_config, algorithm_params):
    """Execute the selected algorithm"""
    solver = Solver(problem_type, selected_algorithm)

    solution = solver.solve(
        st.session_state[f"customers_{problem_type}"],
        st.session_state[f"depot_{problem_type}"],
        vehicle_config,
        algorithm_params,
    )

    return solution


def _store_results(problem_type, selected_algorithm, result):
    """Store solution in session state"""
    # Store as current solution (for single view)
    st.session_state[f"result_{problem_type}"] = result

    # Store in results dictionary (for comparison)
    if f"results_{problem_type}" not in st.session_state:
        st.session_state[f"results_{problem_type}"] = {}

    st.session_state[f"results_{problem_type}"][selected_algorithm] = result

    # Increment chart counter to force refresh
    st.session_state[f"chart_counter_{problem_type}"] += 1


# Keep backward compatibility with old single-algorithm function
def handle_run_button(
    problem_type, selected_algorithm, system_config, algorithm_params
):
    """Legacy single-algorithm handler (for backward compatibility)"""
    handle_run_button_multi(
        problem_type, [selected_algorithm], system_config, algorithm_params
    )
