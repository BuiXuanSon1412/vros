# ui/handlers/run_handler.py - UPDATED for Multiple Algorithm Support

import streamlit as st
from utils.solver import DummySolver
import time


def handle_run_button_multi(
    problem_type, selected_algorithms, system_config, algorithm_params
):
    """Handle run button for multiple algorithms"""
    # Validate data exists
    if not _validate_data_exists(problem_type):
        st.error("Please upload data first!")
        return

    # Convert system_config to vehicle_config format
    vehicle_config = _convert_to_vehicle_config(problem_type, system_config)

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
            solution = _run_algorithm(
                problem_type, algorithm, vehicle_config, algorithm_params
            )

            # Store results
            _store_results(problem_type, algorithm, solution)

            # Small delay for UI update
            time.sleep(0.3)

        # Update progress
        progress_bar.progress((idx + 1) / total_algorithms)

    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()

    # Success message
    if total_algorithms == 1:
        st.success(f"✅ {selected_algorithms[0]} completed!")
    else:
        st.success(
            f"✅ All {total_algorithms} algorithms completed! Check the Comparison tab to see results."
        )

    # Auto-switch to comparison view if multiple algorithms
    if total_algorithms > 1:
        st.info("💡 **Tip:** Switch to the **Metrics** tab to see detailed comparison")


def _validate_data_exists(problem_type):
    """Validate that required data exists"""
    return (
        st.session_state.get(f"customers_{problem_type}") is not None
        and st.session_state.get(f"depot_{problem_type}") is not None
        and st.session_state.get(f"distance_matrix_{problem_type}") is not None
    )


def _convert_to_vehicle_config(problem_type, system_config):
    """Convert problem-specific system config to vehicle_config format"""
    if problem_type in [1, 2]:
        # Problem 1 & 2: technicians and drones
        return {
            "truck": {
                "count": system_config.get("num_technicians", 2),
                "speed": system_config.get("technician_speed")
                or system_config.get("technician_baseline_speed", 35),
                "capacity": system_config.get("truck_capacity", 100),
                "cost_per_km": 5000,
            },
            "drone": {
                "count": system_config.get("num_drones", 2),
                "speed": system_config.get("drone_cruise_speed")
                or system_config.get("drone_speed", 60),
                "capacity": system_config.get("drone_capacity", 5),
                "cost_per_km": 2000,
                "energy_limit": system_config.get("flight_endurance_limit", 3600)
                / 60,  # convert to minutes
            },
        }
    else:  # Problem 3
        return {
            "truck": {
                "count": system_config.get("num_trucks", 3),
                "speed": system_config.get("truck_speed", 40),
                "capacity": 100,
                "cost_per_km": 5000,
            },
            "drone": {
                "count": system_config.get("num_drones", 3),
                "speed": system_config.get("drone_speed", 60),
                "capacity": system_config.get("drone_capacity", 4),
                "cost_per_km": 2000,
                "energy_limit": system_config.get("flight_endurance_limit", 90),
            },
        }


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
    # Store as current solution (for single view)
    st.session_state[f"solution_{problem_type}"] = solution

    # Store in results dictionary (for comparison)
    if f"results_{problem_type}" not in st.session_state:
        st.session_state[f"results_{problem_type}"] = {}

    st.session_state[f"results_{problem_type}"][selected_algorithm] = solution

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
