# utils/session_state.py - Session State Management

import streamlit as st


def initialize_session_state():
    """Initialize session state for all problems"""
    for problem_id in [1, 2, 3]:
        _initialize_problem_state(problem_id)


def _initialize_problem_state(problem_id):
    """Initialize session state for a specific problem"""
    state_keys = {
        f"customers_{problem_id}": None,
        f"depot_{problem_id}": None,
        f"distance_matrix_{problem_id}": None,
        f"solution_{problem_id}": None,
        f"results_{problem_id}": {},
        f"chart_counter_{problem_id}": 0,
        f"file_vehicle_config_{problem_id}": None,
        f"file_processed_{problem_id}": False,
        f"last_uploaded_file_{problem_id}": None,
    }

    for key, default_value in state_keys.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def get_problem_state(problem_id):
    """Get all state for a specific problem"""
    return {
        "customers": st.session_state.get(f"customers_{problem_id}"),
        "depot": st.session_state.get(f"depot_{problem_id}"),
        "distance_matrix": st.session_state.get(f"distance_matrix_{problem_id}"),
        "solution": st.session_state.get(f"solution_{problem_id}"),
        "results": st.session_state.get(f"results_{problem_id}", {}),
        "file_vehicle_config": st.session_state.get(
            f"file_vehicle_config_{problem_id}"
        ),
        "file_processed": st.session_state.get(f"file_processed_{problem_id}", False),
    }


def update_problem_state(problem_id, **kwargs):
    """Update state for a specific problem"""
    for key, value in kwargs.items():
        st.session_state[f"{key}_{problem_id}"] = value
