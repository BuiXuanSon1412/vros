# ui/config_panel.py - Configuration Panel with Problem-Specific UI

import streamlit as st
from ui.config.dataset_config import render_dataset_config
from ui.handlers.run_handler import handle_run_button

# Problem-specific imports
from ui.config.system_config_p1 import render_system_config_p1
from ui.config.system_config_p2 import render_system_config_p2
from ui.config.system_config_p3 import render_system_config_p3
from ui.config.algorithm_config_p1 import render_algorithm_config_p1
from ui.config.algorithm_config_p2 import render_algorithm_config_p2
from ui.config.algorithm_config_p3 import render_algorithm_config_p3


def render_config_panel(problem_type):
    """Render the complete configuration panel with problem-specific tabs"""
    st.markdown("### Configuration")

    # Configuration tabs
    config_tabs = st.tabs(["System", "Dataset", "Algorithm"])

    # TAB 1: SYSTEM (Problem-specific)
    with config_tabs[0]:
        system_config = _render_system_tab(problem_type)

    # TAB 2: DATASET
    with config_tabs[1]:
        render_dataset_config(problem_type)

    # TAB 3: ALGORITHM (Problem-specific)
    with config_tabs[2]:
        algorithm_params = _render_algorithm_tab(problem_type)

        st.markdown("")
        run_button = st.button(
            "Run Algorithm",
            type="primary",
            key=f"run_{problem_type}",
            use_container_width=True,
        )

    # Handle run button
    if run_button:
        handle_run_button(
            problem_type, algorithm_params["algorithm"], system_config, algorithm_params
        )


def _render_system_tab(problem_type):
    """Render problem-specific system configuration"""
    if problem_type == 1:
        return render_system_config_p1()
    elif problem_type == 2:
        return render_system_config_p2()
    elif problem_type == 3:
        return render_system_config_p3()
    else:
        st.error("Invalid problem type")
        return {}


def _render_algorithm_tab(problem_type):
    """Render problem-specific algorithm configuration"""
    if problem_type == 1:
        return render_algorithm_config_p1()
    elif problem_type == 2:
        return render_algorithm_config_p2()
    elif problem_type == 3:
        return render_algorithm_config_p3()
    else:
        st.error("Invalid problem type")
        return {"algorithm": "Unknown"}
