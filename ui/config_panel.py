# ui/config_panel.py - Configuration Panel

import streamlit as st
from ui.config.vehicle_config import render_vehicle_config
from ui.config.dataset_config import render_dataset_config
from ui.config.algorithm_config import render_algorithm_config
from ui.handlers.run_handler import handle_run_button


def render_config_panel(problem_type):
    """Render the complete configuration panel"""
    st.markdown("### Configuration")

    # Configuration tabs
    config_tabs = st.tabs(["Vehicle System", "Dataset", "Algorithm"])

    # TAB 1: VEHICLE SYSTEM
    with config_tabs[0]:
        vehicle_config = render_vehicle_config(problem_type)

    # TAB 2: DATASET
    with config_tabs[1]:
        render_dataset_config(problem_type)

    # TAB 3: ALGORITHM
    with config_tabs[2]:
        selected_algorithm, algorithm_params = render_algorithm_config(problem_type)

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
            problem_type, selected_algorithm, vehicle_config, algorithm_params
        )
