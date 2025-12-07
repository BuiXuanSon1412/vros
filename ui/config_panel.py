# ui/config_panel.py - UPDATED for Multiple Algorithm Support

import streamlit as st
from ui.handlers.run_handler import handle_run_button_multi

# Problem-specific imports
from ui.config.system_config_p1 import render_system_config_p1
from ui.config.system_config_p2 import render_system_config_p2
from ui.config.system_config_p3 import render_system_config_p3
from ui.config.algorithm_config_p1 import render_algorithm_config_p1
from ui.config.algorithm_config_p2 import render_algorithm_config_p2
from ui.config.algorithm_config_p3 import render_algorithm_config_p3
from ui.config.file_upload import render_file_upload


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
        render_file_upload(problem_type)

    # TAB 3: ALGORITHM (Problem-specific)
    with config_tabs[2]:
        algorithm_params = _render_algorithm_tab(problem_type)

        if algorithm_params is None:
            st.error("Please select at least one algorithm")
            return

        # Show selected algorithms
        selected_algos = algorithm_params.get("algorithms", [])
        if selected_algos:
            st.markdown("")
            st.info(
                f"**{len(selected_algos)} algorithm(s) selected:** {', '.join(selected_algos)}"
            )

        st.markdown("")

        # Run button with dynamic text
        button_text = (
            f"Run {len(selected_algos)} Algorithm(s)"
            if len(selected_algos) > 1
            else "Run Algorithm"
        )
        run_button = st.button(
            button_text,
            type="primary",
            key=f"run_{problem_type}",
            use_container_width=True,
            disabled=not selected_algos,
        )

    # Handle run button
    if run_button and selected_algos:
        handle_run_button_multi(
            problem_type, selected_algos, system_config, algorithm_params
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
        return None
