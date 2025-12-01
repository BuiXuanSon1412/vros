# ui/config/dataset_config.py - Dataset Configuration

import streamlit as st
from ui.config.data_generation import render_data_generation
from ui.config.file_upload import render_file_upload


def render_dataset_config(problem_type):
    """Render dataset configuration UI"""
    # Data source selection
    data_source = st.radio(
        "Data Source",
        ["Generate Data", "Upload File"],
        key=f"data_source_radio_{problem_type}",
        horizontal=True,
    )

    st.markdown("---")

    if data_source == "Generate Data":
        st.session_state[f"data_source_{problem_type}"] = "generate"
        render_data_generation(problem_type)
    else:
        st.session_state[f"data_source_{problem_type}"] = "upload"
        render_file_upload(problem_type)
