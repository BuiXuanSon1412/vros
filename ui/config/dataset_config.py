# ui/config/dataset_config.py - Dataset Configuration (Upload Only)

import streamlit as st
from ui.config.file_upload import render_file_upload


def render_dataset_config(problem_type):
    """Render dataset configuration UI - Upload only"""

    # Remove radio selection, go directly to file upload
    st.session_state[f"data_source_{problem_type}"] = "upload"
    render_file_upload(problem_type)
