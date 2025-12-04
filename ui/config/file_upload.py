# ui/config/file_upload.py - File Upload Form with Dataset Display

import streamlit as st
from ui.handlers.upload_handler import handle_file_upload
from ui.config.dataset_display import render_dataset_info


def render_file_upload(problem_type):
    """Render file upload UI with problem-specific information"""
    # Display expected format
    _show_format_info(problem_type)

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["txt", "dat"],
        key=f"file_upload_{problem_type}",
    )

    # Handle file upload
    if uploaded_file is not None:
        handle_file_upload(problem_type, uploaded_file)

    # Display dataset information after upload
    customers_df = st.session_state.get(f"customers_{problem_type}")
    if customers_df is not None and st.session_state.get(
        f"file_processed_{problem_type}", False
    ):
        render_dataset_info(problem_type, customers_df)


def _show_format_info(problem_type):
    """Display expected file format information"""
    if problem_type == 1:
        st.info(
            "📄 **Expected format for PTDS-DDSS:**\n\n"
            "```\n"
            "Customers N\n"
            "Coordinate X  Coordinate Y  Demand\n"
            "x1  y1  demand1\n"
            "...\n"
            "```"
        )
    elif problem_type == 2:
        st.info(
            "📄 **Expected format for MSSVTDE:**\n\n"
            "```\n"
            "number_staff N\n"
            "number_drone M\n"
            "droneLimitationFightTime(s) T\n"
            "Customers K\n"
            "Coordinate X  Coordinate Y  Demand  OnlyServicedByStaff  ServiceTimeByTruck(s)  ServiceTimeByDrone(s)\n"
            "...\n"
            "```"
        )
    else:  # problem_type == 3
        st.info(
            "📄 **Expected format for VRP-MRDR:**\n\n"
            "```\n"
            "XCOORD  YCOORD  DEMAND  RELEASE_DATE\n"
            "0  0  0  0  (depot)\n"
            "x1  y1  demand1  release1\n"
            "...\n"
            "number_truck  N\n"
            "number_drone  M\n"
            "drone_speed  S1\n"
            "truck_speed  S2\n"
            "M_d  capacity\n"
            "L_d  flight_time\n"
            "Sigma  service_time\n"
            "```"
        )
