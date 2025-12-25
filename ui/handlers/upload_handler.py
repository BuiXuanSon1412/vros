# ui/handlers/upload_handler.py - FIXED: Use version tracker to force widget refresh

import streamlit as st
from utils.file_parser import FileParser


@st.cache_resource
def get_file_parser():
    return FileParser()


def handle_file_upload(problem_type, uploaded_file):
    """Handle file upload and trigger UI refresh"""
    # Check if this is a new file (avoid reprocessing)
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"

    if st.session_state.get(f"last_uploaded_file_{problem_type}") == file_id:
        # File already processed
        if st.session_state.get(f"file_processed_{problem_type}", False):
            customers_df = st.session_state[f"customers_{problem_type}"]
            st.success(f"✓ File loaded: {len(customers_df)} customers")
        return

    # Parse new file
    try:
        file_content = uploaded_file.read().decode("utf-8")
        file_parser = get_file_parser()

        # Parse based on problem type
        parsed_data = _parse_file(file_parser, problem_type, file_content)

        # Store parsed data
        _store_parsed_data(problem_type, parsed_data, file_id)

        # Increment file version to force widget keys to change
        current_version = st.session_state.get(f"file_version_{problem_type}", 0)
        st.session_state[f"file_version_{problem_type}"] = current_version + 1

        st.success(
            f"✓ File uploaded successfully! {len(parsed_data['customers'])} customers loaded."
        )

        # Force Streamlit to rerun so System Config tab reflects new data
        st.rerun()

    except Exception as e:
        st.error(f"❌ Error parsing file: {str(e)}")
        st.info("Please check that your file matches the expected format.")


def _parse_file(file_parser, problem_type, file_content):
    """Parse file based on problem type"""
    if problem_type == 1:
        customers_df, depot = file_parser.parse_problem1_file(file_content)
        return {
            "customers": customers_df,
            "depot": depot,
            "vehicle_config": None,
        }
    elif problem_type == 2:
        customers_df, depot, vehicle_config = file_parser.parse_problem2_file(
            file_content
        )
        return {
            "customers": customers_df,
            "depot": depot,
            "vehicle_config": vehicle_config,
        }
    else:  # problem_type == 3
        customers_df, depot, vehicle_config = file_parser.parse_problem3_file(
            file_content
        )
        return {
            "customers": customers_df,
            "depot": depot,
            "vehicle_config": vehicle_config,
        }


def _store_parsed_data(problem_type, parsed_data, file_id):
    """Store parsed data in session state"""
    st.session_state[f"customers_{problem_type}"] = parsed_data["customers"]
    st.session_state[f"depot_{problem_type}"] = parsed_data["depot"]
    st.session_state[f"file_vehicle_config_{problem_type}"] = parsed_data[
        "vehicle_config"
    ]
    st.session_state[f"last_uploaded_file_{problem_type}"] = file_id
    st.session_state[f"file_processed_{problem_type}"] = True

    # CLEAR OLD SOLUTIONS when new file is uploaded
    st.session_state[f"solution_{problem_type}"] = None
    st.session_state[f"results_{problem_type}"] = {}
    st.session_state[f"chart_counter_{problem_type}"] = 0
