# ui/config/file_upload.py - File Upload Form

import streamlit as st
from ui.handlers.upload_handler import handle_file_upload


def render_file_upload(problem_type):
    """Render file upload UI"""
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

    # Display loaded data metrics
    _show_data_metrics(problem_type)

    # Show current loaded data info
    _show_loaded_data_info(problem_type)


def _show_format_info(problem_type):
    """Display expected file format information"""
    if problem_type == 1:
        st.info(
            "📄 Expected format: **6.5.1.txt**\n\n"
            "Customers N\n"
            "Coordinate X  Coordinate Y  Demand"
        )
    elif problem_type == 2:
        st.info(
            "📄 Expected format: **10.10.1.txt**\n\nnumber_staff N\nnumber_drone M\n..."
        )
    else:  # problem_type == 3
        st.info(
            "📄 Expected format: **10.1.txt**\n\nXCOORD  YCOORD  DEMAND  RELEASE_DATE"
        )


def _show_data_metrics(problem_type):
    """Display data metrics if data is loaded"""
    if st.session_state.get(
        f"customers_{problem_type}"
    ) is not None and st.session_state.get(f"file_processed_{problem_type}", False):
        st.markdown("**Data Metrics:**")
        customers_df = st.session_state[f"customers_{problem_type}"]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Customers", len(customers_df))
        with col2:
            st.metric("Avg Demand", f"{customers_df['demand'].mean():.2f} kg")
        with col3:
            if "service_time" in customers_df.columns:
                st.metric(
                    "Avg Service", f"{customers_df['service_time'].mean():.1f} min"
                )


def _show_loaded_data_info(problem_type):
    """Show information about currently loaded data"""
    if (
        st.session_state.get(f"customers_{problem_type}") is not None
        and st.session_state.get(f"data_source_{problem_type}") == "upload"
    ):
        st.markdown("---")
        st.markdown("**Currently Loaded:**")

        customers = st.session_state[f"customers_{problem_type}"]
        depot = st.session_state.get(f"depot_{problem_type}")

        st.write(f"✓ {len(customers)} customers")
        if depot:
            st.write(f"✓ Depot at ({depot['x']:.2f}, {depot['y']:.2f})")
