# ui/layout.py - Page Layout Components

import streamlit as st


def setup_page_config():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="Vehicle Routing Optimization System",
        page_icon="🚚",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_header():
    """Render application header"""
    st.markdown(
        '<div class="main-header">Vehicle Routing Optimization System</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Optimize delivery schedules for trucks and drones</div>',
        unsafe_allow_html=True,
    )


def render_footer():
    """Render application footer"""
    st.markdown(
        '<div class="footer" style="margin-bottom: 0; padding-bottom: 0.5rem;">'
        "Vehicle Routing Optimization System"
        "</div>",
        unsafe_allow_html=True,
    )
