# ui/config/system_config_p1.py - Problem 1: PTDS-DDSS System Configuration

import streamlit as st
from config.default_config import PROBLEM1_CONFIG


def render_system_config_p1():
    """Render Problem 1 system configuration"""
    st.markdown("**Depot**")
    col1, col2 = st.columns(2)
    with col1:
        depot_x = st.text_input(
            "Coordinate X",
            value=PROBLEM1_CONFIG["system"]["depot"]["x"],
            key="p1_depot_x",
            disabled=True,
        )
    with col2:
        depot_y = st.text_input(
            "Coordinate Y",
            value=PROBLEM1_CONFIG["system"]["depot"]["y"],
            key="p1_depot_y",
            disabled=True,
        )

    # st.markdown("---")
    st.markdown("**Vehicle**")

    col1, col2 = st.columns(2)
    with col1:
        num_technicians = st.text_input(
            "Number of technicians",
            value=PROBLEM1_CONFIG["system"]["num_technicians"],
            key="p1_num_technicians",
        )
        technician_speed = st.text_input(
            "Technician speed",
            value=PROBLEM1_CONFIG["system"]["technician_speed"],
            key="p1_technician_speed",
        )

    with col2:
        num_drones = st.text_input(
            "Number of drones",
            value=PROBLEM1_CONFIG["system"]["num_drones"],
            key="p1_num_drones",
        )
        drone_speed = st.text_input(
            "Drone speed",
            value=PROBLEM1_CONFIG["system"]["drone_speed"],
            key="p1_drone_speed",
        )

    # st.markdown("---")
    st.markdown("**Constraints**")

    col1, col2 = st.columns(2)
    with col1:
        flight_endurance = st.text_input(
            "Flight endurance limit",
            value=PROBLEM1_CONFIG["system"]["flight_endurance_limit"],
            key="p1_flight_endurance",
        )

    with col2:
        waiting_limit = st.text_input(
            "Sample waiting limit",
            value=PROBLEM1_CONFIG["system"]["sample_waiting_limit"],
            key="p1_waiting_limit",
        )

    return {
        "depot": {"x": depot_x, "y": depot_y},
        "num_technicians": num_technicians,
        "num_drones": num_drones,
        "technician_speed": technician_speed,
        "drone_speed": drone_speed,
        "flight_endurance_limit": flight_endurance,
        "sample_waiting_limit": waiting_limit,
    }
