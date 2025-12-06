# ui/config/system_config_p1.py - Problem 1: PTDS-DDSS System Configuration

import streamlit as st
from config.default_config import PROBLEM1_CONFIG


def render_system_config_p1():
    """Render Problem 1 system configuration"""
    st.markdown("**Depot Configuration**")
    col1, col2 = st.columns(2)
    with col1:
        depot_x = st.number_input("Coordinate X", value=0.0, key="p1_depot_x")
    with col2:
        depot_y = st.number_input("Coordinate Y", value=0.0, key="p1_depot_y")

    # st.markdown("---")
    st.markdown("**Vehicle Configuration**")

    col1, col2 = st.columns(2)
    with col1:
        num_technicians = st.number_input(
            "Number of technicians",
            min_value=1,
            max_value=10,
            value=PROBLEM1_CONFIG["system"]["num_technicians"],
            key="p1_num_technicians",
        )
        technician_speed = st.number_input(
            "Technician speed",
            min_value=10,
            max_value=100,
            value=PROBLEM1_CONFIG["system"]["technician_speed"],
            key="p1_technician_speed",
        )

    with col2:
        num_drones = st.number_input(
            "Number of drones",
            min_value=0,
            max_value=10,
            value=PROBLEM1_CONFIG["system"]["num_drones"],
            key="p1_num_drones",
        )
        drone_speed = st.number_input(
            "Drone speed",
            min_value=10,
            max_value=120,
            value=PROBLEM1_CONFIG["system"]["drone_speed"],
            key="p1_drone_speed",
        )

    # st.markdown("---")
    st.markdown("**Constraints**")

    col1, col2 = st.columns(2)
    with col1:
        flight_endurance = st.number_input(
            "Flight endurance limit",
            min_value=600,
            max_value=7200,
            value=PROBLEM1_CONFIG["system"]["flight_endurance_limit"],
            key="p1_flight_endurance",
        )

    with col2:
        waiting_limit = st.number_input(
            "Sample waiting limit",
            min_value=10,
            max_value=180,
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
