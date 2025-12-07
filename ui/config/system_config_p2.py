# ui/config/system_config_p2.py - Problem 2: MSSVTDE System Configuration

import streamlit as st
from config.default_config import PROBLEM2_CONFIG


def render_system_config_p2():
    """Render Problem 2 system configuration"""
    st.markdown("**Depot**")
    col1, col2 = st.columns(2)
    with col1:
        depot_x = st.number_input("Coordinate X", value=0.0, key="p2_depot_x")
    with col2:
        depot_y = st.number_input("Coordinate Y", value=0.0, key="p2_depot_y")

    # st.markdown("---")
    st.markdown("**Vehicle**")

    col1, col2 = st.columns(2)
    with col1:
        num_technicians = st.number_input(
            "Number of technicians",
            min_value=1,
            max_value=10,
            value=PROBLEM2_CONFIG["system"]["num_technicians"],
            key="p2_num_technicians",
        )
        baseline_speed = st.number_input(
            "Technician baseline speed",
            min_value=10,
            max_value=100,
            value=PROBLEM2_CONFIG["system"]["technician_baseline_speed"],
            key="p2_baseline_speed",
        )

    with col2:
        num_drones = st.number_input(
            "Number of drones",
            min_value=0,
            max_value=10,
            value=PROBLEM2_CONFIG["system"]["num_drones"],
            key="p2_num_drones",
        )

    st.markdown("**Congestion Factor Range**")
    col1, col2 = st.columns(2)
    with col1:
        congestion_min = st.number_input(
            "Min congestion factor",
            min_value=0.1,
            max_value=1.0,
            value=PROBLEM2_CONFIG["system"]["congestion_factor_min"],
            step=0.1,
            key="p2_congestion_min",
        )
    with col2:
        congestion_max = st.number_input(
            "Max congestion factor",
            min_value=0.1,
            max_value=1.0,
            value=PROBLEM2_CONFIG["system"]["congestion_factor_max"],
            step=0.1,
            key="p2_congestion_max",
        )

    st.markdown("**Drone Speed**")
    col1, col2, col3 = st.columns(3)
    with col1:
        takeoff_speed = st.number_input(
            "Takeoff speed",
            min_value=5,
            max_value=50,
            value=PROBLEM2_CONFIG["system"]["drone_takeoff_speed"],
            key="p2_takeoff_speed",
        )
    with col2:
        cruise_speed = st.number_input(
            "Cruise speed",
            min_value=20,
            max_value=120,
            value=PROBLEM2_CONFIG["system"]["drone_cruise_speed"],
            key="p2_cruise_speed",
        )
    with col3:
        landing_speed = st.number_input(
            "Landing speed",
            min_value=5,
            max_value=50,
            value=PROBLEM2_CONFIG["system"]["drone_landing_speed"],
            key="p2_landing_speed",
        )

    # st.markdown("---")
    st.markdown("**Capacity & Constraints**")

    col1, col2 = st.columns(2)
    with col1:
        truck_capacity = st.number_input(
            "Truck capacity",
            min_value=10,
            max_value=500,
            value=PROBLEM2_CONFIG["system"]["truck_capacity"],
            key="p2_truck_capacity",
        )
        flight_endurance = st.number_input(
            "Flight endurance limit",
            min_value=600,
            max_value=7200,
            value=PROBLEM2_CONFIG["system"]["flight_endurance_limit"],
            key="p2_flight_endurance",
        )

    with col2:
        drone_capacity = st.number_input(
            "Drone capacity",
            min_value=1,
            max_value=50,
            value=PROBLEM2_CONFIG["system"]["drone_capacity"],
            key="p2_drone_capacity",
        )
        waiting_limit = st.number_input(
            "Sample waiting limit",
            min_value=10,
            max_value=180,
            value=PROBLEM2_CONFIG["system"]["sample_waiting_limit"],
            key="p2_waiting_limit",
        )

    return {
        "depot": {"x": depot_x, "y": depot_y},
        "num_technicians": num_technicians,
        "num_drones": num_drones,
        "technician_baseline_speed": baseline_speed,
        "congestion_factor_min": congestion_min,
        "congestion_factor_max": congestion_max,
        "drone_takeoff_speed": takeoff_speed,
        "drone_cruise_speed": cruise_speed,
        "drone_landing_speed": landing_speed,
        "truck_capacity": truck_capacity,
        "drone_capacity": drone_capacity,
        "flight_endurance_limit": flight_endurance,
        "sample_waiting_limit": waiting_limit,
    }
