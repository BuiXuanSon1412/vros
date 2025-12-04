# ui/config/system_config_p3.py - Problem 3: VRP-MRDR System Configuration

import streamlit as st
from config.default_config import PROBLEM3_CONFIG


def render_system_config_p3():
    """Render Problem 3 system configuration"""
    st.markdown("**Depot Configuration**")
    col1, col2 = st.columns(2)
    with col1:
        depot_x = st.number_input("Depot X", value=0.0, key="p3_depot_x")
    with col2:
        depot_y = st.number_input("Depot Y", value=0.0, key="p3_depot_y")

    st.markdown("---")
    st.markdown("**Vehicle Configuration**")

    col1, col2 = st.columns(2)
    with col1:
        num_trucks = st.number_input(
            "Number of trucks",
            min_value=1,
            max_value=10,
            value=PROBLEM3_CONFIG["system"]["num_trucks"],
            key="p3_num_trucks",
        )
        truck_speed = st.number_input(
            "Truck speed (km/h)",
            min_value=10,
            max_value=100,
            value=PROBLEM3_CONFIG["system"]["truck_speed"],
            key="p3_truck_speed",
        )

    with col2:
        num_drones = st.number_input(
            "Number of drones",
            min_value=0,
            max_value=10,
            value=PROBLEM3_CONFIG["system"]["num_drones"],
            key="p3_num_drones",
        )
        drone_speed = st.number_input(
            "Drone speed (km/h)",
            min_value=10,
            max_value=120,
            value=PROBLEM3_CONFIG["system"]["drone_speed"],
            key="p3_drone_speed",
        )

    st.markdown("---")
    st.markdown("**Drone Capacity & Constraints**")

    col1, col2 = st.columns(2)
    with col1:
        drone_capacity = st.selectbox(
            "Drone capacity (kg)",
            options=PROBLEM3_CONFIG["system"]["drone_capacity_options"],
            index=PROBLEM3_CONFIG["system"]["drone_capacity_options"].index(
                PROBLEM3_CONFIG["system"]["drone_capacity_default"]
            ),
            key="p3_drone_capacity",
        )
        flight_endurance = st.number_input(
            "Flight endurance limit (minutes)",
            min_value=10,
            max_value=180,
            value=PROBLEM3_CONFIG["system"]["flight_endurance_limit"],
            key="p3_flight_endurance",
        )

    with col2:
        waiting_limit = st.number_input(
            "Sample waiting limit (minutes)",
            min_value=10,
            max_value=180,
            value=PROBLEM3_CONFIG["system"]["sample_waiting_limit"],
            key="p3_waiting_limit",
        )

    return {
        "depot": {"x": depot_x, "y": depot_y},
        "num_trucks": num_trucks,
        "num_drones": num_drones,
        "truck_speed": truck_speed,
        "drone_speed": drone_speed,
        "drone_capacity": drone_capacity,
        "flight_endurance_limit": flight_endurance,
        "sample_waiting_limit": waiting_limit,
    }
