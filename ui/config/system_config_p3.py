# ui/config/system_config_p3.py - Problem 3: VRP-MRDR System Configuration

import streamlit as st
from config.default_config import PROBLEM3_CONFIG


def render_system_config_p3():
    """Render Problem 3 system configuration"""

    # Check if file is loaded and has depot info
    depot_data = st.session_state.get("depot_3")
    file_loaded = depot_data is not None

    st.markdown("**Depot Configuration**")
    col1, col2 = st.columns(2)
    with col1:
        depot_x = st.number_input(
            "Depot X",
            value=depot_data["x"] if file_loaded else 0.0,
            disabled=file_loaded,
            key="p3_depot_x",
        )
    with col2:
        depot_y = st.number_input(
            "Depot Y",
            value=depot_data["y"] if file_loaded else 0.0,
            disabled=file_loaded,
            key="p3_depot_y",
        )

    if file_loaded:
        st.caption(
            f"ℹ️ Depot location loaded from file: ({depot_data['x']}, {depot_data['y']})"
        )

    st.markdown("---")
    st.markdown("**Vehicle Configuration**")

    # Get vehicle config from uploaded file
    file_vehicle_config = st.session_state.get("file_vehicle_config_3")

    col1, col2 = st.columns(2)
    with col1:
        default_trucks = (
            file_vehicle_config.get(
                "number_truck", PROBLEM3_CONFIG["system"]["num_trucks"]
            )
            if file_vehicle_config
            else PROBLEM3_CONFIG["system"]["num_trucks"]
        )
        num_trucks = st.number_input(
            "Number of trucks",
            min_value=1,
            max_value=10,
            value=int(default_trucks),
            disabled=file_loaded,
            key="p3_num_trucks",
        )

        default_truck_speed = (
            file_vehicle_config.get(
                "truck_speed", PROBLEM3_CONFIG["system"]["truck_speed"]
            )
            if file_vehicle_config
            else PROBLEM3_CONFIG["system"]["truck_speed"]
        )
        truck_speed = st.number_input(
            "Truck speed (km/h)",
            min_value=0.1,
            max_value=100.0,
            value=float(default_truck_speed),
            step=0.1,
            disabled=file_loaded,
            key="p3_truck_speed",
        )

    with col2:
        default_drones = (
            file_vehicle_config.get(
                "number_drone", PROBLEM3_CONFIG["system"]["num_drones"]
            )
            if file_vehicle_config
            else PROBLEM3_CONFIG["system"]["num_drones"]
        )
        num_drones = st.number_input(
            "Number of drones",
            min_value=0,
            max_value=10,
            value=int(default_drones),
            disabled=file_loaded,
            key="p3_num_drones",
        )

        default_drone_speed = (
            file_vehicle_config.get(
                "drone_speed", PROBLEM3_CONFIG["system"]["drone_speed"]
            )
            if file_vehicle_config
            else PROBLEM3_CONFIG["system"]["drone_speed"]
        )
        drone_speed = st.number_input(
            "Drone speed (km/h)",
            min_value=0.1,
            max_value=120.0,
            value=float(default_drone_speed),
            step=0.1,
            disabled=file_loaded,
            key="p3_drone_speed",
        )

    st.markdown("---")
    st.markdown("**Drone Capacity & Constraints**")

    col1, col2 = st.columns(2)
    with col1:
        default_capacity = (
            file_vehicle_config.get(
                "M_d", PROBLEM3_CONFIG["system"]["drone_capacity_default"]
            )
            if file_vehicle_config
            else PROBLEM3_CONFIG["system"]["drone_capacity_default"]
        )

        if file_loaded and file_vehicle_config:
            # When file is loaded, just show the value as a number input (disabled)
            drone_capacity = st.number_input(
                "Drone capacity (kg)",
                min_value=1,
                max_value=50,
                value=int(default_capacity),
                disabled=True,
                key="p3_drone_capacity",
            )
        else:
            # When no file, show selectbox with options
            drone_capacity = st.selectbox(
                "Drone capacity (kg)",
                options=PROBLEM3_CONFIG["system"]["drone_capacity_options"],
                index=PROBLEM3_CONFIG["system"]["drone_capacity_options"].index(
                    PROBLEM3_CONFIG["system"]["drone_capacity_default"]
                ),
                key="p3_drone_capacity",
            )

        default_flight_time = (
            file_vehicle_config.get(
                "L_d", PROBLEM3_CONFIG["system"]["flight_endurance_limit"]
            )
            if file_vehicle_config
            else PROBLEM3_CONFIG["system"]["flight_endurance_limit"]
        )
        flight_endurance = st.number_input(
            "Flight endurance limit (minutes)",
            min_value=10,
            max_value=180,
            value=int(default_flight_time),
            disabled=file_loaded,
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
