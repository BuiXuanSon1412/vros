# ui/config/system_config_p3.py - Problem 3: VRP-MRDR System Configuration

import streamlit as st
from config.default_config import PROBLEM3_CONFIG


def render_system_config_p3():
    """Render Problem 3 system configuration"""

    # Check if file is loaded and has depot info
    depot_data = st.session_state.get("depot_3")
    # file_loaded = depot_data is not None
    file_loaded = depot_data is not None and st.session_state.get(
        "file_processed_3", False
    )
    st.markdown("**Depot**")
    col1, col2 = st.columns(2)
    with col1:
        depot_x = st.text_input(
            "Coordinate X",
            value=depot_data["x"] if depot_data and file_loaded else 0.0,
            disabled=file_loaded,
            key="p3_depot_x",
        )
    with col2:
        depot_y = st.text_input(
            "Coordinate Y",
            value=depot_data["y"] if depot_data and file_loaded else 0.0,
            disabled=file_loaded,
            key="p3_depot_y",
        )

    if depot_data and file_loaded:
        st.caption(
            f"Depot location loaded from file: ({depot_data['x']}, {depot_data['y']})"
        )

    # st.markdown("---")
    st.markdown("**Vehicle**")

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
        num_trucks = st.text_input(
            "Number of trucks",
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
        truck_speed = st.text_input(
            "Truck speed",
            value=float(default_truck_speed),
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
        num_drones = st.text_input(
            "Number of drones",
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
        drone_speed = st.text_input(
            "Drone speed",
            value=float(default_drone_speed),
            disabled=file_loaded,
            key="p3_drone_speed",
        )

    # st.markdown("---")
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
            drone_capacity = st.text_input(
                "Drone capacity",
                value=int(default_capacity),
                disabled=True,
                key="p3_drone_capacity",
            )
        else:
            # When no file, show selectbox with options
            drone_capacity = st.selectbox(
                "Drone capacity",
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
        flight_endurance = st.text_input(
            "Flight endurance limit",
            value=int(default_flight_time),
            disabled=file_loaded,
            key="p3_flight_endurance",
        )

    with col2:
        waiting_limit = st.text_input(
            "Sample waiting limit",
            value=PROBLEM3_CONFIG["system"]["sample_waiting_limit"],
            key="p3_waiting_limit",
            disabled=file_loaded,
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
