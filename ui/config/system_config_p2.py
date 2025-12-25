# ui/config/system_config_p2.py - Problem 2: MSSVTDE System Configuration

import streamlit as st
from config.default_config import PROBLEM2_CONFIG


def render_system_config_p2():
    """Render Problem 2 system configuration"""

    # Check if file is loaded
    file_loaded = st.session_state.get("file_processed_2", False)
    depot_data = st.session_state.get("depot_2")
    file_vehicle_config = st.session_state.get("file_vehicle_config_2")

    # Get file version to create unique keys
    file_version = st.session_state.get("file_version_2", 0)

    st.markdown("**Depot**")
    col1, col2 = st.columns(2)
    with col1:
        # Determine depot X value
        if file_loaded and depot_data:
            depot_x_value = float(depot_data["x"])
        else:
            depot_x_value = float(PROBLEM2_CONFIG["system"]["depot"]["x"])

        depot_x = st.text_input(
            "Coordinate X",
            value=depot_x_value,
            key=f"p2_depot_x_{file_version}",
            disabled=True,
        )
    with col2:
        # Determine depot Y value
        if file_loaded and depot_data:
            depot_y_value = float(depot_data["y"])
        else:
            depot_y_value = float(PROBLEM2_CONFIG["system"]["depot"]["y"])

        depot_y = st.text_input(
            "Coordinate Y",
            value=depot_y_value,
            key=f"p2_depot_y_{file_version}",
            disabled=True,
        )

    # st.markdown("---")
    st.markdown("**Vehicle**")

    col1, col2 = st.columns(2)
    with col1:
        # Number of technicians from file
        if file_loaded and file_vehicle_config and "num_staff" in file_vehicle_config:
            num_technicians_value = int(file_vehicle_config["num_staff"])
        else:
            num_technicians_value = PROBLEM2_CONFIG["system"]["num_technicians"]

        num_technicians = st.text_input(
            "Number of technicians",
            value=num_technicians_value,
            key=f"p2_num_technicians_{file_version}",
            # disabled=file_loaded,
        )

        baseline_speed = st.text_input(
            "Technician baseline speed",
            value=PROBLEM2_CONFIG["system"]["technician_baseline_speed"],
            key=f"p2_baseline_speed_{file_version}",
        )

    with col2:
        # Number of drones from file
        if file_loaded and file_vehicle_config and "num_drone" in file_vehicle_config:
            num_drones_value = int(file_vehicle_config["num_drone"])
        else:
            num_drones_value = PROBLEM2_CONFIG["system"]["num_drones"]

        num_drones = st.text_input(
            "Number of drones",
            value=num_drones_value,
            key=f"p2_num_drones_{file_version}",
            # disabled=file_loaded,
        )

    if file_loaded and file_vehicle_config:
        st.caption(
            f"✓ Vehicle configuration loaded: {num_technicians_value} technicians, {num_drones_value} drones"
        )

    st.markdown("**Congestion Factor Range**")
    col1, col2 = st.columns(2)
    with col1:
        congestion_min = st.text_input(
            "Min congestion factor",
            value=PROBLEM2_CONFIG["system"]["congestion_factor_min"],
            key=f"p2_congestion_min_{file_version}",
        )
    with col2:
        congestion_max = st.text_input(
            "Max congestion factor",
            value=PROBLEM2_CONFIG["system"]["congestion_factor_max"],
            key=f"p2_congestion_max_{file_version}",
        )

    st.markdown("**Drone Speed**")
    col1, col2, col3 = st.columns(3)
    with col1:
        takeoff_speed = st.text_input(
            "Takeoff speed",
            value=PROBLEM2_CONFIG["system"]["drone_takeoff_speed"],
            key=f"p2_takeoff_speed_{file_version}",
        )
    with col2:
        cruise_speed = st.text_input(
            "Cruise speed",
            value=PROBLEM2_CONFIG["system"]["drone_cruise_speed"],
            key=f"p2_cruise_speed_{file_version}",
        )
    with col3:
        landing_speed = st.text_input(
            "Landing speed",
            value=PROBLEM2_CONFIG["system"]["drone_landing_speed"],
            key=f"p2_landing_speed_{file_version}",
        )

    # st.markdown("---")
    # st.markdown("**Drone Constraints**")

    col1, col2 = st.columns(2)
    with col1:
        # Flight endurance from file (droneLimitationFightTime)
        if (
            file_loaded
            and file_vehicle_config
            and "drone_flight_time" in file_vehicle_config
        ):
            flight_endurance_value = int(file_vehicle_config["drone_flight_time"])
        else:
            flight_endurance_value = PROBLEM2_CONFIG["system"]["flight_endurance_limit"]

        flight_endurance = st.text_input(
            "Flight endurance limit",
            value=flight_endurance_value,
            key=f"p2_flight_endurance_{file_version}",
            # disabled=file_loaded,
        )

    with col2:
        drone_capacity = st.text_input(
            "Drone capacity",
            value=PROBLEM2_CONFIG["system"]["drone_capacity"],
            key=f"p2_drone_capacity_{file_version}",
        )

    if file_loaded and file_vehicle_config:
        st.caption(
            f"✓ Flight endurance loaded from file: {flight_endurance_value} seconds"
        )

    # Commented out fields (as per user request - DO NOT REMOVE)
    #    truck_capacity = st.text_input(
    #        "Truck capacity",
    #        value=PROBLEM2_CONFIG["system"]["truck_capacity"],
    #        key="p2_truck_capacity",
    #    )

    # waiting_limit = st.text_input(
    #    "Sample waiting limit",
    #    value=PROBLEM2_CONFIG["system"]["sample_waiting_limit"],
    #    key="p2_waiting_limit",
    # )

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
        # "truck_capacity": truck_capacity,
        "drone_capacity": drone_capacity,
        "flight_endurance_limit": flight_endurance,
        # "sample_waiting_limit": waiting_limit,
    }
