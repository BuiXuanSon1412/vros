# ui/config/system_config_p3.py - FIXED with dynamic keys

import streamlit as st
from config.default_config import PROBLEM3_CONFIG


def render_system_config_p3():
    """Render Problem 3 system configuration"""

    # Check if file is loaded
    file_loaded = st.session_state.get("file_processed_3", False)
    depot_data = st.session_state.get("depot_3")
    file_vehicle_config = st.session_state.get("file_vehicle_config_3")

    # Get file version to create unique keys
    file_version = st.session_state.get("file_version_3", 0)

    st.markdown("**Depot**")
    col1, col2 = st.columns(2)

    with col1:
        # Determine depot X value
        if file_loaded and depot_data:
            depot_x_value = float(depot_data["x"])
        else:
            depot_x_value = float(PROBLEM3_CONFIG["system"]["depot"]["x"])

        depot_x = st.text_input(
            "Coordinate X",
            value=depot_x_value,
            disabled=file_loaded,
            key=f"p3_depot_x_{file_version}",  # Dynamic key based on file version
        )

    with col2:
        # Determine depot Y value
        if file_loaded and depot_data:
            depot_y_value = float(depot_data["y"])
        else:
            depot_y_value = float(PROBLEM3_CONFIG["system"]["depot"]["y"])

        depot_y = st.text_input(
            "Coordinate Y",
            value=depot_y_value,
            disabled=file_loaded,
            key=f"p3_depot_y_{file_version}",  # Dynamic key
        )

    if file_loaded and depot_data:
        st.caption(
            f"✓ Depot location loaded from file: ({depot_data['x']}, {depot_data['y']})"
        )

    st.markdown("**Vehicle**")

    col1, col2 = st.columns(2)

    with col1:
        # Number of trucks
        if (
            file_loaded
            and file_vehicle_config
            and "number_truck" in file_vehicle_config
        ):
            num_trucks_value = int(file_vehicle_config["number_truck"])
        else:
            num_trucks_value = PROBLEM3_CONFIG["system"]["num_trucks"]

        num_trucks = st.text_input(
            "Number of trucks",
            value=num_trucks_value,
            # disabled=file_loaded,
            key=f"p3_num_trucks_{file_version}",  # Dynamic key
        )

        # Truck speed
        if file_loaded and file_vehicle_config and "truck_speed" in file_vehicle_config:
            truck_speed_value = float(file_vehicle_config["truck_speed"])
        else:
            truck_speed_value = PROBLEM3_CONFIG["system"]["truck_speed"]

        truck_speed = st.text_input(
            "Truck speed",
            value=truck_speed_value,
            # disabled=file_loaded,
            key=f"p3_truck_speed_{file_version}",  # Dynamic key
        )

    with col2:
        # Number of drones
        if (
            file_loaded
            and file_vehicle_config
            and "number_drone" in file_vehicle_config
        ):
            num_drones_value = int(file_vehicle_config["number_drone"])
        else:
            num_drones_value = PROBLEM3_CONFIG["system"]["num_drones"]

        num_drones = st.text_input(
            "Number of drones",
            value=num_drones_value,
            # disabled=file_loaded,
            key=f"p3_num_drones_{file_version}",  # Dynamic key
        )

        # Drone speed
        if file_loaded and file_vehicle_config and "drone_speed" in file_vehicle_config:
            drone_speed_value = float(file_vehicle_config["drone_speed"])
        else:
            drone_speed_value = PROBLEM3_CONFIG["system"]["drone_speed"]

        drone_speed = st.text_input(
            "Drone speed",
            value=drone_speed_value,
            # disabled=file_loaded,
            key=f"p3_drone_speed_{file_version}",  # Dynamic key
        )

    if file_loaded and file_vehicle_config:
        st.caption(
            f"✓ Vehicle configuration loaded: {num_trucks_value} trucks, {num_drones_value} drones"
        )

    st.markdown("**Drone Capacity & Constraints**")

    col1, col2 = st.columns(2)

    with col1:
        # Drone capacity
        if file_loaded and file_vehicle_config and "M_d" in file_vehicle_config:
            drone_capacity_value = int(file_vehicle_config["M_d"])
            # When file is loaded, show as disabled text input
            drone_capacity = st.text_input(
                "Drone capacity",
                value=drone_capacity_value,
                # disabled=True,
                key=f"p3_drone_capacity_{file_version}",  # Dynamic key
            )
        else:
            # When no file, show selectbox
            drone_capacity_value = PROBLEM3_CONFIG["system"]["drone_capacity_default"]
            try:
                default_index = PROBLEM3_CONFIG["system"][
                    "drone_capacity_options"
                ].index(drone_capacity_value)
            except ValueError:
                default_index = 0

            drone_capacity = st.selectbox(
                "Drone capacity",
                options=PROBLEM3_CONFIG["system"]["drone_capacity_options"],
                index=default_index,
                key=f"p3_drone_capacity_{file_version}",  # Dynamic key
            )

        # Flight endurance
        if file_loaded and file_vehicle_config and "L_d" in file_vehicle_config:
            flight_endurance_value = int(file_vehicle_config["L_d"])
        else:
            flight_endurance_value = PROBLEM3_CONFIG["system"]["flight_endurance_limit"]

        flight_endurance = st.text_input(
            "Flight endurance limit",
            value=flight_endurance_value,
            # disabled=file_loaded,
            key=f"p3_flight_endurance_{file_version}",  # Dynamic key
        )

    with col2:
        # Waiting limit (Note: This might not be in file, using Sigma or default)
        if file_loaded and file_vehicle_config and "Sigma" in file_vehicle_config:
            waiting_limit_value = int(file_vehicle_config["Sigma"])
        else:
            waiting_limit_value = PROBLEM3_CONFIG["system"]["sample_waiting_limit"]

        waiting_limit = st.text_input(
            "Sample waiting limit",
            value=waiting_limit_value,
            # disabled=file_loaded,
            key=f"p3_waiting_limit_{file_version}",  # Dynamic key
        )

    if file_loaded and file_vehicle_config:
        capacity_display = (
            drone_capacity
            if isinstance(drone_capacity, (int, float))
            else drone_capacity_value
        )
        st.caption(
            f"✓ Constraints loaded: Capacity={capacity_display}, Flight={flight_endurance_value} min"
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
