# ui/config/vehicle_config.py - Vehicle Configuration (No Checkboxes)

import streamlit as st
from config.default_config import DEFAULT_VEHICLE_CONFIG


def render_vehicle_config(problem_type):
    """Render vehicle configuration UI and return config dict"""
    is_file_loaded = st.session_state.get(f"data_source_{problem_type}") == "upload"
    file_config = st.session_state.get(f"file_vehicle_config_{problem_type}")

    # Two columns for trucks and drones
    col_truck, col_drone = st.columns(2, gap="medium")

    with col_truck:
        st.markdown("**Trucks**")
        truck_params = _render_truck_config(problem_type, is_file_loaded, file_config)

    with col_drone:
        st.markdown("**Drones**")
        drone_params = _render_drone_config(problem_type, is_file_loaded, file_config)

    return {
        "truck": truck_params,
        "drone": drone_params,
    }


def _render_truck_config(problem_type, is_file_loaded, file_config):
    """Render truck configuration section"""
    # Get default values from file if loaded
    if is_file_loaded and file_config:
        if problem_type == 2:
            default_count = file_config.get("num_staff", 2)
            default_speed = DEFAULT_VEHICLE_CONFIG["truck"]["speed"]
        elif problem_type == 3:
            default_count = file_config.get("number_truck", 2)
            default_speed = file_config.get("truck_speed", 0.5)
        else:
            default_count = DEFAULT_VEHICLE_CONFIG["truck"]["count"]
            default_speed = DEFAULT_VEHICLE_CONFIG["truck"]["speed"]
    else:
        default_count = DEFAULT_VEHICLE_CONFIG["truck"]["count"]
        default_speed = DEFAULT_VEHICLE_CONFIG["truck"]["speed"]

    # Count
    truck_count = st.number_input(
        "Count",
        min_value=1,
        max_value=10,
        value=int(default_count),
        disabled=is_file_loaded,
        key=f"truck_count_{problem_type}",
    )

    # Speed
    truck_speed = st.number_input(
        "Speed",
        min_value=10,
        max_value=100,
        value=int(
            default_speed
            if default_speed >= 10
            else DEFAULT_VEHICLE_CONFIG["truck"]["speed"]
        ),
        disabled=is_file_loaded,
        key=f"truck_speed_{problem_type}",
    )

    # Capacity
    truck_capacity = st.number_input(
        "Capacity",
        min_value=10,
        max_value=500,
        value=DEFAULT_VEHICLE_CONFIG["truck"]["capacity"],
        disabled=is_file_loaded,
        key=f"truck_capacity_{problem_type}",
    )

    # Cost
    truck_cost = st.number_input(
        "Cost",
        min_value=1000,
        max_value=20000,
        step=500,
        value=DEFAULT_VEHICLE_CONFIG["truck"]["cost_per_km"],
        disabled=is_file_loaded,
        key=f"truck_cost_{problem_type}",
    )

    return {
        "count": truck_count,
        "capacity": truck_capacity,
        "speed": truck_speed,
        "cost_per_km": truck_cost,
    }


def _render_drone_config(problem_type, is_file_loaded, file_config):
    """Render drone configuration section"""
    # Get default values from file if loaded
    if is_file_loaded and file_config:
        if problem_type == 2:
            default_count = file_config.get("num_drone", 3)
            default_speed = DEFAULT_VEHICLE_CONFIG["drone"]["speed"]
            default_capacity = DEFAULT_VEHICLE_CONFIG["drone"]["capacity"]
            default_energy = file_config.get("drone_flight_time", 3600) / 60
        elif problem_type == 3:
            default_count = file_config.get("number_drone", 3)
            default_speed = file_config.get("drone_speed", 1.0)
            default_capacity = file_config.get("M_d", 5)
            default_energy = file_config.get("L_d", 90)
        else:
            default_count = DEFAULT_VEHICLE_CONFIG["drone"]["count"]
            default_speed = DEFAULT_VEHICLE_CONFIG["drone"]["speed"]
            default_capacity = DEFAULT_VEHICLE_CONFIG["drone"]["capacity"]
            default_energy = DEFAULT_VEHICLE_CONFIG["drone"]["energy_limit"]
    else:
        default_count = DEFAULT_VEHICLE_CONFIG["drone"]["count"]
        default_speed = DEFAULT_VEHICLE_CONFIG["drone"]["speed"]
        default_capacity = DEFAULT_VEHICLE_CONFIG["drone"]["capacity"]
        default_energy = DEFAULT_VEHICLE_CONFIG["drone"]["energy_limit"]

    # Count
    drone_count = st.number_input(
        "Count",
        min_value=0,
        max_value=10,
        value=int(default_count),
        disabled=is_file_loaded,
        key=f"drone_count_{problem_type}",
    )

    # Speed
    drone_speed = st.number_input(
        "Speed",
        min_value=10,
        max_value=120,
        value=int(default_speed)
        if default_speed >= 10
        else DEFAULT_VEHICLE_CONFIG["drone"]["speed"],
        disabled=is_file_loaded,
        key=f"drone_speed_{problem_type}",
    )

    # Capacity
    drone_capacity = st.number_input(
        "Capacity",
        min_value=1,
        max_value=50,
        value=int(default_capacity),
        disabled=is_file_loaded,
        key=f"drone_capacity_{problem_type}",
    )

    # Energy limit
    drone_energy = st.number_input(
        "Energy limit",
        min_value=10,
        max_value=120,
        value=int(default_energy),
        disabled=is_file_loaded,
        key=f"drone_energy_{problem_type}",
    )

    # Cost
    drone_cost = st.number_input(
        "Cost",
        min_value=500,
        max_value=10000,
        step=250,
        value=DEFAULT_VEHICLE_CONFIG["drone"]["cost_per_km"],
        disabled=is_file_loaded,
        key=f"drone_cost_{problem_type}",
    )

    return {
        "count": drone_count,
        "capacity": drone_capacity,
        "speed": drone_speed,
        "energy_limit": drone_energy,
        "cost_per_km": drone_cost,
    }
