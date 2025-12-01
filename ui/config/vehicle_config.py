# ui/config/vehicle_config.py - Vehicle Configuration

import streamlit as st
from config.default_config import DEFAULT_VEHICLE_CONFIG


def render_vehicle_config(problem_type):
    """Render vehicle configuration UI and return config dict"""
    is_file_loaded = st.session_state.get(f"data_source_{problem_type}") == "upload"
    file_config = st.session_state.get(f"file_vehicle_config_{problem_type}")

    # Two columns for trucks and drones
    col_truck, col_drone = st.columns(2, gap="medium")

    with col_truck:
        st.markdown("**🚛 Trucks**")
        truck_count = _render_truck_count(problem_type, is_file_loaded, file_config)
        truck_speed = _render_truck_speed(problem_type, is_file_loaded, file_config)
        truck_capacity = _render_truck_capacity(problem_type, is_file_loaded)
        truck_cost = _render_truck_cost(problem_type, is_file_loaded)

    with col_drone:
        st.markdown("**✈️ Drones**")
        drone_count = _render_drone_count(problem_type, is_file_loaded, file_config)
        drone_speed = _render_drone_speed(problem_type, is_file_loaded, file_config)
        drone_capacity = _render_drone_capacity(
            problem_type, is_file_loaded, file_config
        )
        drone_energy = _render_drone_energy(problem_type, is_file_loaded, file_config)
        drone_cost = _render_drone_cost(problem_type, is_file_loaded)

    return {
        "truck": {
            "count": truck_count,
            "capacity": truck_capacity,
            "speed": truck_speed,
            "cost_per_km": truck_cost,
        },
        "drone": {
            "count": drone_count,
            "capacity": drone_capacity,
            "speed": drone_speed,
            "energy_limit": drone_energy,
            "cost_per_km": drone_cost,
        },
    }


def _render_truck_count(problem_type, is_file_loaded, file_config):
    """Render truck count input"""
    if is_file_loaded and file_config:
        if problem_type == 2:
            value = file_config.get("num_staff", 2)
        elif problem_type == 3:
            value = file_config.get("number_truck", 2)
        else:
            return st.number_input(
                "Count",
                min_value=1,
                max_value=10,
                value=DEFAULT_VEHICLE_CONFIG["truck"]["count"],
                key=f"truck_count_{problem_type}",
            )
        return st.number_input(
            "Count",
            value=value,
            disabled=True,
            key=f"truck_count_{problem_type}",
        )
    else:
        return st.number_input(
            "Count",
            min_value=1,
            max_value=10,
            value=DEFAULT_VEHICLE_CONFIG["truck"]["count"],
            key=f"truck_count_{problem_type}",
        )


def _render_truck_speed(problem_type, is_file_loaded, file_config):
    """Render truck speed input"""
    if is_file_loaded and problem_type == 3 and file_config:
        return st.number_input(
            "Speed (km/h)",
            value=float(file_config.get("truck_speed", 0.5)),
            disabled=True,
            key=f"truck_speed_{problem_type}",
        )
    else:
        return st.number_input(
            "Speed (km/h)",
            min_value=10,
            max_value=100,
            value=DEFAULT_VEHICLE_CONFIG["truck"]["speed"],
            key=f"truck_speed_{problem_type}",
        )


def _render_truck_capacity(problem_type, is_file_loaded):
    """Render truck capacity input"""
    return st.number_input(
        "Capacity (kg)",
        min_value=10,
        max_value=500,
        value=DEFAULT_VEHICLE_CONFIG["truck"]["capacity"],
        disabled=is_file_loaded,
        key=f"truck_capacity_{problem_type}",
    )


def _render_truck_cost(problem_type, is_file_loaded):
    """Render truck cost input"""
    return st.number_input(
        "Cost/km (VND)",
        min_value=1000,
        max_value=20000,
        value=DEFAULT_VEHICLE_CONFIG["truck"]["cost_per_km"],
        disabled=is_file_loaded,
        key=f"truck_cost_{problem_type}",
    )


def _render_drone_count(problem_type, is_file_loaded, file_config):
    """Render drone count input"""
    if is_file_loaded and file_config:
        if problem_type == 2:
            value = file_config.get("num_drone", 3)
        elif problem_type == 3:
            value = file_config.get("number_drone", 3)
        else:
            return st.number_input(
                "Count",
                min_value=0,
                max_value=10,
                value=DEFAULT_VEHICLE_CONFIG["drone"]["count"],
                key=f"drone_count_{problem_type}",
            )
        return st.number_input(
            "Count",
            value=value,
            disabled=True,
            key=f"drone_count_{problem_type}",
        )
    else:
        return st.number_input(
            "Count",
            min_value=0,
            max_value=10,
            value=DEFAULT_VEHICLE_CONFIG["drone"]["count"],
            key=f"drone_count_{problem_type}",
        )


def _render_drone_speed(problem_type, is_file_loaded, file_config):
    """Render drone speed input"""
    if is_file_loaded and problem_type == 3 and file_config:
        return st.number_input(
            "Speed (km/h)",
            value=float(file_config.get("drone_speed", 1.0)),
            disabled=True,
            key=f"drone_speed_{problem_type}",
        )
    else:
        return st.number_input(
            "Speed (km/h)",
            min_value=10,
            max_value=120,
            value=DEFAULT_VEHICLE_CONFIG["drone"]["speed"],
            key=f"drone_speed_{problem_type}",
        )


def _render_drone_capacity(problem_type, is_file_loaded, file_config):
    """Render drone capacity input"""
    if is_file_loaded and problem_type == 3 and file_config:
        return st.number_input(
            "Capacity (kg)",
            value=float(file_config.get("M_d", 5)),
            disabled=True,
            key=f"drone_capacity_{problem_type}",
        )
    else:
        return st.number_input(
            "Capacity (kg)",
            min_value=1,
            max_value=50,
            value=DEFAULT_VEHICLE_CONFIG["drone"]["capacity"],
            disabled=is_file_loaded,
            key=f"drone_capacity_{problem_type}",
        )


def _render_drone_energy(problem_type, is_file_loaded, file_config):
    """Render drone energy limit input"""
    if is_file_loaded and file_config:
        if problem_type == 2:
            energy_val = file_config.get("drone_flight_time", 3600) / 60
        elif problem_type == 3:
            energy_val = file_config.get("L_d", 90)
        else:
            return st.number_input(
                "Energy (min)",
                min_value=10,
                max_value=120,
                value=DEFAULT_VEHICLE_CONFIG["drone"]["energy_limit"],
                key=f"drone_energy_{problem_type}",
            )
        return st.number_input(
            "Energy (min)",
            value=float(energy_val),
            disabled=True,
            key=f"drone_energy_{problem_type}",
        )
    else:
        return st.number_input(
            "Energy (min)",
            min_value=10,
            max_value=120,
            value=DEFAULT_VEHICLE_CONFIG["drone"]["energy_limit"],
            key=f"drone_energy_{problem_type}",
        )


def _render_drone_cost(problem_type, is_file_loaded):
    """Render drone cost input"""
    return st.number_input(
        "Cost/km (VND)",
        min_value=500,
        max_value=10000,
        value=DEFAULT_VEHICLE_CONFIG["drone"]["cost_per_km"],
        disabled=is_file_loaded,
        key=f"drone_cost_{problem_type}",
    )
