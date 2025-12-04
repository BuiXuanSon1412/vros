# ui/config/data_generation.py - Data Generation Form (No Checkboxes)

import streamlit as st
from ui.handlers.generate_handler import handle_generate_button


def render_data_generation(problem_type):
    """Render data generation UI - Customer parameters only"""

    customer_params = _render_customer_metrics(problem_type)

    # Generate button
    st.markdown("")
    generate_button = st.button(
        "Generate Data",
        key=f"generate_{problem_type}",
        use_container_width=True,
    )

    if generate_button:
        # Get vehicle config from Vehicle System tab
        vehicle_config = _get_vehicle_config_from_session(problem_type)

        # Combine parameters
        gen_params = {**customer_params, **vehicle_config}

        handle_generate_button(problem_type, gen_params)


def _get_vehicle_config_from_session(problem_type):
    """Extract vehicle configuration from session state (set by Vehicle System tab)"""
    # These values are set by the Vehicle System tab inputs
    if problem_type in [1, 2]:
        return {
            "staff_velocity": st.session_state.get(f"truck_speed_{problem_type}", 40),
            "drone_velocity": st.session_state.get(f"drone_speed_{problem_type}", 60),
            "num_staffs": st.session_state.get(f"truck_count_{problem_type}", 2),
            "num_drones": st.session_state.get(f"drone_count_{problem_type}", 3),
            "drone_flight_time": st.session_state.get(
                f"drone_energy_{problem_type}", 60
            ),
        }
    else:  # Problem 3
        return {
            "truck_velocity": st.session_state.get(f"truck_speed_{problem_type}", 40),
            "drone_velocity": st.session_state.get(f"drone_speed_{problem_type}", 60),
            "num_trucks": st.session_state.get(f"truck_count_{problem_type}", 3),
            "num_drones": st.session_state.get(f"drone_count_{problem_type}", 3),
            "drone_capacity": st.session_state.get(f"drone_capacity_{problem_type}", 4),
            "drone_flight_time": st.session_state.get(
                f"drone_energy_{problem_type}", 90
            ),
        }


def _render_customer_metrics(problem_type):
    """Render customer metrics section"""

    # Number of customers
    num_customers = st.number_input(
        "Number of customers",
        min_value=5,
        max_value=100,
        value=20,
        key=f"num_customers_{problem_type}",
    )

    # Coordinate range
    coord_col1, coord_col2 = st.columns(2)
    with coord_col1:
        coord_min = st.number_input(
            "Min coordinate",
            min_value=-500,
            max_value=0,
            value=-100,
            key=f"coord_min_{problem_type}",
        )
    with coord_col2:
        coord_max = st.number_input(
            "Max coordinate",
            min_value=0,
            max_value=500,
            value=100,
            key=f"coord_max_{problem_type}",
        )

    # Demand range
    dem_col1, dem_col2 = st.columns(2)
    with dem_col1:
        demand_min = st.number_input(
            "Min demand",
            min_value=0.01,
            max_value=10.0,
            value=0.02,
            step=0.01,
            format="%.2f",
            key=f"demand_min_{problem_type}",
        )
    with dem_col2:
        demand_max = st.number_input(
            "Max demand",
            min_value=0.01,
            max_value=10.0,
            value=0.1,
            step=0.01,
            format="%.2f",
            key=f"demand_max_{problem_type}",
        )

    # Problem-specific parameters
    extra_params = {}

    if problem_type in [1, 2]:
        # Staff-only ratio
        ratio_staff_only = st.number_input(
            "Ratio of staff-only (%)",
            min_value=0,
            max_value=100,
            value=50,
            key=f"ratio_staff_{problem_type}",
        )

        # Service time
        st.markdown("**Service time (seconds)**")
        st_col1, st_col2 = st.columns(2)
        with st_col1:
            service_time_truck = st.number_input(
                "Truck",
                min_value=10,
                max_value=300,
                value=60,
                key=f"service_truck_{problem_type}",
            )
        with st_col2:
            service_time_drone = st.number_input(
                "Drone",
                min_value=10,
                max_value=300,
                value=30,
                key=f"service_drone_{problem_type}",
            )

        extra_params = {
            "ratio_staff_only": ratio_staff_only / 100,
            "service_time_truck": service_time_truck,
            "service_time_drone": service_time_drone,
        }

    else:  # Problem 3
        # Release range
        release_min = st.number_input(
            "Min release date",
            min_value=0,
            max_value=100,
            value=0,
            key=f"release_min_{problem_type}",
        )

        release_max = st.number_input(
            "Max release date",
            min_value=0,
            max_value=100,
            value=20,
            key=f"release_max_{problem_type}",
        )

        extra_params = {
            "release_range": (release_min, release_max),
        }

    return {
        "num_customers": num_customers,
        "coord_range": (coord_min, coord_max),
        "demand_range": (demand_min, demand_max),
        **extra_params,
    }
