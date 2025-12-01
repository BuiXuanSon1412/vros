# ui/config/data_generation.py - Data Generation Form (Fixed)

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
    c1, c2 = st.columns([1, 6])
    with c1:
        en_num_cust = st.checkbox(
            "✓", value=True, key=f"en_num_cust_{problem_type}", help="Enable"
        )
    with c2:
        num_customers = st.number_input(
            "Number of customers",
            min_value=5,
            max_value=100,
            value=20,
            disabled=not en_num_cust,
            key=f"num_customers_{problem_type}",
        )

    # Coordinate range
    coord_col1, coord_col2, coord_col3 = st.columns([1, 3, 3])
    with coord_col1:
        en_coord = st.checkbox(
            "✓", value=True, key=f"en_coord_{problem_type}", help="Enable"
        )
    with coord_col2:
        coord_min = st.number_input(
            "Min coordinate",
            min_value=-500,
            max_value=0,
            value=-100,
            disabled=not en_coord,
            key=f"coord_min_{problem_type}",
        )
    with coord_col3:
        coord_max = st.number_input(
            "Max coordinate",
            min_value=0,
            max_value=500,
            value=100,
            disabled=not en_coord,
            key=f"coord_max_{problem_type}",
        )

    # Demand range
    dem_col1, dem_col2, dem_col3 = st.columns([1, 3, 3])
    with dem_col1:
        en_demand = st.checkbox(
            "✓", value=True, key=f"en_demand_{problem_type}", help="Enable"
        )
    with dem_col2:
        demand_min = st.number_input(
            "Min demand",
            min_value=0.01,
            max_value=10.0,
            value=0.02,
            step=0.01,
            format="%.2f",
            disabled=not en_demand,
            key=f"demand_min_{problem_type}",
        )
    with dem_col3:
        demand_max = st.number_input(
            "Max demand",
            min_value=0.01,
            max_value=10.0,
            value=0.1,
            step=0.01,
            format="%.2f",
            disabled=not en_demand,
            key=f"demand_max_{problem_type}",
        )

    # Problem-specific parameters
    extra_params = {}

    if problem_type in [1, 2]:
        # Staff-only ratio
        s1, s2 = st.columns([1, 6])
        with s1:
            en_staff_only = st.checkbox(
                "✓", value=False, key=f"en_staff_only_{problem_type}", help="Enable"
            )
        with s2:
            ratio_staff_only = st.number_input(
                "Ratio of staff-only (%)",
                min_value=0,
                max_value=100,
                value=50,
                disabled=not en_staff_only,
                key=f"ratio_staff_{problem_type}",
            )

        # Service time
        st.markdown("**Service time (seconds)**")
        st_col1, st_col2, st_col3 = st.columns([1, 3, 3])
        with st_col1:
            en_service = st.checkbox(
                "✓", value=True, key=f"en_service_{problem_type}", help="Enable"
            )
        with st_col2:
            service_time_truck = st.number_input(
                "Truck",
                min_value=10,
                max_value=300,
                value=60,
                disabled=not en_service,
                key=f"service_truck_{problem_type}",
            )
        with st_col3:
            service_time_drone = st.number_input(
                "Drone",
                min_value=10,
                max_value=300,
                value=30,
                disabled=not en_service,
                key=f"service_drone_{problem_type}",
            )

        extra_params = {
            "ratio_staff_only": ratio_staff_only / 100 if en_staff_only else 0.5,
            "service_time_truck": service_time_truck if en_service else 60,
            "service_time_drone": service_time_drone if en_service else 30,
        }

    else:  # Problem 3
        # Release range
        r1, r2 = st.columns([1, 6])
        with r1:
            en_release_min = st.checkbox(
                "✓", value=True, key=f"en_release_min_{problem_type}", help="Enable"
            )
        with r2:
            release_min = st.number_input(
                "Min release date",
                min_value=0,
                max_value=100,
                value=0,
                disabled=not en_release_min,
                key=f"release_min_{problem_type}",
            )

        r3, r4 = st.columns([1, 6])
        with r3:
            en_release_max = st.checkbox(
                "✓", value=True, key=f"en_release_max_{problem_type}", help="Enable"
            )
        with r4:
            release_max = st.number_input(
                "Max release date",
                min_value=0,
                max_value=100,
                value=20,
                disabled=not en_release_max,
                key=f"release_max_{problem_type}",
            )

        en_release = en_release_min and en_release_max
        extra_params = {
            "release_range": (release_min, release_max) if en_release else (0, 20),
        }

    return {
        "num_customers": num_customers if en_num_cust else 20,
        "coord_range": (coord_min, coord_max) if en_coord else (-100, 100),
        "demand_range": (demand_min, demand_max) if en_demand else (0.02, 0.1),
        **extra_params,
    }
