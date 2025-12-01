# ui/config/data_generation.py - Data Generation Form

import streamlit as st
from ui.handlers.generate_handler import handle_generate_button


def render_data_generation(problem_type):
    """Render data generation UI"""
    # System Metrics Section
    st.markdown("#### 🔧 System Metrics")
    system_params = _render_system_metrics(problem_type)

    st.markdown("---")

    # Customer Metrics Section
    st.markdown("#### 👥 Customer Metrics")
    customer_params = _render_customer_metrics(problem_type)

    # Combine parameters
    gen_params = {**system_params, **customer_params}

    # Generate button
    st.markdown("")
    generate_button = st.button(
        "Generate Data",
        key=f"generate_{problem_type}",
        use_container_width=True,
    )

    if generate_button:
        handle_generate_button(problem_type, gen_params)


def _render_system_metrics(problem_type):
    """Render system metrics section"""
    if problem_type in [1, 2]:
        return _render_system_metrics_p1_p2(problem_type)
    else:
        return _render_system_metrics_p3(problem_type)


def _render_system_metrics_p1_p2(problem_type):
    """Render system metrics for Problem 1 & 2"""
    col1, col2 = st.columns([1, 3])

    # Checkboxes
    with col1:
        en_staff_vel = st.checkbox("✓", value=True, key=f"en_staff_vel_{problem_type}")
        en_drone_vel = st.checkbox("✓", value=True, key=f"en_drone_vel_{problem_type}")
        en_num_staff = st.checkbox("✓", value=True, key=f"en_num_staff_{problem_type}")
        en_num_drone = st.checkbox("✓", value=True, key=f"en_num_drone_{problem_type}")
        en_drone_flight = st.checkbox(
            "✓", value=True, key=f"en_drone_flight_{problem_type}"
        )
        en_sample_wait = st.checkbox(
            "✓", value=False, key=f"en_sample_wait_{problem_type}"
        )

    # Input fields
    with col2:
        staff_velocity = st.number_input(
            "Staff velocity (km/h)",
            min_value=10,
            max_value=100,
            value=40,
            disabled=not en_staff_vel,
            key=f"staff_vel_{problem_type}",
        )
        drone_velocity = st.number_input(
            "Drone velocity (km/h)",
            min_value=10,
            max_value=120,
            value=60,
            disabled=not en_drone_vel,
            key=f"drone_vel_{problem_type}",
        )
        num_staffs = st.number_input(
            "Number of staffs",
            min_value=1,
            max_value=10,
            value=2,
            disabled=not en_num_staff,
            key=f"num_staffs_{problem_type}",
        )
        num_drones = st.number_input(
            "Number of drones",
            min_value=0,
            max_value=10,
            value=3,
            disabled=not en_num_drone,
            key=f"num_drones_{problem_type}",
        )
        drone_flight_time = st.number_input(
            "Drone flight time limit (min)",
            min_value=10,
            max_value=120,
            value=60,
            disabled=not en_drone_flight,
            key=f"drone_flight_{problem_type}",
        )
        sample_wait_time = st.number_input(
            "Sample waiting time limit (min)",
            min_value=5,
            max_value=120,
            value=30,
            disabled=not en_sample_wait,
            key=f"sample_wait_{problem_type}",
        )

    return {
        "staff_velocity": staff_velocity if en_staff_vel else 40,
        "drone_velocity": drone_velocity if en_drone_vel else 60,
        "num_staffs": num_staffs if en_num_staff else 2,
        "num_drones": num_drones if en_num_drone else 3,
        "drone_flight_time": drone_flight_time if en_drone_flight else 60,
        "sample_wait_time": sample_wait_time if en_sample_wait else 30,
    }


def _render_system_metrics_p3(problem_type):
    """Render system metrics for Problem 3"""
    col1, col2 = st.columns([1, 3])

    # Checkboxes
    with col1:
        en_truck_vel = st.checkbox("✓", value=True, key=f"en_truck_vel_{problem_type}")
        en_drone_vel = st.checkbox("✓", value=True, key=f"en_drone_vel_{problem_type}")
        en_num_truck = st.checkbox("✓", value=True, key=f"en_num_truck_{problem_type}")
        en_num_drone = st.checkbox("✓", value=True, key=f"en_num_drone_{problem_type}")
        en_drone_cap = st.checkbox("✓", value=True, key=f"en_drone_cap_{problem_type}")
        en_drone_flight = st.checkbox(
            "✓", value=True, key=f"en_drone_flight_{problem_type}"
        )

    # Input fields
    with col2:
        truck_velocity = st.number_input(
            "Truck velocity (km/h)",
            min_value=10,
            max_value=100,
            value=40,
            disabled=not en_truck_vel,
            key=f"truck_vel_{problem_type}",
        )
        drone_velocity = st.number_input(
            "Drone velocity (km/h)",
            min_value=10,
            max_value=120,
            value=60,
            disabled=not en_drone_vel,
            key=f"drone_vel_{problem_type}",
        )
        num_trucks = st.number_input(
            "Number of trucks",
            min_value=1,
            max_value=10,
            value=3,
            disabled=not en_num_truck,
            key=f"num_trucks_{problem_type}",
        )
        num_drones = st.number_input(
            "Number of drones",
            min_value=0,
            max_value=10,
            value=3,
            disabled=not en_num_drone,
            key=f"num_drones_{problem_type}",
        )
        drone_capacity = st.number_input(
            "Drone capacity (kg)",
            min_value=1,
            max_value=50,
            value=4,
            disabled=not en_drone_cap,
            key=f"drone_cap_{problem_type}",
        )
        drone_flight_time = st.number_input(
            "Drone flight time limit (min)",
            min_value=10,
            max_value=180,
            value=90,
            disabled=not en_drone_flight,
            key=f"drone_flight_{problem_type}",
        )

    return {
        "truck_velocity": truck_velocity if en_truck_vel else 40,
        "drone_velocity": drone_velocity if en_drone_vel else 60,
        "num_trucks": num_trucks if en_num_truck else 3,
        "num_drones": num_drones if en_num_drone else 3,
        "drone_capacity": drone_capacity if en_drone_cap else 4,
        "drone_flight_time": drone_flight_time if en_drone_flight else 90,
    }


def _render_customer_metrics(problem_type):
    """Render customer metrics section"""
    col1, col2 = st.columns([1, 3])

    en_staff_only = None
    en_service_truck = None
    en_service_drone = None
    en_release = None
    # Checkboxes
    with col1:
        en_num_cust = st.checkbox("✓", value=True, key=f"en_num_cust_{problem_type}")
        en_coord = st.checkbox("✓", value=True, key=f"en_coord_{problem_type}")
        en_demand = st.checkbox("✓", value=True, key=f"en_demand_{problem_type}")

        if problem_type in [1, 2]:
            en_staff_only = st.checkbox(
                "✓", value=False, key=f"en_staff_only_{problem_type}"
            )
            en_service_truck = st.checkbox(
                "✓", value=True, key=f"en_service_truck_{problem_type}"
            )
            en_service_drone = st.checkbox(
                "✓", value=True, key=f"en_service_drone_{problem_type}"
            )
        else:
            en_release = st.checkbox("✓", value=True, key=f"en_release_{problem_type}")

    # Input fields
    with col2:
        num_customers = st.number_input(
            "Number of customers",
            min_value=5,
            max_value=100,
            value=20,
            disabled=not en_num_cust,
            key=f"num_customers_{problem_type}",
        )

        # Coordinate range
        st.markdown("**Coordinate range (km)**")
        coord_col1, coord_col2 = st.columns(2)
        with coord_col1:
            coord_min = st.number_input(
                "Min",
                min_value=-500,
                max_value=0,
                value=-100,
                disabled=not en_coord,
                key=f"coord_min_{problem_type}",
            )
        with coord_col2:
            coord_max = st.number_input(
                "Max",
                min_value=0,
                max_value=500,
                value=100,
                disabled=not en_coord,
                key=f"coord_max_{problem_type}",
            )

        # Demand range
        st.markdown("**Demand range (kg)**")
        demand_col1, demand_col2 = st.columns(2)
        with demand_col1:
            demand_min = st.number_input(
                "Min",
                min_value=0.01,
                max_value=10.0,
                value=0.02,
                step=0.01,
                format="%.2f",
                disabled=not en_demand,
                key=f"demand_min_{problem_type}",
            )
        with demand_col2:
            demand_max = st.number_input(
                "Max",
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
            ratio_staff_only = st.slider(
                "Ratio of served-only-by-staff (%)",
                min_value=0,
                max_value=100,
                value=50,
                disabled=not en_staff_only,
                key=f"ratio_staff_{problem_type}",
            )
            service_time_truck = st.number_input(
                "Service time by truck (seconds)",
                min_value=10,
                max_value=300,
                value=60,
                disabled=not en_service_truck,
                key=f"service_truck_{problem_type}",
            )
            service_time_drone = st.number_input(
                "Service time by drone (seconds)",
                min_value=10,
                max_value=300,
                value=30,
                disabled=not en_service_drone,
                key=f"service_drone_{problem_type}",
            )
            extra_params = {
                "ratio_staff_only": ratio_staff_only / 100 if en_staff_only else 0.5,
                "service_time_truck": service_time_truck if en_service_truck else 60,
                "service_time_drone": service_time_drone if en_service_drone else 30,
            }
        else:
            st.markdown("**Release date range (time units)**")
            release_col1, release_col2 = st.columns(2)
            with release_col1:
                release_min = st.number_input(
                    "Min",
                    min_value=0,
                    max_value=100,
                    value=0,
                    disabled=not en_release,
                    key=f"release_min_{problem_type}",
                )
            with release_col2:
                release_max = st.number_input(
                    "Max",
                    min_value=0,
                    max_value=100,
                    value=20,
                    disabled=not en_release,
                    key=f"release_max_{problem_type}",
                )
            extra_params = {
                "release_range": (release_min, release_max) if en_release else (0, 20),
            }

    return {
        "num_customers": num_customers if en_num_cust else 20,
        "coord_range": (coord_min, coord_max) if en_coord else (-100, 100),
        "demand_range": (demand_min, demand_max) if en_demand else (0.02, 0.1),
        **extra_params,
    }
