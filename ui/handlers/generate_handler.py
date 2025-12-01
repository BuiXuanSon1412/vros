# ui/handlers/generate_handler.py - Generate Data Handler

import streamlit as st
from utils.data_generator import DataGenerator


@st.cache_resource
def get_data_generator():
    return DataGenerator()


def handle_generate_button(problem_type, gen_params):
    """Handle generate data button click"""

    # Validate that vehicle configuration exists
    if not _validate_vehicle_config(problem_type, gen_params):
        st.error(
            "⚠️ Please configure vehicle parameters in the 'Vehicle System' tab first!"
        )
        return

    with st.spinner("Generating..."):
        data_gen = get_data_generator()

        # Generate customers using the custom method
        customers = data_gen.generate_customers_custom(problem_type, gen_params)

        # Generate depot
        coord_range = gen_params.get("coord_range", (-100, 100))
        coord_span = coord_range[1] - coord_range[0]
        depot = data_gen.generate_depot(coord_span)

        # Calculate distance matrix
        distance_matrix = data_gen.calculate_distance_matrix(customers, depot)

        # Store in session state
        _store_generated_data(
            problem_type, customers, depot, distance_matrix, gen_params
        )

        st.success("✅ Data generated successfully!")


def _validate_vehicle_config(problem_type, gen_params):
    """Validate that vehicle configuration parameters exist"""
    if problem_type in [1, 2]:
        required = ["staff_velocity", "drone_velocity", "num_staffs", "num_drones"]
    else:
        required = ["truck_velocity", "drone_velocity", "num_trucks", "num_drones"]

    return all(key in gen_params for key in required)


def _store_generated_data(problem_type, customers, depot, distance_matrix, gen_params):
    """Store generated data in session state"""
    st.session_state[f"customers_{problem_type}"] = customers
    st.session_state[f"depot_{problem_type}"] = depot
    st.session_state[f"distance_matrix_{problem_type}"] = distance_matrix
    st.session_state[f"file_vehicle_config_{problem_type}"] = None
    st.session_state[f"file_processed_{problem_type}"] = False
    st.session_state[f"last_uploaded_file_{problem_type}"] = None
    st.session_state[f"generation_params_{problem_type}"] = gen_params
