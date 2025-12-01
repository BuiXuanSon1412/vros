# ui/handlers/generate_handler.py - Generate Data Handler

import streamlit as st
from utils.data_generator import DataGenerator


@st.cache_resource
def get_data_generator():
    return DataGenerator()


def handle_generate_button(problem_type, gen_params):
    """Handle generate data button click"""
    with st.spinner("Generating..."):
        data_gen = get_data_generator()

        # Generate customers
        customers = data_gen.generate_customers(problem_type, gen_params)

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

        st.success("Data generated!")


def _store_generated_data(problem_type, customers, depot, distance_matrix, gen_params):
    """Store generated data in session state"""
    st.session_state[f"customers_{problem_type}"] = customers
    st.session_state[f"depot_{problem_type}"] = depot
    st.session_state[f"distance_matrix_{problem_type}"] = distance_matrix
    st.session_state[f"file_vehicle_config_{problem_type}"] = None
    st.session_state[f"file_processed_{problem_type}"] = False
    st.session_state[f"last_uploaded_file_{problem_type}"] = None
    st.session_state[f"generation_params_{problem_type}"] = gen_params
