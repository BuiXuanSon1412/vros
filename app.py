# app.py - Main Streamlit Application with File Upload Support

import streamlit as st
from config.default_config import (
    ALGORITHMS,
    DEFAULT_VEHICLE_CONFIG,
)
from utils.data_generator import DataGenerator
from utils.visualizer import Visualizer
from utils.solver import DummySolver, AlgorithmRunner
from utils.file_parser import FileParser

# Page config
st.set_page_config(
    page_title="Vehicle Routing Optimization System",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# [CSS styling remains the same - omitted for brevity]
st.markdown(
    """
<style>

/* Remove Streamlit header completely */
    header[data-testid="stHeader"] {
        display: none;
    }
    
    /* Aggressively remove all top and bottom padding */
    .main .block-container {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        max-width: 100%;
    }
    
    /* Remove padding from main container */
    .main {
        padding: 0 !important;
    }
    
    /* Target the outermost container */
    section[data-testid="stAppViewContainer"] > .main {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    
    /* Remove default Streamlit padding */
    .element-container {
        margin: 0;
    }
    
    /* Remove space above block container */
    .block-container {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    
    /* Hide footer */
    footer {visibility: hidden;}
    
    /* Global styling */
    .main {
        background-color: #ffffff;
    }
    
    /* Header styling - Minimalist & Compact */
    .main-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #0f172a;
        text-align: center;
        margin-bottom: 0.15rem;
        margin-top: 0.5rem;
        padding-top: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .sub-header {
        font-size: 0.875rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 0.75rem;
        margin-top: 0;
        font-weight: 400;
    }
    
    /* Enhanced section titles - Minimalist */
    .config-section-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #1e293b;
        margin-top: 0.75rem;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 0.25rem;
        padding-left: 0;
    }
    
    .config-section-title::before {
        display: none;
    }
    
    /* Remove field descriptions to save space */
    .field-description {
        display: none;
    }
    
    /* Minimalist button styling */
    .stButton>button {
        width: 100%;
        background: #2563eb;
        color: white;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        border: none;
        transition: all 0.2s ease;
        font-size: 0.875rem;
        box-shadow: none;
    }
    
    .stButton>button:hover {
        background: #1d4ed8;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.2);
    }
    
    .stButton>button:active {
        transform: scale(0.98);
    }
    
    /* Secondary button */
    .stButton>button:not([kind="primary"]) {
        background: #f1f5f9;
        color: #1e293b;
        border: 1px solid #e2e8f0;
    }
    
    .stButton>button:not([kind="primary"]):hover {
        background: #e2e8f0;
        border-color: #cbd5e1;
    }
    
    /* Minimalist expander styling */
    div[data-testid="stExpander"] {
        background: #ffffff;
        border-radius: 6px;
        border: 1px solid #e2e8f0;
        margin-bottom: 0.75rem;
        transition: border-color 0.2s ease;
    }
    
    div[data-testid="stExpander"]:hover {
        border-color: #cbd5e1;
    }
    
    div[data-testid="stExpander"] summary {
        padding: 0.5rem 0.75rem;
        font-size: 0.875rem;
        font-weight: 600;
        color: #1e293b;
        background-color: transparent;
    }
    
    div[data-testid="stExpander"][aria-expanded="true"] {
        border-color: #3b82f6;
    }
    
    div[data-testid="stExpander"] > div:last-child {
        padding: 0.75rem;
        background-color: #ffffff;
    }
    
    /* Minimalist input fields */
    .stNumberInput label, .stSelectbox label {
        font-size: 0.8rem;
        margin-bottom: 0.25rem;
        font-weight: 500;
        color: #475569;
    }
    
    .stNumberInput, .stSelectbox {
        margin-bottom: 0.5rem;
    }
    
    /* Style input fields */
    .stNumberInput > div > div > input {
        border: 1px solid #e2e8f0;
        border-radius: 4px;
        padding: 0.4rem 0.6rem;
        font-size: 0.875rem;
        background-color: #ffffff;
        transition: border-color 0.2s ease;
    }
    
    .stNumberInput > div > div > input:hover {
        border-color: #cbd5e1;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.1);
        background-color: #ffffff;
    }
    
    /* Style selectbox */
    .stSelectbox > div > div > div {
        border: 1px solid #e2e8f0;
        border-radius: 4px;
        background-color: #ffffff;
        transition: border-color 0.2s ease;
    }
    
    .stSelectbox > div > div > div:hover {
        border-color: #cbd5e1;
    }
    
    /* Input increment/decrement buttons */
    .stNumberInput button {
        border-radius: 3px;
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        color: #64748b;
        transition: all 0.2s ease;
    }
    
    .stNumberInput button:hover {
        background-color: #f1f5f9;
        border-color: #cbd5e1;
        color: #475569;
    }
    
    /* Metric cards - minimalist */
    div[data-testid="stMetricValue"] {
        font-size: 1.25rem;
        font-weight: 600;
        color: #0f172a;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 0.75rem;
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Info/Success/Error boxes - minimalist */
    .stAlert {
        border-radius: 4px;
        border-left-width: 3px;
        padding: 0.5rem 0.75rem;
        font-size: 0.875rem;
    }
    
    /* Minimalist tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: transparent;
        padding: 0;
        border-bottom: 1px solid #e2e8f0;
        margin-top: 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 0;
        padding: 0.4rem 0.8rem;
        font-weight: 500;
        color: #64748b;
        font-size: 0.875rem;
        border-bottom: 2px solid transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: transparent;
        color: #2563eb;
        border-bottom: 2px solid #2563eb;
    }
    
    /* Remove tab content top padding */
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 0.75rem;
    }
    
    /* Dataframe - minimalist */
    .stDataFrame {
        border-radius: 4px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Footer - minimalist */
    .footer {
        text-align: center;
        color: #94a3b8;
        padding: 0.5rem 0;
        margin-top: 1rem;
        border-top: 1px solid #e2e8f0;
        font-size: 0.75rem;
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-top-color: #2563eb;
    }
    
    /* Charts container - minimalist */
    .js-plotly-plot {
        border-radius: 4px;
    }
    
    /* Reduce column gaps */
    .row-widget.stHorizontal {
        gap: 0.5rem;
    }
    
    /* Compact column spacing */
    div[data-testid="column"] {
        padding: 0 0.25rem;
    }

</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state for each problem independently
for problem_id in [1, 2, 3]:
    if f"customers_{problem_id}" not in st.session_state:
        st.session_state[f"customers_{problem_id}"] = None
    if f"depot_{problem_id}" not in st.session_state:
        st.session_state[f"depot_{problem_id}"] = None
    if f"distance_matrix_{problem_id}" not in st.session_state:
        st.session_state[f"distance_matrix_{problem_id}"] = None
    if f"solution_{problem_id}" not in st.session_state:
        st.session_state[f"solution_{problem_id}"] = None
    if f"results_{problem_id}" not in st.session_state:
        st.session_state[f"results_{problem_id}"] = {}
    if f"chart_counter_{problem_id}" not in st.session_state:
        st.session_state[f"chart_counter_{problem_id}"] = 0
    if f"data_source_{problem_id}" not in st.session_state:
        st.session_state[f"data_source_{problem_id}"] = (
            "generate"  # "generate" or "upload"
        )
    if f"file_vehicle_config_{problem_id}" not in st.session_state:
        st.session_state[f"file_vehicle_config_{problem_id}"] = None
    if f"file_processed_{problem_id}" not in st.session_state:
        st.session_state[f"file_processed_{problem_id}"] = False
    if f"last_uploaded_file_{problem_id}" not in st.session_state:
        st.session_state[f"last_uploaded_file_{problem_id}"] = None


# Initialize utilities
@st.cache_resource
def get_visualizer():
    return Visualizer()


@st.cache_resource
def get_data_generator():
    return DataGenerator()


@st.cache_resource
def get_file_parser():
    return FileParser()


viz = get_visualizer()
data_gen = get_data_generator()
file_parser = get_file_parser()

# Header
st.markdown(
    '<div class="main-header">Vehicle Routing Optimization</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">Optimize delivery schedules for trucks and drones</div>',
    unsafe_allow_html=True,
)

# Tabs for problem selection
problem_tabs = st.tabs(
    ["Problem 1: Min-Timespan", "Problem 2: Bi-Objective", "Problem 3: Resupply"]
)

for tab_idx, tab in enumerate(problem_tabs):
    problem_type = tab_idx + 1

    with tab:
        # Layout: Left (65%) - Visualization | Right (35%) - Config
        col_left, col_right = st.columns([65, 35])

        # ==================== RIGHT SIDE: CONFIGURATION ====================
        with col_right:
            st.markdown("### Configuration")

            # Configuration tabs
            config_tabs = st.tabs(["Vehicle System", "Dataset", "Algorithm"])

            # TAB 1: VEHICLE SYSTEM
            with config_tabs[0]:
                # Check if data was loaded from file
                is_file_loaded = (
                    st.session_state[f"data_source_{problem_type}"] == "upload"
                )
                file_config = st.session_state[f"file_vehicle_config_{problem_type}"]

                # Two columns for trucks and drones
                col_truck, col_drone = st.columns(2, gap="medium")

                with col_truck:
                    st.markdown("**Trucks**")
                    truck_count = st.number_input(
                        "Count",
                        min_value=1,
                        max_value=10,
                        value=DEFAULT_VEHICLE_CONFIG["truck"]["count"],
                        key=f"truck_count_{problem_type}",
                    )
                    truck_capacity = st.number_input(
                        "Capacity (kg)",
                        min_value=10,
                        max_value=500,
                        value=DEFAULT_VEHICLE_CONFIG["truck"]["capacity"],
                        disabled=is_file_loaded,
                        key=f"truck_capacity_{problem_type}",
                    )

                    truck_cost = st.number_input(
                        "Cost/km (VND)",
                        min_value=1000,
                        max_value=20000,
                        value=DEFAULT_VEHICLE_CONFIG["truck"]["cost_per_km"],
                        disabled=is_file_loaded,
                        key=f"truck_cost_{problem_type}",
                    )

                with col_drone:
                    st.markdown("**Drones**")
                    drone_count = st.number_input(
                        "Count",
                        min_value=0,
                        max_value=10,
                        value=DEFAULT_VEHICLE_CONFIG["drone"]["count"],
                        key=f"drone_count_{problem_type}",
                    )
                    drone_capacity = st.number_input(
                        "Capacity (kg)",
                        min_value=1,
                        max_value=50,
                        value=DEFAULT_VEHICLE_CONFIG["drone"]["capacity"],
                        key=f"drone_capacity_{problem_type}",
                    )
                    drone_speed = st.number_input(
                        "Speed (km/h)",
                        min_value=10,
                        max_value=120,
                        value=DEFAULT_VEHICLE_CONFIG["drone"]["speed"],
                        key=f"drone_speed_{problem_type}",
                    )
                    drone_energy = st.number_input(
                        "Energy (min)",
                        min_value=10,
                        max_value=120,
                        value=DEFAULT_VEHICLE_CONFIG["drone"]["energy_limit"],
                        key=f"drone_energy_{problem_type}",
                    )
                    drone_cost = st.number_input(
                        "Cost/km (VND)",
                        min_value=500,
                        max_value=10000,
                        value=DEFAULT_VEHICLE_CONFIG["drone"]["cost_per_km"],
                        disabled=is_file_loaded,
                        key=f"drone_cost_{problem_type}",
                    )

            vehicle_config = {
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

            # TAB 2: DATASET (Previously CUSTOMER)
            with config_tabs[1]:
                # Data source selection
                data_source = st.radio(
                    "Data Source",
                    ["Generate Data", "Upload File"],
                    key=f"data_source_radio_{problem_type}",
                    horizontal=True,
                )

                st.markdown("---")

                if data_source == "Generate Data":
                    st.session_state[f"data_source_{problem_type}"] = "generate"

                    # === SYSTEM METRICS SECTION ===
                    st.markdown("#### 🔧 System Metrics")

                    if problem_type in [1, 2]:
                        # Problem 1 & 2: Staff/Drone system
                        col1, col2 = st.columns([1, 3])

                        with col1:
                            enable_staff_vel = st.checkbox(
                                "✓", value=True, key=f"en_staff_vel_{problem_type}"
                            )
                            enable_drone_vel = st.checkbox(
                                "✓", value=True, key=f"en_drone_vel_{problem_type}"
                            )
                            enable_num_staff = st.checkbox(
                                "✓", value=True, key=f"en_num_staff_{problem_type}"
                            )
                            enable_num_drone = st.checkbox(
                                "✓", value=True, key=f"en_num_drone_{problem_type}"
                            )
                            enable_drone_flight = st.checkbox(
                                "✓", value=True, key=f"en_drone_flight_{problem_type}"
                            )
                            enable_sample_wait = st.checkbox(
                                "✓", value=False, key=f"en_sample_wait_{problem_type}"
                            )

                        with col2:
                            staff_velocity = st.number_input(
                                "Staff velocity (km/h)",
                                min_value=10,
                                max_value=100,
                                value=40,
                                disabled=not enable_staff_vel,
                                key=f"staff_vel_{problem_type}",
                            )
                            drone_velocity = st.number_input(
                                "Drone velocity (km/h)",
                                min_value=10,
                                max_value=120,
                                value=60,
                                disabled=not enable_drone_vel,
                                key=f"drone_vel_{problem_type}",
                            )
                            num_staffs = st.number_input(
                                "Number of staffs",
                                min_value=1,
                                max_value=10,
                                value=2,
                                disabled=not enable_num_staff,
                                key=f"num_staffs_{problem_type}",
                            )
                            num_drones = st.number_input(
                                "Number of drones",
                                min_value=0,
                                max_value=10,
                                value=3,
                                disabled=not enable_num_drone,
                                key=f"num_drones_{problem_type}",
                            )
                            drone_flight_time = st.number_input(
                                "Drone flight time limit (min)",
                                min_value=10,
                                max_value=120,
                                value=60,
                                disabled=not enable_drone_flight,
                                key=f"drone_flight_{problem_type}",
                            )
                            sample_wait_time = st.number_input(
                                "Sample waiting time limit (min)",
                                min_value=5,
                                max_value=120,
                                value=30,
                                disabled=not enable_sample_wait,
                                key=f"sample_wait_{problem_type}",
                            )

                    else:  # problem_type == 3
                        # Problem 3: Truck/Drone system with release dates
                        col1, col2 = st.columns([1, 3])

                        with col1:
                            enable_truck_vel = st.checkbox(
                                "✓", value=True, key=f"en_truck_vel_{problem_type}"
                            )
                            enable_drone_vel = st.checkbox(
                                "✓", value=True, key=f"en_drone_vel_{problem_type}"
                            )
                            enable_num_truck = st.checkbox(
                                "✓", value=True, key=f"en_num_truck_{problem_type}"
                            )
                            enable_num_drone = st.checkbox(
                                "✓", value=True, key=f"en_num_drone_{problem_type}"
                            )
                            enable_drone_capacity = st.checkbox(
                                "✓", value=True, key=f"en_drone_cap_{problem_type}"
                            )
                            enable_drone_flight = st.checkbox(
                                "✓", value=True, key=f"en_drone_flight_{problem_type}"
                            )

                        with col2:
                            truck_velocity = st.number_input(
                                "Truck velocity (km/h)",
                                min_value=10,
                                max_value=100,
                                value=40,
                                disabled=not enable_truck_vel,
                                key=f"truck_vel_{problem_type}",
                            )
                            drone_velocity = st.number_input(
                                "Drone velocity (km/h)",
                                min_value=10,
                                max_value=120,
                                value=60,
                                disabled=not enable_drone_vel,
                                key=f"drone_vel_{problem_type}",
                            )
                            num_trucks = st.number_input(
                                "Number of trucks",
                                min_value=1,
                                max_value=10,
                                value=3,
                                disabled=not enable_num_truck,
                                key=f"num_trucks_{problem_type}",
                            )
                            num_drones = st.number_input(
                                "Number of drones",
                                min_value=0,
                                max_value=10,
                                value=3,
                                disabled=not enable_num_drone,
                                key=f"num_drones_{problem_type}",
                            )
                            drone_capacity = st.number_input(
                                "Drone capacity (kg)",
                                min_value=1,
                                max_value=50,
                                value=4,
                                disabled=not enable_drone_capacity,
                                key=f"drone_cap_{problem_type}",
                            )
                            drone_flight_time = st.number_input(
                                "Drone flight time limit (min)",
                                min_value=10,
                                max_value=180,
                                value=90,
                                disabled=not enable_drone_flight,
                                key=f"drone_flight_{problem_type}",
                            )

                    st.markdown("---")

                    # === CUSTOMER METRICS SECTION ===
                    st.markdown("#### 👥 Customer Metrics")

                    col1, col2 = st.columns([1, 3])

                    with col1:
                        enable_num_customers = st.checkbox(
                            "✓", value=True, key=f"en_num_cust_{problem_type}"
                        )
                        enable_coord_range = st.checkbox(
                            "✓", value=True, key=f"en_coord_{problem_type}"
                        )
                        enable_demand = st.checkbox(
                            "✓", value=True, key=f"en_demand_{problem_type}"
                        )
                        if problem_type in [1, 2]:
                            enable_staff_only = st.checkbox(
                                "✓", value=False, key=f"en_staff_only_{problem_type}"
                            )
                            enable_service_truck = st.checkbox(
                                "✓", value=True, key=f"en_service_truck_{problem_type}"
                            )
                            enable_service_drone = st.checkbox(
                                "✓", value=True, key=f"en_service_drone_{problem_type}"
                            )
                        else:
                            enable_release_date = st.checkbox(
                                "✓", value=True, key=f"en_release_{problem_type}"
                            )

                    with col2:
                        num_customers = st.number_input(
                            "Number of customers",
                            min_value=5,
                            max_value=100,
                            value=20,
                            disabled=not enable_num_customers,
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
                                disabled=not enable_coord_range,
                                key=f"coord_min_{problem_type}",
                            )
                        with coord_col2:
                            coord_max = st.number_input(
                                "Max",
                                min_value=0,
                                max_value=500,
                                value=100,
                                disabled=not enable_coord_range,
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
                                disabled=not enable_demand,
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
                                disabled=not enable_demand,
                                key=f"demand_max_{problem_type}",
                            )

                        if problem_type in [1, 2]:
                            # Problem 1 & 2 specific
                            ratio_staff_only = st.slider(
                                "Ratio of served-only-by-staff (%)",
                                min_value=0,
                                max_value=100,
                                value=50,
                                disabled=not enable_staff_only,
                                key=f"ratio_staff_{problem_type}",
                            )

                            service_time_truck = st.number_input(
                                "Service time by truck (seconds)",
                                min_value=10,
                                max_value=300,
                                value=60,
                                disabled=not enable_service_truck,
                                key=f"service_truck_{problem_type}",
                            )

                            service_time_drone = st.number_input(
                                "Service time by drone (seconds)",
                                min_value=10,
                                max_value=300,
                                value=30,
                                disabled=not enable_service_drone,
                                key=f"service_drone_{problem_type}",
                            )
                        else:
                            # Problem 3 specific
                            st.markdown("**Release date range (time units)**")
                            release_col1, release_col2 = st.columns(2)
                            with release_col1:
                                release_min = st.number_input(
                                    "Min",
                                    min_value=0,
                                    max_value=100,
                                    value=0,
                                    disabled=not enable_release_date,
                                    key=f"release_min_{problem_type}",
                                )
                            with release_col2:
                                release_max = st.number_input(
                                    "Max",
                                    min_value=0,
                                    max_value=100,
                                    value=20,
                                    disabled=not enable_release_date,
                                    key=f"release_max_{problem_type}",
                                )

                    st.markdown("")
                    generate_button = st.button(
                        "Generate Data",
                        key=f"generate_{problem_type}",
                        use_container_width=True,
                    )

                    if generate_button:
                        with st.spinner("Generating..."):
                            # Collect generation parameters
                            gen_params = {
                                "num_customers": num_customers
                                if enable_num_customers
                                else 20,
                                "coord_range": (coord_min, coord_max)
                                if enable_coord_range
                                else (-100, 100),
                                "demand_range": (demand_min, demand_max)
                                if enable_demand
                                else (0.02, 0.1),
                            }

                            if problem_type in [1, 2]:
                                gen_params.update(
                                    {
                                        "staff_velocity": staff_velocity
                                        if enable_staff_vel
                                        else 40,
                                        "drone_velocity": drone_velocity
                                        if enable_drone_vel
                                        else 60,
                                        "num_staffs": num_staffs
                                        if enable_num_staff
                                        else 2,
                                        "num_drones": num_drones
                                        if enable_num_drone
                                        else 3,
                                        "drone_flight_time": drone_flight_time
                                        if enable_drone_flight
                                        else 60,
                                        "ratio_staff_only": ratio_staff_only / 100
                                        if enable_staff_only
                                        else 0.5,
                                        "service_time_truck": service_time_truck
                                        if enable_service_truck
                                        else 60,
                                        "service_time_drone": service_time_drone
                                        if enable_service_drone
                                        else 30,
                                    }
                                )
                            else:
                                gen_params.update(
                                    {
                                        "truck_velocity": truck_velocity
                                        if enable_truck_vel
                                        else 40,
                                        "drone_velocity": drone_velocity
                                        if enable_drone_vel
                                        else 60,
                                        "num_trucks": num_trucks
                                        if enable_num_truck
                                        else 3,
                                        "num_drones": num_drones
                                        if enable_num_drone
                                        else 3,
                                        "drone_capacity": drone_capacity
                                        if enable_drone_capacity
                                        else 4,
                                        "drone_flight_time": drone_flight_time
                                        if enable_drone_flight
                                        else 90,
                                        "release_range": (release_min, release_max)
                                        if enable_release_date
                                        else (0, 20),
                                    }
                                )

                            # Generate data with custom parameters
                            st.session_state[f"customers_{problem_type}"] = (
                                data_gen.generate_customers_custom(
                                    problem_type, gen_params
                                )
                            )
                            st.session_state[f"depot_{problem_type}"] = (
                                data_gen.generate_depot(
                                    coord_max - coord_min if enable_coord_range else 200
                                )
                            )
                            st.session_state[f"distance_matrix_{problem_type}"] = (
                                data_gen.calculate_distance_matrix(
                                    st.session_state[f"customers_{problem_type}"],
                                    st.session_state[f"depot_{problem_type}"],
                                )
                            )
                            st.session_state[f"file_vehicle_config_{problem_type}"] = (
                                None
                            )
                            st.session_state[f"file_processed_{problem_type}"] = False
                            st.session_state[f"last_uploaded_file_{problem_type}"] = (
                                None
                            )
                            st.session_state[f"generation_params_{problem_type}"] = (
                                gen_params
                            )
                            st.success("Data generated!")

                else:  # Upload File
                    st.session_state[f"data_source_{problem_type}"] = "upload"

                    # Display expected format
                    if problem_type == 1:
                        st.info(
                            "📄 Expected format: **6.5.1.txt**\n\nCustomers N\nCoordinate X  Coordinate Y  Demand"
                        )
                        expected_ext = ".txt"
                    elif problem_type == 2:
                        st.info(
                            "📄 Expected format: **10.10.1.txt**\n\nnumber_staff N\nnumber_drone M\n..."
                        )
                        expected_ext = ".txt"
                    else:  # problem_type == 3
                        st.info(
                            "📄 Expected format: **10.1.txt**\n\nXCOORD  YCOORD  DEMAND  RELEASE_DATE"
                        )
                        expected_ext = ".txt"

                    uploaded_file = st.file_uploader(
                        "Choose a file",
                        type=["txt", "dat"],
                        key=f"file_upload_{problem_type}",
                    )

                    if uploaded_file is not None:
                        # Check if this is a new file (avoid reprocessing)
                        file_id = f"{uploaded_file.name}_{uploaded_file.size}"

                        if (
                            st.session_state[f"last_uploaded_file_{problem_type}"]
                            != file_id
                        ):
                            try:
                                # Read file content
                                file_content = uploaded_file.read().decode("utf-8")

                                # Parse based on problem type
                                if problem_type == 1:
                                    customers_df, depot, distance_matrix = (
                                        file_parser.parse_problem1_file(file_content)
                                    )
                                    vehicle_config_from_file = None
                                elif problem_type == 2:
                                    (
                                        customers_df,
                                        depot,
                                        distance_matrix,
                                        vehicle_config_from_file,
                                    ) = file_parser.parse_problem2_file(file_content)
                                else:  # problem_type == 3
                                    (
                                        customers_df,
                                        depot,
                                        distance_matrix,
                                        vehicle_config_from_file,
                                    ) = file_parser.parse_problem3_file(file_content)

                                # Store in session state
                                st.session_state[f"customers_{problem_type}"] = (
                                    customers_df
                                )
                                st.session_state[f"depot_{problem_type}"] = depot
                                st.session_state[f"distance_matrix_{problem_type}"] = (
                                    distance_matrix
                                )
                                st.session_state[
                                    f"file_vehicle_config_{problem_type}"
                                ] = vehicle_config_from_file
                                st.session_state[
                                    f"last_uploaded_file_{problem_type}"
                                ] = file_id
                                st.session_state[f"file_processed_{problem_type}"] = (
                                    True
                                )

                                st.success(
                                    f"✅ File uploaded successfully! {len(customers_df)} customers loaded."
                                )

                            except Exception as e:
                                st.error(f"❌ Error parsing file: {str(e)}")
                                st.info(
                                    "Please check that your file matches the expected format."
                                )
                        else:
                            # File already processed, just show success message
                            if st.session_state[f"file_processed_{problem_type}"]:
                                customers_df = st.session_state[
                                    f"customers_{problem_type}"
                                ]
                                st.success(
                                    f"✅ File loaded: {len(customers_df)} customers"
                                )

                    # Display data metrics if data exists
                    if (
                        st.session_state[f"customers_{problem_type}"] is not None
                        and st.session_state[f"file_processed_{problem_type}"]
                    ):
                        st.markdown("**Data Metrics:**")
                        customers_df = st.session_state[f"customers_{problem_type}"]
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Customers", len(customers_df))
                        with col2:
                            st.metric(
                                "Avg Demand", f"{customers_df['demand'].mean():.2f} kg"
                            )
                        with col3:
                            if "service_time" in customers_df.columns:
                                st.metric(
                                    "Avg Service",
                                    f"{customers_df['service_time'].mean():.1f} min",
                                )

                    # Show current loaded data info if exists
                    if (
                        st.session_state[f"customers_{problem_type}"] is not None
                        and st.session_state[f"data_source_{problem_type}"] == "upload"
                    ):
                        st.markdown("---")
                        st.markdown("**Currently Loaded:**")
                        customers = st.session_state[f"customers_{problem_type}"]
                        depot = st.session_state[f"depot_{problem_type}"]
                        st.write(f"✓ {len(customers)} customers")
                        st.write(f"✓ Depot at ({depot['x']:.2f}, {depot['y']:.2f})")

            # TAB 3: ALGORITHM
            with config_tabs[2]:
                available_algorithms = ALGORITHMS[problem_type]

                selected_algorithm = st.selectbox(
                    "Algorithm", available_algorithms, key=f"algorithm_{problem_type}"
                )

                max_iterations = st.number_input(
                    "Max iterations",
                    min_value=10,
                    max_value=10000,
                    value=1000,
                    key=f"iterations_{problem_type}",
                )

                if "Tabu" in selected_algorithm:
                    col1, col2 = st.columns(2)
                    with col1:
                        tabu_tenure = st.number_input(
                            "Tabu tenure",
                            min_value=5,
                            max_value=50,
                            value=10,
                            key=f"tabu_tenure_{problem_type}",
                        )
                    with col2:
                        neighborhood_size = st.number_input(
                            "Neighborhood",
                            min_value=10,
                            max_value=100,
                            value=20,
                            key=f"neighborhood_{problem_type}",
                        )

                    algorithm_params = {
                        "max_iterations": max_iterations,
                        "tabu_tenure": tabu_tenure,
                        "neighborhood_size": neighborhood_size,
                    }
                elif "NSGA" in selected_algorithm or "MOEA" in selected_algorithm:
                    population_size = st.number_input(
                        "Population size",
                        min_value=20,
                        max_value=500,
                        value=100,
                        key=f"population_{problem_type}",
                    )

                    col1, col2 = st.columns(2)
                    with col1:
                        crossover_prob = st.number_input(
                            "Crossover prob",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.9,
                            step=0.1,
                            key=f"crossover_{problem_type}",
                        )
                    with col2:
                        mutation_prob = st.number_input(
                            "Mutation prob",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.1,
                            step=0.1,
                            key=f"mutation_{problem_type}",
                        )

                    algorithm_params = {
                        "max_iterations": max_iterations,
                        "population_size": population_size,
                        "crossover_prob": crossover_prob,
                        "mutation_prob": mutation_prob,
                    }
                else:
                    algorithm_params = {"max_iterations": max_iterations}

                st.markdown("")
                run_button = st.button(
                    "Run Algorithm",
                    type="primary",
                    key=f"run_{problem_type}",
                    use_container_width=True,
                )

            # Handle run button
            if run_button:
                if (
                    st.session_state[f"customers_{problem_type}"] is None
                    or st.session_state[f"depot_{problem_type}"] is None
                    or st.session_state[f"distance_matrix_{problem_type}"] is None
                ):
                    st.error("Please generate or upload data first!")
                else:
                    with st.spinner(f"Running {selected_algorithm}..."):
                        solver = DummySolver(problem_type, selected_algorithm)
                        solution = solver.solve(
                            st.session_state[f"customers_{problem_type}"],
                            st.session_state[f"depot_{problem_type}"],
                            st.session_state[f"distance_matrix_{problem_type}"],
                            vehicle_config,
                            algorithm_params,
                        )
                        st.session_state[f"solution_{problem_type}"] = solution
                        st.session_state[f"results_{problem_type}"][
                            selected_algorithm
                        ] = solution
                        st.session_state[f"chart_counter_{problem_type}"] += 1
                        st.success("Completed!")

        # ==================== LEFT SIDE: VISUALIZATION ====================
        with col_left:
            st.markdown("### Results & Visualization")

            if st.session_state[f"customers_{problem_type}"] is not None:
                # Tabs for different visualizations
                viz_tabs = st.tabs(
                    ["Map View", "Metrics", "Convergence", "Timeline", "Comparison"]
                )

                # TAB 1: MAP
                with viz_tabs[0]:
                    if (
                        st.session_state[f"solution_{problem_type}"] is not None
                        and st.session_state[f"depot_{problem_type}"] is not None
                    ):
                        fig_map = viz.plot_routes_2d(
                            st.session_state[f"customers_{problem_type}"],
                            st.session_state[f"depot_{problem_type}"],
                            st.session_state[f"solution_{problem_type}"]["routes"],
                            title=f"Routes - {st.session_state[f'solution_{problem_type}']['algorithm']}",
                        )
                        st.plotly_chart(
                            fig_map,
                            use_container_width=True,
                            key=f"map_routes_{problem_type}_{st.session_state[f'chart_counter_{problem_type}']}",
                        )
                    elif st.session_state[f"depot_{problem_type}"] is not None:
                        fig_map = viz.plot_routes_2d(
                            st.session_state[f"customers_{problem_type}"],
                            st.session_state[f"depot_{problem_type}"],
                            {},
                            title="Customer Locations",
                        )
                        st.plotly_chart(
                            fig_map,
                            use_container_width=True,
                            key=f"map_no_routes_{problem_type}_{st.session_state[f'chart_counter_{problem_type}']}",
                        )
                        st.info("Run algorithm to see routes")

                # TAB 2: METRICS
                with viz_tabs[1]:
                    if st.session_state[f"solution_{problem_type}"] is not None:
                        sol = st.session_state[f"solution_{problem_type}"]

                        metric_cols = st.columns(4)
                        with metric_cols[0]:
                            st.metric("Makespan", f"{sol['makespan']:.1f} min")
                        with metric_cols[1]:
                            st.metric("Cost", f"{sol['cost']:,.0f} VND")
                        with metric_cols[2]:
                            st.metric("Distance", f"{sol['total_distance']:.1f} km")
                        with metric_cols[3]:
                            st.metric("Time", f"{sol['computation_time']:.2f}s")

                        st.markdown("---")

                        st.markdown("**Route Details:**")
                        for vehicle_id, route in sol["routes"].items():
                            if route:
                                st.write(
                                    f"**{vehicle_id}:** {len(route)} customers - {route}"
                                )
                    else:
                        st.info("Run algorithm to see metrics")

                # TAB 3: CONVERGENCE
                with viz_tabs[2]:
                    if (
                        st.session_state[f"solution_{problem_type}"] is not None
                        and st.session_state[f"solution_{problem_type}"][
                            "convergence_history"
                        ]
                    ):
                        iterations, fitness = zip(
                            *st.session_state[f"solution_{problem_type}"][
                                "convergence_history"
                            ]
                        )
                        fig_conv = viz.plot_convergence(
                            list(iterations),
                            list(fitness),
                            title=f"Convergence - {st.session_state[f'solution_{problem_type}']['algorithm']}",
                        )
                        st.plotly_chart(
                            fig_conv,
                            use_container_width=True,
                            key=f"convergence_{problem_type}_{st.session_state[f'chart_counter_{problem_type}']}",
                        )

                        if (
                            problem_type == 2
                            and st.session_state[f"solution_{problem_type}"][
                                "pareto_front"
                            ]
                        ):
                            st.markdown("---")
                            fig_pareto = viz.plot_pareto_front(
                                st.session_state[f"solution_{problem_type}"][
                                    "pareto_front"
                                ],
                                title="Pareto Front",
                            )
                            st.plotly_chart(
                                fig_pareto,
                                use_container_width=True,
                                key=f"pareto_{problem_type}_{st.session_state[f'chart_counter_{problem_type}']}",
                            )
                    else:
                        st.info("Run algorithm to see convergence")

                # TAB 4: TIMELINE
                with viz_tabs[3]:
                    if (
                        st.session_state[f"solution_{problem_type}"] is not None
                        and st.session_state[f"solution_{problem_type}"]["schedule"]
                    ):
                        fig_gantt = viz.plot_gantt_chart(
                            st.session_state[f"solution_{problem_type}"]["schedule"],
                            title="Schedule Timeline",
                        )
                        st.plotly_chart(
                            fig_gantt,
                            use_container_width=True,
                            key=f"timeline_{problem_type}_{st.session_state[f'chart_counter_{problem_type}']}",
                        )
                    else:
                        st.info("Run algorithm to see timeline")

                # TAB 5: COMPARISON
                with viz_tabs[4]:
                    if len(st.session_state[f"results_{problem_type}"]) > 1:
                        runner = AlgorithmRunner(problem_type)
                        runner.results = st.session_state[f"results_{problem_type}"]
                        comparison_df = runner.get_comparison_summary()

                        st.markdown("**Comparison Table:**")
                        st.dataframe(comparison_df, use_container_width=True)

                        st.markdown("---")

                        col1, col2 = st.columns(2)
                        with col1:
                            fig_makespan = viz.plot_metrics_comparison(
                                comparison_df, "Makespan"
                            )
                            st.plotly_chart(
                                fig_makespan,
                                use_container_width=True,
                                key=f"result_{problem_type}_{st.session_state[f'chart_counter_{problem_type}']}",
                            )
                        with col2:
                            fig_cost = viz.plot_metrics_comparison(
                                comparison_df, "Cost"
                            )
                            st.plotly_chart(fig_cost, use_container_width=True)
                    else:
                        st.info("Run multiple algorithms to see comparison")

            else:
                st.info("Generate or upload data in the configuration panel to begin")

# Footer
st.markdown(
    '<div class="footer" style="margin-bottom: 0; padding-bottom: 0.5rem;">'
    "Vehicle Routing Optimization System"
    "</div>",
    unsafe_allow_html=True,
)
