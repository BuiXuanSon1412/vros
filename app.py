# app.py - Main Streamlit Application with Compact Configuration

import streamlit as st
from config.default_config import (
    ALGORITHMS,
    DEFAULT_VEHICLE_CONFIG,
)
from utils.data_generator import DataGenerator
from utils.visualizer import Visualizer
from utils.solver import DummySolver, AlgorithmRunner

# Page config
st.set_page_config(
    page_title="Vehicle Routing Optimization System",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional Neutral Theme CSS
st.markdown(
    """
<style>
    /* Remove Streamlit header completely */
    header[data-testid="stHeader"] {
        display: none;
    }
    
    /* Remove top padding caused by header */
    .main .block-container {
        padding-top: 2rem;
    }
    
    /* Hide footer */
    footer {visibility: hidden;}
    
    /* Global styling */
    .main {
        background-color: #ffffff;
    }
    
    /* Header styling - Compact */
    .main-header {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0f172a;
        text-align: center;
        margin-bottom: 0.2rem;
        letter-spacing: -0.025em;
    }
    
    .sub-header {
        font-size: 0.95rem;
        color: #475569;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 400;
    }
    
    /* Enhanced section titles with icon-like effect */
    .config-section-title {
        font-size: 1rem;
        font-weight: 600;
        color: #1e40af;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        border-bottom: 2px solid #3b82f6;
        padding-bottom: 0.35rem;
        position: relative;
        padding-left: 0.75rem;
    }
    
    .config-section-title::before {
        content: '';
        position: absolute;
        left: 0;
        top: 50%;
        transform: translateY(-50%);
        width: 4px;
        height: 60%;
        background: linear-gradient(to bottom, #3b82f6, #2563eb);
        border-radius: 2px;
    }
    
    /* Remove field descriptions to save space */
    .field-description {
        display: none;
    }
    
    /* Enhanced button styling with gradient and effects */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        border-radius: 8px;
        padding: 0.65rem 1.25rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2);
        font-size: 0.9rem;
        letter-spacing: 0.01em;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
        transform: translateY(-2px);
    }
    
    .stButton>button:active {
        transform: translateY(0);
        box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2);
    }
    
    /* Secondary button (Generate Data) */
    .stButton>button:not([kind="primary"]) {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        color: #1e293b;
        border: 1.5px solid #cbd5e1;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    .stButton>button:not([kind="primary"]):hover {
        background: linear-gradient(135deg, #e2e8f0 0%, #cbd5e1 100%);
        border-color: #94a3b8;
        box-shadow: 0 3px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Enhanced expander styling with gradient */
    div[data-testid="stExpander"] {
        background: linear-gradient(to bottom, #ffffff, #f8fafc);
        border-radius: 8px;
        border: 1.5px solid #e2e8f0;
        margin-bottom: 0.75rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease;
    }
    
    div[data-testid="stExpander"]:hover {
        border-color: #cbd5e1;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
    }
    
    div[data-testid="stExpander"] summary {
        padding: 0.65rem 0.75rem;
        font-size: 0.95rem;
        font-weight: 600;
        color: #1e293b;
        background-color: rgba(248, 250, 252, 0.5);
        border-radius: 7px 7px 0 0;
        transition: background-color 0.2s ease;
    }
    
    div[data-testid="stExpander"] summary:hover {
        background-color: rgba(241, 245, 249, 0.8);
    }
    
    div[data-testid="stExpander"][aria-expanded="true"] {
        border-color: #3b82f6;
    }
    
    div[data-testid="stExpander"] > div:last-child {
        padding: 1rem;
        background-color: #ffffff;
    }
    
    /* Enhanced input fields with better styling */
    .stNumberInput label, .stSelectbox label {
        font-size: 0.85rem;
        margin-bottom: 0.3rem;
        font-weight: 500;
        color: #334155;
    }
    
    .stNumberInput, .stSelectbox {
        margin-bottom: 0.75rem;
    }
    
    /* Style input fields */
    .stNumberInput > div > div > input {
        border: 1.5px solid #e2e8f0;
        border-radius: 6px;
        padding: 0.5rem 0.75rem;
        font-size: 0.9rem;
        background-color: #ffffff;
        transition: all 0.2s ease;
    }
    
    .stNumberInput > div > div > input:hover {
        border-color: #cbd5e1;
        background-color: #f8fafc;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        background-color: #ffffff;
    }
    
    /* Style selectbox */
    .stSelectbox > div > div > div {
        border: 1.5px solid #e2e8f0;
        border-radius: 6px;
        background-color: #ffffff;
        transition: all 0.2s ease;
    }
    
    .stSelectbox > div > div > div:hover {
        border-color: #cbd5e1;
        background-color: #f8fafc;
    }
    
    /* Input increment/decrement buttons */
    .stNumberInput button {
        border-radius: 4px;
        background-color: #f1f5f9;
        border: 1px solid #e2e8f0;
        color: #475569;
        transition: all 0.2s ease;
    }
    
    .stNumberInput button:hover {
        background-color: #e2e8f0;
        border-color: #cbd5e1;
        color: #1e293b;
    }
    
    /* Metric cards styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 600;
        color: #0f172a;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 0.8rem;
        color: #64748b;
        font-weight: 500;
    }
    
    /* Info/Success/Error boxes - more compact */
    .stAlert {
        border-radius: 6px;
        border-left-width: 3px;
        padding: 0.5rem 0.75rem;
        font-size: 0.9rem;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #f8fafc;
        padding: 0.4rem;
        border-radius: 6px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px;
        padding: 0.4rem 0.8rem;
        font-weight: 500;
        color: #475569;
        font-size: 0.9rem;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        color: #2563eb;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 6px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        color: #64748b;
        padding: 1.5rem 0;
        margin-top: 2rem;
        border-top: 1px solid #e2e8f0;
        font-size: 0.85rem;
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-top-color: #2563eb;
    }
    
    /* Charts container */
    .js-plotly-plot {
        border-radius: 6px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
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

# Initialize session state
if "customers" not in st.session_state:
    st.session_state.customers = None
if "depot" not in st.session_state:
    st.session_state.depot = None
if "distance_matrix" not in st.session_state:
    st.session_state.distance_matrix = None
if "solution" not in st.session_state:
    st.session_state.solution = None
if "results" not in st.session_state:
    st.session_state.results = {}
if "chart_counter" not in st.session_state:
    st.session_state.chart_counter = 0


# Initialize utilities
@st.cache_resource
def get_visualizer():
    return Visualizer()


@st.cache_resource
def get_data_generator():
    return DataGenerator()


viz = get_visualizer()
data_gen = get_data_generator()

# Header
st.markdown(
    '<div class="main-header">Vehicle Routing Optimization System</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">Optimizing delivery schedules for trucks and drones</div>',
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

            # ACTION BUTTONS AT TOP
            available_algorithms = ALGORITHMS[problem_type]

            if len(available_algorithms) > 1:
                # Three buttons layout
                col_btn1, col_btn2, col_btn3 = st.columns(3)
                with col_btn1:
                    generate_button = st.button(
                        "Generate Data", key=f"generate_{problem_type}"
                    )
                with col_btn2:
                    run_button = st.button(
                        "Run Algorithm", type="primary", key=f"run_{problem_type}"
                    )
                with col_btn3:
                    compare_button = st.button(
                        "Compare All Algorithms", key=f"compare_{problem_type}"
                    )
            else:
                # Two buttons layout
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    generate_button = st.button(
                        "Generate Data", key=f"generate_{problem_type}"
                    )
                with col_btn2:
                    run_button = st.button(
                        "Run Algorithm", type="primary", key=f"run_{problem_type}"
                    )
                compare_button = False

            st.markdown("---")

            # VEHICLE CONFIGURATION - Using Expanders for compactness
            with st.expander("Vehicle System", expanded=True):
                # Two columns for trucks and drones with visual separation
                col_truck, col_drone = st.columns(2, gap="medium")

                with col_truck:
                    st.markdown("**🚛 Trucks**")
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
                        key=f"truck_capacity_{problem_type}",
                    )
                    truck_speed = st.number_input(
                        "Speed (km/h)",
                        min_value=10,
                        max_value=100,
                        value=DEFAULT_VEHICLE_CONFIG["truck"]["speed"],
                        key=f"truck_speed_{problem_type}",
                    )
                    truck_cost = st.number_input(
                        "Cost/km (VND)",
                        min_value=1000,
                        max_value=20000,
                        value=DEFAULT_VEHICLE_CONFIG["truck"]["cost_per_km"],
                        key=f"truck_cost_{problem_type}",
                    )

                with col_drone:
                    st.markdown("**✈️ Drones**")
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

            # CUSTOMER CONFIGURATION
            with st.expander("Customer", expanded=True):
                num_customers = st.number_input(
                    "Number of customers",
                    min_value=5,
                    max_value=100,
                    value=20,
                    key=f"num_customers_{problem_type}",
                )

                area_size = st.number_input(
                    "Service area size (km × km)",
                    min_value=10,
                    max_value=100,
                    value=50,
                    key=f"area_size_{problem_type}",
                )

                col1, col2 = st.columns(2)
                with col1:
                    demand_min = st.number_input(
                        "Min demand (kg)",
                        min_value=1,
                        max_value=50,
                        value=1,
                        key=f"demand_min_{problem_type}",
                    )
                    service_min = st.number_input(
                        "Min service (min)",
                        min_value=1,
                        max_value=60,
                        value=5,
                        key=f"service_min_{problem_type}",
                    )
                with col2:
                    demand_max = st.number_input(
                        "Max demand (kg)",
                        min_value=1,
                        max_value=50,
                        value=10,
                        key=f"demand_max_{problem_type}",
                    )
                    service_max = st.number_input(
                        "Max service (min)",
                        min_value=1,
                        max_value=60,
                        value=15,
                        key=f"service_max_{problem_type}",
                    )

                priority_levels = st.number_input(
                    "Priority levels",
                    min_value=1,
                    max_value=5,
                    value=3,
                    key=f"priority_{problem_type}",
                )

            # ALGORITHM CONFIGURATION
            with st.expander("Algorithm", expanded=True):
                selected_algorithm = st.selectbox(
                    "Algorithm",
                    available_algorithms,
                    key=f"algorithm_{problem_type}",
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

            # Handle button actions
            if generate_button:
                with st.spinner("Generating..."):
                    st.session_state.customers = data_gen.generate_customers(
                        num_customers,
                        area_size,
                        (demand_min, demand_max),
                        (service_min, service_max),
                        priority_levels,
                    )
                    st.session_state.depot = data_gen.generate_depot(area_size)
                    st.session_state.distance_matrix = (
                        data_gen.calculate_distance_matrix(
                            st.session_state.customers, st.session_state.depot
                        )
                    )
                    st.success("Data generated!")

            if run_button:
                if (
                    st.session_state.customers is None
                    or st.session_state.depot is None
                    or st.session_state.distance_matrix is None
                ):
                    st.error("Please generate data first!")
                else:
                    with st.spinner(f"Running {selected_algorithm}..."):
                        solver = DummySolver(problem_type, selected_algorithm)
                        solution = solver.solve(
                            st.session_state.customers,
                            st.session_state.depot,
                            st.session_state.distance_matrix,
                            vehicle_config,
                            algorithm_params,
                        )
                        st.session_state.solution = solution
                        st.session_state.results[selected_algorithm] = solution
                        st.session_state.chart_counter += 1
                        st.success("Completed!")

            # Handle compare button
            if compare_button:
                if (
                    st.session_state.customers is None
                    or st.session_state.depot is None
                    or st.session_state.distance_matrix is None
                ):
                    st.error("Please generate data first!")
                else:
                    with st.spinner("Running all algorithms..."):
                        runner = AlgorithmRunner(problem_type)
                        all_results = runner.run_multiple_algorithms(
                            available_algorithms,
                            st.session_state.customers,
                            st.session_state.depot,
                            st.session_state.distance_matrix,
                            vehicle_config,
                            algorithm_params,
                        )
                        st.session_state.results = all_results
                        st.success("Comparison completed!")

        # ==================== LEFT SIDE: VISUALIZATION ====================
        with col_left:
            st.markdown("### Results & Visualization")

            if st.session_state.customers is not None:
                # Tabs for different visualizations
                viz_tabs = st.tabs(
                    ["Map View", "Metrics", "Convergence", "Timeline", "Comparison"]
                )

                # TAB 1: MAP
                with viz_tabs[0]:
                    if (
                        st.session_state.solution is not None
                        and st.session_state.depot is not None
                    ):
                        fig_map = viz.plot_routes_2d(
                            st.session_state.customers,
                            st.session_state.depot,
                            st.session_state.solution["routes"],
                            title=f"Routes - {st.session_state.solution['algorithm']}",
                        )
                        st.plotly_chart(
                            fig_map,
                            use_container_width=True,
                            key=f"map_routes_{problem_type}_{st.session_state.chart_counter}",
                        )
                    elif st.session_state.depot is not None:
                        fig_map = viz.plot_routes_2d(
                            st.session_state.customers,
                            st.session_state.depot,
                            {},
                            title="Customer Locations",
                        )
                        st.plotly_chart(
                            fig_map,
                            use_container_width=True,
                            key=f"map_no_routes_{problem_type}_{st.session_state.chart_counter}",
                        )
                        st.info("Run algorithm to see routes")

                # TAB 2: METRICS
                with viz_tabs[1]:
                    if st.session_state.solution is not None:
                        sol = st.session_state.solution

                        # Display metrics
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

                        # Routes detail
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
                        st.session_state.solution is not None
                        and st.session_state.solution["convergence_history"]
                    ):
                        iterations, fitness = zip(
                            *st.session_state.solution["convergence_history"]
                        )
                        fig_conv = viz.plot_convergence(
                            list(iterations),
                            list(fitness),
                            title=f"Convergence - {st.session_state.solution['algorithm']}",
                        )
                        st.plotly_chart(
                            fig_conv,
                            use_container_width=True,
                            key=f"convergence_{problem_type}_{st.session_state.chart_counter}",
                        )

                        # Pareto front for multi-objective
                        if (
                            problem_type == 2
                            and st.session_state.solution["pareto_front"]
                        ):
                            st.markdown("---")
                            fig_pareto = viz.plot_pareto_front(
                                st.session_state.solution["pareto_front"],
                                title="Pareto Front",
                            )
                            st.plotly_chart(
                                fig_pareto,
                                use_container_width=True,
                                key=f"pareto_{problem_type}_{st.session_state.chart_counter}",
                            )
                    else:
                        st.info("Run algorithm to see convergence")

                # TAB 4: TIMELINE
                with viz_tabs[3]:
                    if (
                        st.session_state.solution is not None
                        and st.session_state.solution["schedule"]
                    ):
                        fig_gantt = viz.plot_gantt_chart(
                            st.session_state.solution["schedule"],
                            title="Schedule Timeline",
                        )
                        st.plotly_chart(
                            fig_gantt,
                            use_container_width=True,
                            key=f"timeline_{problem_type}_{st.session_state.chart_counter}",
                        )
                    else:
                        st.info("Run algorithm to see timeline")

                # TAB 5: COMPARISON
                with viz_tabs[4]:
                    if len(st.session_state.results) > 1:
                        runner = AlgorithmRunner(problem_type)
                        runner.results = st.session_state.results
                        comparison_df = runner.get_comparison_summary()

                        st.markdown("**Comparison Table:**")
                        st.dataframe(
                            comparison_df,
                            use_container_width=True,
                        )

                        st.markdown("---")

                        col1, col2 = st.columns(2)
                        with col1:
                            fig_makespan = viz.plot_metrics_comparison(
                                comparison_df, "Makespan"
                            )
                            st.plotly_chart(
                                fig_makespan,
                                use_container_width=True,
                                key=f"result_{problem_type}_{st.session_state.chart_counter}",
                            )
                        with col2:
                            fig_cost = viz.plot_metrics_comparison(
                                comparison_df, "Cost"
                            )
                            st.plotly_chart(
                                fig_cost,
                                use_container_width=True,
                            )
                    else:
                        st.info("Run 'Compare All Algorithms' to see comparison")

            else:
                st.info("Generate data in the configuration panel to begin")

# Footer
st.markdown("---")
st.markdown(
    '<div class="footer">'
    "Vehicle Routing Optimization System | Built with Streamlit & Python"
    "</div>",
    unsafe_allow_html=True,
)
