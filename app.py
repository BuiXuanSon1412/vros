# app.py - Main Streamlit Application with Improved Config Panel

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

# Custom CSS - Removed dark backgrounds, improved readability
st.markdown(
    """
<style>
    .main-header {
        font-size: 1.5rem;  /* Changed from 2.5rem */
        font-weight: 700;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 0.3rem;  /* Changed from 0.5rem */
    }
    .sub-header {
        font-size: 0.95rem;  /* Changed from 1.2rem */
        color: #64748b;
        text-align: center;
        margin-bottom: 1rem;  /* Changed from 2rem */
    }

    /* Force light text for all labels and content */
    .stNumberInput label, 
    .stSlider label, 
    .stSelectbox label,
    div[data-testid="stExpander"] p,
    div[data-testid="stExpander"] strong {
        color: #f3f4f6 !important;  /* Light gray text */
        font-weight: 600 !important;
    }

    /* Make expander content area have dark background explicitly */
    div[data-testid="stExpander"] div[role="region"] {
        background-color: #1f2937 !important;
        color: #f3f4f6 !important;
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
    '<div class="main-header">Vehicle Routing Optimization</div>',
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
        # Layout: Left (70%) - Visualization | Right (30%) - Config
        col_left, col_right = st.columns([7, 3])

        # ==================== RIGHT SIDE: CONFIGURATION ====================
        with col_right:
            # ========== STICKY ACTION BUTTONS AT TOP ==========
            st.markdown('<div class="action-container">', unsafe_allow_html=True)
            st.markdown("### Quick Actions")

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                generate_button = st.button(
                    "Generate Data",
                    key=f"generate_{problem_type}",
                    use_container_width=True,
                    help="Generate random customer data",
                    type="secondary",
                )
            with col_btn2:
                run_button = st.button(
                    "Run Algorithm",
                    type="primary",
                    key=f"run_{problem_type}",
                    use_container_width=True,
                    help="Run the selected algorithm",
                )

            # Show data status
            if st.session_state.customers is not None:
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    st.metric(
                        "Customers",
                        len(st.session_state.customers),
                        label_visibility="visible",
                    )
                with col_s2:
                    total_vehicles = st.session_state.get(
                        f"truck_count_{problem_type}", 2
                    ) + st.session_state.get(f"drone_count_{problem_type}", 3)
                    st.metric("Vehicles", total_vehicles)
                with col_s3:
                    if st.session_state.solution:
                        st.metric("Status", "Ready")
                    else:
                        st.metric("Status", "Waiting")

            st.markdown("</div>", unsafe_allow_html=True)

            # ========== PRESET SELECTOR ==========
            st.markdown("### Configuration")

            preset = st.selectbox(
                "Quick Start Preset",
                [
                    "Custom",
                    "Urban (Small Fleet)",
                    "Regional (Large Fleet)",
                    "Drone-Heavy Mix",
                ],
                key=f"preset_{problem_type}",
                help="Choose a preset configuration or customize your own",
            )

            # Load preset values based on selection
            if preset == "Urban (Small Fleet)":
                default_truck_count, default_drone_count = 1, 2
                default_customers, default_area = 15, 30
            elif preset == "Regional (Large Fleet)":
                default_truck_count, default_drone_count = 4, 2
                default_customers, default_area = 40, 70
            elif preset == "Drone-Heavy Mix":
                default_truck_count, default_drone_count = 1, 5
                default_customers, default_area = 25, 50
            else:  # Custom
                default_truck_count = DEFAULT_VEHICLE_CONFIG["truck"]["count"]
                default_drone_count = DEFAULT_VEHICLE_CONFIG["drone"]["count"]
                default_customers, default_area = 20, 50

            # ========== COLLAPSIBLE SECTIONS ==========

            # 1. FLEET CONFIGURATION
            with st.expander("Fleet Configuration", expanded=True):
                st.markdown("**Trucks**")
                col1, col2 = st.columns(2)
                with col1:
                    truck_count = st.number_input(
                        "Count",
                        1,
                        10,
                        default_truck_count,
                        key=f"truck_count_{problem_type}",
                    )
                    truck_speed = st.number_input(
                        "Speed (km/h)",
                        10,
                        100,
                        DEFAULT_VEHICLE_CONFIG["truck"]["speed"],
                        key=f"truck_speed_{problem_type}",
                    )
                with col2:
                    truck_capacity = st.number_input(
                        "Capacity (kg)",
                        10,
                        500,
                        DEFAULT_VEHICLE_CONFIG["truck"]["capacity"],
                        key=f"truck_capacity_{problem_type}",
                    )
                    truck_cost = st.number_input(
                        "Cost/km",
                        1000,
                        20000,
                        DEFAULT_VEHICLE_CONFIG["truck"]["cost_per_km"],
                        key=f"truck_cost_{problem_type}",
                    )

                st.markdown("---")
                st.markdown("**Drones**")
                col1, col2 = st.columns(2)
                with col1:
                    drone_count = st.number_input(
                        "Count",
                        0,
                        10,
                        default_drone_count,
                        key=f"drone_count_{problem_type}",
                    )
                    drone_speed = st.number_input(
                        "Speed (km/h)",
                        10,
                        120,
                        DEFAULT_VEHICLE_CONFIG["drone"]["speed"],
                        key=f"drone_speed_{problem_type}",
                    )
                with col2:
                    drone_capacity = st.number_input(
                        "Capacity (kg)",
                        1,
                        50,
                        DEFAULT_VEHICLE_CONFIG["drone"]["capacity"],
                        key=f"drone_capacity_{problem_type}",
                    )
                    drone_energy = st.number_input(
                        "Battery (min)",
                        10,
                        120,
                        DEFAULT_VEHICLE_CONFIG["drone"]["energy_limit"],
                        key=f"drone_energy_{problem_type}",
                    )

                drone_cost = st.number_input(
                    "Cost per km (VND)",
                    500,
                    10000,
                    DEFAULT_VEHICLE_CONFIG["drone"]["cost_per_km"],
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

            # 2. CUSTOMER CONFIGURATION
            with st.expander("Customer & Area", expanded=False):
                num_customers = st.slider(
                    "Number of customers",
                    5,
                    100,
                    default_customers,
                    key=f"num_customers_{problem_type}",
                )

                area_size = st.slider(
                    "Service area (km x km)",
                    10,
                    100,
                    default_area,
                    key=f"area_size_{problem_type}",
                )

                st.markdown("**Package Details**")
                col1, col2 = st.columns(2)
                with col1:
                    demand_min = st.number_input(
                        "Min (kg)", 1, 50, 1, key=f"demand_min_{problem_type}"
                    )
                    service_min = st.number_input(
                        "Min time (min)", 1, 60, 5, key=f"service_min_{problem_type}"
                    )
                with col2:
                    demand_max = st.number_input(
                        "Max (kg)", 1, 50, 10, key=f"demand_max_{problem_type}"
                    )
                    service_max = st.number_input(
                        "Max time (min)", 1, 60, 15, key=f"service_max_{problem_type}"
                    )

                priority_levels = st.slider(
                    "Priority levels",
                    1,
                    5,
                    3,
                    key=f"priority_{problem_type}",
                    help="1 = all same, 5 = five tiers",
                )

            # 3. ALGORITHM CONFIGURATION
            available_algorithms = ALGORITHMS[problem_type]

            with st.expander("Algorithm Settings", expanded=False):
                selected_algorithm = st.selectbox(
                    "Algorithm", available_algorithms, key=f"algorithm_{problem_type}"
                )

                max_iterations = st.slider(
                    "Max iterations",
                    10,
                    10000,
                    1000,
                    key=f"iterations_{problem_type}",
                    help="More = better solution but slower",
                )

                # Algorithm-specific parameters
                if "Tabu" in selected_algorithm:
                    col1, col2 = st.columns(2)
                    with col1:
                        tabu_tenure = st.number_input(
                            "Tabu tenure", 5, 50, 10, key=f"tabu_tenure_{problem_type}"
                        )
                    with col2:
                        neighborhood_size = st.number_input(
                            "Neighborhood",
                            10,
                            100,
                            20,
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
                        20,
                        500,
                        100,
                        key=f"population_{problem_type}",
                    )
                    col1, col2 = st.columns(2)
                    with col1:
                        crossover_prob = st.slider(
                            "Crossover", 0.0, 1.0, 0.9, key=f"crossover_{problem_type}"
                        )
                    with col2:
                        mutation_prob = st.slider(
                            "Mutation", 0.0, 1.0, 0.1, key=f"mutation_{problem_type}"
                        )

                    algorithm_params = {
                        "max_iterations": max_iterations,
                        "population_size": population_size,
                        "crossover_prob": crossover_prob,
                        "mutation_prob": mutation_prob,
                    }
                else:
                    algorithm_params = {"max_iterations": max_iterations}

            # COMPARE BUTTON (if multiple algorithms available)
            if len(available_algorithms) > 1:
                st.markdown("---")
                compare_button = st.button(
                    "Compare All Algorithms",
                    key=f"compare_{problem_type}",
                    use_container_width=True,
                )
            else:
                compare_button = False

            # ========== BUTTON LOGIC ==========

            # Generate button logic
            if generate_button:
                with st.spinner("Generating data..."):
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
                    st.success("Data generated successfully!")
                    st.rerun()

            # Run button logic
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
                        st.success("Algorithm completed successfully!")
                        st.rerun()

            # Compare button logic
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
                        st.rerun()

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
                            width="stretch",
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
                            width="stretch",
                            key=f"map_no_routes_{problem_type}_{st.session_state.chart_counter}",
                        )
                        st.info("Run algorithm to see routes")

                # TAB 2: METRICS
                with viz_tabs[1]:
                    if st.session_state.solution is not None:
                        sol = st.session_state.solution

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
                            width="stretch",
                            key=f"convergence_{problem_type}_{st.session_state.chart_counter}",
                        )

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
                                width="stretch",
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
                            width="stretch",
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
                        st.dataframe(comparison_df, width="stretch")

                        st.markdown("---")
                        col1, col2 = st.columns(2)
                        with col1:
                            fig_makespan = viz.plot_metrics_comparison(
                                comparison_df, "Makespan"
                            )
                            st.plotly_chart(fig_makespan, width="stretch")
                        with col2:
                            fig_cost = viz.plot_metrics_comparison(
                                comparison_df, "Cost"
                            )
                            st.plotly_chart(fig_cost, width="stretch")
                    else:
                        st.info("Run 'Compare All Algorithms' to see comparison")

            else:
                st.info("Generate data in the configuration panel to begin")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b;'>"
    "Vehicle Routing Optimization System | Built with Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
