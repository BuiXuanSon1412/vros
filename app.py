# app.py - Main Streamlit Application

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
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 2rem;
    }
    .config-section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1e40af;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        border-bottom: 2px solid #3b82f6;
        padding-bottom: 0.3rem;
    }
    .field-description {
        font-size: 0.85rem;
        color: #64748b;
        font-style: italic;
        margin-top: -0.5rem;
        margin-bottom: 0.5rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #3b82f6;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #2563eb;
    }
    div[data-testid="stExpander"] {
        background-color: #f8fafc;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
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
        # Layout: Left (70%) - Visualization | Right (30%) - Config
        col_left, col_right = st.columns([7, 3])

        # ==================== RIGHT SIDE: CONFIGURATION ====================
        with col_right:
            st.markdown("### Configuration Panel")

            # 1. VEHICLE CONFIGURATION
            st.markdown(
                '<div class="config-section-title">Vehicle Configuration</div>',
                unsafe_allow_html=True,
            )

            # TRUCKS
            st.markdown("**Trucks**")

            truck_count = st.number_input(
                "Number of trucks",
                min_value=1,
                max_value=10,
                value=DEFAULT_VEHICLE_CONFIG["truck"]["count"],
                key=f"truck_count_{problem_type}",
                help="Total number of trucks available for delivery",
            )
            st.markdown(
                '<div class="field-description">How many trucks are available in the fleet</div>',
                unsafe_allow_html=True,
            )

            truck_capacity = st.number_input(
                "Truck capacity (kg)",
                min_value=10,
                max_value=500,
                value=DEFAULT_VEHICLE_CONFIG["truck"]["capacity"],
                key=f"truck_capacity_{problem_type}",
                help="Maximum weight each truck can carry",
            )
            st.markdown(
                '<div class="field-description">Maximum load capacity for each truck in kilograms</div>',
                unsafe_allow_html=True,
            )

            truck_speed = st.number_input(
                "Truck speed (km/h)",
                min_value=10,
                max_value=100,
                value=DEFAULT_VEHICLE_CONFIG["truck"]["speed"],
                key=f"truck_speed_{problem_type}",
                help="Average speed of trucks during delivery",
            )
            st.markdown(
                '<div class="field-description">Average traveling speed of trucks in kilometers per hour</div>',
                unsafe_allow_html=True,
            )

            truck_cost = st.number_input(
                "Truck cost per km (VND)",
                min_value=1000,
                max_value=20000,
                value=DEFAULT_VEHICLE_CONFIG["truck"]["cost_per_km"],
                key=f"truck_cost_{problem_type}",
                help="Operating cost per kilometer for trucks",
            )
            st.markdown(
                '<div class="field-description">Operational cost for trucks per kilometer traveled</div>',
                unsafe_allow_html=True,
            )

            st.markdown("---")

            # DRONES
            st.markdown("**Drones (UAVs)**")

            drone_count = st.number_input(
                "Number of drones",
                min_value=0,
                max_value=10,
                value=DEFAULT_VEHICLE_CONFIG["drone"]["count"],
                key=f"drone_count_{problem_type}",
                help="Total number of drones available for delivery",
            )
            st.markdown(
                '<div class="field-description">How many drones are available in the fleet</div>',
                unsafe_allow_html=True,
            )

            drone_capacity = st.number_input(
                "Drone capacity (kg)",
                min_value=1,
                max_value=50,
                value=DEFAULT_VEHICLE_CONFIG["drone"]["capacity"],
                key=f"drone_capacity_{problem_type}",
                help="Maximum weight each drone can carry",
            )
            st.markdown(
                '<div class="field-description">Maximum payload capacity for each drone in kilograms</div>',
                unsafe_allow_html=True,
            )

            drone_speed = st.number_input(
                "Drone speed (km/h)",
                min_value=10,
                max_value=120,
                value=DEFAULT_VEHICLE_CONFIG["drone"]["speed"],
                key=f"drone_speed_{problem_type}",
                help="Average speed of drones during delivery",
            )
            st.markdown(
                '<div class="field-description">Average flying speed of drones in kilometers per hour</div>',
                unsafe_allow_html=True,
            )

            drone_energy = st.number_input(
                "Drone energy limit (minutes)",
                min_value=10,
                max_value=120,
                value=DEFAULT_VEHICLE_CONFIG["drone"]["energy_limit"],
                key=f"drone_energy_{problem_type}",
                help="Maximum flight time before battery runs out",
            )
            st.markdown(
                '<div class="field-description">Maximum continuous flight time for drones before needing recharge</div>',
                unsafe_allow_html=True,
            )

            drone_cost = st.number_input(
                "Drone cost per km (VND)",
                min_value=500,
                max_value=10000,
                value=DEFAULT_VEHICLE_CONFIG["drone"]["cost_per_km"],
                key=f"drone_cost_{problem_type}",
                help="Operating cost per kilometer for drones",
            )
            st.markdown(
                '<div class="field-description">Operational cost for drones per kilometer traveled</div>',
                unsafe_allow_html=True,
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
            st.markdown(
                '<div class="config-section-title">Customer Configuration</div>',
                unsafe_allow_html=True,
            )

            num_customers = st.slider(
                "Number of customers",
                min_value=5,
                max_value=100,
                value=20,
                key=f"num_customers_{problem_type}",
                help="Total number of delivery locations",
            )
            st.markdown(
                '<div class="field-description">Total number of customer locations to be serviced</div>',
                unsafe_allow_html=True,
            )

            area_size = st.slider(
                "Service area size (km x km)",
                min_value=10,
                max_value=100,
                value=50,
                key=f"area_size_{problem_type}",
                help="Size of the square delivery area",
            )
            st.markdown(
                '<div class="field-description">Dimensions of the square service area in kilometers</div>',
                unsafe_allow_html=True,
            )

            col1, col2 = st.columns(2)
            with col1:
                demand_min = st.number_input(
                    "Min demand (kg)",
                    min_value=1,
                    max_value=50,
                    value=1,
                    key=f"demand_min_{problem_type}",
                    help="Minimum package weight",
                )
            with col2:
                demand_max = st.number_input(
                    "Max demand (kg)",
                    min_value=1,
                    max_value=50,
                    value=10,
                    key=f"demand_max_{problem_type}",
                    help="Maximum package weight",
                )
            st.markdown(
                '<div class="field-description">Range of package weights that customers may request (in kg)</div>',
                unsafe_allow_html=True,
            )

            col3, col4 = st.columns(2)
            with col3:
                service_min = st.number_input(
                    "Min service time (min)",
                    min_value=1,
                    max_value=60,
                    value=5,
                    key=f"service_min_{problem_type}",
                    help="Minimum time to serve a customer",
                )
            with col4:
                service_max = st.number_input(
                    "Max service time (min)",
                    min_value=1,
                    max_value=60,
                    value=15,
                    key=f"service_max_{problem_type}",
                    help="Maximum time to serve a customer",
                )
            st.markdown(
                '<div class="field-description">Range of time needed to complete delivery at each customer location (in minutes)</div>',
                unsafe_allow_html=True,
            )

            priority_levels = st.slider(
                "Number of priority levels",
                min_value=1,
                max_value=5,
                value=3,
                key=f"priority_{problem_type}",
                help="How many priority tiers for customers (1=all same, 5=more tiers)",
            )
            st.markdown(
                '<div class="field-description">Customer priority levels (1 = highest priority, higher numbers = lower priority)</div>',
                unsafe_allow_html=True,
            )

            if st.button("Generate Random Data", key=f"generate_{problem_type}"):
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

            # 3. ALGORITHM CONFIGURATION
            st.markdown(
                '<div class="config-section-title">Algorithm Configuration</div>',
                unsafe_allow_html=True,
            )

            available_algorithms = ALGORITHMS[problem_type]
            selected_algorithm = st.selectbox(
                "Select algorithm",
                available_algorithms,
                key=f"algorithm_{problem_type}",
                help="Choose which optimization algorithm to use",
            )
            st.markdown(
                '<div class="field-description">Optimization algorithm to solve the routing problem</div>',
                unsafe_allow_html=True,
            )

            max_iterations = st.number_input(
                "Maximum iterations",
                min_value=10,
                max_value=10000,
                value=1000,
                key=f"iterations_{problem_type}",
                help="Maximum number of iterations the algorithm will run",
            )
            st.markdown(
                '<div class="field-description">Maximum number of iterations before the algorithm stops</div>',
                unsafe_allow_html=True,
            )

            if "Tabu" in selected_algorithm:
                tabu_tenure = st.number_input(
                    "Tabu tenure",
                    min_value=5,
                    max_value=50,
                    value=10,
                    key=f"tabu_tenure_{problem_type}",
                    help="Number of iterations a move stays in tabu list",
                )
                st.markdown(
                    '<div class="field-description">How long a solution stays forbidden in the tabu list</div>',
                    unsafe_allow_html=True,
                )

                neighborhood_size = st.number_input(
                    "Neighborhood size",
                    min_value=10,
                    max_value=100,
                    value=20,
                    key=f"neighborhood_{problem_type}",
                    help="Number of neighbor solutions to explore each iteration",
                )
                st.markdown(
                    '<div class="field-description">Number of neighboring solutions to explore in each iteration</div>',
                    unsafe_allow_html=True,
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
                    help="Number of solutions in each generation",
                )
                st.markdown(
                    '<div class="field-description">Number of candidate solutions maintained in each generation</div>',
                    unsafe_allow_html=True,
                )

                crossover_prob = st.slider(
                    "Crossover probability",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.9,
                    key=f"crossover_{problem_type}",
                    help="Probability of combining two parent solutions",
                )
                st.markdown(
                    '<div class="field-description">Probability that two parent solutions will be combined to create offspring</div>',
                    unsafe_allow_html=True,
                )

                mutation_prob = st.slider(
                    "Mutation probability",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.1,
                    key=f"mutation_{problem_type}",
                    help="Probability of random changes to a solution",
                )
                st.markdown(
                    '<div class="field-description">Probability that random modifications will be applied to a solution</div>',
                    unsafe_allow_html=True,
                )

                algorithm_params = {
                    "max_iterations": max_iterations,
                    "population_size": population_size,
                    "crossover_prob": crossover_prob,
                    "mutation_prob": mutation_prob,
                }
            else:
                algorithm_params = {"max_iterations": max_iterations}

            # 4. RUN BUTTON
            st.markdown("---")
            run_button = st.button(
                "Run Algorithm", type="primary", key=f"run_{problem_type}"
            )

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

            # 5. COMPARE BUTTON (for multiple algorithms)
            if len(available_algorithms) > 1:
                st.markdown("---")
                compare_button = st.button(
                    "Compare All Algorithms", key=f"compare_{problem_type}"
                )

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
                            width="stretch",
                            key=f"map_routes_{problem_type}_{st.session_state.chart_counter}",
                        )
                    elif st.session_state.depot is not None:
                        # Show customers and depot without routes
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

                        # Display metrics
                        metric_cols = st.columns(4)
                        with metric_cols[0]:
                            st.metric("Makespan", f"{sol['makespan']:.1f} min")
                        with metric_cols[1]:
                            st.metric("Cost", f"{sol['cost']:,.0f} VND")
                        with metric_cols[2]:
                            st.metric(
                                "Total Distance", f"{sol['total_distance']:.1f} km"
                            )
                        with metric_cols[3]:
                            st.metric(
                                "Computation Time", f"{sol['computation_time']:.2f}s"
                            )

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
                            width="stretch",
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
                        st.dataframe(
                            comparison_df,
                            width="stretch",
                        )

                        st.markdown("---")

                        col1, col2 = st.columns(2)
                        with col1:
                            fig_makespan = viz.plot_metrics_comparison(
                                comparison_df, "Makespan"
                            )
                            st.plotly_chart(
                                fig_makespan,
                                width="stretch",
                                key=f"result_{problem_type}_{st.session_state.chart_counter}",
                            )
                        with col2:
                            fig_cost = viz.plot_metrics_comparison(
                                comparison_df, "Cost"
                            )
                            st.plotly_chart(
                                fig_cost,
                                width="stretch",
                            )
                    else:
                        st.info("Run 'Compare All Algorithms' to see comparison")

            else:
                st.info("Generate data in the configuration panel to begin")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b;'>"
    "Vehicle Routing Optimization System | Developed with Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
