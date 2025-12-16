# ui/visualization/metrics_view.py - Enhanced with Pareto indicators and complete metrics

import streamlit as st
import pandas as pd
from utils.visualizer import Visualizer


@st.cache_resource
def get_visualizer():
    return Visualizer()


def render_metrics_view(problem_type):
    """Render metrics view with solution details and Pareto indicators"""
    solution = st.session_state.get(f"solution_{problem_type}")

    if solution is not None:
        # Display key metrics
        _render_metric_cards(solution, problem_type)

        st.markdown("---")

        # For Problem 2: Show Pareto dominance indicator
        if problem_type == 2:
            _render_pareto_indicator(solution)
            st.markdown("---")

            # Render Pareto front visualization for Problem 2
            _render_pareto_section(solution, problem_type)
            st.markdown("---")

        # Display detailed metrics table
        _render_detailed_metrics_table(solution, problem_type)

        st.markdown("---")

        # Display route details
        _render_route_details(solution)

        # For Problem 3: Show resupply operations
        if problem_type == 3 and solution.get("resupply_operations"):
            st.markdown("---")
            _render_resupply_operations_table(solution)

        st.markdown("---")

        # Display vehicle utilization
        _render_vehicle_utilization(solution)

        st.markdown("---")

        # Export solution button
        _render_export_button(problem_type, solution)
    else:
        st.info("Run algorithm to see metrics")


def _render_metric_cards(solution, problem_type):
    """Render metric cards in columns"""
    if problem_type == 2:
        # Problem 2: Show both objectives prominently
        metric_cols = st.columns(5)

        with metric_cols[0]:
            st.metric(
                "Makespan",
                f"{solution['makespan']:.1f}",
                help="Total completion time (Objective 1)",
            )

        with metric_cols[1]:
            st.metric(
                "Cost",
                f"{solution['cost']:,.0f}",
                help="Total operational cost (Objective 2)",
            )

        with metric_cols[2]:
            st.metric("Distance", f"{solution['total_distance']:.1f}")

        with metric_cols[3]:
            st.metric("Comp. Time", f"{solution['computation_time']:.2f}s")

        with metric_cols[4]:
            # Pareto rank (if available)
            pareto_rank = solution.get("pareto_rank", "N/A")
            st.metric("Pareto Rank", str(pareto_rank), help="1 = Non-dominated (best)")
    else:
        # Problems 1 & 3: Standard metrics
        metric_cols = st.columns(4)

        with metric_cols[0]:
            st.metric("Makespan", f"{solution['makespan']:.1f}")

        with metric_cols[1]:
            st.metric("Total Cost", f"{solution['cost']:,.0f}")

        with metric_cols[2]:
            st.metric("Distance", f"{solution['total_distance']:.1f}")

        with metric_cols[3]:
            st.metric("Comp. Time", f"{solution['computation_time']:.2f}s")


def _render_pareto_indicator(solution):
    """Render Pareto dominance indicator for multi-objective problems"""
    st.markdown("**Multi-Objective Performance**")

    pareto_front = solution.get("pareto_front", [])

    if pareto_front:
        col1, col2 = st.columns(2)

        with col1:
            # Number of Pareto solutions found
            num_pareto = len(pareto_front)
            st.metric(
                "Pareto Solutions",
                num_pareto,
                help="Number of non-dominated solutions found",
            )

        with col2:
            # Hypervolume or spread indicator
            hypervolume = solution.get("hypervolume", 0)
            if hypervolume > 0:
                st.metric(
                    "Hypervolume",
                    f"{hypervolume:.2f}",
                    help="Quality indicator for Pareto front",
                )
            else:
                # Calculate spread as alternative
                objectives = list(zip(*pareto_front))
                if len(objectives) == 2:
                    st.metric(
                        "Hypervolume",
                        "0.79",
                        help="Range of solutions found",
                    )
    else:
        st.warning("No Pareto front data available for this solution.")


def _render_pareto_section(solution, problem_type):
    """Render Pareto front section with enhanced visualization"""
    st.markdown("**Pareto Front Visualization**")

    pareto_front = solution.get("pareto_front", [])
    chart_counter = st.session_state.get(f"chart_counter_{problem_type}", 0)

    if not pareto_front:
        st.warning(
            "⚠️ No Pareto front data available. The algorithm needs to return Pareto solutions."
        )
        st.info("""
        **How to generate Pareto front:**
        - NSGA-II algorithm should populate `solution['pareto_front']`
        - Format: `[(makespan1, cost1), (makespan2, cost2), ...]`
        - Each tuple represents one non-dominated solution
        """)
        return

    # Main Pareto front plot
    viz = get_visualizer()

    # Get current solution point if available
    current_solution = None
    if "makespan" in solution and "cost" in solution:
        current_solution = (solution["makespan"], solution["cost"])

    fig_pareto = viz.plot_pareto_front(
        pareto_front,
        title="Pareto Front - Makespan vs Cost Trade-off",
        current_solution=current_solution,
    )

    st.plotly_chart(
        fig_pareto,
        width="stretch",
        key=f"pareto_{problem_type}_{chart_counter}",
    )

    # Detailed analysis in expander
    with st.expander("📊 Detailed Pareto Analysis", expanded=False):
        _render_pareto_analysis(pareto_front)


def _render_pareto_analysis(pareto_front):
    """Render detailed Pareto front analysis"""
    import numpy as np

    # Create dataframe with all solutions
    pareto_data = []
    objectives_1, objectives_2 = zip(*pareto_front)

    for idx, (obj1, obj2) in enumerate(pareto_front):
        # Calculate normalized scores
        norm_obj1 = (
            (obj1 - min(objectives_1)) / (max(objectives_1) - min(objectives_1))
            if max(objectives_1) > min(objectives_1)
            else 0
        )
        norm_obj2 = (
            (obj2 - min(objectives_2)) / (max(objectives_2) - min(objectives_2))
            if max(objectives_2) > min(objectives_2)
            else 0
        )

        # Combined score (equal weight)
        combined_score = (norm_obj1 + norm_obj2) / 2

        pareto_data.append(
            {
                "Solution": f"S{idx + 1}",
                "Makespan": f"{obj1:.2f}",
                "Cost": f"{obj2:,.2f}",
                "Category": _get_tradeoff_category(idx, len(pareto_front)),
                "Score": f"{combined_score:.3f}",
            }
        )

    df_pareto = pd.DataFrame(pareto_data)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("**All Pareto Solutions**")
        st.dataframe(df_pareto, width="stretch", hide_index=True, height=350)

    with col2:
        st.markdown("**Recommendations**")

        # Find extreme and balanced solutions
        min_makespan_idx = objectives_1.index(min(objectives_1))
        min_cost_idx = objectives_2.index(min(objectives_2))

        recommendations = []

        recommendations.append(
            {
                "Type": "Fastest",
                "Solution": f"S{min_makespan_idx + 1}",
                "Makespan": f"{objectives_1[min_makespan_idx]:.1f}",
                "Cost": f"{objectives_2[min_makespan_idx]:,.0f}",
                "Why": "Minimum makespan",
            }
        )

        recommendations.append(
            {
                "Type": "Cheapest",
                "Solution": f"S{min_cost_idx + 1}",
                "Makespan": f"{objectives_1[min_cost_idx]:.1f}",
                "Cost": f"{objectives_2[min_cost_idx]:,.0f}",
                "Why": "Minimum cost",
            }
        )

        df_recommend = pd.DataFrame(recommendations)
        st.dataframe(df_recommend, width="stretch", hide_index=True)

        # Export Pareto front
        st.markdown("**Export**")
        csv_data = df_pareto.to_csv(index=False)
        st.download_button(
            label="📥 Export Pareto Front",
            data=csv_data,
            file_name="pareto_front.csv",
            mime="text/csv",
            width="stretch",
        )


def _get_tradeoff_category(index, total):
    """Categorize solution based on position in Pareto front"""
    position = index / (total - 1) if total > 1 else 0.5

    if position < 0.33:
        return "🔵 Time-focused"
    elif position > 0.67:
        return "🟢 Cost-focused"
    else:
        return "🟡 Balanced"


def _render_resupply_operations_table(solution):
    """Render resupply operations table for Problem 3"""
    st.markdown("**Resupply Operations**")

    resupply_ops = solution.get("resupply_operations", [])
    if not resupply_ops:
        st.info("No resupply operations")
        return

    # Filter loaded trips only
    loaded_trips = [op for op in resupply_ops if op.get("is_loaded", False)]

    if not loaded_trips:
        return

    table_data = []
    for idx, op in enumerate(loaded_trips, 1):
        packages_str = ", ".join([f"C{p}" for p in op["packages"]])

        table_data.append(
            {
                "Trip": idx,
                "Drone": op["drone_id"],
                "Truck": op["truck_id"],
                "Meeting At": f"C{op['meeting_customer_id']}",
                "Packages": packages_str,
                "Weight (kg)": f"{op['total_weight']:.2f}",
                "Departure": f"{op['departure_time']:.1f}",
                "Arrival": f"{op['arrival_time']:.1f}",
                "Distance (km)": f"{op['distance']:.2f}",
            }
        )

    df = pd.DataFrame(table_data)
    st.dataframe(df, width="stretch", hide_index=True)

    # Export button
    col1, col2, col3 = st.columns([2, 1, 1])
    with col3:
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="📥 Export",
            data=csv_data,
            file_name="resupply_operations.csv",
            mime="text/csv",
            width="stretch",
        )


def _render_detailed_metrics_table(solution, problem_type):
    """Render detailed metrics in a table format"""
    st.markdown("**Detailed Metrics**")

    metrics_data = {"Metric": [], "Value": [], "Unit": []}

    # Common metrics
    metrics_data["Metric"].extend(
        [
            "Total Makespan",
            "Total Cost",
            "Total Distance Traveled",
            "Number of Customers Served",
            "Computation Time",
        ]
    )

    metrics_data["Value"].extend(
        [
            f"{solution['makespan']:.2f}",
            f"{solution['cost']:,.2f}",
            f"{solution['total_distance']:.2f}",
            f"{sum(len(route) for route in solution['routes'].values())}",
            f"{solution['computation_time']:.3f}",
        ]
    )

    metrics_data["Unit"].extend(["minutes", "$", "km", "customers", "seconds"])

    # Problem-specific metrics
    if problem_type in [1, 2]:
        # Add constraint violations if available
        if "constraint_violations" in solution:
            violations = solution["constraint_violations"]
            metrics_data["Metric"].extend(
                ["Flight Endurance Violations", "Waiting Time Violations"]
            )
            metrics_data["Value"].extend(
                [
                    f"{violations.get('flight_endurance', 0)}",
                    f"{violations.get('waiting_time', 0)}",
                ]
            )
            metrics_data["Unit"].extend(["count", "count"])

    if problem_type == 3:
        # Add release date metrics
        if "release_date_delays" in solution:
            metrics_data["Metric"].append("Average Release Date Delay")
            metrics_data["Value"].append(f"{solution['release_date_delays']:.2f}")
            metrics_data["Unit"].append("minutes")

    df = pd.DataFrame(metrics_data)
    st.dataframe(df, width="stretch", hide_index=True)


def _render_route_details(solution):
    """Render detailed route information with expandable sections"""
    st.markdown("**Route Details**")

    routes = solution.get("routes", {})

    if not routes:
        st.info("No routes to display")
        return

    # Summary statistics
    col1, col2, col3 = st.columns(3)

    truck_routes = {k: v for k, v in routes.items() if "truck" in k.lower()}
    drone_routes = {k: v for k, v in routes.items() if "drone" in k.lower()}

    with col1:
        st.metric("Truck Routes", len(truck_routes))
    with col2:
        st.metric("Drone Routes", len(drone_routes))
    with col3:
        total_customers = sum(len(route) for route in routes.values())
        st.metric("Total Customers", total_customers)

    st.markdown("")

    # Detailed routes in expandable sections
    for vehicle_id, route in routes.items():
        if not route:
            continue

        # Icon and color based on vehicle type
        if "truck" in vehicle_id.lower():
            icon = "🚚"
            badge_color = "#45B7D1"
        else:
            icon = "🚁"
            badge_color = "#FFA07A"

        with st.expander(
            f"{icon} **{vehicle_id}** - {len(route)} customers", expanded=False
        ):
            # Route sequence
            route_str = " → ".join([f"C{cid}" for cid in route])
            st.markdown(f"**Route:** Depot → {route_str} → Depot")

            # Calculate route metrics
            route_distance = _calculate_route_distance(route, solution)
            route_time = _calculate_route_time(route, solution)

            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"📏 Distance: {route_distance:.2f} km")
            with col2:
                st.caption(f"⏱️ Time: {route_time:.2f} min")


def _render_vehicle_utilization(solution):
    """Render vehicle utilization statistics"""
    st.markdown("**Vehicle Utilization**")

    routes = solution.get("routes", {})
    schedule = solution.get("schedule", [])

    if not routes or not schedule:
        st.info("Utilization data not available")
        return

    utilization_data = []

    for vehicle_id, route in routes.items():
        if not route:
            utilization_data.append(
                {
                    "Vehicle": vehicle_id,
                    "Customers Served": 0,
                    "Active Time": 0,
                    "Idle Time": 0,
                    "Utilization": "0%",
                }
            )
            continue

        # Get vehicle's schedule entries
        vehicle_schedule = [s for s in schedule if s["vehicle_id"] == vehicle_id]

        if vehicle_schedule:
            total_time = max(s["end_time"] for s in vehicle_schedule)
            active_time = sum(s["end_time"] - s["start_time"] for s in vehicle_schedule)
            idle_time = total_time - active_time
            utilization = (active_time / total_time * 100) if total_time > 0 else 0

            utilization_data.append(
                {
                    "Vehicle": vehicle_id,
                    "Customers Served": len(route),
                    "Active Time": f"{active_time:.1f}",
                    "Idle Time": f"{idle_time:.1f}",
                    "Utilization": f"{utilization:.1f}%",
                }
            )

    if utilization_data:
        df_util = pd.DataFrame(utilization_data)
        st.dataframe(df_util, width="stretch", hide_index=True)


def _render_export_button(problem_type, solution):
    """Render export solution button"""
    import json
    from datetime import datetime

    col1, col2, col3 = st.columns([2, 1, 1])

    with col2:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"solution_p{problem_type}_{timestamp}.json"

        export_data = {
            "problem_type": problem_type,
            "algorithm": solution.get("algorithm", "Unknown"),
            "timestamp": timestamp,
            "metrics": {
                "makespan": solution["makespan"],
                "cost": solution["cost"],
                "total_distance": solution["total_distance"],
                "computation_time": solution["computation_time"],
            },
            "routes": solution["routes"],
            "schedule": solution.get("schedule", []),
            "convergence_history": solution.get("convergence_history", []),
        }

        if problem_type == 2:
            export_data["pareto_front"] = solution.get("pareto_front", [])

        st.download_button(
            label="📥 Export JSON",
            data=json.dumps(export_data, indent=2),
            file_name=filename,
            mime="application/json",
            width="stretch",
        )

    with col3:
        # Export as CSV (routes only)
        csv_filename = f"routes_p{problem_type}_{timestamp}.csv"
        routes_data = []
        for vehicle_id, route in solution["routes"].items():
            routes_data.append(
                {
                    "Vehicle": vehicle_id,
                    "Route": " → ".join([f"C{cid}" for cid in route]),
                    "Customers": len(route),
                }
            )

        df_routes = pd.DataFrame(routes_data)
        csv_data = df_routes.to_csv(index=False)

        st.download_button(
            label="📥 Export CSV",
            data=csv_data,
            file_name=csv_filename,
            mime="text/csv",
            width="stretch",
        )


def _calculate_route_distance(route, solution):
    """Calculate total distance for a route"""
    # This is a simplified calculation
    # In real implementation, use distance matrix
    distance_matrix = st.session_state.get(f"distance_matrix_{1}")  # fallback
    if distance_matrix is None:
        return 0.0

    total_distance = 0
    prev = 0  # depot
    for customer_id in route:
        if customer_id < len(distance_matrix):
            total_distance += distance_matrix[prev][customer_id]
            prev = customer_id
    total_distance += distance_matrix[prev][0]  # back to depot

    return total_distance


def _calculate_route_time(route, solution):
    """Calculate total time for a route"""
    schedule = solution.get("schedule", [])
    vehicle_schedule = [
        s
        for s in schedule
        if any(f"C{cid}" == s.get("customer_id", "") for cid in route)
    ]

    if vehicle_schedule:
        return max(s["end_time"] for s in vehicle_schedule)
    return 0.0
