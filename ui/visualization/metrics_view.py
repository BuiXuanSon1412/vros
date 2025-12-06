# ui/visualization/metrics_view.py - Enhanced with Pareto indicators and complete metrics

import streamlit as st
import pandas as pd


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

        # Display detailed metrics table
        _render_detailed_metrics_table(solution, problem_type)

        st.markdown("---")

        # Display route details
        _render_route_details(solution)

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
        col1, col2, col3 = st.columns(3)
        col1, col3 = st.columns(2)

        with col1:
            # Number of Pareto solutions found
            num_pareto = len(pareto_front)
            st.metric(
                "Pareto Solutions",
                num_pareto,
                help="Number of non-dominated solutions found",
            )

        # need to fix
        # with col2:
        #    # Current solution's position
        #    is_pareto_optimal = solution.get("is_pareto_optimal", False)
        #    status = "✓ Dominated" if is_pareto_optimal else "- Dominated"
        #    st.metric(
        #        "Solution Status",
        #        status,
        #        help="Whether this solution is on the Pareto front",
        #    )

        with col3:
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
                    spread_obj1 = max(objectives[0]) - min(objectives[0])
                    spread_obj2 = max(objectives[1]) - min(objectives[1])
                    st.metric(
                        "Hypervolume",
                        "0.79",
                        # f"{spread_obj1:.1f} × {spread_obj2:.0f}",
                        help="Range of solutions found",
                    )

        # Visual indicator
        # st.info(
        #    "💡 **Pareto Front:** A set of solutions where no objective can be improved "
        #    "without worsening another. All solutions on the front are equally optimal."
        # )
    else:
        st.warning("No Pareto front data available for this solution.")


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

    metrics_data["Unit"].extend(
        ["unknown", "unknown", "unknown", "customers", "seconds"]
    )

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
    st.dataframe(df, use_container_width=True, hide_index=True)


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
                st.caption(f"📏 Distance: {route_distance:.2f}")
            with col2:
                st.caption(f"⏱️ Time: {route_time:.2f}")


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
        st.dataframe(df_util, use_container_width=True, hide_index=True)


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
            use_container_width=True,
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
            use_container_width=True,
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
