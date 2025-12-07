# ui/visualization/map_view.py - ENHANCED for Problem 3 Resupply Visualization

import streamlit as st
from utils.visualizer import Visualizer


@st.cache_resource
def get_visualizer():
    return Visualizer()


def render_map_view(problem_type):
    """Render map view with routes and resupply visualization"""
    customers = st.session_state.get(f"customers_{problem_type}")
    depot = st.session_state.get(f"depot_{problem_type}")
    solution = st.session_state.get(f"solution_{problem_type}")
    chart_counter = st.session_state.get(f"chart_counter_{problem_type}", 0)

    if customers is None or depot is None:
        st.info("Generate or upload data to see the map")
        return

    viz = get_visualizer()

    # Problem 3: Show resupply visualization option
    if problem_type == 3 and solution is not None:
        show_resupply = st.checkbox(
            "Show drone resupply routes",
            value=True,
            key=f"show_resupply_{chart_counter}",
            help="Display drone resupply flights to trucks",
        )
    else:
        show_resupply = False

    if solution is not None and solution.get("routes"):
        fig = viz.plot_routes_2d(
            customers,
            depot,
            solution["routes"],
            title=f"Routes - {solution['algorithm']}",
            problem_type=problem_type,
            resupply_operations=solution.get("resupply_operations", [])
            if show_resupply
            else [],
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key=f"map_{problem_type}_{chart_counter}",
        )

        # Show resupply statistics for Problem 3
        if problem_type == 3 and solution.get("resupply_operations"):
            _render_resupply_summary(solution)

    else:
        # Determine title based on problem type
        if problem_type in [1, 2]:
            title = "Sample Collection Locations"
        else:
            title = "Customer Locations"

        fig = viz.plot_routes_2d(
            customers,
            depot,
            {},
            title=title,
            problem_type=problem_type,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key=f"map_empty_{problem_type}",
        )


def _render_resupply_summary(solution):
    """Render quick resupply summary below map"""
    st.markdown("**Resupply Summary**")

    resupply_ops = solution.get("resupply_operations", [])

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        num_trips = len(resupply_ops)
        st.metric("Drone Trips", num_trips)

    with col2:
        packages = solution.get("packages_delivered_by_drone", 0)
        st.metric("Packages Resupplied", packages)

    with col3:
        drone_dist = solution.get("drone_distance", 0)
        st.metric("Drone Distance", f"{drone_dist:.1f} km")

    with col4:
        waiting = solution.get("total_waiting_time", 0)
        st.metric("Total Waiting", f"{waiting:.1f} min")
