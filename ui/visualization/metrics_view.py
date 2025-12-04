# ui/visualization/metrics_view.py - Metrics Display

import streamlit as st


def render_metrics_view(problem_type):
    """Render metrics view with solution details"""
    solution = st.session_state.get(f"solution_{problem_type}")

    if solution is not None:
        # Display key metrics
        _render_metric_cards(solution)

        st.markdown("---")

        # Display route details
        _render_route_details(solution)
    else:
        st.info("Run algorithm to see metrics")


def _render_metric_cards(solution):
    """Render metric cards in columns"""
    metric_cols = st.columns(4)

    with metric_cols[0]:
        st.metric("Makespan", f"{solution['makespan']:.1f} min")

    with metric_cols[1]:
        st.metric("Cost", f"{solution['cost']:,.0f}")

    with metric_cols[2]:
        st.metric("Distance", f"{solution['total_distance']:.1f}")

    with metric_cols[3]:
        st.metric("Time", f"{solution['computation_time']:.2f}")


def _render_route_details(solution):
    """Render detailed route information"""
    st.markdown("**Route Details:**")

    for vehicle_id, route in solution["routes"].items():
        if route:
            st.write(f"**{vehicle_id}:** {len(route)} customers - {route}")
