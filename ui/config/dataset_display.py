# ui/config/dataset_display.py - Display dataset information after upload

import streamlit as st
import pandas as pd


def render_dataset_info(problem_type, customers_df):
    """Render dataset information based on problem type"""

    if customers_df is None or len(customers_df) == 0:
        return

    # st.markdown("---")
    st.markdown("**Dataset Information**")

    # Common metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("No. Customers", len(customers_df))

    with col2:
        coord_range = (
            min(customers_df["x"].min(), customers_df["y"].min()),
            max(customers_df["x"].max(), customers_df["y"].max()),
        )
        st.metric("Coordinate range", f"[{coord_range[0]:.1f}, {coord_range[1]:.1f}]")

    with col3:
        demand_range = (customers_df["demand"].min(), customers_df["demand"].max())
        st.metric("Demand range", f"[{demand_range[0]:.2f}, {demand_range[1]:.2f}]")

    # Problem-specific information
    if problem_type in [1, 2]:
        _render_technician_drone_info(customers_df)
    elif problem_type == 3:
        _render_release_date_info(customers_df)


def _render_technician_drone_info(customers_df):
    """Display technician/drone distribution info for Problem 1 & 2"""
    st.markdown("**Distribution Information**")

    col1, col2 = st.columns(2)

    if "only_staff" in customers_df.columns:
        only_tech_count = customers_df["only_staff"].sum()
        only_tech_pct = (only_tech_count / len(customers_df)) * 100

        with col1:
            st.info(
                f"**Only-technician customers:** {only_tech_count} ({only_tech_pct:.1f}%)"
            )

        with col2:
            both_count = len(customers_df) - only_tech_count
            both_pct = 100 - only_tech_pct
            st.info(f"**Can use both:** {both_count} ({both_pct:.1f}%)")

    # Show coordinate distribution
    st.markdown("**Coordinate Distribution:**")
    st.caption(
        f"X: [{customers_df['x'].min():.1f}, {customers_df['x'].max():.1f}], "
        f"Y: [{customers_df['y'].min():.1f}, {customers_df['y'].max():.1f}]"
    )

    # Show demand distribution
    st.markdown("**Demand Distribution:**")
    st.caption(
        f"Min: {customers_df['demand'].min():.3f} kg, "
        f"Max: {customers_df['demand'].max():.3f} kg, "
        f"Mean: {customers_df['demand'].mean():.3f} kg"
    )


def _render_release_date_info(customers_df):
    """Display release date info for Problem 3"""
    st.markdown("**Distribution Information**")

    col1, col2 = st.columns(2)

    if "release_date" in customers_df.columns:
        release_range = (
            customers_df["release_date"].min(),
            customers_df["release_date"].max(),
        )

        with col1:
            st.info(
                f"**Release date range:** [{release_range[0]:.0f}, {release_range[1]:.0f}]"
            )

        with col2:
            avg_release = customers_df["release_date"].mean()
            st.info(f"**Average release date:** {avg_release:.1f}")

    # Show coordinate distribution
    st.markdown("**Coordinate Distribution:**")
    st.caption(
        f"X: [{customers_df['x'].min():.1f}, {customers_df['x'].max():.1f}], "
        f"Y: [{customers_df['y'].min():.1f}, {customers_df['y'].max():.1f}]"
    )

    # Show demand distribution
    st.markdown("**Demand Distribution:**")
    st.caption(
        f"Min: {customers_df['demand'].min():.3f} kg, "
        f"Max: {customers_df['demand'].max():.3f} kg, "
        f"Mean: {customers_df['demand'].mean():.3f} kg"
    )
