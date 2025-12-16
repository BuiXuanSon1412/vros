# ui/config/algorithm_config_p3.py - UPDATED with Multiple Algorithm Selection

import streamlit as st
from config.default_config import PROBLEM3_CONFIG


def render_algorithm_config_p3():
    """Render Problem 3 algorithm configuration with multiple selection"""

    st.markdown("**Algorithm Selection**")

    # Available algorithms for Problem 3
    available_algorithms = ["Adaptive Tabu Search"]

    selected_algorithms = st.multiselect(
        "Select algorithms to run",
        options=available_algorithms,
        default=["Adaptive Tabu Search"],
        key="p3_selected_algorithms",
        help="You can select multiple algorithms to compare results",
    )

    if not selected_algorithms:
        st.warning("⚠️ Please select at least one algorithm")
        return None

    st.markdown("**Algorithm Parameters**")
    st.caption("These parameters apply to all selected algorithms")

    st.markdown("**Score Factors**")
    col1, col2 = st.columns(2)
    with col1:
        gamma1 = st.number_input(
            "Score factor 1 (γ₁)",
            min_value=0.0,
            max_value=5.0,
            value=PROBLEM3_CONFIG["algorithm"]["gamma1"],
            step=0.1,
            key="p3_gamma1",
        )
        gamma2 = st.number_input(
            "Score factor 2 (γ₂)",
            min_value=0.0,
            max_value=5.0,
            value=PROBLEM3_CONFIG["algorithm"]["gamma2"],
            step=0.1,
            key="p3_gamma2",
        )

    with col2:
        gamma3 = st.number_input(
            "Score factor 3 (γ₃)",
            min_value=0.0,
            max_value=5.0,
            value=PROBLEM3_CONFIG["algorithm"]["gamma3"],
            step=0.1,
            key="p3_gamma3",
        )
        gamma4 = st.number_input(
            "Score factor 4 (γ₄)",
            min_value=0.0,
            max_value=5.0,
            value=PROBLEM3_CONFIG["algorithm"]["gamma4"],
            step=0.1,
            key="p3_gamma4",
        )

    st.markdown("**Iteration Parameters**")

    col1, col2, col3 = st.columns(3)
    with col1:
        eta = st.number_input(
            "Variable max iteration (η)",
            min_value=10,
            max_value=500,
            value=PROBLEM3_CONFIG["algorithm"]["eta"],
            step=10,
            key="p3_eta",
        )

    with col2:
        loop = st.number_input(
            "Fixed max iteration (LOOP)",
            min_value=100,
            max_value=5000,
            value=PROBLEM3_CONFIG["algorithm"]["loop"],
            step=100,
            key="p3_loop",
        )

    with col3:
        seg = st.number_input(
            "Number of segments (SEG)",
            min_value=1,
            max_value=20,
            value=PROBLEM3_CONFIG["algorithm"]["seg"],
            step=1,
            key="p3_seg",
        )

    return {
        "algorithms": selected_algorithms,
        "gamma1": gamma1,
        "gamma2": gamma2,
        "gamma3": gamma3,
        "gamma4": gamma4,
        "eta": eta,
        "loop": loop,
        "seg": seg,
    }
