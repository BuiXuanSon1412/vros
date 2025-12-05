# ui/problem_info_panel.py - Problem-specific information and help

import streamlit as st


def render_problem_info(problem_type):
    """Render problem-specific information panel"""

    with st.expander("ℹ️ Problem Information", expanded=False):
        if problem_type == 1:
            _render_problem1_info()
        elif problem_type == 2:
            _render_problem2_info()
        elif problem_type == 3:
            _render_problem3_info()


def _render_problem1_info():
    """PTDS-DDSS Problem Information"""
    st.markdown("""
    ### 🚚🚁 Parallel Technician-Drone Delivery System (PTDS-DDSS)
    
    **Problem Description:**
    - Minimize the total completion time (makespan) for medical sample collection
    - Technicians and drones work in parallel
    - Some customers can only be served by technicians
    
    **Constraints:**
    - ✈️ Drone flight endurance limit (battery constraint)
    - ⏱️ Sample waiting time limit (freshness constraint)
    - 👤 Some customers require technician service only
    
    **Objectives:**
    - Minimize makespan (single objective)
    
    **Algorithm:**
    - Tabu Search (TS) - Local search metaheuristic
    """)

    st.markdown("---")

    st.markdown("""
    **Key Parameters:**
    - `α₁`: Penalty weight for flight endurance violations
    - `α₂`: Penalty weight for waiting time violations
    - `β`: Penalty multiplication factor
    """)


def _render_problem2_info():
    """MSSVTDE Problem Information"""
    st.markdown("""
    ### 🏥 Medical Sample Scheduling with Vehicle and Drone (MSSVTDE)
    
    **Problem Description:**
    - **Bi-objective optimization**: Minimize both makespan AND cost
    - Consider traffic congestion affecting vehicle speeds
    - Drone has three-phase flight: takeoff → cruise → landing
    
    **Constraints:**
    - 🚛 Truck capacity limit
    - ✈️ Drone capacity and flight time limits
    - 🚦 Congestion-dependent truck travel times
    - ⏱️ Sample waiting time constraints
    
    **Objectives:**
    1. Minimize makespan (completion time)
    2. Minimize total operational cost
    
    **Algorithm:**
    - HNSGAII-TS - Hybrid NSGA-II with Tabu Search
    - Produces a Pareto front of non-dominated solutions
    """)

    st.markdown("---")

    st.markdown("""
    **What is a Pareto Front?**
    
    A Pareto front is a set of solutions where:
    - No solution can improve one objective without worsening another
    - All solutions are equally "optimal" depending on preferences
    - You choose based on your priority (speed vs. cost)
    
    Example:
    - Solution A: Fast (60 min) but expensive ($5000)
    - Solution B: Slow (90 min) but cheap ($3000)
    - Both are Pareto optimal!
    """)


def _render_problem3_info():
    """VRP-MRDR Problem Information"""
    st.markdown("""
    ### 📅 Vehicle Routing with Multiple Release Dates (VRP-MRDR)
    
    **Problem Description:**
    - Customers become available at different times (release dates)
    - Vehicles cannot serve a customer before their release date
    - Must wait if arriving too early
    
    **Constraints:**
    - 📅 Release date constraints (cannot serve before ready)
    - ✈️ Drone capacity and flight endurance limits
    - ⏱️ Sample waiting time limits
    
    **Objectives:**
    - Minimize makespan (single objective)
    - Handle dynamic customer availability
    
    **Algorithm:**
    - ATS (Adaptive Tabu Search)
    - Adapts search strategy based on solution quality
    """)

    st.markdown("---")

    st.markdown("""
    **Key Features:**
    - `γ₁, γ₂, γ₃, γ₄`: Score factors for adaptive neighborhood selection
    - `η`: Variable maximum iteration parameter
    - `LOOP`: Fixed maximum iteration limit
    - `SEG`: Number of segments for adaptive strategy
    """)


def render_algorithm_help(algorithm_name):
    """Render algorithm-specific help"""

    st.markdown("**Algorithm Help**")

    if "Tabu" in algorithm_name:
        st.info("""
        **Tabu Search** is a local search metaheuristic that:
        - Explores neighborhood solutions iteratively
        - Maintains a "tabu list" to prevent cycling
        - Accepts worse solutions to escape local optima
        
        **Tips:**
        - Higher iterations → better quality (but slower)
        - Adjust penalty parameters (α, β) if constraints are violated
        """)

    elif "NSGA" in algorithm_name:
        st.info("""
        **NSGA-II (Non-dominated Sorting Genetic Algorithm II)**:
        - Multi-objective evolutionary algorithm
        - Produces a Pareto front of solutions
        - Uses crossover and mutation operators
        
        **Tips:**
        - Larger population → better diversity
        - More generations → better convergence
        - Check Pareto front for trade-off visualization
        """)

    elif "ATS" in algorithm_name:
        st.info("""
        **Adaptive Tabu Search**:
        - Dynamically adjusts search strategy
        - Changes neighborhood based on progress
        - Handles dynamic constraints (release dates)
        
        **Tips:**
        - Adjust γ parameters to balance exploration vs. exploitation
        - SEG parameter controls adaptation frequency
        """)


def render_metrics_help():
    """Render help for understanding metrics"""

    with st.expander("📊 Understanding Metrics", expanded=False):
        st.markdown("""
        ### Key Metrics Explained
        
        **Makespan**
        - Total time to complete all deliveries
        - Measured from depot departure to last vehicle return
        - Lower is better ✓
        
        **Cost**
        - Total operational cost (fuel, labor, etc.)
        - Cost per km × total distance traveled
        - Different rates for trucks vs. drones
        - Lower is better ✓
        
        **Total Distance**
        - Sum of all vehicle travel distances
        - Includes: depot → customers → depot
        - Lower usually means better efficiency ✓
        
        **Computation Time**
        - Time taken by algorithm to find solution
        - Trade-off: quality vs. speed
        - Faster is preferred for large instances
        
        **Vehicle Utilization**
        - Percentage of time vehicles are actively working
        - Higher utilization = less idle time ✓
        - Target: > 80% for good efficiency
        
        **Pareto Rank** (Problem 2 only)
        - Rank 1 = Non-dominated (on Pareto front) ✓
        - Rank 2+ = Dominated by other solutions
        """)


def render_constraint_violations_panel(solution):
    """Render constraint violations if any exist"""

    violations = solution.get("constraint_violations", {})

    if not violations or all(v == 0 for v in violations.values()):
        st.success("✅ All constraints satisfied!")
        return

    st.warning("⚠️ Constraint Violations Detected")

    violation_data = []

    for constraint, count in violations.items():
        if count > 0:
            violation_data.append(
                {
                    "Constraint": constraint.replace("_", " ").title(),
                    "Violations": count,
                    "Severity": "High"
                    if count > 5
                    else "Medium"
                    if count > 2
                    else "Low",
                }
            )

    if violation_data:
        import pandas as pd

        df_violations = pd.DataFrame(violation_data)

        # Color code severity
        def color_severity(val):
            if val == "High":
                return "background-color: #f8d7da; color: #721c24"
            elif val == "Medium":
                return "background-color: #fff3cd; color: #856404"
            else:
                return "background-color: #d4edda; color: #155724"

        styled_violations = df_violations.style.applymap(
            color_severity, subset=["Severity"]
        )

        st.dataframe(styled_violations, use_container_width=True, hide_index=True)

        st.info("""
        **How to fix:**
        - Increase penalty parameters (α₁, α₂, β)
        - Increase vehicle count or capacity
        - Relax constraint limits if feasible
        """)
