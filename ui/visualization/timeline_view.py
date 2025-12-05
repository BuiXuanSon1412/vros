# ui/visualization/timeline_view.py - Enhanced Timeline/Gantt Charts

import streamlit as st
from utils.visualizer import Visualizer
import pandas as pd


@st.cache_resource
def get_visualizer():
    return Visualizer()


def render_timeline_view(problem_type):
    """Render enhanced timeline/Gantt chart view with analysis"""
    solution = st.session_state.get(f"solution_{problem_type}")
    chart_counter = st.session_state.get(f"chart_counter_{problem_type}", 0)

    if solution is not None and solution.get("schedule"):
        # Schedule statistics
        _render_schedule_statistics(solution)

        st.markdown("---")

        # Main Gantt chart
        _render_gantt_chart(solution, problem_type, chart_counter)

        st.markdown("---")

        # Schedule details table
        _render_schedule_details(solution)

        st.markdown("---")

        # Timeline analysis
        _render_timeline_analysis(solution, problem_type)
    else:
        st.info("Run algorithm to see timeline")


def _render_schedule_statistics(solution):
    """Render schedule overview statistics"""
    st.markdown("**Schedule Overview**")

    schedule = solution.get("schedule", [])
    routes = solution.get("routes", {})

    if not schedule:
        return

    # Calculate statistics
    total_tasks = len(schedule)
    unique_vehicles = len(set(task["vehicle_id"] for task in schedule))
    total_makespan = solution.get("makespan", 0)

    # Vehicle type breakdown
    truck_tasks = sum(1 for task in schedule if "truck" in task["vehicle_id"].lower())
    drone_tasks = sum(1 for task in schedule if "drone" in task["vehicle_id"].lower())

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Tasks", total_tasks)

    with col2:
        st.metric("Active Vehicles", unique_vehicles)

    with col3:
        st.metric("Total Makespan", f"{total_makespan:.1f} min")

    with col4:
        st.metric("🚚 Truck Tasks", truck_tasks)

    with col5:
        st.metric("🚁 Drone Tasks", drone_tasks)


def _render_gantt_chart(solution, problem_type, chart_counter):
    """Render Gantt chart for schedule"""
    viz = get_visualizer()

    fig_gantt = viz.plot_gantt_chart(
        solution["schedule"],
        title="Schedule Timeline - All Vehicles",
    )
    st.plotly_chart(
        fig_gantt,
        use_container_width=True,
        key=f"timeline_{problem_type}_{chart_counter}",
    )

    # Add view options
    with st.expander("⚙️ View Options", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            show_trucks_only = st.checkbox(
                "Show Trucks Only", key=f"trucks_only_{problem_type}"
            )

        with col2:
            show_drones_only = st.checkbox(
                "Show Drones Only", key=f"drones_only_{problem_type}"
            )

        if show_trucks_only:
            truck_schedule = [
                s for s in solution["schedule"] if "truck" in s["vehicle_id"].lower()
            ]
            if truck_schedule:
                fig_trucks = viz.plot_gantt_chart(
                    truck_schedule, title="Truck Schedule"
                )
                st.plotly_chart(fig_trucks, use_container_width=True)

        if show_drones_only:
            drone_schedule = [
                s for s in solution["schedule"] if "drone" in s["vehicle_id"].lower()
            ]
            if drone_schedule:
                fig_drones = viz.plot_gantt_chart(
                    drone_schedule, title="Drone Schedule"
                )
                st.plotly_chart(fig_drones, use_container_width=True)


def _render_schedule_details(solution):
    """Render detailed schedule table"""
    st.markdown("**Detailed Schedule**")

    schedule = solution.get("schedule", [])

    if not schedule:
        st.info("No schedule data available")
        return

    # Create detailed dataframe
    schedule_data = []

    for task in schedule:
        schedule_data.append(
            {
                "Vehicle": task["vehicle_id"],
                "Customer": task["customer_id"],
                "Start (min)": f"{task['start_time']:.1f}",
                "End (min)": f"{task['end_time']:.1f}",
                "Duration (min)": f"{task['end_time'] - task['start_time']:.1f}",
                "Service Time (min)": f"{task.get('service_time', 0):.1f}",
            }
        )

    df_schedule = pd.DataFrame(schedule_data)

    # Add filtering
    col1, col2 = st.columns([1, 3])

    with col1:
        vehicle_filter = st.multiselect(
            "Filter by Vehicle",
            options=df_schedule["Vehicle"].unique(),
            default=None,
            key=f"vehicle_filter_{solution.get('algorithm', 'unknown')}",
        )

    # Apply filter
    if vehicle_filter:
        df_filtered = df_schedule[df_schedule["Vehicle"].isin(vehicle_filter)]
    else:
        df_filtered = df_schedule

    # Display table
    st.dataframe(df_filtered, use_container_width=True, hide_index=True, height=400)

    # Export schedule
    col1, col2, col3 = st.columns([2, 1, 1])
    with col3:
        csv_data = df_schedule.to_csv(index=False)
        st.download_button(
            label="📥 Export Schedule",
            data=csv_data,
            file_name=f"schedule_{solution.get('algorithm', 'solution')}.csv",
            mime="text/csv",
            use_container_width=True,
        )


def _render_timeline_analysis(solution, problem_type):
    """Render timeline analysis and insights"""
    st.markdown("**Timeline Analysis**")

    schedule = solution.get("schedule", [])

    if not schedule:
        return

    # Group by vehicle
    vehicle_schedules = {}
    for task in schedule:
        vehicle_id = task["vehicle_id"]
        if vehicle_id not in vehicle_schedules:
            vehicle_schedules[vehicle_id] = []
        vehicle_schedules[vehicle_id].append(task)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Vehicle Efficiency**")

        efficiency_data = []

        for vehicle_id, tasks in vehicle_schedules.items():
            if not tasks:
                continue

            total_time = max(t["end_time"] for t in tasks)
            active_time = sum(t["end_time"] - t["start_time"] for t in tasks)
            idle_time = total_time - active_time
            efficiency = (active_time / total_time * 100) if total_time > 0 else 0

            efficiency_data.append(
                {
                    "Vehicle": vehicle_id,
                    "Efficiency": f"{efficiency:.1f}%",
                    "Active": f"{active_time:.1f} min",
                    "Idle": f"{idle_time:.1f} min",
                }
            )

        df_efficiency = pd.DataFrame(efficiency_data)

        # Color code by efficiency
        def color_efficiency(val):
            if "%" in str(val):
                percentage = float(val.replace("%", ""))
                if percentage >= 80:
                    return "background-color: #d4edda"
                elif percentage >= 60:
                    return "background-color: #fff3cd"
                else:
                    return "background-color: #f8d7da"
            return ""

        styled_efficiency = df_efficiency.style.applymap(
            color_efficiency, subset=["Efficiency"]
        )

        st.dataframe(styled_efficiency, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("**Workload Distribution**")

        workload_data = []

        for vehicle_id, tasks in vehicle_schedules.items():
            num_tasks = len(tasks)
            total_service_time = sum(t.get("service_time", 0) for t in tasks)

            workload_data.append(
                {
                    "Vehicle": vehicle_id,
                    "Tasks": num_tasks,
                    "Service Time": f"{total_service_time:.1f} min",
                    "Avg per Task": f"{total_service_time / num_tasks:.1f} min"
                    if num_tasks > 0
                    else "0 min",
                }
            )

        df_workload = pd.DataFrame(workload_data)
        st.dataframe(df_workload, use_container_width=True, hide_index=True)

    # Insights
    st.markdown("**💡 Insights**")

    insights = _generate_timeline_insights(vehicle_schedules, solution)

    for insight in insights:
        st.info(insight)


def _generate_timeline_insights(vehicle_schedules, solution):
    """Generate insights from timeline analysis"""
    insights = []

    # Calculate average efficiency
    efficiencies = []
    for vehicle_id, tasks in vehicle_schedules.items():
        if tasks:
            total_time = max(t["end_time"] for t in tasks)
            active_time = sum(t["end_time"] - t["start_time"] for t in tasks)
            efficiency = (active_time / total_time * 100) if total_time > 0 else 0
            efficiencies.append(efficiency)

    if efficiencies:
        avg_efficiency = sum(efficiencies) / len(efficiencies)

        if avg_efficiency >= 80:
            insights.append(
                f"✓ Excellent vehicle utilization: {avg_efficiency:.1f}% average efficiency"
            )
        elif avg_efficiency >= 60:
            insights.append(
                f"⚠ Moderate vehicle utilization: {avg_efficiency:.1f}% - Consider optimizing idle times"
            )
        else:
            insights.append(
                f"⚠ Low vehicle utilization: {avg_efficiency:.1f}% - Significant idle time detected"
            )

    # Check workload balance
    task_counts = [len(tasks) for tasks in vehicle_schedules.values()]
    if task_counts:
        max_tasks = max(task_counts)
        min_tasks = min(task_counts)

        if max_tasks - min_tasks > 3:
            insights.append(
                f"⚠ Workload imbalance: Some vehicles handle {max_tasks} tasks while others handle {min_tasks}"
            )
        else:
            insights.append(
                f"✓ Balanced workload: Tasks distributed evenly across vehicles"
            )

    # Check for parallel execution
    truck_schedules = {
        k: v for k, v in vehicle_schedules.items() if "truck" in k.lower()
    }
    drone_schedules = {
        k: v for k, v in vehicle_schedules.items() if "drone" in k.lower()
    }

    if truck_schedules and drone_schedules:
        insights.append(
            "✓ Parallel execution: Using both trucks and drones for improved efficiency"
        )

    return insights
