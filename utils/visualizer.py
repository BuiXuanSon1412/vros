# utils/visualizer.py - FIXED with Problem 3 Resupply Visualization

import plotly.graph_objects as go
import pandas as pd
from typing import List, Dict, Tuple
from config.default_config import COLORS
import numpy as np
from collections import defaultdict


# ------------------------------
# Helper: generate curved edges
# ------------------------------
def curved_edge(x0, y0, x1, y1, k=0.15):
    """Quadratic bezier curve from (x0,y0) → (x1,y1). k controls curvature."""
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    dx, dy = x1 - x0, y1 - y0
    px, py = -dy, dx  # perpendicular

    cx = mx + k * px
    cy = my + k * py

    t = np.linspace(0, 1, 50)
    xs = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t**2 * x1
    ys = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t**2 * y1
    return xs, ys


class Visualizer:
    """Class for result visualization with resupply support"""

    def __init__(self):
        self.colors_truck = COLORS["truck"]
        self.colors_drone = COLORS["drone"]
        self.color_depot = COLORS["depot"]
        self.color_customer = COLORS["customer"]

    def plot_routes_2d(
        self,
        customers: pd.DataFrame,
        depot: Dict,
        routes: Dict,
        title: str = "Vehicle Routes",
        problem_type: int = 1,
        resupply_operations: List[Dict] = None,
    ) -> go.Figure:
        if resupply_operations is None:
            resupply_operations = []

        fig = go.Figure()

        # Determine depot type based on problem
        if problem_type in [1, 2]:
            depot_icon = "🏥"
            depot_label = "Medical Center"
            depot_hover = "Medical Center (Depot)"
        else:
            depot_icon = "🏢"
            depot_label = "Depot"
            depot_hover = "Depot"

        # ============== LAYER 1: TRUCK & DRONE ROUTE LINES WITH ARROWS ==============
        first_truck = next((vid for vid in routes if "truck" in vid.lower()), None)
        first_drone = next((vid for vid in routes if "drone" in vid.lower()), None)

        for vehicle_id, route in routes.items():
            if not route:
                continue

            is_truck = "truck" in vehicle_id.lower()

            if is_truck:
                color = self.colors_truck[0]
                vehicle_icon = "🚑" if problem_type in [1, 2] else "🚚"
                show_in_legend = vehicle_id == first_truck
            else:  # drone
                color = self.colors_drone[0]
                vehicle_icon = "🚁"
                show_in_legend = vehicle_id == first_drone

            # Build route path
            xs = [depot["x"]]
            ys = [depot["y"]]

            for cust_id in route:
                row = customers.loc[customers["id"] == cust_id].iloc[0]
                xs.append(row["x"])
                ys.append(row["y"])

            xs.append(depot["x"])
            ys.append(depot["y"])

            # Add arrows for route segments
            for i in range(len(xs) - 1):
                x0, y0 = xs[i], ys[i]
                x1, y1 = xs[i + 1], ys[i + 1]

                # Use dashed line for drones in Problems 1 & 2
                line_dash = (
                    "dot" if not is_truck and problem_type in [1, 2] else "solid"
                )
                arrow_size = 1.5 if is_truck else 1.2

                fig.add_annotation(
                    x=x1,
                    y=y1,
                    ax=x0,
                    ay=y0,
                    xref="x",
                    yref="y",
                    axref="x",
                    ayref="y",
                    showarrow=True,
                    arrowhead=3 if is_truck else 2,
                    arrowsize=arrow_size,
                    arrowwidth=2,
                    arrowcolor=color,
                    opacity=0.9 if is_truck else 0.7,
                )

        # ============== LAYER 2: DRONE RESUPPLY ROUTES (Problem 3) ==============
        if problem_type == 3 and resupply_operations:
            for idx, op in enumerate(resupply_operations):
                if not op.get("packages"):
                    continue

                meeting_id = op["meeting_customer_id"]
                meeting_customer = customers[customers["id"] == meeting_id].iloc[0]

                # Drone path: depot -> meeting point -> depot
                drone_xs = [depot["x"], meeting_customer["x"], depot["x"]]
                drone_ys = [depot["y"], meeting_customer["y"], depot["y"]]

                # Draw drone path with dashed line
                # drone_color = self.colors_drone[idx % len(self.colors_drone)]
                drone_color = self.colors_drone[0]
                fig.add_trace(
                    go.Scatter(
                        x=drone_xs,
                        y=drone_ys,
                        mode="lines",
                        line=dict(color=drone_color, width=2, dash="dot"),
                        showlegend=idx == 0,
                        name="Drone Resupply" if idx == 0 else "",
                        hoverinfo="skip",
                    )
                )

                # Add directional arrows
                for i in range(len(drone_xs) - 1):
                    fig.add_annotation(
                        x=drone_xs[i + 1],
                        y=drone_ys[i + 1],
                        ax=drone_xs[i],
                        ay=drone_ys[i],
                        xref="x",
                        yref="y",
                        axref="x",
                        ayref="y",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1.2,
                        arrowwidth=1.5,
                        arrowcolor=drone_color,
                        opacity=0.7,
                    )

                # Add package info at meeting point
                packages_str = ", ".join([f"C{p}" for p in op["packages"]])
                hover_text = (
                    f"<b>{op['drone_id']} Resupply</b><br>"
                    f"Meeting: C{meeting_id}<br>"
                    f"Packages: {packages_str}<br>"
                    f"Weight: {op['total_weight']:.2f} kg<br>"
                    f"Arrival: {op['arrival_time']:.1f} min"
                )

                fig.add_trace(
                    go.Scatter(
                        x=[meeting_customer["x"]],
                        y=[meeting_customer["y"]],
                        mode="markers",
                        marker=dict(
                            size=25,
                            color=drone_color,
                            # symbol="diamond",
                            opacity=0.5,
                            line=dict(width=2, color="white"),
                        ),
                        hovertemplate=hover_text + "<extra></extra>",
                        showlegend=False,
                    )
                )

        # ======== CUSTOM LEGEND ========
        truck_legend_name = "Ambulance" if problem_type in [1, 2] else "Truck"
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color=self.colors_truck[0], width=2, dash="solid"),
                name=f"{truck_legend_name} Route",
                showlegend=True,
            )
        )

        # ============== LAYER 3: DEPOT ICON ==============
        fig.add_trace(
            go.Scatter(
                x=[depot["x"]],
                y=[depot["y"]],
                mode="markers+text",
                text=[depot_icon],
                textfont=dict(size=30),
                textposition="middle center",
                marker=dict(size=1, color="rgba(0,0,0,0)"),
                name=f"{depot_icon} {depot_label}",
                hovertemplate=f"<b>{depot_hover}</b><br>Coordinates: (%{{x:.1f}}, %{{y:.1f}})<extra></extra>",
                showlegend=True,
            )
        )

        # ============== LAYER 4: CUSTOMER ICONS ==============
        if problem_type in [1, 2]:
            customer_icon = "🧪"
            customer_label = "Sample"
        else:
            customer_icon = "📦"
            customer_label = "Customer"

        hover_text = []
        customer_icons = []
        for _, row in customers.iterrows():
            text = f"<b>{customer_label} {int(row['id'])}</b><br>Coordinates: ({row['x']:.1f}, {row['y']:.1f})<br>Demand: {row['demand']:.2f} kg"
            if "service_time" in row:
                text += f"<br>Service: {row['service_time']} min"
            if "release_date" in row and problem_type == 3:
                text += f"<br>Release: {row['release_date']} min"
            hover_text.append(text)
            customer_icons.append(customer_icon)

        fig.add_trace(
            go.Scatter(
                x=customers["x"],
                y=customers["y"],
                mode="markers+text",
                text=customer_icons,
                textfont=dict(size=20),
                textposition="middle center",
                marker=dict(size=1, color="rgba(0,0,0,0)"),
                name=f"{customer_icon} {customer_label}s",
                hovertemplate="%{hovertext}<extra></extra>",
                hovertext=hover_text,
                showlegend=True,
            )
        )

        # ============== LAYER 5: CUSTOMER ID LABELS ==============
        fig.add_trace(
            go.Scatter(
                x=customers["x"],
                y=customers["y"],
                mode="text",
                text=customers["id"],
                textfont=dict(size=7, color="white", family="Arial Black"),
                textposition="middle center",
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Layout
        fig.update_layout(
            title=dict(text=title, font=dict(size=16, color="#0f172a"), x=0.5),
            xaxis=dict(
                title="X Coordinate",
                gridcolor="#e2e8f0",
                showgrid=True,
                zeroline=True,
                zerolinecolor="#cbd5e1",
                zerolinewidth=2,
            ),
            yaxis=dict(
                title="Y Coordinate",
                gridcolor="#e2e8f0",
                showgrid=True,
                zeroline=True,
                zerolinecolor="#cbd5e1",
                zerolinewidth=2,
            ),
            hovermode="closest",
            template="plotly_white",
            height=600,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                bgcolor="rgba(255,255,255,0.95)",
                bordercolor="#e2e8f0",
                borderwidth=1,
                font=dict(size=11),
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        return fig

    # ======================= Other plots remain unchanged =======================
    def plot_gantt_chart(
        self, schedule: List[Dict], title: str = "Schedule Timeline"
    ) -> go.Figure:
        """Create Gantt chart from schedule data"""

        if not schedule:
            fig = go.Figure()
            fig.add_annotation(
                text="No schedule data available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16, color="gray"),
            )
            fig.update_layout(height=400, template="plotly_white", title=title)
            return fig

        fig = go.Figure()

        vehicle_tasks = {}
        for task in schedule:
            vehicle_id = task.get("vehicle_id", "Unknown")
            if vehicle_id not in vehicle_tasks:
                vehicle_tasks[vehicle_id] = []
            vehicle_tasks[vehicle_id].append(task)

        sorted_vehicles = sorted(vehicle_tasks.keys())

        color_map = {}
        for vehicle in sorted_vehicles:
            if "truck" in vehicle.lower():
                color_map[vehicle] = self.colors_truck[0]
            else:
                color_map[vehicle] = self.colors_drone[0]

        for idx, vehicle_id in enumerate(sorted_vehicles):
            tasks = vehicle_tasks[vehicle_id]
            color = color_map.get(vehicle_id, "#2563eb")

            for task in tasks:
                start = task.get("start_time", 0)
                end = task.get("end_time", 0)
                customer = task.get("customer_id", "?")
                action = task.get("action", "Service")

                fig.add_trace(
                    go.Bar(
                        x=[end - start],
                        y=[vehicle_id],
                        base=start,
                        orientation="h",
                        marker=dict(color=color, line=dict(color="white", width=1)),
                        name=vehicle_id,
                        showlegend=False,
                        text=f"{customer}" if action != "Return" else "Return",
                        textposition="inside",
                        textfont=dict(color="white", size=10),
                        hovertemplate=(
                            f"<b>{vehicle_id}</b><br>"
                            f"Customer: {customer}<br>"
                            f"Action: {action}<br>"
                            f"Start: {start:.1f}<br>"
                            f"End: {end:.1f}<br>"
                            f"Duration: {end - start:.1f}<br>"
                            "<extra></extra>"
                        ),
                        width=0.5,
                    )
                )

        fig.update_layout(
            title=dict(
                text=title, font=dict(size=16, color="#0f172a"), x=0.5, xanchor="center"
            ),
            xaxis=dict(title="Time (minutes)", gridcolor="#e2e8f0", showgrid=True),
            yaxis=dict(
                title="Vehicle", categoryorder="array", categoryarray=sorted_vehicles
            ),
            height=max(400, len(sorted_vehicles) * 60),
            template="plotly_white",
            hovermode="closest",
            barmode="overlay",
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        return fig

    def plot_convergence(
        self,
        iterations: List[int],
        fitness_values: List[float],
        title: str = "Algorithm Convergence",
    ) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=fitness_values,
                mode="lines+markers",
                name="Best Fitness",
                line=dict(color="#2563eb", width=3),
                marker=dict(size=6, color="#2563eb"),
                fill="tozeroy",
                fillcolor="rgba(37, 99, 235, 0.1)",
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title="Iteration",
            yaxis_title="Fitness Value (Makespan)",
            height=400,
            template="plotly_white",
            hovermode="x unified",
        )
        return fig

    def plot_pareto_front(self, solutions, title="Pareto Front", current_solution=None):
        fig = go.Figure()

        if not solutions:
            fig.update_layout(
                title="No Pareto front data available",
                height=500,
                template="plotly_white",
            )
            return fig

        obj1_values, obj2_values = zip(*solutions)
        obj1_values = list(obj1_values)
        obj2_values = list(obj2_values)

        sorted_indices = np.argsort(obj1_values)
        obj1_sorted = [obj1_values[i] for i in sorted_indices]
        obj2_sorted = [obj2_values[i] for i in sorted_indices]

        fig.add_trace(
            go.Scatter(
                x=obj1_sorted,
                y=obj2_sorted,
                mode="lines",
                line=dict(color="rgba(99, 110, 250, 0.3)", width=2, dash="dash"),
                name="Pareto Front",
                showlegend=True,
                hoverinfo="skip",
            )
        )

        hover_texts = []
        for i, (obj1, obj2) in enumerate(zip(obj1_values, obj2_values)):
            hover_text = (
                f"<b>Solution {i + 1}</b><br>"
                f"Makespan: {obj1:.2f}<br>"
                f"Cost: ${obj2:,.2f}<br>"
                f"<extra></extra>"
            )
            hover_texts.append(hover_text)

        fig.add_trace(
            go.Scatter(
                x=obj1_values,
                y=obj2_values,
                mode="markers",
                marker=dict(
                    size=14,
                    color="#6366f1",
                    line=dict(width=2, color="white"),
                    opacity=0.8,
                ),
                name="Pareto Solutions",
                hovertemplate="%{hovertext}",
                hovertext=hover_texts,
                showlegend=True,
            )
        )

        fig.update_layout(
            title=dict(text=title, font=dict(size=18, color="#0f172a"), x=0.5),
            xaxis_title="<b>Objective 1: Makespan</b>",
            yaxis_title="<b>Objective 2: Cost</b>",
            height=550,
            template="plotly_white",
            hovermode="closest",
            showlegend=True,
        )

        return fig

    def create_comparison_table(self, results: Dict[str, Dict]) -> pd.DataFrame:
        df = pd.DataFrame(results).T
        df.index.name = "Algorithm"
        return df.reset_index()

    def plot_metrics_comparison(
        self, comparison_df: pd.DataFrame, metric: str = "Makespan"
    ) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=comparison_df["Algorithm"],
                y=comparison_df[metric],
                marker_color="#2563eb",
                text=comparison_df[metric],
                texttemplate="%{text:.2f}",
                textposition="outside",
            )
        )
        fig.update_layout(
            title=f"Comparison: {metric}",
            xaxis_title="Algorithm",
            yaxis_title=metric,
            height=400,
            template="plotly_white",
        )
        return fig

    def plot_routes_2d1(
        self,
        customers: pd.DataFrame,
        depot: Dict,
        routes: Dict,
        title: str = "Vehicle Routes",
        problem_type: int = 1,
        resupply_operations: List[Dict] = None,
    ) -> go.Figure:
        if resupply_operations is None:
            resupply_operations = []

        fig = go.Figure()

        # Determine depot type based on problem
        if problem_type in [1, 2]:
            depot_icon = "🏥"
            depot_label = "Medical Center"
            depot_hover = "Medical Center (Depot)"
        else:
            depot_icon = "🏢"
            depot_label = "Depot"
            depot_hover = "Depot"

        # ============== LAYER 1: TRUCK ROUTE LINES WITH ARROWS ==============
        first_truck = next((vid for vid in routes if "truck" in vid.lower()), None)
        first_drone = next((vid for vid in routes if "drone" in vid.lower()), None)

        # Track overlap counts
        edge_count = defaultdict(int)

        # Curvature offsets for overlapping edges
        CURVE_OFFSETS = [-0.25, -0.12, 0, 0.12, 0.25, 0.35, -0.35]

        # =====================================================
        # LAYER 1: TRUCK ROUTES with curved edges
        # =====================================================
        for vehicle_id, route in routes.items():
            if not route:
                continue

            if "truck" in vehicle_id.lower():
                color = self.colors_truck[0]
                vehicle_icon = "🚑" if problem_type in [1, 2] else "🚚"
                legend_name = "Truck Route"
                show_in_legend = vehicle_id == first_truck

                # Build truck route path
                xs = [depot["x"]]
                ys = [depot["y"]]

                for cust_id in route:
                    row = customers.loc[customers["id"] == cust_id].iloc[0]
                    xs.append(row["x"])
                    ys.append(row["y"])

                xs.append(depot["x"])
                ys.append(depot["y"])

                # Draw curved edges
                for i in range(len(xs) - 1):
                    x0, y0 = xs[i], ys[i]
                    x1, y1 = xs[i + 1], ys[i + 1]

                    # Overlap detection
                    key = tuple(sorted([(x0, y0), (x1, y1)]))
                    count = edge_count[key]
                    edge_count[key] += 1
                    k = CURVE_OFFSETS[count % len(CURVE_OFFSETS)]

                    # Generate curved line
                    xc, yc = curved_edge(x0, y0, x1, y1, k=k)

                    # Draw curve
                    fig.add_trace(
                        go.Scatter(
                            x=xc,
                            y=yc,
                            mode="lines",
                            line=dict(color=color, width=2),
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )

                    # Draw arrow at end of curve
                    fig.add_annotation(
                        x=xc[-1],
                        y=yc[-1],
                        ax=xc[-2],
                        ay=yc[-2],
                        xref="x",
                        yref="y",
                        axref="x",
                        ayref="y",
                        showarrow=True,
                        arrowhead=3,
                        arrowsize=1.5,
                        arrowwidth=2,
                        arrowcolor=color,
                        opacity=0.9,
                    )

        # =====================================================
        # LAYER 2: DRONE RESUPPLY ROUTES with curved edges
        # =====================================================
        if problem_type == 3 and resupply_operations:
            for idx, op in enumerate(resupply_operations):
                if not op.get("packages"):
                    continue

                meeting_id = op["meeting_customer_id"]
                meeting_customer = customers[customers["id"] == meeting_id].iloc[0]

                # Drone path: depot -> meeting point -> depot
                drone_xs = [depot["x"], meeting_customer["x"], depot["x"]]
                drone_ys = [depot["y"], meeting_customer["y"], depot["y"]]

                drone_color = self.colors_drone[0]

                # Curved edges for drones
                for i in range(len(drone_xs) - 1):
                    x0, y0 = drone_xs[i], drone_ys[i]
                    x1, y1 = drone_xs[i + 1], drone_ys[i + 1]

                    # Overlap detection
                    key = tuple(sorted([(x0, y0), (x1, y1)]))
                    count = edge_count[key]
                    edge_count[key] += 1
                    k = CURVE_OFFSETS[count % len(CURVE_OFFSETS)]

                    # Compute curved path
                    xc, yc = curved_edge(x0, y0, x1, y1, k=k)

                    # Draw curve (dashed style)
                    fig.add_trace(
                        go.Scatter(
                            x=xc,
                            y=yc,
                            mode="lines",
                            line=dict(color=drone_color, width=2, dash="dot"),
                            hoverinfo="skip",
                            showlegend=(idx == 0),
                            name="Drone Resupply" if idx == 0 else "",
                        )
                    )

                    # Arrow on curved path
                    fig.add_annotation(
                        x=xc[-1],
                        y=yc[-1],
                        ax=xc[-2],
                        ay=yc[-2],
                        xref="x",
                        yref="y",
                        axref="x",
                        ayref="y",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1.2,
                        arrowwidth=1.5,
                        arrowcolor=drone_color,
                        opacity=0.7,
                    )

                # Meeting point marker
                packages_str = ", ".join([f"C{p}" for p in op["packages"]])
                hover_text = (
                    f"<b>{op['drone_id']} Resupply</b><br>"
                    f"Meeting: C{meeting_id}<br>"
                    f"Packages: {packages_str}<br>"
                    f"Weight: {op['total_weight']:.2f} kg<br>"
                    f"Arrival: {op['arrival_time']:.1f} min"
                )

                fig.add_trace(
                    go.Scatter(
                        x=[meeting_customer["x"]],
                        y=[meeting_customer["y"]],
                        mode="markers",
                        marker=dict(
                            size=25,
                            color=drone_color,
                            opacity=0.5,
                            line=dict(width=2, color="white"),
                        ),
                        hovertemplate=hover_text + "<extra></extra>",
                        showlegend=False,
                    )
                )

            # ======== CUSTOM LEGEND ========
        truck_legend_name = "Ambulance" if problem_type in [1, 2] else "Truck"
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color=self.colors_truck[0], width=2, dash="solid"),
                name=f"{truck_legend_name} Route",
                showlegend=True,
            )
        )

        # ============== LAYER 3: DEPOT ICON ==============
        fig.add_trace(
            go.Scatter(
                x=[depot["x"]],
                y=[depot["y"]],
                mode="markers+text",
                text=[depot_icon],
                textfont=dict(size=30),
                textposition="middle center",
                marker=dict(size=1, color="rgba(0,0,0,0)"),
                name=f"{depot_icon} {depot_label}",
                hovertemplate=f"<b>{depot_hover}</b><br>Coordinates: (%{{x:.1f}}, %{{y:.1f}})<extra></extra>",
                showlegend=True,
            )
        )

        # ============== LAYER 4: CUSTOMER ICONS ==============
        if problem_type in [1, 2]:
            customer_icon = "🧪"
            customer_label = "Sample"
        else:
            customer_icon = "📦"
            customer_label = "Customer"

        hover_text = []
        customer_icons = []
        for _, row in customers.iterrows():
            text = f"<b>{customer_label} {int(row['id'])}</b><br>Coordinates: ({row['x']:.1f}, {row['y']:.1f})<br>Demand: {row['demand']:.2f} kg"
            if "service_time" in row:
                text += f"<br>Service: {row['service_time']} min"
            if "release_date" in row and problem_type == 3:
                text += f"<br>Release: {row['release_date']} min"
            hover_text.append(text)
            customer_icons.append(customer_icon)

        fig.add_trace(
            go.Scatter(
                x=customers["x"],
                y=customers["y"],
                mode="markers+text",
                text=customer_icons,
                textfont=dict(size=20),
                textposition="middle center",
                marker=dict(size=1, color="rgba(0,0,0,0)"),
                name=f"{customer_icon} {customer_label}s",
                hovertemplate="%{hovertext}<extra></extra>",
                hovertext=hover_text,
                showlegend=True,
            )
        )

        # ============== LAYER 5: CUSTOMER ID LABELS ==============
        fig.add_trace(
            go.Scatter(
                x=customers["x"],
                y=customers["y"],
                mode="text",
                text=customers["id"],
                textfont=dict(size=7, color="white", family="Arial Black"),
                textposition="middle center",
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Layout
        fig.update_layout(
            title=dict(text=title, font=dict(size=16, color="#0f172a"), x=0.5),
            xaxis=dict(
                title="X Coordinate",
                gridcolor="#e2e8f0",
                showgrid=True,
                zeroline=True,
                zerolinecolor="#cbd5e1",
                zerolinewidth=2,
            ),
            yaxis=dict(
                title="Y Coordinate",
                gridcolor="#e2e8f0",
                showgrid=True,
                zeroline=True,
                zerolinecolor="#cbd5e1",
                zerolinewidth=2,
            ),
            hovermode="closest",
            template="plotly_white",
            height=600,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                bgcolor="rgba(255,255,255,0.95)",
                bordercolor="#e2e8f0",
                borderwidth=1,
                font=dict(size=11),
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        return fig
