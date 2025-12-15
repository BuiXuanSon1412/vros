# utils/visualizer.py - FIXED with Problem 3 Resupply Visualization

import plotly.graph_objects as go
import pandas as pd
from typing import List, Dict
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


def _add_marker_arrow(
    fig,
    x0,
    y0,
    x1,
    y1,
    color,
    size=9,
    opacity=1.0,
):
    angle = np.degrees(np.arctan2(y1 - y0, x1 - x0))

    fig.add_trace(
        go.Scatter(
            x=[x1],
            y=[y1],
            mode="markers",
            marker=dict(
                symbol="triangle-up",
                size=size,
                color=color,
                angle=angle,
                opacity=opacity,
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )


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
        resupply_operations: List[Dict] | None = None,
    ) -> go.Figure:
        if resupply_operations is None:
            resupply_operations = []

        fig = go.Figure()

        # ==========================================================
        # DEPOT INFO
        # ==========================================================
        if problem_type in [1, 2]:
            depot_icon, depot_label, depot_hover = (
                "🏥",
                "Medical Center",
                "Medical Center (Depot)",
            )
        else:
            depot_icon, depot_label, depot_hover = "🏢", "Depot", "Depot"

        first_truck = next((v for v in routes if "truck" in v.lower()), None)
        first_drone = next((v for v in routes if "drone" in v.lower()), None)

        use_curvature = problem_type == 3

        CURVE_OFFSETS = [-0.15, -0.08, 0.0, 0.08, 0.15]
        truck_edges = defaultdict(int)
        drone_edges = defaultdict(int)

        # ==========================================================
        # LAYER 1: ROUTES FROM `routes`
        # ==========================================================
        for vehicle_id, route in routes.items():
            if not route:
                continue

            is_truck = "truck" in vehicle_id.lower()
            is_drone = "drone" in vehicle_id.lower()

            color = self.colors_truck[0] if is_truck else self.colors_drone[0]
            dash = "solid" if is_truck else "dot"

            edge_counter = truck_edges if is_truck else drone_edges
            legend_name = "Truck Route" if is_truck else "Drone Route"
            show_legend = (
                vehicle_id == first_truck if is_truck else vehicle_id == first_drone
            )

            # Build path
            xs = [depot["x"]]
            ys = [depot["y"]]

            for cid in route:
                row = customers.loc[customers["id"] == cid].iloc[0]
                xs.append(row["x"])
                ys.append(row["y"])

            xs.append(depot["x"])
            ys.append(depot["y"])

            # Draw segments
            for i in range(len(xs) - 1):
                x0, y0 = xs[i], ys[i]
                x1, y1 = xs[i + 1], ys[i + 1]

                if use_curvature:
                    key = tuple(sorted([(x0, y0), (x1, y1)]))
                    k = CURVE_OFFSETS[edge_counter[key] % len(CURVE_OFFSETS)]
                    edge_counter[key] += 1

                    xc, yc = curved_edge(x0, y0, x1, y1, k=k)

                    fig.add_trace(
                        go.Scatter(
                            x=xc,
                            y=yc,
                            mode="lines",
                            line=dict(color=color, width=2, dash=dash),
                            hoverinfo="skip",
                            showlegend=show_legend and i == 0,
                            name=legend_name if show_legend and i == 0 else "",
                        )
                    )

                    ai = int(len(xc) * 0.85)
                    _add_marker_arrow(
                        fig,
                        xc[ai - 1],
                        yc[ai - 1],
                        xc[ai],
                        yc[ai],
                        color,
                        size=10 if is_truck else 8,
                        opacity=1.0 if is_truck else 0.8,
                    )

                else:
                    fig.add_trace(
                        go.Scatter(
                            x=[x0, x1],
                            y=[y0, y1],
                            mode="lines",
                            line=dict(color=color, width=2, dash=dash),
                            hoverinfo="skip",
                            showlegend=show_legend and i == 0,
                            name=legend_name if show_legend and i == 0 else "",
                        )
                    )

                    _add_marker_arrow(
                        fig,
                        x0,
                        y0,
                        x1,
                        y1,
                        color,
                        size=10 if is_truck else 8,
                        opacity=1.0 if is_truck else 0.8,
                    )

        # ==========================================================
        # LAYER 2: DRONE RESUPPLY (Problem 3)
        # ==========================================================
        if problem_type == 3:
            for idx, op in enumerate(resupply_operations):
                meet_id = op.get("meeting_customer_id")
                if meet_id is None:
                    continue

                meet = customers.loc[customers["id"] == meet_id].iloc[0]
                color = self.colors_drone[0]

                xs = [depot["x"], meet["x"], depot["x"]]
                ys = [depot["y"], meet["y"], depot["y"]]

                for i in range(2):
                    xc, yc = curved_edge(xs[i], ys[i], xs[i + 1], ys[i + 1], k=0.12)

                    fig.add_trace(
                        go.Scatter(
                            x=xc,
                            y=yc,
                            mode="lines",
                            line=dict(color=color, width=2, dash="dot"),
                            hoverinfo="skip",
                            showlegend=idx == 0 and i == 0,
                            name="Drone Resupply" if idx == 0 and i == 0 else "",
                        )
                    )

                    ai = int(len(xc) * 0.85)
                    _add_marker_arrow(
                        fig,
                        xc[ai - 1],
                        yc[ai - 1],
                        xc[ai],
                        yc[ai],
                        color,
                        size=8,
                        opacity=0.75,
                    )

        # ==========================================================
        # LAYER 3: DEPOT
        # ==========================================================
        fig.add_trace(
            go.Scatter(
                x=[depot["x"]],
                y=[depot["y"]],
                mode="markers+text",
                text=[depot_icon],
                textfont=dict(size=30),
                marker=dict(size=1, color="rgba(0,0,0,0)"),
                textposition="middle center",
                name=f"{depot_icon} {depot_label}",
                hovertemplate=f"<b>{depot_hover}</b><br>(%{{x:.1f}}, %{{y:.1f}})<extra></extra>",
                showlegend=True,
            )
        )

        # ==========================================================
        # LAYER 4: CUSTOMERS
        # ==========================================================
        cust_icon = "🧪" if problem_type in [1, 2] else "📦"
        cust_label = "Sample" if problem_type in [1, 2] else "Customer"

        fig.add_trace(
            go.Scatter(
                x=customers["x"],
                y=customers["y"],
                mode="markers+text",
                text=[cust_icon] * len(customers),
                textfont=dict(size=18),
                marker=dict(size=1, color="rgba(0,0,0,0)"),
                textposition="middle center",
                name=f"{cust_icon} {cust_label}s",
                showlegend=True,
                hovertemplate=(
                    # f"<b>{cust_label} %{text}</b><br>"
                    f"<b>{cust_label}</b><br>"
                    "Coordinates: (%{x:.1f}, %{y:.1f})<extra></extra>"
                ),
                # customdata=customers["id"],
            )
        )

        # ==========================================================
        # LAYOUT
        # ==========================================================
        fig.update_layout(
            title=dict(text=title, x=0.5),
            template="plotly_white",
            height=600,
            hovermode="closest",
            margin=dict(l=60, r=120, t=60, b=60),
            legend=dict(x=1.02, y=1, bgcolor="rgba(255,255,255,0.95)"),
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
