# utils/visualizer.py - Improved with proper layering

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from config.default_config import COLORS


class Visualizer:
    """Class for result visualization"""

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
    ) -> go.Figure:
        """
        Plot 2D route map with proper layering - routes behind, icons on top
        """
        fig = go.Figure()

        # ============== LAYER 1: ROUTE LINES (BACKGROUND) ==============
        for idx, (vehicle_id, route) in enumerate(routes.items()):
            if not route:
                continue

            # Choose color based on vehicle type
            if "truck" in vehicle_id.lower():
                colors = self.colors_truck
                vehicle_icon = "🚚"
            else:
                colors = self.colors_drone
                vehicle_icon = "🚁"

            color = colors[idx % len(colors)]

            # Create route path
            xs = [depot["x"]]
            ys = [depot["y"]]
            segments = ["Depot"]

            for cust_id in route:
                c = customers.loc[customers["id"] == cust_id].iloc[0]
                xs.append(c["x"])
                ys.append(c["y"])
                segments.append(f"C{cust_id}")

            xs.append(depot["x"])
            ys.append(depot["y"])
            segments.append("Depot")

            # Create hover text for route
            route_info = f"<b>{vehicle_icon} {vehicle_id}</b><br>"
            route_info += f"Customers: {len(route)}<br>"
            route_info += f"Path: {' → '.join(segments)}"

            # Add route line (will be drawn first = behind)
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color=color, width=3),
                    name=f"{vehicle_icon} {vehicle_id}",
                    hovertemplate=route_info + "<extra></extra>",
                    showlegend=True,
                    legendgroup=vehicle_id,
                )
            )

        # ============== LAYER 2: DEPOT BACKGROUND CIRCLE ==============
        fig.add_trace(
            go.Scatter(
                x=[depot["x"]],
                y=[depot["y"]],
                mode="markers",
                marker=dict(
                    size=35, color="white", line=dict(width=3, color=self.color_depot)
                ),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # ============== LAYER 3: DEPOT ICON ==============
        fig.add_trace(
            go.Scatter(
                x=[depot["x"]],
                y=[depot["y"]],
                mode="text",
                text=["🏢"],
                textfont=dict(size=20),
                textposition="middle center",
                name="Depot",
                hovertemplate="<b>🏢 Depot</b><br>Coordinates: (%{x:.1f}, %{y:.1f})<extra></extra>",
                showlegend=True,
            )
        )

        # ============== LAYER 4: CUSTOMER BACKGROUND CIRCLES ==============
        fig.add_trace(
            go.Scatter(
                x=customers["x"],
                y=customers["y"],
                mode="markers",
                marker=dict(
                    size=28,
                    color="white",
                    line=dict(width=2, color=self.color_customer),
                ),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # ============== LAYER 5: CUSTOMER ICONS ==============
        hover_text = []
        customer_icons = []

        for _, row in customers.iterrows():
            text = f"<b>Customer {row['id']}</b><br>"
            text += f"Coordinates: ({row['x']:.1f}, {row['y']:.1f})<br>"
            text += f"Demand: {row['demand']:.2f} kg"
            if "service_time" in row:
                text += f"<br>Service: {row['service_time']} min"
            hover_text.append(text)

            # Customize icon based on customer type
            # if row.get("only_staff", 0) == 1:
            #    customer_icons.append("🚚")  # Truck-only
            # else:
            #    customer_icons.append("📦")  # Regular package
            customer_icons.append("📦")  # Regular package

        fig.add_trace(
            go.Scatter(
                x=customers["x"],
                y=customers["y"],
                mode="text",
                text=customer_icons,
                textfont=dict(size=16),
                textposition="middle center",
                name="Customers",
                hovertemplate="%{hovertext}<extra></extra>",
                hovertext=hover_text,
                showlegend=True,
            )
        )

        # ============== LAYER 6: CUSTOMER ID LABELS ==============
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

        # Layout with better styling
        fig.update_layout(
            title=dict(
                text=title, font=dict(size=16, color="#0f172a"), x=0.5, xanchor="center"
            ),
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

        # Equal aspect ratio for accurate visualization
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

        return fig

    def plot_gantt_chart(
        self, schedule: List[Dict], title: str = "Schedule Timeline"
    ) -> go.Figure:
        """Plot Gantt chart for schedule"""
        if not schedule:
            fig = go.Figure()
            fig.update_layout(title="No schedule data available")
            return fig

        df = pd.DataFrame(schedule)

        # Create custom color mapping
        unique_vehicles = df["vehicle_id"].unique()
        color_map = {}
        for i, vehicle in enumerate(unique_vehicles):
            if "truck" in vehicle.lower():
                color_map[vehicle] = self.colors_truck[i % len(self.colors_truck)]
            else:
                color_map[vehicle] = self.colors_drone[i % len(self.colors_drone)]

        fig = px.timeline(
            df,
            x_start="start_time",
            x_end="end_time",
            y="vehicle_id",
            color="vehicle_id",
            text="customer_id",
            title=title,
            color_discrete_map=color_map,
        )

        fig.update_yaxes(categoryorder="total ascending")
        fig.update_layout(
            xaxis_title="Time (minutes)",
            yaxis_title="Vehicle",
            height=400,
            showlegend=True,
            template="plotly_white",
        )

        return fig

    def plot_convergence(
        self,
        iterations: List[int],
        fitness_values: List[float],
        title: str = "Algorithm Convergence",
    ) -> go.Figure:
        """Plot algorithm convergence chart"""
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

    def plot_pareto_front(
        self, solutions: List[Tuple[float, float]], title: str = "Pareto Front"
    ) -> go.Figure:
        """Plot Pareto front for multi-objective problem"""
        if not solutions:
            fig = go.Figure()
            fig.update_layout(title="No Pareto front data available")
            return fig

        obj1, obj2 = zip(*solutions)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=obj1,
                y=obj2,
                mode="markers",
                marker=dict(
                    size=12,
                    color="#6C5CE7",
                    line=dict(width=2, color="white"),
                    opacity=0.8,
                ),
                name="Solutions",
                hovertemplate="<b>Solution</b><br>Makespan: %{x:.2f}<br>Cost: %{y:.2f}<extra></extra>",
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Objective 1 (Makespan)",
            yaxis_title="Objective 2 (Cost)",
            height=400,
            template="plotly_white",
        )

        return fig

    def create_comparison_table(self, results: Dict[str, Dict]) -> pd.DataFrame:
        """Create comparison table for algorithm results"""
        df = pd.DataFrame(results).T
        df.index.name = "Algorithm"
        return df.reset_index()

    def plot_metrics_comparison(
        self, comparison_df: pd.DataFrame, metric: str = "Makespan"
    ) -> go.Figure:
        """Plot metrics comparison between algorithms"""
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
