# utils/visualizer.py

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import folium
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
        Plot 2D route map with:
        - Different colors for truck vs drone
        - Hover highlight: fade other routes, highlight active route
        """
        fig = go.Figure()

        # --- Depot ---
        fig.add_trace(
            go.Scatter(
                x=[depot["x"]],
                y=[depot["y"]],
                mode="markers",
                marker=dict(size=20, color=self.color_depot, symbol="square"),
                name="Depot",
                text=["Depot"],
                hovertemplate="<b>Depot</b><br>(%{x:.1f}, %{y:.1f})<extra></extra>",
                customdata=[{"type": "depot"}],
            )
        )

        # --- Customers ---
        fig.add_trace(
            go.Scatter(
                x=customers["x"],
                y=customers["y"],
                mode="markers+text",
                marker=dict(size=10, color=self.color_customer, symbol="circle"),
                text=customers["id"],
                textposition="top center",
                name="Customers",
                hovertemplate="<b>Customer %{text}</b><br>(%{x:.1f}, %{y:.1f})<extra></extra>",
                customdata=[{"type": "customer"} for _ in customers["id"]],
            )
        )

        # -------- ROUTES --------
        for idx, (vehicle_id, route) in enumerate(routes.items()):
            if not route:
                continue

            # Choose color
            if "truck" in vehicle_id.lower():
                colors = self.colors_truck
            else:
                colors = self.colors_drone
            color = colors[idx % len(colors)]

            # Create route path (depot → customers → depot)
            xs = [depot["x"]]
            ys = [depot["y"]]

            for cust_id in route:
                c = customers.loc[customers["id"] == cust_id].iloc[0]
                xs.append(c["x"])
                ys.append(c["y"])

            xs.append(depot["x"])
            ys.append(depot["y"])

            # Add trace
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines+markers",
                    line=dict(color=color, width=2),
                    marker=dict(size=6, color=color),
                    name=f"{vehicle_id}",
                    customdata=[{"route_id": vehicle_id}] * len(xs),
                    hovertemplate="<b>Route: %s</b><extra></extra>" % vehicle_id,
                    opacity=1.0,  # default
                )
            )

        # Layout
        fig.update_layout(
            title=title,
            xaxis_title="X",
            yaxis_title="Y",
            hovermode="closest",
            template="plotly_white",
            height=600,
            showlegend=True,
        )

        # ---------- HOVER BEHAVIOR ----------
        # Fade other routes on hover, restore on unhover
        fig.update_layout(
            clickmode="none",
            hovermode="closest",
        )

        fig.add_layout_image(
            dict(
                source="",
                xref="paper",
                yref="paper",
                x=0,
                y=0,
                sizex=0.0001,
                sizey=0.0001,
            )
        )

        fig._js_on_hover = """
        function(e) {
            let hoverRoute = e.points[0].data.customdata[0].route_id;
            let update_opacity = [];

            for (let i = 0; i < this.data.length; i++) {
                const trace = this.data[i];
                if (!trace.customdata || !trace.customdata[0].route_id) {
                    update_opacity.push(1.0); // depot/customers
                    continue;
                }
                update_opacity.push(
                    trace.customdata[0].route_id === hoverRoute ? 1.0 : 0.15
                );
            }

            Plotly.restyle(this, {opacity: update_opacity});
        }
        """

        fig._js_on_unhover = """
        function(e) {
            let update_opacity = this.data.map(_ => 1.0);
            Plotly.restyle(this, {opacity: update_opacity});
        }
        """
        return fig

    def plot_gantt_chart(
        self, schedule: List[Dict], title: str = "Schedule Timeline"
    ) -> go.Figure:
        """
        Plot Gantt chart for schedule

        Args:
            schedule: List of dicts with keys: vehicle_id, customer_id, start_time, end_time
            title: Chart title

        Returns:
            Plotly Figure
        """
        if not schedule:
            fig = go.Figure()
            fig.update_layout(title="No schedule data available")
            return fig

        df = pd.DataFrame(schedule)

        fig = px.timeline(
            df,
            x_start="start_time",
            x_end="end_time",
            y="vehicle_id",
            color="vehicle_id",
            text="customer_id",
            title=title,
        )

        fig.update_yaxes(categoryorder="total ascending")
        fig.update_layout(
            xaxis_title="Time (minutes)",
            yaxis_title="Vehicle",
            height=400,
            showlegend=True,
        )

        return fig

    def plot_convergence(
        self,
        iterations: List[int],
        fitness_values: List[float],
        title: str = "Algorithm Convergence",
    ) -> go.Figure:
        """
        Plot algorithm convergence chart

        Args:
            iterations: List of iterations
            fitness_values: List of fitness values
            title: Chart title

        Returns:
            Plotly Figure
        """
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=fitness_values,
                mode="lines+markers",
                name="Best Fitness",
                line=dict(color="#0984E3", width=2),
                marker=dict(size=4),
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
        """
        Plot Pareto front for multi-objective problem

        Args:
            solutions: List of tuples (objective1, objective2)
            title: Chart title

        Returns:
            Plotly Figure
        """
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
                    size=10, color="#6C5CE7", line=dict(width=1, color="white")
                ),
                name="Solutions",
                hovertemplate="Objective 1: %{x:.2f}<br>Objective 2: %{y:.2f}<extra></extra>",
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
        """
        Create comparison table for algorithm results

        Args:
            results: Dict {algorithm_name: {metric: value}}

        Returns:
            DataFrame
        """
        df = pd.DataFrame(results).T
        df.index.name = "Algorithm"
        return df.reset_index()

    def plot_metrics_comparison(
        self, comparison_df: pd.DataFrame, metric: str = "makespan"
    ) -> go.Figure:
        """
        Plot metrics comparison between algorithms

        Args:
            comparison_df: Results DataFrame
            metric: Metric to compare

        Returns:
            Plotly Figure
        """
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=comparison_df["Algorithm"],
                y=comparison_df[metric],
                marker_color="#00B894",
                text=comparison_df[metric],
                texttemplate="%{text:.2f}",
                textposition="outside",
            )
        )

        fig.update_layout(
            title=f"Comparison: {metric.capitalize()}",
            xaxis_title="Algorithm",
            yaxis_title=metric.capitalize(),
            height=400,
            template="plotly_white",
        )

        return fig


# Test
if __name__ == "__main__":
    viz = Visualizer()
    print("Visualizer initialized successfully!")
