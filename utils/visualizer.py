# utils/visualizer.py - Patched version

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import List, Dict, Tuple
from config.default_config import COLORS
import numpy as np


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
        fig = go.Figure()

        # ============== LAYER 1: ROUTE LINES (TRUCKS & DRONES SAME COLOR EACH) ==============
        # Find first truck route and first drone route for legend
        first_truck = next((vid for vid in routes if "truck" in vid.lower()), None)
        first_drone = next((vid for vid in routes if "drone" in vid.lower()), None)

        for vehicle_id, route in routes.items():
            if not route:
                continue

            if "truck" in vehicle_id.lower():
                color = self.colors_truck[0]
                vehicle_icon = "🚚"
                legend_name = "🚚 Truck Routes"
                legend_group = "truck"
                dash = "solid"
                show_in_legend = vehicle_id == first_truck
            else:
                color = self.colors_drone[0]
                vehicle_icon = "🚁"
                legend_name = "🚁 Drone Routes"
                legend_group = "drone"
                dash = "dash"
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
            print((xs[i], ys[i]) for i in range(len(xs)))

            """
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color=color, width=3, dash=dash),
                    name=legend_name,
                    showlegend=show_in_legend,
                    legendgroup=legend_group,
                    hovertemplate=f"<b>{vehicle_icon} {vehicle_id}</b><extra></extra>",
                )
            )
            """
            for i in range(len(xs) - 1):
                x0, y0 = xs[i], ys[i]  # start
                x1, y1 = xs[i + 1], ys[i + 1]  # end

                fig.add_annotation(
                    x=x1,
                    y=y1,  # tip (end of segment)
                    ax=x0,
                    ay=y0,  # tail very close to tip
                    xref="x",
                    yref="y",
                    axref="x",
                    ayref="y",
                    showarrow=True,
                    arrowhead=3,
                    arrowsize=1.5,
                    arrowwidth=1.5,
                    arrowcolor=color,
                    opacity=0.9,
                )

        # ======== CUSTOM LEGEND ========

        # Truck legend entry (solid line + arrowhead)
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color=self.colors_truck[0], width=2, dash="solid"),
                name="Truck Route",
                showlegend=True,
            )
        )

        # Drone legend entry (dashed line + arrowhead)
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color=self.colors_drone[0], width=2, dash="solid"),
                name="Drone Route",
                showlegend=True,
            )
        )

        # Depot legend entry
        """
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers+text",
                text=["🏢"],
                textfont=dict(size=20),
                marker=dict(size=1, color="rgba(0,0,0,0)"),
                name="🏢 Depot",
                showlegend=True,
            )
        )
        """
        # Customer legend entry
        """
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers+text",
                text=["📦"],
                textfont=dict(size=20),
                marker=dict(size=1, color="rgba(0,0,0,0)"),
                name="📦 Customer",
                showlegend=True,
            )
        )
        """
        # ============== LAYER 2: DEPOT BACKGROUND CIRCLE ==============
        """
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
        """
        # ============== LAYER 3: DEPOT ICON ==============
        fig.add_trace(
            go.Scatter(
                x=[depot["x"]],
                y=[depot["y"]],
                mode="markers+text",
                text=["🏢"],
                textfont=dict(size=25),
                textposition="middle center",
                marker=dict(size=1, color="rgba(0,0,0,0)"),
                name="🏢 Depot",
                hovertemplate="<b>🏢 Depot</b><br>Coordinates: (%{x:.1f}, %{y:.1f})<extra></extra>",
                showlegend=True,
            )
        )

        # ============== LAYER 4: CUSTOMER BACKGROUND CIRCLES ==============
        """
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
        """
        # ============== LAYER 5: CUSTOMER ICONS ==============
        hover_text = []
        customer_icons = []
        for _, row in customers.iterrows():
            text = f"<b>Customer {int(row['id'])}</b><br>Coordinates: ({row['x']:.1f}, {row['y']:.1f})<br>Demand: {row['demand']:.2f} kg"
            if "service_time" in row:
                text += f"<br>Service: {row['service_time']} min"
            hover_text.append(text)
            customer_icons.append("📦")

        fig.add_trace(
            go.Scatter(
                x=customers["x"],
                y=customers["y"],
                mode="markers+text",
                text=customer_icons,
                textfont=dict(size=25),
                textposition="middle center",
                marker=dict(size=1, color="rgba(0,0,0,0)"),
                name="📦 Customers",
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

        # Create figure manually for more control
        fig = go.Figure()

        # Group tasks by vehicle
        vehicle_tasks = {}
        for task in schedule:
            vehicle_id = task.get("vehicle_id", "Unknown")
            if vehicle_id not in vehicle_tasks:
                vehicle_tasks[vehicle_id] = []
            vehicle_tasks[vehicle_id].append(task)

        # Sort vehicles
        sorted_vehicles = sorted(vehicle_tasks.keys())

        # Assign colors
        color_map = {}
        for vehicle in sorted_vehicles:
            if "truck" in vehicle.lower():
                color_map[vehicle] = self.colors_truck[0]
            else:
                color_map[vehicle] = self.colors_drone[0]

        # Plot each vehicle's tasks
        for idx, vehicle_id in enumerate(sorted_vehicles):
            tasks = vehicle_tasks[vehicle_id]
            color = color_map.get(vehicle_id, "#2563eb")

            for task in tasks:
                start = task.get("start_time", 0)
                end = task.get("end_time", 0)
                customer = task.get("customer_id", "?")

                # Add bar for this task
                fig.add_trace(
                    go.Bar(
                        x=[end - start],
                        y=[vehicle_id],
                        base=start,
                        orientation="h",
                        marker=dict(color=color, line=dict(color="white", width=1)),
                        name=vehicle_id,
                        showlegend=False,
                        text=customer,
                        textposition="inside",
                        textfont=dict(color="white", size=10),
                        hovertemplate=(
                            f"<b>{vehicle_id}</b><br>"
                            f"Customer: {customer}<br>"
                            f"Start: {start:.1f} min<br>"
                            f"End: {end:.1f} min<br>"
                            f"Duration: {end - start:.1f} min<br>"
                            "<extra></extra>"
                        ),
                        width=0.5,
                    )
                )

        # Update layout
        fig.update_layout(
            title=dict(
                text=title, font=dict(size=16, color="#0f172a"), x=0.5, xanchor="center"
            ),
            xaxis=dict(
                title="Time (minutes)",
                gridcolor="#e2e8f0",
                showgrid=True,
            ),
            yaxis=dict(
                title="Vehicle",
                categoryorder="array",
                categoryarray=sorted_vehicles,
            ),
            height=max(
                400, len(sorted_vehicles) * 60
            ),  # Dynamic height based on vehicles
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
        """
        Plot Pareto front with enhanced visualization

        Args:
            solutions: List of (obj1, obj2) tuples - Pareto optimal solutions
            title: Chart title
            current_solution: Optional (obj1, obj2) tuple for highlighting current solution
        """
        fig = go.Figure()

        if not solutions:
            fig.update_layout(
                title="No Pareto front data available",
                height=500,
                template="plotly_white",
            )
            return fig

        # Separate objectives
        obj1_values, obj2_values = zip(*solutions)
        obj1_values = list(obj1_values)
        obj2_values = list(obj2_values)

        # Sort by first objective for better visualization
        sorted_indices = np.argsort(obj1_values)
        obj1_sorted = [obj1_values[i] for i in sorted_indices]
        obj2_sorted = [obj2_values[i] for i in sorted_indices]

        # ============== LAYER 1: Pareto Front Line ==============
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

        # ============== LAYER 2: Solution Points ==============
        # Create hover text with solution details
        hover_texts = []
        for i, (obj1, obj2) in enumerate(zip(obj1_values, obj2_values)):
            hover_text = (
                f"<b>Solution {i + 1}</b><br>"
                f"Makespan: {obj1:.2f} min<br>"
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
                    symbol="circle",
                ),
                name="Pareto Solutions",
                hovertemplate="%{hovertext}",
                hovertext=hover_texts,
                showlegend=True,
            )
        )

        # ============== LAYER 3: Extreme Points Annotations ==============
        # Find and annotate extreme solutions
        min_obj1_idx = obj1_values.index(min(obj1_values))
        min_obj2_idx = obj2_values.index(min(obj2_values))

        # Fastest solution (minimum makespan)
        fig.add_trace(
            go.Scatter(
                x=[obj1_values[min_obj1_idx]],
                y=[obj2_values[min_obj1_idx]],
                mode="markers",
                marker=dict(
                    size=20,
                    color="#10b981",
                    symbol="star",
                    line=dict(width=2, color="white"),
                ),
                name="⚡ Fastest",
                hovertemplate=(
                    f"<b>⚡ Fastest Solution</b><br>"
                    f"Makespan: {obj1_values[min_obj1_idx]:.2f} min<br>"
                    f"Cost: ${obj2_values[min_obj1_idx]:,.2f}<br>"
                    f"<extra></extra>"
                ),
                showlegend=True,
            )
        )

        # Add annotation for fastest
        fig.add_annotation(
            x=obj1_values[min_obj1_idx],
            y=obj2_values[min_obj1_idx],
            text="⚡ Fastest",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#10b981",
            ax=0,
            ay=-50,
            font=dict(size=12, color="#10b981", family="Arial Black"),
            bgcolor="white",
            bordercolor="#10b981",
            borderwidth=2,
            borderpad=4,
        )

        # Cheapest solution (minimum cost)
        fig.add_trace(
            go.Scatter(
                x=[obj1_values[min_obj2_idx]],
                y=[obj2_values[min_obj2_idx]],
                mode="markers",
                marker=dict(
                    size=20,
                    color="#f59e0b",
                    symbol="star",
                    line=dict(width=2, color="white"),
                ),
                name="💰 Cheapest",
                hovertemplate=(
                    f"<b>💰 Cheapest Solution</b><br>"
                    f"Makespan: {obj1_values[min_obj2_idx]:.2f} min<br>"
                    f"Cost: ${obj2_values[min_obj2_idx]:,.2f}<br>"
                    f"<extra></extra>"
                ),
                showlegend=True,
            )
        )

        # Add annotation for cheapest
        fig.add_annotation(
            x=obj1_values[min_obj2_idx],
            y=obj2_values[min_obj2_idx],
            text="💰 Cheapest",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#f59e0b",
            ax=0,
            ay=50,
            font=dict(size=12, color="#f59e0b", family="Arial Black"),
            bgcolor="white",
            bordercolor="#f59e0b",
            borderwidth=2,
            borderpad=4,
        )

        # ============== LAYER 4: Balanced Solution (Middle point) ==============
        if len(obj1_values) >= 3:
            mid_idx = len(obj1_values) // 2

            fig.add_trace(
                go.Scatter(
                    x=[obj1_values[mid_idx]],
                    y=[obj2_values[mid_idx]],
                    mode="markers",
                    marker=dict(
                        size=18,
                        color="#8b5cf6",
                        symbol="diamond",
                        line=dict(width=2, color="white"),
                    ),
                    name="⚖️ Balanced",
                    hovertemplate=(
                        f"<b>⚖️ Balanced Solution</b><br>"
                        f"Makespan: {obj1_values[mid_idx]:.2f} min<br>"
                        f"Cost: ${obj2_values[mid_idx]:,.2f}<br>"
                        f"<extra></extra>"
                    ),
                    showlegend=True,
                )
            )

        # ============== LAYER 5: Current Solution Highlight ==============
        if current_solution:
            fig.add_trace(
                go.Scatter(
                    x=[current_solution[0]],
                    y=[current_solution[1]],
                    mode="markers",
                    marker=dict(
                        size=22,
                        color="#ef4444",
                        symbol="x",
                        line=dict(width=3, color="white"),
                    ),
                    name="📍 Current",
                    hovertemplate=(
                        f"<b>📍 Current Solution</b><br>"
                        f"Makespan: {current_solution[0]:.2f} min<br>"
                        f"Cost: ${current_solution[1]:,.2f}<br>"
                        f"<extra></extra>"
                    ),
                    showlegend=True,
                )
            )

        # ============== LAYER 6: Trade-off Region Shading ==============
        # Add shaded region to show trade-off space
        fig.add_trace(
            go.Scatter(
                x=obj1_sorted + obj1_sorted[::-1],
                y=obj2_sorted + [max(obj2_values)] * len(obj2_sorted),
                fill="toself",
                fillcolor="rgba(99, 110, 250, 0.05)",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
                name="Feasible Region",
            )
        )

        # ============== Layout Configuration ==============
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=18, color="#0f172a", family="Arial Black"),
                x=0.5,
                xanchor="center",
            ),
            xaxis=dict(
                title="<b>Objective 1: Makespan (minutes)</b>",
                title_font=dict(size=14, color="#475569"),
                gridcolor="#e2e8f0",
                showgrid=True,
                zeroline=False,
                tickfont=dict(size=12),
            ),
            yaxis=dict(
                title="<b>Objective 2: Cost (USD)</b>",
                title_font=dict(size=14, color="#475569"),
                gridcolor="#e2e8f0",
                showgrid=True,
                zeroline=False,
                tickfont=dict(size=12),
            ),
            height=550,
            template="plotly_white",
            hovermode="closest",
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="right",
                x=1.15,
                bgcolor="rgba(255,255,255,0.95)",
                bordercolor="#e2e8f0",
                borderwidth=1,
                font=dict(size=11),
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=80, r=150, t=80, b=80),
        )

        # Add grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#e2e8f0")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#e2e8f0")

        return fig

    def plot_pareto_comparison(
        self, pareto_fronts_dict, title="Pareto Front Comparison"
    ):
        """
        Compare Pareto fronts from multiple algorithms

        Args:
            pareto_fronts_dict: Dict of {algorithm_name: [(obj1, obj2), ...]}
            title: Chart title
        """
        fig = go.Figure()

        colors = ["#6366f1", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6"]
        symbols = ["circle", "square", "diamond", "cross", "star"]

        for idx, (algo_name, solutions) in enumerate(pareto_fronts_dict.items()):
            if not solutions:
                continue

            obj1_values, obj2_values = zip(*solutions)
            color = colors[idx % len(colors)]
            symbol = symbols[idx % len(symbols)]

            # Sort for line connection
            sorted_indices = np.argsort(obj1_values)
            obj1_sorted = [obj1_values[i] for i in sorted_indices]
            obj2_sorted = [obj2_values[i] for i in sorted_indices]

            # Add line
            fig.add_trace(
                go.Scatter(
                    x=obj1_sorted,
                    y=obj2_sorted,
                    mode="lines",
                    line=dict(color=color, width=1, dash="dash"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            # Add points
            fig.add_trace(
                go.Scatter(
                    x=list(obj1_values),
                    y=list(obj2_values),
                    mode="markers",
                    marker=dict(
                        size=12,
                        color=color,
                        symbol=symbol,
                        line=dict(width=2, color="white"),
                        opacity=0.8,
                    ),
                    name=algo_name,
                    hovertemplate=(
                        f"<b>{algo_name}</b><br>"
                        "Makespan: %{x:.2f} min<br>"
                        "Cost: $%{y:,.2f}<br>"
                        "<extra></extra>"
                    ),
                )
            )

        fig.update_layout(
            title=dict(text=title, font=dict(size=18, color="#0f172a"), x=0.5),
            xaxis_title="<b>Makespan (minutes)</b>",
            yaxis_title="<b>Cost (USD)</b>",
            height=550,
            template="plotly_white",
            hovermode="closest",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        return fig

    def plot_pareto_3d(self, solutions, obj3_values, title="3D Pareto Front"):
        """
        Plot 3D Pareto front for three objectives

        Args:
            solutions: List of (obj1, obj2) tuples
            obj3_values: List of third objective values
            title: Chart title
        """
        if not solutions or not obj3_values:
            fig = go.Figure()
            fig.update_layout(title="No 3D Pareto data available")
            return fig

        obj1_values, obj2_values = zip(*solutions)

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=list(obj1_values),
                    y=list(obj2_values),
                    z=obj3_values,
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=obj3_values,
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title="Objective 3"),
                        line=dict(width=1, color="white"),
                    ),
                    hovertemplate=(
                        "<b>Solution</b><br>"
                        "Obj 1: %{x:.2f}<br>"
                        "Obj 2: %{y:.2f}<br>"
                        "Obj 3: %{z:.2f}<br>"
                        "<extra></extra>"
                    ),
                )
            ]
        )

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="Objective 1",
                yaxis_title="Objective 2",
                zaxis_title="Objective 3",
            ),
            height=600,
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
