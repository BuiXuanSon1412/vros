# utils/visualizer.py - FIXED with Problem 3 Resupply Visualization

from operator import pos
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
    angle,
    color,
    size=16,
    opacity=1.0,
    position=0.5,  # New parameter: where along the line (0 to 1)
):
    """Add a directional marker arrow at a specified position along a segment"""
    # Calculate arrow position
    arrow_x = x0 + (x1 - x0) * position
    arrow_y = y0 + (y1 - y0) * position

    fig.add_trace(
        go.Scatter(
            x=[arrow_x],
            y=[arrow_y],
            mode="markers",
            marker=dict(
                symbol="triangle-up",
                size=size,
                color=color,
                angle=angle,
                opacity=opacity,
                line=dict(width=1, color="white"),
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
        # DEPOT META
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

        # ==========================================================
        # PASS 1 — COUNT UNDIRECTED EDGE USAGE (ALL SOURCES)
        # ==========================================================
        edge_usage = defaultdict(int)

        # Main routes
        for route in routes.values():
            if not route:
                continue
            path = ["depot"] + route + ["depot"]
            for i in range(len(path) - 1):
                a, b = str(path[i]), str(path[i + 1])
                edge_usage[tuple(sorted((a, b)))] += 1

        # Resupply routes (Problem 3)
        if problem_type == 3:
            for op in resupply_operations:
                meet = op.get("meeting_customer_id")
                if meet is None:
                    continue
                path = ["depot", meet, "depot"]
                for i in range(2):
                    a, b = str(path[i]), str(path[i + 1])
                    edge_usage[tuple(sorted((a, b)))] += 1

        # ==========================================================
        # PASS 2 — DRAW ROUTES
        # ==========================================================
        CURVE_OFFSETS = [-0.3, -0.18, -0.1, 0.1, 0.18, 0.3]
        edge_draw_count = defaultdict(int)

        def draw_edge(
            id0, x0, y0, id1, x1, y1, color, dash, show_legend, legend_name, arrow_size
        ):
            key = tuple(sorted((str(id0), str(id1))))
            usage = edge_usage[key]
            angle = np.degrees(np.arctan2(x1 - x0, y1 - y0))

            # ---------- straight ----------
            if usage == 1:
                fig.add_trace(
                    go.Scatter(
                        x=[x0, x1],
                        y=[y0, y1],
                        mode="lines",
                        line=dict(color=color, width=2, dash=dash),
                        hoverinfo="skip",
                        showlegend=show_legend,
                        name=legend_name if show_legend else "",
                    )
                )
                _add_marker_arrow(
                    fig, x0, y0, x1, y1, angle, color, size=arrow_size, position=0.5
                )

            # ---------- curved ----------
            else:
                k = CURVE_OFFSETS[edge_draw_count[key] % len(CURVE_OFFSETS)]
                edge_draw_count[key] += 1

                xc, yc = curved_edge(x0, y0, x1, y1, k=k)
                fig.add_trace(
                    go.Scatter(
                        x=xc,
                        y=yc,
                        mode="lines",
                        line=dict(color=color, width=2, dash=dash),
                        hoverinfo="skip",
                        showlegend=show_legend,
                        name=legend_name if show_legend else "",
                    )
                )

                mid = len(xc) // 2
                _add_marker_arrow(
                    fig,
                    xc[mid - 1],
                    yc[mid - 1],
                    xc[mid],
                    yc[mid],
                    angle,
                    color,
                    size=arrow_size,
                    position=0.5,  # arrow at the middle of these 4 points
                )

        # ---------- resupply routes ----------
        if problem_type == 3:
            for idx, op in enumerate(resupply_operations):
                meet_id = op.get("meeting_customer_id")
                if meet_id is None:
                    continue

                meet = customers.loc[customers["id"] == meet_id].iloc[0]
                color = self.colors_drone[0]

                draw_edge(
                    "depot",
                    depot["x"],
                    depot["y"],
                    meet_id,
                    meet["x"],
                    meet["y"],
                    color=color,
                    dash="dot",
                    show_legend=idx == 0,
                    legend_name="Drone Resupply",
                    arrow_size=10,
                )
                draw_edge(
                    meet_id,
                    meet["x"],
                    meet["y"],
                    "depot",
                    depot["x"],
                    depot["y"],
                    color=color,
                    dash="dot",
                    show_legend=False,
                    legend_name="",
                    arrow_size=10,
                )

        # ---------- main routes ----------
        for vehicle_id, route in routes.items():
            if not route:
                continue

            is_truck = "truck" in vehicle_id.lower()

            color = self.colors_truck[0] if is_truck else self.colors_drone[0]
            dash = "solid" if is_truck else "dot"

            show_legend = vehicle_id == (first_truck if is_truck else first_drone)
            legend_name = "Truck Route" if is_truck else "Drone Route"
            if problem_type == 3 and legend_name == "Truck Route":
                legend_name = "Technician Route"

            nodes = [("depot", depot["x"], depot["y"])]
            for cid in route:
                r = customers.loc[customers["id"] == cid].iloc[0]
                nodes.append((cid, r["x"], r["y"]))
            nodes.append(("depot", depot["x"], depot["y"]))

            for i in range(len(nodes) - 1):
                draw_edge(
                    *nodes[i],
                    *nodes[i + 1],
                    color=color,
                    dash=dash,
                    show_legend=show_legend and i == 0,
                    legend_name=legend_name,
                    arrow_size=12 if is_truck else 10,
                )

        # ==========================================================
        # DEPOT + CUSTOMERS
        # ==========================================================
        # Depot hover text
        depot_hover_text = f"<b>{depot_hover}</b><br>"
        depot_hover_text += f"Location: ({depot['x']:.2f}, {depot['y']:.2f})<br>"

        # Count vehicles starting here
        num_trucks = sum(1 for v in routes if "truck" in v.lower() and routes[v])
        num_drones = sum(1 for v in routes if "drone" in v.lower() and routes[v])

        if problem_type == 3:
            depot_hover_text += f"Technicians: {num_trucks}<br>"
        else:
            depot_hover_text += f"Trucks: {num_trucks}<br>"
        depot_hover_text += f"Drones: {num_drones}"

        fig.add_trace(
            go.Scatter(
                x=[depot["x"]],
                y=[depot["y"]],
                mode="markers+text",
                text=[depot_icon],
                textfont=dict(size=24),
                marker=dict(size=1, color="rgba(0,0,0,0)"),
                textposition="middle center",
                name=f"{depot_icon} {depot_label}",
                showlegend=True,
                hovertemplate="%{customdata}<extra></extra>",
                customdata=[depot_hover_text],
            )
        )

        cust_icon = "🧪" if problem_type in [1, 2] else "📦"
        cust_label = "Sample" if problem_type in [1, 2] else "Customer"

        # Build hover text for each customer
        hover_texts = []
        for _, customer in customers.iterrows():
            cust_id = customer["id"]

            # Find which vehicle serves this customer
            serving_vehicle = "Unassigned"
            for vehicle_id, route in routes.items():
                if cust_id in route:
                    serving_vehicle = vehicle_id
                    break

            # Base hover info
            hover_text = f"<b>Customer {int(cust_id)}</b><br>"
            hover_text += f"  Location: ({customer['x']:.2f}, {customer['y']:.2f})<br>"
            hover_text += f"  Demand: {customer['demand']:.2f} kg<br>"

            # Problem-specific info

            # Problem 3: Check if customer is served via resupply
            if problem_type == 3 and "release_date" in customer:
                hover_text += f"  Release Date: {customer['release_date']:.0f}<br>"

                # Find resupply operation for this customer
                resupply_info = None
                for op in resupply_operations:
                    if cust_id in op.get("packages", []):
                        resupply_info = op
                        break

                if resupply_info:
                    # This customer's package was delivered by drone
                    hover_text += "<b>Drone Resupply:</b><br>"
                    hover_text += f"  {resupply_info['drone_id']} → {resupply_info['truck_id']}<br>"
                    hover_text += (
                        f"  Meeting Point: C{resupply_info['meeting_customer_id']}<br>"
                    )

                    # Show all packages in this drone trip
                    packages = resupply_info["packages"]
                    # if len(packages) > 1:
                    package_list = ", ".join([f"C{p}" for p in packages])
                    hover_text += f"  Batch: {package_list}<br>"
                    hover_text += (
                        f"  Total Weight: {resupply_info['total_weight']:.2f} kg<br>"
                    )

                    hover_text += (
                        f"  Arrival: {resupply_info['arrival_time']:.1f} min<br>"
                    )

                elif serving_vehicle != "Unassigned":
                    # Direct service by truck
                    hover_text += f"<b>Direct Service:</b> {serving_vehicle}<br>"

            elif problem_type in [1, 2]:
                if "only_staff" in customer and customer["only_staff"] == 1:
                    hover_text += "Service: Technician Only<br>"
                hover_text += (
                    f"Service Time: {customer.get('service_time', 10)} min<br>"
                )

            if serving_vehicle != "Unassigned":
                hover_text += f"<b>Assigned to:</b> {serving_vehicle}"

            hover_texts.append(hover_text)

        # Plot customers with hover info
        fig.add_trace(
            go.Scatter(
                x=customers["x"],
                y=customers["y"],
                mode="markers+text",
                text=[cust_icon] * len(customers),
                textfont=dict(size=18),
                marker=dict(size=1, color="rgba(0,0,0,0)"),
                textposition="middle center",
                name=f"{cust_icon} {cust_label}",
                showlegend=True,
                hovertemplate="%{customdata}<extra></extra>",
                customdata=hover_texts,
            )
        )

        fig.update_layout(
            title=dict(text=title, x=0.5),
            template="plotly_white",
            height=600,
            hovermode="closest",
            margin=dict(l=60, r=120, t=60, b=60),
            legend=dict(x=1.02, y=1),
        )

        fig.update_xaxes(
            showgrid=True,
            gridcolor="#e5e7eb",
            gridwidth=1,
            zeroline=True,
            zerolinecolor="#cbd5e1",
        )

        fig.update_yaxes(
            showgrid=True,
            gridcolor="#e5e7eb",
            gridwidth=1,
            zeroline=True,
            zerolinecolor="#cbd5e1",
            scaleanchor="x",
            scaleratio=1,
        )
        return fig

    def plot_routes_3d(
        self,
        customers: pd.DataFrame,
        depot: Dict,
        solution_data: List,
        title: str = "Vehicle Routes with Drone Resupply",
    ) -> go.Figure:
        """
        Plot Problem 3 routes from ATSSolver format
        solution_data format: [truck_routes, drone_operations, drone_route_details]
        """
        if not solution_data or len(solution_data) < 2:
            return self._create_empty_plot(customers, depot, title)

        truck_routes = solution_data[0]  # List of truck routes
        drone_operations = solution_data[1]  # List of drone operations

        fig = go.Figure()

        # ==========================================================
        # PASS 1 — COUNT UNDIRECTED EDGE USAGE
        # ==========================================================
        edge_usage = defaultdict(int)

        # Count truck route edges
        for truck_route in truck_routes:
            path = [0] + [e["node"] for e in truck_route] + [0]
            for i in range(len(path) - 1):
                a, b = str(path[i]), str(path[i + 1])
                edge_usage[tuple(sorted((a, b)))] += 1

        # Count drone resupply edges
        for drone_op in drone_operations:
            stops = drone_op.get("stops", [])
            if not stops:
                continue

            # Build complete drone path: depot -> stop1 -> stop2 -> ... -> depot
            drone_path = [0]
            for stop in stops:
                node = stop["node"]
                if node != 0:
                    drone_path.append(node)
            drone_path.append(0)

            # Count edges in the drone route
            for i in range(len(drone_path) - 1):
                a, b = str(drone_path[i]), str(drone_path[i + 1])
                edge_usage[tuple(sorted((a, b)))] += 1

        # ==========================================================
        # PASS 2 — DRAW ROUTES
        # ==========================================================
        CURVE_OFFSETS = [-0.3, -0.18, -0.1, 0.1, 0.18, 0.3]
        edge_draw_count = defaultdict(int)

        def draw_edge(
            id0, x0, y0, id1, x1, y1, color, dash, show_legend, legend_name, arrow_size
        ):
            key = tuple(sorted((str(id0), str(id1))))
            usage = edge_usage[key]
            angle = np.degrees(np.arctan2(x1 - x0, y1 - y0))

            if usage == 1:
                fig.add_trace(
                    go.Scatter(
                        x=[x0, x1],
                        y=[y0, y1],
                        mode="lines",
                        line=dict(color=color, width=2, dash=dash),
                        hoverinfo="skip",
                        showlegend=show_legend,
                        name=legend_name if show_legend else "",
                    )
                )
                _add_marker_arrow(
                    fig, x0, y0, x1, y1, angle, color, size=arrow_size, position=0.5
                )
            else:
                k = CURVE_OFFSETS[edge_draw_count[key] % len(CURVE_OFFSETS)]
                edge_draw_count[key] += 1
                xc, yc = curved_edge(x0, y0, x1, y1, k=k)
                fig.add_trace(
                    go.Scatter(
                        x=xc,
                        y=yc,
                        mode="lines",
                        line=dict(color=color, width=2, dash=dash),
                        hoverinfo="skip",
                        showlegend=show_legend,
                        name=legend_name if show_legend else "",
                    )
                )
                mid = len(xc) // 2
                _add_marker_arrow(
                    fig,
                    xc[mid - 1],
                    yc[mid - 1],
                    xc[mid],
                    yc[mid],
                    angle,
                    color,
                    size=arrow_size,
                    position=0.5,
                )

        # Draw drone resupply routes first (so they appear behind truck routes)
        for idx, drone_op in enumerate(drone_operations):
            stops = drone_op.get("stops", [])
            if not stops:
                continue

            color = self.colors_drone[idx % len(self.colors_drone)]

            # Build complete drone route: depot -> stop1 -> stop2 -> ... -> depot
            drone_path = [(0, depot["x"], depot["y"])]

            for stop in stops:
                node_id = stop["node"]
                if node_id == 0:
                    continue

                node_data = customers.loc[customers["id"] == node_id]
                if node_data.empty:
                    continue

                node_row = node_data.iloc[0]
                drone_path.append((node_id, node_row["x"], node_row["y"]))

            # Return to depot
            drone_path.append((0, depot["x"], depot["y"]))

            # Draw all edges in the drone route
            for i in range(len(drone_path) - 1):
                draw_edge(
                    *drone_path[i],
                    *drone_path[i + 1],
                    color=color,
                    dash="dot",
                    show_legend=(idx == 0 and i == 0),
                    legend_name="Drone Resupply" if idx == 0 and i == 0 else "",
                    arrow_size=10,
                )

        # Draw truck routes
        for truck_idx, truck_route in enumerate(truck_routes):
            nodes = truck_route.get("packages", [])
            if not nodes:
                continue

            color = self.colors_truck[truck_idx % len(self.colors_truck)]

            # Build path with coordinates
            path_coords = [(0, depot["x"], depot["y"])]
            for node_info in nodes:
                node_id = node_info["node"]
                if node_id == 0:
                    continue
                node_data = customers.loc[customers["id"] == node_id]
                if not node_data.empty:
                    node_row = node_data.iloc[0]
                    path_coords.append((node_id, node_row["x"], node_row["y"]))
            path_coords.append((0, depot["x"], depot["y"]))

            # Draw edges
            for i in range(len(path_coords) - 1):
                draw_edge(
                    *path_coords[i],
                    *path_coords[i + 1],
                    color=color,
                    dash="solid",
                    show_legend=(truck_idx == 0 and i == 0),
                    legend_name="Truck Route" if truck_idx == 0 else "",
                    arrow_size=12,
                )

        # ==========================================================
        # DEPOT + CUSTOMERS
        # ==========================================================
        depot_hover = (
            f"<b>Depot</b><br>"
            f"Location: ({depot['x']:.2f}, {depot['y']:.2f})<br>"
            f"Trucks: {len(truck_routes)}<br>"
            f"Drones: {len(drone_operations)}"
        )

        fig.add_trace(
            go.Scatter(
                x=[depot["x"]],
                y=[depot["y"]],
                mode="markers+text",
                text=["🏢"],
                textfont=dict(size=24),
                marker=dict(size=1, color="rgba(0,0,0,0)"),
                textposition="middle center",
                name="🏢 Depot",
                showlegend=True,
                hovertemplate="%{customdata}<extra></extra>",
                customdata=[depot_hover],
            )
        )

        # Build customer hover texts
        hover_texts = []
        for _, customer in customers.iterrows():
            cust_id = customer["id"]
            hover_text = f"<b>Customer {int(cust_id)}</b><br>"
            hover_text += f"Location: ({customer['x']:.2f}, {customer['y']:.2f})<br>"
            hover_text += f"Demand: {customer['demand']:.0f} kg<br>"

            if "release_date" in customer:
                hover_text += f"Release: {customer['release_date']:.0f} min<br>"

            # Check if served by truck (direct service)
            serving_truck = None
            for truck_idx, truck_route in enumerate(truck_routes):
                if any(n["node"] == cust_id for n in truck_route.get("packages", [])):
                    serving_truck = f"Truck {truck_idx}"
                    break

            # Check if served by drone resupply
            serving_drone = None
            drone_route_info = None
            for drone_idx, drone_op in enumerate(drone_operations):
                stops = drone_op.get("stops", [])
                for stop_idx, stop in enumerate(stops):
                    if cust_id in stop.get("packages", []) or stop["node"] == cust_id:
                        serving_drone = f"Drone {drone_idx}"
                        # Get the full drone route sequence
                        route_sequence = [s["node"] for s in stops]
                        drone_route_info = {
                            "position": stop_idx + 1,
                            "total_stops": len(stops),
                            "route": route_sequence,
                            "packages": stop.get("packages", []),
                        }
                        break
                if serving_drone:
                    break

            if serving_truck:
                hover_text += f"<b>Served by:</b> {serving_truck}<br>"

            if serving_drone and drone_route_info:
                hover_text += f"<b>Resupply by:</b> {serving_drone}<br>"
                hover_text += f"  Stop {drone_route_info['position']}/{drone_route_info['total_stops']}<br>"

                # Show drone route
                route_str = " → ".join([f"C{n}" for n in drone_route_info["route"]])
                hover_text += f"  Drone Route: Depot → {route_str} → Depot<br>"

                # Show packages delivered at this stop
                if drone_route_info["packages"]:
                    pkg_str = ", ".join([f"C{p}" for p in drone_route_info["packages"]])
                    hover_text += f"  Packages: {pkg_str}"

            hover_texts.append(hover_text)

        fig.add_trace(
            go.Scatter(
                x=customers["x"],
                y=customers["y"],
                mode="markers+text",
                text=["📦"] * len(customers),
                textfont=dict(size=18),
                marker=dict(size=1, color="rgba(0,0,0,0)"),
                textposition="middle center",
                name="📦 Customer",
                showlegend=True,
                hovertemplate="%{customdata}<extra></extra>",
                customdata=hover_texts,
            )
        )

        fig.update_layout(
            title=dict(text=title, x=0.5),
            template="plotly_white",
            height=600,
            hovermode="closest",
            margin=dict(l=60, r=120, t=60, b=60),
            legend=dict(x=1.02, y=1),
        )

        fig.update_xaxes(
            showgrid=True, gridcolor="#e5e7eb", zeroline=True, zerolinecolor="#cbd5e1"
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor="#e5e7eb",
            zeroline=True,
            zerolinecolor="#cbd5e1",
            scaleanchor="x",
            scaleratio=1,
        )

        return fig

    def _create_empty_plot(
        self, customers: pd.DataFrame, depot: Dict, title: str
    ) -> go.Figure:
        """Create empty plot with just customers and depot"""
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=[depot["x"]],
                y=[depot["y"]],
                mode="markers+text",
                text=["🏢"],
                textfont=dict(size=24),
                marker=dict(size=1, color="rgba(0,0,0,0)"),
                name="Depot",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=customers["x"],
                y=customers["y"],
                mode="markers+text",
                text=["📦"] * len(customers),
                textfont=dict(size=18),
                marker=dict(size=1, color="rgba(0,0,0,0)"),
                name="Customer",
            )
        )

        fig.update_layout(
            title=title,
            template="plotly_white",
            height=600,
        )

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
