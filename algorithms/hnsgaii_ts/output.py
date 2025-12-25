import json
import os
from . import parameters as params


def output_graph_data(obj1: float, obj2: float, gen: int):
    """Saves generation and objective values for graphing in JSON format."""
    data = {"generation": gen + 1, "fitness1": obj1, "fitness2": obj2}
    # We append to a list of objects in the file
    file_path = params.output_graph  # Ensure this ends in .json

    existing_data = []
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []

    existing_data.append(data)
    with open(file_path, "w") as f:
        json.dump(existing_data, f, indent=4)


def output_logs(gen: int, pareto: list):
    """Logs the full route and fitness of all individuals in the current Pareto front."""
    log_entry = {
        "generation": gen,
        "num_solutions": len(pareto),
        "solutions": [
            {
                "route": individual.route,
                "fitness1": individual.fitness1,
                "fitness2": individual.fitness2,
            }
            for individual in pareto
        ],
    }

    file_path = params.output_log
    existing_logs = []
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            try:
                existing_logs = json.load(f)
            except json.JSONDecodeError:
                existing_logs = []

    existing_logs.append(log_entry)
    with open(file_path, "w") as f:
        json.dump(existing_logs, f, indent=4)


def output(
    pareto: list,
    time_val: float,
    end_iter: int,
    tb_time: float,
    last_time: float,
    last_update: int,
):
    """Outputs execution metadata and formatted routes into a structured JSON file."""
    output_data = {
        "metadata": {
            "total_time": time_val,
            "last_iteration": end_iter,
            "last_update_at_iter": last_update,
            "tabu_time": tb_time,
            "last_solution_time": last_time,
            "pareto_size": len(pareto),
        },
        "solutions": [],
    }

    for individual in pareto:
        sol_data = {
            "fitness1": individual.fitness1,
            "fitness2": individual.fitness2,
            "raw_route": individual.route,
            "formatted_route": format_route_string(individual.route),
        }
        output_data["solutions"].append(sol_data)

    with open(params.output_filename, "w") as f:
        json.dump(output_data, f, indent=4)


def output_pareto(pareto: list):
    """Outputs detailed solution breakdown by assignments in JSON."""
    detailed_pareto = []

    for individual in pareto:
        route = individual.route
        assignment = {"trucks": [], "drones": []}

        current_truck_route = []
        current_drone_tracks = []
        current_track = []

        # Logic to separate truck and drone routes based on separators
        for node in route:
            if node == 0:  # Separator between Trucks and Drones
                if current_truck_route:
                    assignment["trucks"].append(current_truck_route)
                current_truck_route = []
            elif node > params.NUM_CUS:
                if node <= params.NUM_CUS + params.NUM_TRUCKS:
                    # Truck route separator
                    if current_truck_route:
                        assignment["trucks"].append(current_truck_route)
                    current_truck_route = []
                else:
                    # Drone track separator
                    if current_track:
                        current_drone_tracks.append(current_track)
                    current_track = []
            else:
                # Actual customer node
                # Determining if currently in truck or drone phase
                # (Simple heuristic based on if we've passed the '0' marker)
                # For more complex logic, mirror your original parsing exactly
                current_truck_route.append(node)

        detailed_pareto.append(
            {
                "fitness1": individual.fitness1,
                "fitness2": individual.fitness2,
                "assignments": assignment,
                "raw_route": route,
            }
        )

    with open(params.output_pareto_filename, "w") as f:
        json.dump(detailed_pareto, f, indent=4)


def format_route_string(route: list) -> str:
    """Helper to maintain the visual route format inside JSON."""
    formatted = []
    for j in range(len(route)):
        if route[j] == 0:
            formatted.append("||")
        elif route[j] > params.NUM_CUS:
            if j > 0 and route[j - 1] == params.NUM_CUS:
                formatted.append("|")
            if j > 0 and j != len(route) - 1:
                if (route[j] - route[j - 1] != 1) and (route[j - 1] != 0):
                    formatted.append("|")
        else:
            formatted.append(f"{route[j]} ")
    return "".join(formatted).strip()


def output_pareto_result(pareto_records: list, k: int):
    """Outputs ranked Pareto results with criteria metadata."""
    criteria_label = (
        f"Criteria {k}: {0.25 * k} fitness1 and {0.25 * (3 - k)} fitness2"
        if k != 0
        else "General Ranking"
    )

    records = []
    limit = min(len(pareto_records), 5)

    for i in range(limit):
        records.append(
            {
                "rank": i + 1,
                "fitness1": pareto_records[i].indi.fitness1,
                "fitness2": pareto_records[i].indi.fitness2,
                "formatted_route": format_route_string(pareto_records[i].indi.route),
            }
        )

    result_entry = {"criteria": criteria_label, "top_solutions": records}

    file_path = params.output_pareto_selection_filename
    existing_results = []
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            try:
                existing_results = json.load(f)
            except json.JSONDecodeError:
                existing_results = []

    existing_results.append(result_entry)
    with open(file_path, "w") as f:
        json.dump(existing_results, f, indent=4)
