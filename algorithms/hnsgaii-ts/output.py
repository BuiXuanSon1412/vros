import parameters as params


def output_graph_data(obj1: float, obj2: float, gen: int):
    """Saves generation and objective values for graphing."""
    with open(params.output_graph, "a") as output_file:
        output_file.write(f"{gen + 1} {obj1} {obj2}\n")


def output_logs(gen: int, pareto: list):
    """Logs the full route and fitness of all individuals in the current Pareto front."""
    with open(params.output_log, "a") as output_file:
        output_file.write(f"Generation {gen}: {len(pareto)} sols\n")
        for individual in pareto:
            # Join route elements with spaces
            route_str = " ".join(map(str, individual.route))
            output_file.write(f"{route_str}\n")
            output_file.write(f"{individual.fitness1} {individual.fitness2}\n")
        output_file.write("\n")


def output(
    pareto: list,
    time_val: float,
    end_iter: int,
    tb_time: float,
    last_time: float,
    last_update: int,
):
    """Outputs a summarized result including execution metadata and formatted routes."""
    with open(params.output_filename, "a") as output_file:
        output_file.write(f"Time:{time_val}\n")
        output_file.write(f"Last Iter:{end_iter}\n")
        output_file.write(f"Last Update:{last_update}\n")
        output_file.write(f"Tabu:{tb_time}\n")
        output_file.write(f"Last Time{last_time}\n")
        output_file.write(f"{len(pareto)}\n")

        for individual in pareto:
            route = individual.route
            formatted_route = []
            for j in range(len(route)):
                if route[j] == 0:
                    formatted_route.append("||")
                elif route[j] > params.NUM_CUS:
                    if j > 0 and route[j - 1] == params.NUM_CUS:
                        formatted_route.append("|")

                    # Logic for non-sequential drone tracks
                    if j > 0 and j != len(route) - 1:
                        if (route[j] - route[j - 1] != 1) and (route[j - 1] != 0):
                            formatted_route.append("|")
                else:
                    formatted_route.append(f"{route[j]} ")

            output_file.write("".join(formatted_route) + "\n")
            output_file.write(f"{individual.fitness1} {individual.fitness2}\n")
        output_file.write("\n")


def output_pareto(pareto: list):
    """Outputs detailed solution breakdown by Truck and Drone assignments."""
    with open(params.output_pareto_filename, "w") as output_file:
        for i, individual in enumerate(pareto):
            output_file.write(f"Solution {i + 1}:\n")
            route = individual.route
            num_drone = 1
            output_file.write("Trucks:\n")

            for j in range(len(route)):
                if route[j] > params.NUM_CUS:
                    if route[j] > params.NUM_CUS + params.NUM_TRUCKS:
                        a = route[j] - params.NUM_CUS - params.NUM_TRUCKS + 1
                        if a % params.drone_max_tracks == 0:
                            if j > 0 and route[j] - route[j - 1] != 1:
                                output_file.write("\n")
                            num_drone += 1
                            output_file.write(f"Drone {num_drone}:\n")
                        else:
                            if j > 0 and route[j] - route[j - 1] != 1:
                                output_file.write("\n")
                    else:
                        output_file.write("\n")
                elif route[j] == 0:
                    output_file.write("\n")
                    output_file.write("Drone 1:\n")
                else:
                    output_file.write(f"{route[j]} ")

            output_file.write("\n")
            # Raw CSV-style route output
            output_file.write(", ".join(map(str, route)) + "\n")
            output_file.write(f"{individual.fitness1} {individual.fitness2}\n\n")


def output_pareto_result(pareto_records: list, k: int):
    """Outputs ranked Pareto results based on specific criteria weights."""
    with open(params.output_pareto_selection_filename, "a") as output_file:
        limit = min(len(pareto_records), 5)

        if k != 0:
            output_file.write(
                f"Criteria {k}: {0.25 * k} fitness 1 and {0.25 * (3 - k)} fitness2\n"
            )
        else:
            output_file.write("General Ranking:\n")

        for i in range(limit):
            route = pareto_records[i].indi.route
            formatted_route = []
            for j in range(len(route)):
                if route[j] == 0:
                    formatted_route.append("||")
                elif route[j] > params.NUM_CUS:
                    if j > 0 and route[j - 1] == params.NUM_CUS:
                        formatted_route.append("|")
                    if j > 0 and j != len(route) - 1:
                        if (route[j] - route[j - 1] != 1) and (route[j - 1] != 0):
                            formatted_route.append("|")
                else:
                    formatted_route.append(f"{route[j]} ")

            output_file.write("".join(formatted_route) + "\n")
            output_file.write(
                f"{pareto_records[i].indi.fitness1} {pareto_records[i].indi.fitness2}\n"
            )
