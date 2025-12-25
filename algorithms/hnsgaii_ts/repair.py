from algorithms.hnsgaii_ts.route_check import feasible_drone_route
from algorithms.hnsgaii_ts.fitness import calculate_truck_time
from . import parameters as params


def fix_route(drone_route: list) -> list:
    """
    Identifies customers causing drone infeasibility by incrementally
    building the route and catching the point of failure.
    """
    temp = []
    error_customers = []
    for customer in drone_route:
        temp.append(customer)
        if feasible_drone_route(temp) != 0:
            temp.pop()
            error_customers.append(customer)
    return error_customers


def repair_route(route: list) -> list:
    """
    Main repair logic:
    1. Extracts drone segments.
    2. Uses fix_route to remove infeasible customers.
    3. Re-inserts those customers into the truck portion of the route
       at the position that minimizes (Makespan + Wait Time).
    """
    new_route = []
    try:
        depot_pos = route.index(0)
    except ValueError:
        return route  # Should not happen in a valid chromosome

    # Keep the truck portion (up to the depot marker)
    for i in range(depot_pos + 1):
        new_route.append(route[i])

    drone_route_segment = []
    all_errors = []

    # Identify and remove infeasible drone customers
    for i in range(depot_pos + 1, len(route)):
        if route[i] > params.NUM_CUS:  # Marker for new sortie/track
            if drone_route_segment:
                error_list = fix_route(drone_route_segment)
                # Remove errors from current segment
                drone_route_segment = [
                    c for c in drone_route_segment if c not in error_list
                ]
                all_errors.extend(error_list)
                new_route.extend(drone_route_segment)
            new_route.append(route[i])
            drone_route_segment = []
        else:
            drone_route_segment.append(route[i])

    # Final segment check
    if drone_route_segment:
        error_list = fix_route(drone_route_segment)
        drone_route_segment = [c for c in drone_route_segment if c not in error_list]
        all_errors.extend(error_list)
        new_route.extend(drone_route_segment)

    # RE-INSERTION: Place errors into the truck routes greedily
    for er_cust in all_errors:
        best_pos = 0
        min_cost = float("inf")

        # Get current truck part (elements before the 0 marker)
        truck_part = new_route[:depot_pos]

        for j in range(len(truck_part) + 1):
            temp_truck_part = list(truck_part)
            temp_truck_part.insert(j, er_cust)

            # Evaluate cost of this truck route insertion
            fit1, fit2 = 0.0, 0.0
            sub_route = []
            for val in temp_truck_part:
                if val <= params.NUM_CUS:
                    sub_route.append(val)
                else:
                    if sub_route:
                        t_need, t_wait = calculate_truck_time(sub_route)
                        fit2 += t_wait
                        fit1 = max(fit1, t_need)
                        sub_route = []

            if sub_route:
                t_need, t_wait = calculate_truck_time(sub_route)
                fit2 += t_wait
                fit1 = max(fit1, t_need)

            current_total = fit1 + fit2
            if current_total < min_cost:
                min_cost = current_total
                best_pos = j

        # Insert the customer at the best found position in the actual route
        new_route.insert(best_pos, er_cust)
        depot_pos += 1  # Depot shifts right because we inserted a customer before it

    return new_route


def repair_position(route: list) -> list:
    """
    Ensures markers (0 and values > NUM_CUS) are in their
    designated logical positions within the chromosome.
    """
    trucks_done = 1
    depot_found = 0
    drones_done = 0

    for i in range(len(route)):
        if route[i] == 0 or route[i] > params.NUM_CUS:
            # If we've already placed all truck markers
            if trucks_done == params.NUM_TRUCKS:
                if depot_found == 0:
                    # Place the depot (0) marker
                    try:
                        pos = route.index(0)
                        route[i], route[pos] = route[pos], route[i]
                        depot_found += 1
                    except ValueError:
                        pass
                else:
                    # Place drone track markers
                    marker = params.NUM_CUS + params.NUM_TRUCKS + drones_done
                    try:
                        pos = route.index(marker)
                        route[i], route[pos] = route[pos], route[i]
                        drones_done += 1
                    except ValueError:
                        pass

            # Still placing truck route markers
            elif trucks_done < params.NUM_TRUCKS:
                marker = params.NUM_CUS + trucks_done
                try:
                    pos = route.index(marker)
                    route[i], route[pos] = route[pos], route[i]
                    trucks_done += 1
                except ValueError:
                    pass

    return route
