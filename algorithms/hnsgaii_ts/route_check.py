from . import parameters as params


def feasible_route(route: list) -> bool:
    """Basic bounds check to ensure all node indices are valid."""
    for node in route:
        if node >= params.total_node or node < 0:
            return False
    return True


def feasible_drone_route(drone_route: list) -> int:
    """
    Checks physical constraints for a single drone sortie.
    Returns:
        0: Feasible
        1: Contains a customer restricted to trucks only
        2: Exceeds drone weight capacity
        3: Exceeds battery/energy capacity
    """
    if not drone_route:
        return 0

    # Sortie starts and ends at the depot (index 0)
    copy_route = [0] + drone_route + [0]

    total_weight = 0.0
    current_weight = 0.0
    total_energy = 0.0

    # 1. Check if any customer is truck-only
    for node in drone_route:
        if params.customers[node].only_by_truck == 1:
            return 1

    # 2. Check total weight capacity
    for node in drone_route:
        total_weight += params.customers[node].demand

    if total_weight > params.CAPACITY_C:
        return 2

    # 3. Check Energy Consumption
    # Note: Weight decreases as drone drops off items?
    # Logic follows C++: weight starts at 0 and adds demand of next node
    for i in range(len(copy_route) - 1):
        energy_per_sec = params.GAMA + (params.BETA_B * current_weight)
        distance = params.M[copy_route[i]][copy_route[i + 1]]
        time = distance / params.CRUISE_SPEED

        total_energy += (
            params.TAKEOFF_TIME + params.LANDING_TIME + time
        ) * energy_per_sec

        if total_energy > params.BATTERY_POWER:
            return 3

        # Accumulate weight for the next leg
        current_weight += params.customers[copy_route[i + 1]].demand

    return 0


def check_route(temp_route: list) -> bool:
    """
    Main validator for the entire chromosome.
    Extracts drone segments (after the depot marker '0') and validates each.
    """
    try:
        # The first '0' splits truck section from drone section
        pos_depot = temp_route.index(0)
    except ValueError:
        return False  # A valid chromosome must contain a depot marker

    drone_segment = []

    # Iterate through the drone portion of the route
    for i in range(pos_depot + 1, len(temp_route)):
        # Values > num_cus are markers for a new drone sortie or track
        if temp_route[i] > params.NUM_CUS:
            if drone_segment:
                if feasible_drone_route(drone_segment) > 0:
                    return False
            drone_segment = []
        else:
            drone_segment.append(temp_route[i])

    # Check the final segment
    if drone_segment:
        if feasible_drone_route(drone_segment) > 0:
            return False

    return True
