from typing import List, Tuple

# This imports everything (params.V_MAX_TRUCK, M, NUM_CUS, etc.) directly into the namespace
import parameters as params

TRUCK_HOUR = [0.7, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0, 0.7, 0.6, 0.5, 0.7, 0.8]


def calculate_truck_time(truck_route: List[int]) -> Tuple[float, float]:
    """Calculates makespan and total wait time for a single truck route."""
    copy_route = [0] + truck_route + [0]
    time_needed = 0.0
    total_time_wait = 0.0
    time_wait = []

    for i in range(len(copy_route) - 1):
        # Accessing global distance matrix M and global params.V_MAX_TRUCK
        distance = params.M[copy_route[i]][copy_route[i + 1]]
        hour = int(time_needed / 3600)

        while True:
            # hour % 12 handles the 12-hour speed cycle
            speed_factor = TRUCK_HOUR[hour % 12]
            current_speed = params.V_MAX_TRUCK * speed_factor
            time_left_in_hour = (hour + 1) * 3600 - time_needed
            max_dist_this_hour = time_left_in_hour * current_speed

            if distance > max_dist_this_hour:
                distance -= max_dist_this_hour
                time_needed = (hour + 1) * 3600
                hour = int(time_needed / 3600)
            else:
                # Accessing global params.TRUCK_SERVICE_TIME
                time_needed += distance / current_speed + params.TRUCK_SERVICE_TIME
                break

        time_wait.append(time_needed)

    time_needed -= params.TRUCK_SERVICE_TIME
    for i in range(len(truck_route)):
        total_time_wait += time_needed - time_wait[i]

    return time_needed, total_time_wait


def calculate_drone_time(drone_route: List[int]) -> Tuple[float, float]:
    """Calculates makespan and total wait time for a single drone sortie."""
    copy_route = [0] + drone_route + [0]
    time_needed = 0.0
    time_wait = []

    for i in range(len(copy_route) - 1):
        distance = params.M[copy_route[i]][copy_route[i + 1]]
        travel_time = distance / params.CRUISE_SPEED

        # Accessing global physics constants
        time_needed += (
            params.TAKEOFF_TIME + params.LANDING_TIME + travel_time
        ) + params.DRONE_SERVICE_TIME
        time_wait.append(time_needed)

    time_needed -= params.DRONE_SERVICE_TIME
    total_time_wait = sum((time_needed - tw) for tw in time_wait[:-1])

    return time_needed, total_time_wait


def calculate_fitness(route: List[int]) -> Tuple[float, float]:
    """
    Decodes the chromosome into sub-routes and calculates fitness.
    Objective 1: Makespan (time_needed)
    Objective 2: Customer Dissatisfaction (total_time_wait)
    """
    time_needed = 0.0
    total_time_wait = 0.0

    try:
        pos_depot = route.index(0)
    except ValueError:
        return params.M_VALUE, params.M_VALUE

    # --- Section 1: Truck Routes (indices before the 0 marker) ---
    current_truck_route = []
    for i in range(pos_depot):
        if route[i] <= params.NUM_CUS:
            current_truck_route.append(route[i])
        else:
            if current_truck_route:
                t_need, t_wait = calculate_truck_time(current_truck_route)
                total_time_wait += t_wait
                time_needed = max(time_needed, t_need)
                current_truck_route = []

    if current_truck_route:
        t_need, t_wait = calculate_truck_time(current_truck_route)
        total_time_wait += t_wait
        time_needed = max(time_needed, t_need)

    # --- Section 2: Drone Routes (indices after the 0 marker) ---
    current_drone_route = []
    drone_accumulator_time = 0.0

    for i in range(pos_depot + 1, len(route)):
        if route[i] <= params.NUM_CUS:
            current_drone_route.append(route[i])
        else:
            if current_drone_route:
                d_need, d_wait = calculate_drone_time(current_drone_route)
                drone_accumulator_time += d_need
                total_time_wait += d_wait

            # Use globals for vehicle counts and track constraints
            t_val = route[i] - params.NUM_CUS - params.NUM_TRUCKS + 1
            if t_val > 0 and t_val % params.drone_max_tracks == 0:
                time_needed = max(time_needed, drone_accumulator_time)
                drone_accumulator_time = 0.0

            current_drone_route = []

        if i == len(route) - 1:
            if current_drone_route:
                d_need, d_wait = calculate_drone_time(current_drone_route)
                drone_accumulator_time += d_need
                total_time_wait += d_wait
            time_needed = max(time_needed, drone_accumulator_time)

    return time_needed, total_time_wait
