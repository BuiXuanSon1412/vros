import parameters as params


def calculate_hash(route: list) -> int:
    """
    Decodes the route into truck and drone structures and generates
     a unique hash based on the resulting sub-routes.
    """
    try:
        pos_depot = route.index(0)
    except ValueError:
        return hash(tuple(route))  # Fallback for invalid routes

    # --- Group Truck Routes ---
    all_truck_routes = []
    current_truck_route = []
    for i in range(pos_depot):
        if route[i] <= params.NUM_CUS:
            current_truck_route.append(route[i])
        else:
            if current_truck_route:
                # v2Hashing logic: add depot (0) to start and end
                all_truck_routes.append(tuple([0] + current_truck_route + [0]))
                current_truck_route = []

    if current_truck_route:
        all_truck_routes.append(tuple([0] + current_truck_route + [0]))

    # --- Group Drone Routes ---
    all_drone_routes = []  # This will be a list of lists of tuples
    single_drone_track = []
    current_drone_route = []

    for i in range(pos_depot + 1, len(route)):
        if route[i] <= params.NUM_CUS:
            current_drone_route.append(route[i])
        else:
            if current_drone_route:
                # add depot (0) to start and end
                single_drone_track.append(tuple([0] + current_drone_route + [0]))

            # Drone track logic
            t_val = route[i] - params.NUM_CUS - params.NUM_TRUCKS + 1
            if t_val > 0 and t_val % params.drone_max_tracks == 0:
                all_drone_routes.append(tuple(single_drone_track))
                single_drone_track = []

            current_drone_route = []

        if i == len(route) - 1:
            if current_drone_route:
                single_drone_track.append(tuple([0] + current_drone_route + [0]))
            if single_drone_track:
                all_drone_routes.append(tuple(single_drone_track))

    # Python's built-in hash() handles nested tuples by combining
    # their internal hashes exactly like the C++ manual bit-shifting code.
    final_truck_hash = hash(tuple(all_truck_routes))
    final_drone_hash = hash(tuple(all_drone_routes))

    return hash((final_truck_hash, final_drone_hash))
