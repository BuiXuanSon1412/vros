from . import parameters as params
from algorithms.hnsgaii_ts.fitness import calculate_truck_time, calculate_drone_time
from algorithms.hnsgaii_ts.route_check import feasible_drone_route


def split_tracks(copy_tour: list):
    """
    Decomposes a full chromosome into separate tracks (truck routes and drone groups).
    Returns:
        - tracks: A list of lists, where each sub-list contains the customers for that track.
        - endpoints: Indices of the markers (0 or >NUM_CUS) in the original tour.
    """
    tracks = []
    current_track = []
    endpoints = [-1]

    for j in range(params.total_node):
        node = copy_tour[j]

        # Check if node is a marker (Depot 0 or Vehicle Delimiter)
        if node == 0 or node > params.NUM_CUS:
            # If it's a truck delimiter or the main depot marker
            if node == 0 or node < (params.NUM_CUS + params.NUM_TRUCKS):
                endpoints.append(j)
                tracks.append(current_track)
                current_track = []
            else:
                # Logic for drone track grouping based on drone_max_tracks
                marker_val = node - params.NUM_CUS - params.NUM_TRUCKS + 1
                if marker_val % params.drone_max_tracks == 0:
                    endpoints.append(j)
                    tracks.append(current_track)
                    current_track = []
                else:
                    # Delimiters that don't trigger a track split are kept inside the track
                    current_track.append(node)
        else:
            # Standard customer node
            current_track.append(node)

    tracks.append(current_track)
    endpoints.append(params.total_node)
    return tracks, endpoints


def track_result(track: list, track_num: int):
    """
    Calculates the (Makespan, WaitTime) for a specific track.
    If it's a drone track and becomes infeasible, returns (0, 0).
    """
    if track_num < params.NUM_TRUCKS:
        # Evaluate as a truck route
        return calculate_truck_time(track)
    else:
        # Evaluate as a drone track (potentially multiple sorties)
        track_obj1 = 0.0
        track_obj2 = 0.0
        drone_route = []

        for node in track:
            if node <= params.NUM_CUS:
                drone_route.append(node)
            else:
                # Sub-delimiter within the drone track
                if feasible_drone_route(drone_route) != 0:
                    return 0.0, 0.0

                if drone_route:
                    d_res1, d_res2 = calculate_drone_time(drone_route)
                    track_obj1 += d_res1
                    track_obj2 += d_res2
                drone_route = []

        # Final sortie check
        if feasible_drone_route(drone_route) != 0:
            return 0.0, 0.0

        if drone_route:
            d_res1, d_res2 = calculate_drone_time(drone_route)
            track_obj1 += d_res1
            track_obj2 += d_res2

        return track_obj1, track_obj2


def compute_linear(obj1: float, obj2: float) -> float:
    """
    Computes a weighted scalar value for multi-objective comparison
    using normalization constants from parameters.py.
    """
    # Uses global minobj and objnorm variables
    return (obj1 - params.min_obj1) / params.obj1_norm + (
        obj2 - params.min_obj2
    ) / params.obj2_norm
