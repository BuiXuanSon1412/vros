import math

# Assuming params.customers is a list of Customer objects from the previous step
# params.customers: List[Customer] = []
# M: List[List[float]] = []
from . import parameters as params


def calculate_distance(i: int, j: int) -> float:
    """Calculates Euclidean distance between two customer nodes."""
    x_diff = params.customers[i].x - params.customers[j].x
    y_diff = params.customers[i].y - params.customers[j].y
    return math.sqrt(x_diff**2 + y_diff**2)


def init_matrix():
    """
    Initializes a distance matrix M based on customer coordinates.
    Logic mirrors the provided C++ structure.
    """
    # Initialize M as an empty 2D list (matrix)
    params.M = [[] for _ in range(params.total_node)]

    # First section: Distance from params.customers (0 to params.NUM_CUS) to all other nodes
    for i in range(params.NUM_CUS + 1):
        for j in range(params.total_node):
            if j > params.NUM_CUS:
                # If target is beyond customer count, calculate distance to depot (index 0)
                params.M[i].append(calculate_distance(i, 0))
            else:
                params.M[i].append(calculate_distance(i, j))

    # Second section: Distance from virtual/extra nodes to all other nodes
    for i in range(params.NUM_CUS + 1, params.total_node):
        for j in range(params.total_node):
            if j > params.NUM_CUS:
                # Distance between two virtual nodes is 0
                params.M[i].append(0.0)
            else:
                # Distance from virtual node to customer is distance from depot (0) to customer
                params.M[i].append(calculate_distance(0, j))
