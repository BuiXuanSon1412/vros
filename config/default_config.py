# config/default_config.py

# Default configuration for the system

PROBLEM_TYPES = {
    1: "Problem 1: Min-timespan parallel technician-and-drone (TS/TS-Multilevel)",
    2: "Problem 2: Bi-objective Medical Sampling (NSGA-II+TS/MOEA-D)",
    3: "Problem 3: Resupply with release date (TS)",
}

ALGORITHMS = {
    1: ["Tabu Search", "Tabu Search Multilevel"],
    2: ["NSGA-II + TS", "MOEA/D"],
    3: ["Tabu Search"],
}

# Default vehicle configuration
DEFAULT_VEHICLE_CONFIG = {
    "truck": {
        "count": 2,
        "capacity": 100,  # kg
        "speed": 40,  # km/h
        "cost_per_km": 5000,  # VND
        "energy_limit": None,
    },
    "drone": {
        "count": 3,
        "capacity": 5,  # kg
        "speed": 60,  # km/h
        "cost_per_km": 2000,  # VND
        "energy_limit": 30,  # minutes
    },
}

# Default customer configuration
DEFAULT_CUSTOMER_CONFIG = {
    "num_customers": 20,
    "service_time_range": (5, 15),  # minutes
    "demand_range": (1, 10),  # kg
    "priority_levels": 3,
    "area_size": 50,  # km x km
}

# Default algorithm configuration
DEFAULT_ALGORITHM_CONFIG = {
    "tabu_search": {"max_iterations": 1000, "tabu_tenure": 10, "neighborhood_size": 20},
    "nsga2": {
        "population_size": 100,
        "generations": 200,
        "crossover_prob": 0.9,
        "mutation_prob": 0.1,
    },
    "moead": {"population_size": 100, "generations": 200, "neighborhood_size": 20},
}

# Colors for visualization
COLORS = {
    "truck": ["#45B7D1", "#FF6B6B", "#4ECDC4"],
    "drone": ["#FFA07A", "#98D8C8", "#6C5CE7"],
    "depot": "#2D3436",
    "customer": "#0984E3",
}
