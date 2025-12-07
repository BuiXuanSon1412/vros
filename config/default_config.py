# config/default_config.py - Updated with Algorithm Lists

# Problem names
PROBLEM_TYPES = {
    1: "PTDS-DDSS: Parallel Technician-Drone Delivery System",
    2: "MSSVTDE: Medical Sample Scheduling with Vehicle and Drone",
    3: "VRP-MRDR: Vehicle Routing Problem with Multiple Release Dates",
}

# Algorithms per problem - Exact available algorithms
ALGORITHMS = {
    1: ["Tabu Search", "Tabu Search Multi-Level"],
    2: ["Tabu Search", "MOEA/D", "NSGAII", "HNSGAII-TS"],
    3: ["Adaptive Tabu Search"],
}

# Default selected algorithms
DEFAULT_ALGORITHMS = {
    1: ["Tabu Search"],
    2: ["HNSGAII-TS"],
    3: ["Adaptive Tabu Search"],
}

# Problem 1: PTDS-DDSS
PROBLEM1_CONFIG = {
    "system": {
        "depot": {"x": 0.0, "y": 0.0},
        "num_technicians": 2,
        "num_drones": 2,
        "technician_speed": 35,  # km/h
        "drone_speed": 60,  # km/h
        "flight_endurance_limit": 3600,  # seconds
        "sample_waiting_limit": 60,  # minutes
    },
    "algorithm": {
        "max_iteration": 1000,
        "max_iteration_no_improve": 100,
        "alpha1": 1.0,  # penalty for flight endurance violation
        "alpha2": 1.0,  # penalty for waiting time violation
        "beta": 1.5,  # penalty factor
    },
}

# Problem 2: MSSVTDE
PROBLEM2_CONFIG = {
    "system": {
        "depot": {"x": 0.0, "y": 0.0},
        "num_technicians": 2,
        "num_drones": 2,
        "technician_baseline_speed": 35,  # km/h
        "congestion_factor_min": 0.4,
        "congestion_factor_max": 0.9,
        "drone_takeoff_speed": 20,  # km/h
        "drone_cruise_speed": 60,  # km/h
        "drone_landing_speed": 20,  # km/h
        "truck_capacity": 100,  # kg
        "drone_capacity": 5,  # kg
        "flight_endurance_limit": 3600,  # seconds
        "sample_waiting_limit": 60,  # minutes
    },
    "algorithm": {
        "crossover_rate": 0.9,
        "mutation_rate": 0.05,
        "num_generations": 200,
        "population_size": 100,
        "tabu_search_iterations": 50,
    },
}

# Problem 3: VRP-MRDR
PROBLEM3_CONFIG = {
    "system": {
        "depot": {"x": 0.0, "y": 0.0},
        "num_trucks": 3,
        "num_drones": 3,
        "truck_speed": 40,  # km/h
        "drone_speed": 60,  # km/h
        "drone_capacity_options": [2, 4, 8],  # kg
        "drone_capacity_default": 4,
        "flight_endurance_limit": 90,  # minutes
        "sample_waiting_limit": 60,  # minutes
    },
    "algorithm": {
        "gamma1": 1.0,  # score factor 1
        "gamma2": 1.0,  # score factor 2
        "gamma3": 1.0,  # score factor 3
        "gamma4": 1.0,  # score factor 4
        "eta": 100,  # variable maximum iteration
        "loop": 1000,  # fixed maximum iteration
        "seg": 5,  # number of segments
    },
    "release_factors": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
}

# Colors for visualization
COLORS = {
    "truck": ["#45B7D1", "#FF6B6B", "#4ECDC4"],
    "drone": ["#FFA07A", "#98D8C8", "#6C5CE7"],
    "depot": "#2D3436",
    "customer": "#0984E3",
}
