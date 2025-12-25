# config/default_config.py - Updated with Algorithm Lists

# Problem names
PROBLEM_TYPES = {
    1: "PTDS-DDSS: Parallel Technician-Drone Delivery System",
    2: "MSSVTDE: Medical Sample Scheduling with Vehicle and Drone",
    3: "VRP-MRDR: Vehicle Routing Problem with Multiple Release Dates",
}

# Algorithms per problem - Exact available algorithms
ALGORITHMS = {
    # 1: ["Tabu Search", "Tabu Search Multi-Level"],
    1: ["Tabu Search Multi-Level"],
    # 2: ["Tabu Search", "MOEA/D", "NSGAII", "HNSGAII-TS"],
    2: ["HNSGAII-TS"],
    # 3: ["Adaptive Tabu Search"],
    3: ["Adaptive Tabu Search"],
}

# Default selected algorithms
DEFAULT_ALGORITHMS = {
    1: ["Multi-Level Tabu Search"],
    2: ["HNSGAII-TS"],
    3: ["Adaptive Tabu Search"],
}

# Problem 1: PTDS-DDSS
PROBLEM1_CONFIG = {
    "system": {
        "depot": {"x": 0.0, "y": 0.0},
        "num_technicians": 2,
        "num_drones": 2,
        "technician_speed": 0.58,
        "drone_speed": 0.85,
        "flight_endurance_limit": 120.0,
        "sample_waiting_limit": 60.0,
    },
    "algorithm": {
        "max_iteration": 500,
        "max_iteration_no_improve": 200,
        "alpha1": 1.0,  # penalty for flight endurance violation
        "alpha2": 1.0,  # penalty for waiting time violation
        "beta": 0.5,  # penalty factor
    },
}

# Problem 2: MSSVTDE
PROBLEM2_CONFIG = {
    "system": {
        "depot": {"x": 0.0, "y": 0.0},
        "num_technicians": 2,
        "num_drones": 2,
        "technician_baseline_speed": 15.557,
        "congestion_factor_min": 0.4,
        "congestion_factor_max": 0.9,
        "drone_takeoff_speed": 7.8232,
        "drone_cruise_speed": 15.6464,
        "drone_landing_speed": 3.9116,
        # "truck_capacity": 5,
        "drone_capacity": 2.27,
        "flight_endurance_limit": 562990,
        # "sample_waiting_limit": 60,
    },
    "algorithm": {
        "crossover_rate": 0.9,
        "mutation_rate": 0.05,
        "num_generations": 2000,
        "population_size": 200,
        "tabu_search_iterations": 50,
    },
}

# Problem 3: VRP-MRDR
PROBLEM3_CONFIG = {
    "system": {
        "depot": {"x": 0.0, "y": 0.0},
        "num_trucks": 3,
        "num_drones": 3,
        "truck_speed": 40,
        "drone_speed": 60,
        # "drone_capacity_options": [2, 4, 8],
        # "drone_capacity_default": 4,
        "drone_capacity": 8,
        "flight_endurance_limit": 90,
        # "sample_waiting_limit": 60,
    },
    "algorithm": {
        "gamma1": 0.5,  # score factor 1
        "gamma2": 0.3,  # score factor 2
        "gamma3": 0.1,  # score factor 3
        "gamma4": 0.3,  # score factor 4
        "eta": 2,  # variable maximum iteration
        "loop": 100,  # fixed maximum iteration
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
