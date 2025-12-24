import math
from typing import List

# --- General Simulation Parameters ---
CROSSOVER_MOD = 50
NUM_CUS = 20
NUM_TRUCKS = 2
NUM_DRONES = 2
POPULATION_SIZE = 200
MAX_GENERATIONS = 2000

# --- Service Times ---
DRONE_SERVICE_TIME = 30
TRUCK_SERVICE_TIME = 60

# --- Mutation & Crossover Rates ---
MUTATION_RATE = 0.05
CROSSOVER_RATE = 0.9

# --- Drone & Truck Physics/Constraints ---
V_MAX_TRUCK = 15.557
TAKEOFF_SPEED = 7.8232
CRUISE_SPEED = 15.6464
LANDING_SPEED = 3.9116
CRUISE_ALT = 50

# Calculated Constants (Equivalent to C++ #define macros)
TAKEOFF_TIME = CRUISE_ALT / TAKEOFF_SPEED
LANDING_TIME = CRUISE_ALT / LANDING_SPEED

CAPACITY_C = 2.27
BATTERY_POWER = 562990
BETA_B = 210.8
GAMA = 181.2
M_VALUE = 1e20

# --- File Paths & Strings ---
# PROBLEM = "20.20.2"
RESULT_SRC = "./result/"
input_file = ""

# Output Filenames (to be initialized during execution)
output_filename = ""
output_pareto_filename = ""
output_pareto_selection_filename = ""
output_log = ""
output_graph = ""
output_tblog = ""

# --- Global Data Structures ---
# In Python, we initialize these as empty lists
customers = []  # List[Customer]
M: List[List[float]] = []  # List[List[float]] - Distance Matrix

# --- Variables & Normalization ---
total_node = 0
can_used_drone = 0
drone_max_tracks = 0
max_tabu_iter = 0

max_obj1 = 1000000000.0
max_obj2 = 1000000000.0
min_obj1 = 0.0
min_obj2 = 0.0

obj1_norm = 0.0
obj2_norm = 0.0
time_limit = 0.0

# Initial equal probabilities for the three crossover operators
crossover_proportion = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
