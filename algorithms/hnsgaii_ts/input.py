import math
import os

import parameters as params
from structs import Customer


def check_customer(idx: int) -> bool:
    """
    Checks if a customer can be served by a drone based on demand weight,
    distance to depot, and battery constraints.
    """
    customer = params.customers[idx]
    total_weight = customer.demand

    # Check weight capacity (capacityC comes from parameters)
    if total_weight > params.CAPACITY_C:
        return False

    weight = 0.0
    total_energy = 0.0

    # Distance to depot (Depot is index 0)
    x_diff = params.customers[0].x - customer.x
    y_diff = params.customers[0].y - customer.y
    distance = math.sqrt(x_diff**2 + y_diff**2)

    # Drone travel logic (2 legs: to customer and back)
    for _ in range(2):
        # gama and betaB are battery discharge constants
        energy_per_sec = params.GAMA + (params.BETA_B * weight)
        travel_time = distance / params.CRUISE_SPEED

        # Energy consumption including takeoff and landing
        total_energy += (
            params.TAKEOFF_TIME + params.LANDING_TIME + travel_time
        ) * energy_per_sec

        if total_energy > params.BATTERY_POWER:
            return False

        # On the second leg, the drone is carrying the package weight
        weight = total_weight

    return True


def input_data():
    """
    Parses the input file to populate the customers list and
    calculate how many can be served by drones.
    """
    params.can_used_drone = 0

    # Initialize Depot
    depot = Customer(
        x=0.0,
        y=0.0,
        demand=0.0,
        only_by_truck=0,
        truck_service_time=0.0,
        drone_service_time=0.0,
    )
    params.customers.append(depot)

    if not os.path.exists(params.input_file):
        print(f"Error: Input file {params.input_file} not found.")
        return

    with open(params.input_file, "r") as f:
        # Skip the first 5 lines (header information)
        lines = f.readlines()
        data_lines = lines[5:]

        for i in range(params.NUM_CUS):
            if i >= len(data_lines):
                break

            line = data_lines[i].strip()
            if not line:
                continue

            values = list(map(float, line.split()))

            # Parsing values based on C++ vector 'a' mapping
            # a[0]:x, a[1]:y, a[2]:demand, a[3]:OnlyByTruck,
            # a[4]:TruckServiceTime, a[5]:DroneServiceTime
            new_cust = Customer(
                x=values[0],
                y=values[1],
                demand=values[2],
                only_by_truck=int(values[3]),
                truck_service_time=values[4],
                drone_service_time=values[5],
            )

            if new_cust.only_by_truck == 0:
                params.can_used_drone += 1

            params.customers.append(new_cust)
    for cust in params.customers:
        print(f"({cust.x}, {cust.y})")
    # Post-process: Verify if drone-eligible customers actually fit drone constraints
    print(f"No. Customer: {params.NUM_CUS}")
    for i in range(1, params.NUM_CUS + 1):
        if params.customers[i].only_by_truck == 0:
            if not check_customer(i):
                params.customers[i].only_by_truck = 1
                params.can_used_drone -= 1


def input_time_limit(instance_name: str):
    """
    Reads the time.txt file to set the global time limit
    specific to the instance being solved.
    """
    global timeLimit
    time_file_path = "./data/random_data/time.txt"

    if not os.path.exists(time_file_path):
        return

    with open(time_file_path, "r") as f:
        # Check first 60 lines as per C++ logic
        for _ in range(60):
            line = f.readline()
            if not line:
                break

            if instance_name in line:
                # Format expected: "instance_name;time"
                if ";" in line:
                    parts = line.split(";")
                    try:
                        timeLimit = float(parts[1].strip())
                    except (ValueError, IndexError):
                        pass
                break
