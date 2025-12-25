# utils/file_parser.py

import pandas as pd
import numpy as np
from typing import Dict, Tuple


class FileParser:
    """Parse different file formats for each problem type"""

    @staticmethod
    def parse_problem1_file(file_content: str) -> Tuple[pd.DataFrame, Dict, np.ndarray]:
        """
        Parse Problem 1 file format (6.5.1.txt)
        Format:
        Customers N
        Coordinate X    Coordinate Y    Demand
        x1  y1  demand1
        ...
        """
        lines = file_content.strip().split("\n")

        # Parse header
        num_customers = int(lines[0].split()[1])

        # Parse customer data (skip header line)
        customers = []
        for i in range(2, 2 + num_customers):
            parts = lines[i].split()
            customers.append(
                {
                    "id": i - 1,
                    "x": float(parts[0]),
                    "y": float(parts[1]),
                    "demand": float(parts[2]),
                    "service_time": 10,  # Default value
                    "priority": 1,  # Default value
                    "time_window_start": 0,
                    "time_window_end": 480,
                }
            )

        customers_df = pd.DataFrame(customers)

        # Create depot at origin
        depot = {"id": 0, "x": 0.0, "y": 0.0, "name": "Depot"}

        # Calculate distance matrix
        distance_matrix = FileParser._calculate_distance_matrix(customers_df, depot)

        return customers_df, depot, distance_matrix

    @staticmethod
    def parse_problem2_file(
        file_content: str,
    ) -> Tuple[pd.DataFrame, Dict, np.ndarray, Dict]:
        """
        Parse Problem 2 file format (10.10.1.txt)
        Format:
        number_staff N
        number_drone M
        droneLimitationFightTime(s) T
        Customers K
        Coordinate X    Coordinate Y    Demand  OnlyServicedByStaff ServiceTimeByTruck(s)   ServiceTimeByDrone(s)
        """
        lines = file_content.strip().split("\n")

        # Parse vehicle configuration
        num_staff = int(lines[0].split()[1])
        num_drone = int(lines[1].split()[1])
        drone_flight_time = int(lines[2].split()[1])

        # Parse number of customers
        num_customers = int(lines[3].split()[1])

        # Parse customer data (skip header line)
        customers = []
        for i in range(5, 5 + num_customers):
            parts = lines[i].split()
            customers.append(
                {
                    "id": i - 4,
                    "x": float(parts[0]),
                    "y": float(parts[1]),
                    "demand": float(parts[2]),
                    "only_staff": int(parts[3]),
                    "service_time_truck": int(parts[4]),
                    "service_time_drone": int(parts[5]),
                    "service_time": int(parts[4]),  # Default to truck
                    "priority": 1,
                    "time_window_start": 0,
                    "time_window_end": 480,
                }
            )

        customers_df = pd.DataFrame(customers)

        # Create depot at origin
        depot = {"id": 0, "x": 0.0, "y": 0.0, "name": "Depot"}

        # Calculate distance matrix
        distance_matrix = FileParser._calculate_distance_matrix(customers_df, depot)

        # Vehicle config
        vehicle_config = {
            "num_staff": num_staff,
            "num_drone": num_drone,
            "drone_flight_time": drone_flight_time,
        }

        return customers_df, depot, distance_matrix, vehicle_config

    @staticmethod
    def parse_problem3_file(
        file_content: str,
    ) -> Tuple[pd.DataFrame, Dict, np.ndarray, Dict]:
        """
        Parse Problem 3 file format (10.1.txt)
        Format:
        number_truck    N
        number_drone    M
        truck_speed     S2
        drone_speed     S1
        M_d     capacity
        L_d     flight_time
        Sigma   service_time
        XCOORD  YCOORD  DEMAND  RELEASE_DATE
        40  50  0   0  (depot)
        x1  y1  demand1  release1
        ...drone_capacity = st.text_input(
                "Drone capacity",
                value=drone_capacity_value,
                # disabled=True,
                key=f"p3_drone_capacity_{file_version}",  # Dynamic key
            )
        """
        lines = file_content.strip().split("\n")

        # Find where customer data starts (after seeing XCOORD header)
        data_start = 0
        for i, line in enumerate(lines):
            if "XCOORD" in line.upper() and "YCOORD" in line.upper():
                data_start = i + 1  # Data starts on next line
                break

        # Parse vehicle configuration (lines before data_start)
        vehicle_config = {}
        for i in range(data_start):
            line = lines[i].strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) >= 2:
                key = parts[0]
                try:
                    value = float(parts[1]) if "." in parts[1] else int(parts[1])
                    vehicle_config[key] = value
                except ValueError:
                    continue

        # Parse customer data (lines after data_start)
        customers = []
        depot = None
        customer_id = 1

        for i in range(data_start, len(lines)):
            line = lines[i].strip()
            if not line:
                continue

            parts = line.split()

            # Need at least 4 numeric values
            if len(parts) < 4:
                continue

            try:
                x = float(parts[0])
                y = float(parts[1])
                demand = float(parts[2])
                release = float(parts[3])

                if depot is None:  # First data line is depot
                    depot = {
                        "id": 0,
                        "x": x,
                        "y": y,
                        "name": "Depot",
                    }
                else:  # Customer data
                    customers.append(
                        {
                            "id": customer_id,
                            "x": x,
                            "y": y,
                            "demand": int(demand),  # Keep as integer
                            "release_date": int(release),  # Keep as integer
                            "service_time": vehicle_config.get("Sigma", 5),
                            "priority": 1,
                            "time_window_start": 0,
                            "time_window_end": 480,
                        }
                    )
                    customer_id += 1
            except (ValueError, IndexError):
                continue

        # Ensure we have customers
        if len(customers) == 0:
            raise ValueError("No customer data found in file")

        if depot is None:
            depot = {"id": 0, "x": 0.0, "y": 0.0, "name": "Depot"}

        customers_df = pd.DataFrame(customers)

        # Calculate distance matrix
        distance_matrix = FileParser._calculate_distance_matrix(customers_df, depot)

        return customers_df, depot, distance_matrix, vehicle_config

    @staticmethod
    def _calculate_distance_matrix(
        customers_df: pd.DataFrame, depot: Dict
    ) -> np.ndarray:
        """Calculate Euclidean distance matrix"""
        # Add depot at the beginning
        all_locations = pd.concat(
            [pd.DataFrame([depot]), customers_df[["x", "y"]]], ignore_index=True
        )

        n = len(all_locations)
        distance_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = all_locations.iloc[i]["x"] - all_locations.iloc[j]["x"]
                    dy = all_locations.iloc[i]["y"] - all_locations.iloc[j]["y"]
                    distance_matrix[i][j] = np.sqrt(dx**2 + dy**2)

        return distance_matrix
