# utils/file_parser.py

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import streamlit as st

from config.default_config import PROBLEM2_CONFIG, PROBLEM3_CONFIG


class FileParser:
    """Parse different file formats for each problem type"""

    @staticmethod
    def parse_problem1_file(file_content: str) -> Tuple[pd.DataFrame, Dict]:
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
                }
            )

        customers_df = pd.DataFrame(customers)

        # Create depot at origin
        depot = {"id": 0, "x": 0.0, "y": 0.0, "name": "Depot"}

        # Calculate distance matrix

        return customers_df, depot

    @staticmethod
    def parse_problem2_file(
        file_content: str,
    ) -> Tuple[pd.DataFrame, Dict, Dict]:
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
                }
            )

        customers_df = pd.DataFrame(customers)

        # Create depot at origin
        depot = {"id": 0, "x": 0.0, "y": 0.0, "name": "Depot"}

        # Vehicle config
        vehicle_config = {
            "num_staff": num_staff,
            "num_drone": num_drone,
            "drone_flight_time": drone_flight_time,
        }

        return customers_df, depot, vehicle_config

    @staticmethod
    def parse_problem3_file(
        file_content: str,
    ) -> Tuple[pd.DataFrame, Dict, Dict]:
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

        file_version = st.session_state.get("file_version_3", 0)
        vehicle_config["num_trucks"] = int(
            st.session_state.get(
                f"p3_num_trucks_{file_version}", PROBLEM3_CONFIG["system"]["num_trucks"]
            )
        )
        vehicle_config["num_drones"] = int(
            st.session_state.get(
                f"p3_num_drones_{file_version}", PROBLEM3_CONFIG["system"]["num_drones"]
            )
        )

        vehicle_config["truck_speed"] = float(
            st.session_state.get(
                f"p3_truck_speed_{file_version}",
                PROBLEM3_CONFIG["system"]["truck_speed"],
            )
        )
        vehicle_config["drone_speed"] = float(
            st.session_state.get(
                f"p3_drone_speed_{file_version}",
                PROBLEM3_CONFIG["system"]["drone_speed"],
            )
        )

        vehicle_config["drone_capacity"] = float(
            st.session_state.get(
                f"p3_drone_capacity_{file_version}",
                PROBLEM3_CONFIG["system"]["drone_capacity"],
            )
        )

        vehicle_config["flight_endurance_limit"] = float(
            st.session_state.get(
                f"p3_flight_endurance_{file_version}",
                PROBLEM3_CONFIG["system"]["flight_endurance_limit"],
            )
        )

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
                        "demand": float(demand),
                        "release_date": float(release),
                    }
                else:  # Customer data
                    customers.append(
                        {
                            "id": customer_id,
                            "x": x,
                            "y": y,
                            "demand": float(demand),  # Keep as integer
                            "release_date": float(release),  # Keep as integer
                        }
                    )
                    customer_id += 1
            except (ValueError, IndexError):
                continue

        # Ensure we have customers
        if len(customers) == 0:
            raise ValueError("No customer data found in file")

        if depot is None:
            depot = {"id": 0, "x": 0.0, "y": 0.0, "demand": 0, "release_date": 0}

        customers_df = pd.DataFrame(customers)

        return customers_df, depot, vehicle_config
