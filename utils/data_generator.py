# utils/data_generator.py

import numpy as np
import pandas as pd
from typing import Dict, Tuple


class DataGenerator:
    """Generate sample data for routing problems"""

    def __init__(self, seed=42):
        np.random.seed(seed)

    def generate_customers(
        self,
        num_customers: int,
        area_size: float = 50,
        demand_range: Tuple[int, int] = (1, 10),
        service_time_range: Tuple[int, int] = (5, 15),
        priority_levels: int = 3,
    ) -> pd.DataFrame:
        """
        Generate random customer data (legacy method)
        """
        customers = []

        for i in range(num_customers):
            customer = {
                "id": i + 1,
                "x": np.random.uniform(0, area_size),
                "y": np.random.uniform(0, area_size),
                "demand": np.random.randint(demand_range[0], demand_range[1] + 1),
                "service_time": np.random.randint(
                    service_time_range[0], service_time_range[1] + 1
                ),
                "priority": np.random.randint(1, priority_levels + 1),
                "time_window_start": np.random.randint(0, 480),
            }
            customer["time_window_end"] = customer[
                "time_window_start"
            ] + np.random.randint(60, 240)
            customers.append(customer)

        return pd.DataFrame(customers)

    def generate_customers_custom(
        self, problem_type: int, params: Dict
    ) -> pd.DataFrame:
        """
        Generate custom customer data based on problem type and parameters

        Args:
            problem_type: 1, 2, or 3
            params: Dictionary of generation parameters

        Returns:
            DataFrame containing customer information
        """
        num_customers = params.get("num_customers", 20)
        coord_range = params.get("coord_range", (-100, 100))
        demand_range = params.get("demand_range", (0.02, 0.1))

        customers = []

        for i in range(num_customers):
            customer = {
                "id": i + 1,
                "x": np.random.uniform(coord_range[0], coord_range[1]),
                "y": np.random.uniform(coord_range[0], coord_range[1]),
                "demand": np.random.uniform(demand_range[0], demand_range[1]),
            }

            if problem_type in [1, 2]:
                # Problem 1 & 2: Staff/Drone system
                ratio_staff_only = params.get("ratio_staff_only", 0.5)
                service_time_truck = params.get("service_time_truck", 60)
                service_time_drone = params.get("service_time_drone", 30)

                # Determine if customer can only be served by staff
                only_staff = np.random.random() < ratio_staff_only

                customer.update(
                    {
                        "only_staff": 1 if only_staff else 0,
                        "service_time_truck": service_time_truck,
                        "service_time_drone": service_time_drone,
                        "service_time": service_time_truck,  # Default
                        "priority": 1,
                        "time_window_start": 0,
                        "time_window_end": 480,
                    }
                )

            elif problem_type == 3:
                # Problem 3: Release dates
                release_range = params.get("release_range", (0, 20))

                customer.update(
                    {
                        "release_date": np.random.randint(
                            release_range[0], release_range[1] + 1
                        ),
                        "service_time": 5,  # Default service time
                        "priority": 1,
                        "time_window_start": 0,
                        "time_window_end": 480,
                    }
                )

            customers.append(customer)

        return pd.DataFrame(customers)

    def generate_depot(self, area_size: float = 50) -> Dict:
        """Generate depot information - centered in area"""
        # If area_size is the range (e.g., 200 for -100 to 100), center at 0
        if area_size > 100:
            return {"id": 0, "x": 0.0, "y": 0.0, "name": "Depot"}
        else:
            return {"id": 0, "x": area_size / 2, "y": area_size / 2, "name": "Depot"}

    def calculate_distance_matrix(
        self, locations: pd.DataFrame, depot: Dict
    ) -> np.ndarray:
        """
        Calculate Euclidean distance matrix
        """
        # Add depot at the beginning
        all_locations = pd.concat(
            [pd.DataFrame([depot]), locations[["x", "y"]]], ignore_index=True
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

    def export_to_file(
        self,
        customers: pd.DataFrame,
        depot: Dict,
        problem_type: int,
        params: Dict,
        filename: str,
    ):
        """Export generated data to file in appropriate format"""
        with open(filename, "w") as f:
            if problem_type == 1:
                # Format: 6.5.1.txt
                f.write(f"Customers {len(customers)}\n")
                f.write("Coordinate X\t\tCoordinate Y\t\tDemand\n")
                for _, row in customers.iterrows():
                    f.write(f"{row['x']}\t{row['y']}\t{row['demand']}\n")

            elif problem_type == 2:
                # Format: 10.10.1.txt
                f.write(f"number_staff {params.get('num_staffs', 2)}\n")
                f.write(f"number_drone {params.get('num_drones', 3)}\n")
                f.write(
                    f"droneLimitationFightTime(s) {params.get('drone_flight_time', 3600)}\n"
                )
                f.write(f"Customers {len(customers)}\n")
                f.write(
                    "Coordinate X\t\tCoordinate Y\t\tDemand\t\tOnlyServicedByStaff\t\t"
                    "ServiceTimeByTruck(s)\t\tServiceTimeByDrone(s)\n"
                )
                for _, row in customers.iterrows():
                    f.write(
                        f"{row['x']}\t{row['y']}\t{row['demand']}\t{row['only_staff']}\t"
                        f"{row['service_time_truck']}\t{row['service_time_drone']}\n"
                    )

            elif problem_type == 3:
                # Format: 10.1.txt
                f.write("XCOORD\tYCOORD\tDEMAND\tRELEASE_DATE\n")
                f.write(f"{depot['x']}\t{depot['y']}\t0\t0\n")  # Depot first
                for _, row in customers.iterrows():
                    f.write(
                        f"{row['x']}\t{row['y']}\t{row['demand']}\t{row['release_date']}\n"
                    )

                f.write(f"number_truck\t{params.get('num_trucks', 3)}\n")
                f.write(f"number_drone\t{params.get('num_drones', 3)}\n")
                f.write(f"drone_speed\t{params.get('drone_velocity', 1)}\n")
                f.write(f"truck_speed\t{params.get('truck_velocity', 0.5)}\n")
                f.write(f"M_d\t{params.get('drone_capacity', 4)}\n")
                f.write(f"L_d\t{params.get('drone_flight_time', 90)}\n")
                f.write("Sigma\t5\n")

        print(f"Data exported to file: {filename}")


# Test function
if __name__ == "__main__":
    generator = DataGenerator()

    # Test Problem 2 generation
    params = {
        "num_customers": 10,
        "coord_range": (-100, 100),
        "demand_range": (0.02, 0.1),
        "num_staffs": 2,
        "num_drones": 2,
        "drone_flight_time": 3600,
        "ratio_staff_only": 0.5,
        "service_time_truck": 60,
        "service_time_drone": 30,
    }

    customers = generator.generate_customers_custom(2, params)
    depot = generator.generate_depot(200)

    print("Problem 2 Customers:")
    print(customers.head())
    print(f"\nDepot: {depot}")
