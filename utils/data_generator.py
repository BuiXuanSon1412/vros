# utils/data_generator.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


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
        Generate random customer data

        Args:
            num_customers: Number of customers
            area_size: Area size (km x km)
            demand_range: Demand range (kg)
            service_time_range: Service time range (minutes)
            priority_levels: Number of priority levels

        Returns:
            DataFrame containing customer information
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
                "time_window_start": np.random.randint(0, 480),  # 8 hours = 480 minutes
            }
            # Time window end = start + random duration
            customer["time_window_end"] = customer[
                "time_window_start"
            ] + np.random.randint(60, 240)
            customers.append(customer)

        return pd.DataFrame(customers)

    def generate_depot(self, area_size: float = 50) -> Dict:
        """Generate depot information"""
        return {"id": 0, "x": area_size / 2, "y": area_size / 2, "name": "Depot"}

    def calculate_distance_matrix(
        self, locations: pd.DataFrame, depot: Dict
    ) -> np.ndarray:
        """
        Calculate Euclidean distance matrix

        Args:
            locations: DataFrame containing location coordinates
            depot: Dict containing depot information

        Returns:
            Distance matrix (n+1) x (n+1), with depot as point 0
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

    def generate_release_dates(
        self, num_customers: int, max_time: int = 480
    ) -> List[int]:
        """
        Generate release dates for problem 3 (packages appear at different times)

        Args:
            num_customers: Number of customers
            max_time: Maximum time (minutes)

        Returns:
            List of release times
        """
        return sorted(np.random.randint(0, max_time, num_customers).tolist())

    def export_to_file(
        self,
        customers: pd.DataFrame,
        depot: Dict,
        distance_matrix: np.ndarray,
        filename: str,
    ):
        """Export data to file"""
        with open(filename, "w") as f:
            f.write(f"DEPOT {depot['id']} {depot['x']:.2f} {depot['y']:.2f}\n")
            f.write(f"CUSTOMERS {len(customers)}\n")
            for _, row in customers.iterrows():
                f.write(
                    f"{row['id']} {row['x']:.2f} {row['y']:.2f} "
                    f"{row['demand']} {row['service_time']} "
                    f"{row['priority']} {row['time_window_start']} {row['time_window_end']}\n"
                )
            f.write("DISTANCE_MATRIX\n")
            for row in distance_matrix:
                f.write(" ".join([f"{d:.2f}" for d in row]) + "\n")

        print(f"Data exported to file: {filename}")


# Test function
if __name__ == "__main__":
    generator = DataGenerator()
    customers = generator.generate_customers(20)
    depot = generator.generate_depot()
    distance_matrix = generator.calculate_distance_matrix(customers, depot)

    print("Customers:")
    print(customers.head())
    print(f"\nDepot: {depot}")
    print(f"\nDistance matrix shape: {distance_matrix.shape}")
