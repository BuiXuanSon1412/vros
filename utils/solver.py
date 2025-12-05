# utils/solver.py - Updated with Pareto front generation

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import time


class DummySolver:
    """
    Dummy solver for testing the interface
    Will be replaced with real algorithms later
    """

    def __init__(self, problem_type: int, algorithm: str):
        self.problem_type = problem_type
        self.algorithm = algorithm
        self.best_solution = None
        self.convergence_history = []

    def solve(
        self,
        customers: pd.DataFrame,
        depot: Dict,
        distance_matrix: np.ndarray,
        vehicle_config: Dict,
        algorithm_params: Dict,
    ) -> Dict:
        """
        Solve the problem and return results

        Returns:
            Dict containing: routes, schedule, makespan, cost, convergence_history
        """
        print(
            f"[DEBUG] Solving with {self.algorithm} for problem type {self.problem_type}"
        )

        # Simulate solving process
        num_customers = len(customers)
        num_vehicles = (
            vehicle_config["truck"]["count"] + vehicle_config["drone"]["count"]
        )

        # Generate dummy routes
        routes = self._generate_dummy_routes(num_customers, vehicle_config)

        # Generate dummy schedule
        schedule = self._generate_dummy_schedule(
            routes, customers, depot, distance_matrix, vehicle_config
        )

        # Calculate metrics
        makespan = max([task["end_time"] for task in schedule]) if schedule else 0
        total_distance = self._calculate_total_distance(routes, distance_matrix)
        cost = self._calculate_cost(routes, distance_matrix, vehicle_config)

        # Get max iterations from algorithm params
        max_iterations = self._get_max_iterations(algorithm_params)

        # Generate convergence history
        self.convergence_history = self._generate_convergence_history(
            max_iterations, makespan
        )

        # Initialize result dictionary
        result = {
            "routes": routes,
            "schedule": schedule,
            "makespan": makespan,
            "cost": cost,
            "total_distance": total_distance,
            "convergence_history": self.convergence_history,
            "computation_time": np.random.uniform(1, 5),  # seconds
            "algorithm": self.algorithm,
        }

        # ============================================================
        # ADD PARETO FRONT FOR PROBLEM 2 (BI-OBJECTIVE)
        # ============================================================
        if self.problem_type == 2:
            # Generate Pareto front (20-30 solutions)
            pareto_front = self._generate_pareto_front(makespan, cost, num_solutions=25)
            result["pareto_front"] = pareto_front

            # Add Pareto-specific metrics
            result["pareto_rank"] = 1  # Assume current solution is on front
            result["is_pareto_optimal"] = True

            # Optional: Calculate hypervolume (quality indicator)
            # result['hypervolume'] = self._calculate_hypervolume(pareto_front)
        else:
            result["pareto_front"] = []  # Empty for single-objective problems

        self.best_solution = result
        return result

    def _generate_pareto_front(
        self, base_makespan: float, base_cost: float, num_solutions: int = 25
    ) -> List[Tuple[float, float]]:
        """
        Generate a realistic Pareto front for bi-objective optimization

        Args:
            base_makespan: Reference makespan value
            base_cost: Reference cost value
            num_solutions: Number of Pareto optimal solutions to generate

        Returns:
            List of (makespan, cost) tuples representing Pareto front
        """
        solutions = []

        for i in range(num_solutions):
            # Create trade-off: as we optimize one objective, the other gets worse
            alpha = i / (num_solutions - 1) if num_solutions > 1 else 0.5

            # Makespan: ranges from base*0.8 to base*1.2
            # Lower alpha = faster (lower makespan) but more expensive
            makespan = base_makespan * (0.8 + 0.4 * alpha)

            # Cost: inverse relationship - ranges from base*1.2 to base*0.8
            # Lower alpha = more expensive, higher alpha = cheaper
            cost = base_cost * (1.2 - 0.4 * alpha)

            # Add slight curvature to make it look more realistic (Pareto fronts are often curved)
            curvature = 0.1 * np.sin(alpha * np.pi)
            makespan *= 1 + curvature
            cost *= 1 + curvature

            # Add small random noise for realism
            noise_makespan = np.random.uniform(-0.02, 0.02)
            noise_cost = np.random.uniform(-0.02, 0.02)

            makespan *= 1 + noise_makespan
            cost *= 1 + noise_cost

            # Ensure positive values
            makespan = max(makespan, base_makespan * 0.75)
            cost = max(cost, base_cost * 0.75)

            solutions.append((makespan, cost))

        # Sort by first objective for cleaner visualization
        solutions.sort(key=lambda x: x[0])

        return solutions

    def _get_max_iterations(self, algorithm_params: Dict) -> int:
        """Extract max iterations from algorithm params based on problem type"""
        # Problem 1: max_iteration
        if "max_iteration" in algorithm_params:
            return algorithm_params["max_iteration"]
        # Problem 2: num_generations
        elif "num_generations" in algorithm_params:
            return algorithm_params["num_generations"]
        # Problem 3: loop
        elif "loop" in algorithm_params:
            return algorithm_params["loop"]
        # Fallback
        else:
            return 100

    def _generate_dummy_routes(self, num_customers: int, vehicle_config: Dict) -> Dict:
        """Generate dummy routes"""
        routes = {}
        customer_ids = list(range(1, num_customers + 1))
        np.random.shuffle(customer_ids)

        num_trucks = vehicle_config["truck"]["count"]
        num_drones = vehicle_config["drone"]["count"]

        # Distribute customers to vehicles
        total_vehicles = num_trucks + num_drones
        if total_vehicles == 0:
            return routes

        customers_per_vehicle = max(1, num_customers // total_vehicles)

        idx = 0
        for i in range(num_trucks):
            end_idx = min(idx + customers_per_vehicle, len(customer_ids))
            routes[f"Truck_{i + 1}"] = customer_ids[idx:end_idx]
            idx = end_idx

        for i in range(num_drones):
            end_idx = min(idx + customers_per_vehicle, len(customer_ids))
            routes[f"Drone_{i + 1}"] = customer_ids[idx:end_idx]
            idx = end_idx

        # Distribute remaining customers
        if idx < len(customer_ids) and num_trucks > 0:
            routes["Truck_1"].extend(customer_ids[idx:])

        return routes

    def _generate_dummy_schedule(
        self,
        routes: Dict,
        customers: pd.DataFrame,
        depot: Dict,
        distance_matrix: np.ndarray,
        vehicle_config: Dict,
    ) -> List[Dict]:
        """Generate dummy schedule"""
        schedule = []

        for vehicle_id, route in routes.items():
            if not route:
                continue

            is_truck = "Truck" in vehicle_id
            speed = (
                vehicle_config["truck"]["speed"]
                if is_truck
                else vehicle_config["drone"]["speed"]
            )

            current_time = 0
            current_location = 0  # depot

            for customer_id in route:
                customer = customers[customers["id"] == customer_id].iloc[0]

                # Travel time
                travel_time = (
                    distance_matrix[current_location][customer_id] / speed
                ) * 60  # minutes
                arrival_time = current_time + travel_time

                # Service time
                service_time = customer.get("service_time", 10)
                departure_time = arrival_time + service_time

                schedule.append(
                    {
                        "vehicle_id": vehicle_id,
                        "customer_id": f"C{customer_id}",
                        "start_time": arrival_time,
                        "end_time": departure_time,
                        "service_time": service_time,
                    }
                )

                current_time = departure_time
                current_location = customer_id

            # Return to depot
            travel_time = (distance_matrix[current_location][0] / speed) * 60
            current_time += travel_time

        return schedule

    def _calculate_total_distance(
        self, routes: Dict, distance_matrix: np.ndarray
    ) -> float:
        """Calculate total distance"""
        total = 0
        for route in routes.values():
            if not route:
                continue
            # depot -> first customer
            total += distance_matrix[0][route[0]]
            # between customers
            for i in range(len(route) - 1):
                total += distance_matrix[route[i]][route[i + 1]]
            # last customer -> depot
            total += distance_matrix[route[-1]][0]
        return total

    def _calculate_cost(
        self, routes: Dict, distance_matrix: np.ndarray, vehicle_config: Dict
    ) -> float:
        """Calculate total cost"""
        total_cost = 0
        for vehicle_id, route in routes.items():
            if not route:
                continue

            is_truck = "Truck" in vehicle_id
            cost_per_km = (
                vehicle_config["truck"]["cost_per_km"]
                if is_truck
                else vehicle_config["drone"]["cost_per_km"]
            )

            distance = 0
            distance += distance_matrix[0][route[0]]
            for i in range(len(route) - 1):
                distance += distance_matrix[route[i]][route[i + 1]]
            distance += distance_matrix[route[-1]][0]

            total_cost += distance * cost_per_km

        return total_cost

    def _generate_convergence_history(
        self, max_iterations: int, final_fitness: float
    ) -> List[Tuple[int, float]]:
        """Generate dummy convergence history"""
        history = []
        current_fitness = final_fitness * 1.5  # Start worse

        for i in range(0, max_iterations + 1, max(1, max_iterations // 50)):
            # Simulate improvement
            improvement = (current_fitness - final_fitness) * np.random.uniform(
                0.05, 0.15
            )
            current_fitness = max(final_fitness, current_fitness - improvement)
            history.append((i, current_fitness))

        return history


class AlgorithmRunner:
    """Wrapper to run multiple algorithms and compare"""

    def __init__(self, problem_type: int):
        self.problem_type = problem_type
        self.results = {}

    def run_multiple_algorithms(
        self,
        algorithms: List[str],
        customers: pd.DataFrame,
        depot: Dict,
        distance_matrix: np.ndarray,
        vehicle_config: Dict,
        algorithm_params: Dict,
    ) -> Dict:
        """Run multiple algorithms and save results"""

        for algo in algorithms:
            print(f"Running {algo}...")
            solver = DummySolver(self.problem_type, algo)
            result = solver.solve(
                customers, depot, distance_matrix, vehicle_config, algorithm_params
            )
            self.results[algo] = result
            time.sleep(0.5)  # Simulate computation

        return self.results

    def get_comparison_summary(self) -> pd.DataFrame:
        """Create comparison summary table"""
        summary = []

        for algo, result in self.results.items():
            summary.append(
                {
                    "Algorithm": algo,
                    "Makespan": result["makespan"],
                    "Cost": result["cost"],
                    "Total Distance": result["total_distance"],
                    "Computation Time": result["computation_time"],
                }
            )

        return pd.DataFrame(summary)
