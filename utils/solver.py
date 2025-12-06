# utils/solver.py - Enhanced with Real Pareto Front Tracking

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import time


class ParetoFrontTracker:
    """Track Pareto-optimal solutions during optimization"""

    def __init__(self):
        self.pareto_solutions = []  # List of (makespan, cost, solution_data)

    def add_solution(
        self, makespan: float, cost: float, solution_data: Optional[Dict] = None
    ):
        """Add a solution and update Pareto front"""
        if solution_data is None:
            solution_data = {}
        new_solution = (makespan, cost, solution_data)

        # Check if new solution is dominated by existing solutions
        is_dominated = False
        for existing_makespan, existing_cost, _ in self.pareto_solutions:
            if existing_makespan <= makespan and existing_cost <= cost:
                if existing_makespan < makespan or existing_cost < cost:
                    is_dominated = True
                    break

        if is_dominated:
            return False

        # Remove solutions dominated by new solution
        self.pareto_solutions = [
            (ms, c, data)
            for ms, c, data in self.pareto_solutions
            if not (makespan <= ms and cost <= c and (makespan < ms or cost < c))
        ]

        # Add new solution
        self.pareto_solutions.append(new_solution)
        return True

    def get_pareto_front(self) -> List[Tuple[float, float]]:
        """Get Pareto front as list of (makespan, cost) tuples"""
        front = [(ms, cost) for ms, cost, _ in self.pareto_solutions]
        # Sort by first objective for cleaner visualization
        front.sort(key=lambda x: x[0])
        return front

    def get_best_solution_by_weight(
        self, weight_makespan: float = 0.5
    ) -> Tuple[float, float, Dict]:
        """Get best solution based on weighted sum"""
        if not self.pareto_solutions:
            # Return a dummy solution if no Pareto solutions exist
            return (0.0, 0.0, {})

        weight_cost = 1.0 - weight_makespan

        # Normalize objectives
        makespans = [ms for ms, _, _ in self.pareto_solutions]
        costs = [c for _, c, _ in self.pareto_solutions]

        max_makespan = max(makespans)
        max_cost = max(costs)

        best_score = float("inf")
        best_solution: Tuple[float, float, Dict] = (0.0, 0.0, {})

        for ms, cost, data in self.pareto_solutions:
            norm_ms = ms / max_makespan if max_makespan > 0 else 0
            norm_cost = cost / max_cost if max_cost > 0 else 0

            score = weight_makespan * norm_ms + weight_cost * norm_cost

            if score < best_score:
                best_score = score
                best_solution = (ms, cost, data if data is not None else {})

        return best_solution


class DummySolver:
    """
    Enhanced solver with real Pareto front tracking
    """

    def __init__(self, problem_type: int, algorithm: str):
        self.problem_type = problem_type
        self.algorithm = algorithm
        self.best_solution = None
        self.convergence_history = []
        self.pareto_tracker = ParetoFrontTracker() if problem_type == 2 else None

    def solve(
        self,
        customers: pd.DataFrame,
        depot: Dict,
        distance_matrix: np.ndarray,
        vehicle_config: Dict,
        algorithm_params: Dict,
    ) -> Dict:
        """
        Solve the problem and return results with real Pareto front for Problem 2
        """
        print(
            f"[DEBUG] Solving with {self.algorithm} for problem type {self.problem_type}"
        )

        num_customers = len(customers)
        max_iterations = self._get_max_iterations(algorithm_params)

        # Initialize base solution
        routes = self._generate_dummy_routes(num_customers, vehicle_config)
        schedule = self._generate_dummy_schedule(
            routes, customers, depot, distance_matrix, vehicle_config
        )

        # ============================================================
        # SIMULATE OPTIMIZATION WITH PARETO TRACKING (PROBLEM 2)
        # ============================================================
        if self.problem_type == 2:
            result = self._solve_biobjective(
                routes,
                schedule,
                customers,
                depot,
                distance_matrix,
                vehicle_config,
                max_iterations,
            )
        else:
            result = self._solve_single_objective(
                routes,
                schedule,
                customers,
                depot,
                distance_matrix,
                vehicle_config,
                max_iterations,
            )

        self.best_solution = result
        return result

    def _solve_biobjective(
        self,
        routes,
        schedule,
        customers,
        depot,
        distance_matrix,
        vehicle_config,
        max_iterations,
    ) -> Dict:
        """Solve bi-objective problem with Pareto tracking"""

        # Calculate base metrics
        base_makespan = max([task["end_time"] for task in schedule]) if schedule else 0
        base_distance = self._calculate_total_distance(routes, distance_matrix)
        base_cost = self._calculate_cost(routes, distance_matrix, vehicle_config)

        print(
            f"[DEBUG] Base solution - Makespan: {base_makespan:.2f}, Cost: {base_cost:.2f}"
        )

        # Initialize convergence tracking
        convergence_history = []

        # Simulate iterative optimization
        for iteration in range(0, max_iterations + 1, max(1, max_iterations // 50)):
            # Simulate exploring different trade-offs
            # Generate multiple candidate solutions per iteration
            num_candidates = 5  # Generate 5 candidates per iteration

            for _ in range(num_candidates):
                # Create variation in makespan and cost
                # Random trade-off: sometimes favor speed, sometimes favor cost
                trade_off_factor = np.random.uniform(0, 1)

                # Makespan: improves over iterations but with trade-offs
                improvement_rate = iteration / max_iterations
                makespan_factor = 1.0 - (
                    0.25 * improvement_rate * (1 - trade_off_factor)
                )
                candidate_makespan = base_makespan * makespan_factor

                # Cost: inverse relationship with makespan improvement
                cost_factor = 1.0 - (0.25 * improvement_rate * trade_off_factor)
                candidate_cost = base_cost * cost_factor

                # Add realistic noise
                candidate_makespan *= 1 + np.random.uniform(-0.03, 0.03)
                candidate_cost *= 1 + np.random.uniform(-0.03, 0.03)

                # Ensure positive values
                candidate_makespan = max(base_makespan * 0.7, candidate_makespan)
                candidate_cost = max(base_cost * 0.7, candidate_cost)

                # Add to Pareto tracker
                self.pareto_tracker.add_solution(
                    candidate_makespan,
                    candidate_cost,
                    solution_data={"iteration": iteration},
                )

            # Track convergence (use weighted sum for single-objective tracking)
            current_front = self.pareto_tracker.get_pareto_front()
            if current_front:
                # Use minimum weighted sum as convergence metric
                weights = (0.5, 0.5)  # Equal weights
                makespans, costs = zip(*current_front)

                # Normalize
                max_ms = max(makespans)
                max_cost = max(costs)

                scores = [
                    weights[0] * (ms / max_ms) + weights[1] * (c / max_cost)
                    for ms, c in current_front
                ]
                best_weighted_fitness = min(scores) * max_ms  # Scale back for display

                convergence_history.append((iteration, best_weighted_fitness))

        # Get final Pareto front
        pareto_front = self.pareto_tracker.get_pareto_front()

        print(f"[DEBUG] Found {len(pareto_front)} Pareto-optimal solutions")

        # Select a balanced solution for display
        best_solution = self.pareto_tracker.get_best_solution_by_weight(0.5)
        final_makespan, final_cost, _ = best_solution

        total_distance = self._calculate_total_distance(routes, distance_matrix)

        return {
            "routes": routes,
            "schedule": schedule,
            "makespan": final_makespan,
            "cost": final_cost,
            "total_distance": total_distance,
            "convergence_history": convergence_history,
            "computation_time": np.random.uniform(2, 8),
            "algorithm": self.algorithm,
            "pareto_front": pareto_front,
            "pareto_rank": 1,
            "is_pareto_optimal": True,
            "num_pareto_solutions": len(pareto_front),
        }

    def _solve_single_objective(
        self,
        routes,
        schedule,
        customers,
        depot,
        distance_matrix,
        vehicle_config,
        max_iterations,
    ) -> Dict:
        """Solve single-objective problem"""

        makespan = max([task["end_time"] for task in schedule]) if schedule else 0
        total_distance = self._calculate_total_distance(routes, distance_matrix)
        cost = self._calculate_cost(routes, distance_matrix, vehicle_config)

        # Generate convergence history
        convergence_history = self._generate_convergence_history(
            max_iterations, makespan
        )

        return {
            "routes": routes,
            "schedule": schedule,
            "makespan": makespan,
            "cost": cost,
            "total_distance": total_distance,
            "convergence_history": convergence_history,
            "computation_time": np.random.uniform(1, 5),
            "algorithm": self.algorithm,
            "pareto_front": [],
        }

    def _get_max_iterations(self, algorithm_params: Dict) -> int:
        """Extract max iterations from algorithm params"""
        if "max_iteration" in algorithm_params:
            return algorithm_params["max_iteration"]
        elif "num_generations" in algorithm_params:
            return algorithm_params["num_generations"]
        elif "loop" in algorithm_params:
            return algorithm_params["loop"]
        else:
            return 100

    def _generate_dummy_routes(self, num_customers: int, vehicle_config: Dict) -> Dict:
        """Generate dummy routes"""
        routes = {}
        customer_ids = list(range(1, num_customers + 1))
        np.random.shuffle(customer_ids)

        num_trucks = vehicle_config["truck"]["count"]
        num_drones = vehicle_config["drone"]["count"]
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

        if idx < len(customer_ids) and num_trucks > 0:
            routes["Truck_1"].extend(customer_ids[idx:])

        return routes

    def _generate_dummy_schedule(
        self, routes, customers, depot, distance_matrix, vehicle_config
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
            current_location = 0

            for customer_id in route:
                customer = customers[customers["id"] == customer_id].iloc[0]
                travel_time = (
                    distance_matrix[current_location][customer_id] / speed
                ) * 60
                arrival_time = current_time + travel_time
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

            travel_time = (distance_matrix[current_location][0] / speed) * 60
            current_time += travel_time

        return schedule

    def _calculate_total_distance(self, routes, distance_matrix) -> float:
        """Calculate total distance"""
        total = 0
        for route in routes.values():
            if not route:
                continue
            total += distance_matrix[0][route[0]]
            for i in range(len(route) - 1):
                total += distance_matrix[route[i]][route[i + 1]]
            total += distance_matrix[route[-1]][0]
        return total

    def _calculate_cost(self, routes, distance_matrix, vehicle_config) -> float:
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
        """Generate convergence history for single-objective"""
        history = []
        current_fitness = final_fitness * 1.5

        for i in range(0, max_iterations + 1, max(1, max_iterations // 50)):
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
        algorithms,
        customers,
        depot,
        distance_matrix,
        vehicle_config,
        algorithm_params,
    ) -> Dict:
        """Run multiple algorithms"""
        for algo in algorithms:
            print(f"Running {algo}...")
            solver = DummySolver(self.problem_type, algo)
            result = solver.solve(
                customers, depot, distance_matrix, vehicle_config, algorithm_params
            )
            self.results[algo] = result
            time.sleep(0.5)

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
