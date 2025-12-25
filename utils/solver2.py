# utils/solver.py - FIXED Problem 3 VRP-MRDR Implementation

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import time


class ParetoFrontTracker:
    """Track Pareto-optimal solutions during optimization"""

    def __init__(self):
        self.pareto_solutions = []

    def add_solution(
        self, makespan: float, cost: float, solution_data: Optional[Dict] = None
    ):
        """Add a solution and update Pareto front"""
        if solution_data is None:
            solution_data = {}
        new_solution = (makespan, cost, solution_data)

        is_dominated = False
        for existing_makespan, existing_cost, _ in self.pareto_solutions:
            if existing_makespan <= makespan and existing_cost <= cost:
                if existing_makespan < makespan or existing_cost < cost:
                    is_dominated = True
                    break

        if is_dominated:
            return False

        self.pareto_solutions = [
            (ms, c, data)
            for ms, c, data in self.pareto_solutions
            if not (makespan <= ms and cost <= c and (makespan < ms or cost < c))
        ]

        self.pareto_solutions.append(new_solution)
        return True

    def get_pareto_front(self) -> List[Tuple[float, float]]:
        """Get Pareto front as list of (makespan, cost) tuples"""
        front = [(ms, cost) for ms, cost, _ in self.pareto_solutions]
        front.sort(key=lambda x: x[0])
        return front

    def get_best_solution_by_weight(
        self, weight_makespan: float = 0.5
    ) -> Tuple[float, float, Dict]:
        """Get best solution based on weighted sum"""
        if not self.pareto_solutions:
            return (0.0, 0.0, {})

        weight_cost = 1.0 - weight_makespan

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
    """Enhanced solver with FIXED Problem 3 resupply logic"""

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
        """Solve the problem with correct logic for each problem type"""
        print(f"[DEBUG] Solving Problem {self.problem_type} with {self.algorithm}")

        num_customers = len(customers)
        max_iterations = self._get_max_iterations(algorithm_params)

        # Problem 3: Resupply logic
        if self.problem_type == 3:
            result = self._solve_problem3_resupply(
                customers, depot, distance_matrix, vehicle_config, max_iterations
            )
        # Problem 2: Bi-objective
        elif self.problem_type == 2:
            routes = self._generate_dummy_routes(num_customers, vehicle_config)
            schedule = self._generate_dummy_schedule(
                routes, customers, depot, distance_matrix, vehicle_config
            )
            result = self._solve_biobjective(
                routes,
                schedule,
                customers,
                depot,
                distance_matrix,
                vehicle_config,
                max_iterations,
            )
        # Problem 1: Single objective
        else:
            routes = self._generate_dummy_routes(num_customers, vehicle_config)
            schedule = self._generate_dummy_schedule(
                routes, customers, depot, distance_matrix, vehicle_config
            )
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

    def _solve_problem3_resupply(
        self,
        customers: pd.DataFrame,
        depot: Dict,
        distance_matrix: np.ndarray,
        vehicle_config: Dict,
        max_iterations: int,
    ) -> Dict:
        """
        FIXED Problem 3: VRP with Multi-point Drone Resupply
        - Trucks visit all customers in sequence
        - Drones resupply trucks with packages at customer locations
        - Packages have release dates - drone delivers if not ready at truck departure
        """
        print("[DEBUG] Problem 3: VRP-MRDR mode activated")

        num_trucks = vehicle_config["truck"]["count"]
        num_drones = vehicle_config["drone"]["count"]
        truck_speed = vehicle_config["truck"]["speed"]
        drone_speed = vehicle_config["drone"]["speed"]
        drone_capacity = vehicle_config["drone"]["capacity"]

        # Sort customers by release date and proximity
        customers_sorted = customers.sort_values(
            ["release_date", "x", "y"]
        ).reset_index(drop=True)

        # Generate truck routes (assign customers to trucks)
        truck_routes = self._assign_customers_to_trucks_balanced(
            customers_sorted, num_trucks, depot, distance_matrix
        )

        # Simulate truck operations with intelligent drone resupply
        schedule, resupply_operations = self._simulate_vrp_with_resupply(
            truck_routes,
            customers_sorted,
            depot,
            distance_matrix,
            truck_speed,
            drone_speed,
            num_drones,
            drone_capacity,
        )

        # Calculate metrics
        makespan = max([task["end_time"] for task in schedule]) if schedule else 0
        total_distance = self._calculate_total_distance_p3(
            truck_routes, distance_matrix
        )

        # Calculate costs
        truck_cost = total_distance * vehicle_config["truck"]["cost_per_km"]
        resupply_distance = sum(op["distance"] for op in resupply_operations)
        drone_cost = resupply_distance * vehicle_config["drone"]["cost_per_km"]
        total_cost = truck_cost + drone_cost

        # Generate convergence history
        convergence_history = self._generate_convergence_history(
            max_iterations, makespan
        )

        # Statistics
        num_resupplies = len([op for op in resupply_operations if op.get("packages")])
        total_waiting_time = sum(
            task.get("waiting_time", 0) for task in schedule if "waiting_time" in task
        )
        packages_delivered_by_drone = sum(
            len(op.get("packages", [])) for op in resupply_operations
        )

        print(
            f"[DEBUG] Makespan: {makespan:.2f}, Resupplies: {num_resupplies}, Packages by drone: {packages_delivered_by_drone}"
        )

        return {
            "routes": {f"Truck {i + 1}": route for i, route in enumerate(truck_routes)},
            "schedule": schedule,
            "resupply_operations": resupply_operations,
            "makespan": makespan,
            "cost": total_cost,
            "total_distance": total_distance + resupply_distance,
            "truck_distance": total_distance,
            "drone_distance": resupply_distance,
            "convergence_history": convergence_history,
            "computation_time": np.random.uniform(3, 8),
            "algorithm": self.algorithm,
            "pareto_front": [],
            "num_resupplies": num_resupplies,
            "packages_delivered_by_drone": packages_delivered_by_drone,
            "total_waiting_time": total_waiting_time,
        }

    def _assign_customers_to_trucks_balanced(
        self,
        customers: pd.DataFrame,
        num_trucks: int,
        depot: Dict,
        distance_matrix: np.ndarray,
    ) -> List[List[int]]:
        """Assign customers to trucks with spatial clustering"""
        customer_ids = customers["id"].tolist()

        if num_trucks == 1:
            return [customer_ids]

        # Simple nearest neighbor clustering
        truck_routes = [[] for _ in range(num_trucks)]
        assigned = set()

        # Start each truck with a seed customer (spatially distributed)
        angles = np.linspace(0, 2 * np.pi, num_trucks, endpoint=False)
        for truck_idx, angle in enumerate(angles):
            # Find customer closest to this direction from depot
            best_customer = None
            best_score = -float("inf")

            for cust_id in customer_ids:
                if cust_id in assigned:
                    continue

                cust = customers[customers["id"] == cust_id].iloc[0]
                dx = cust["x"] - depot["x"]
                dy = cust["y"] - depot["y"]
                cust_angle = np.arctan2(dy, dx)

                # Score based on angle difference
                angle_diff = abs(cust_angle - angle)
                if angle_diff > np.pi:
                    angle_diff = 2 * np.pi - angle_diff

                score = -angle_diff

                if score > best_score:
                    best_score = score
                    best_customer = cust_id

            if best_customer is not None:
                truck_routes[truck_idx].append(best_customer)
                assigned.add(best_customer)

        # Assign remaining customers to nearest truck
        for cust_id in customer_ids:
            if cust_id in assigned:
                continue

            # Find truck with closest last customer
            best_truck = 0
            best_distance = float("inf")

            for truck_idx, route in enumerate(truck_routes):
                if not route:
                    distance = distance_matrix[0][cust_id]
                else:
                    distance = distance_matrix[route[-1]][cust_id]

                if distance < best_distance:
                    best_distance = distance
                    best_truck = truck_idx

            truck_routes[best_truck].append(cust_id)
            assigned.add(cust_id)

        return truck_routes

    def _simulate_vrp_with_resupply(
        self,
        truck_routes: List[List[int]],
        customers: pd.DataFrame,
        depot: Dict,
        distance_matrix: np.ndarray,
        truck_speed: float,
        drone_speed: float,
        num_drones: int,
        drone_capacity: int,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        FIXED: Simulate VRP with intelligent drone resupply
        Key logic: Drone only resupplies if package not ready when truck departs depot
        """
        schedule = []
        resupply_operations = []
        drone_available_times = [0.0] * num_drones

        for truck_idx, route in enumerate(truck_routes):
            if not route:
                continue

            truck_id = f"Truck {truck_idx + 1}"
            current_time = 0.0
            current_location = 0  # depot
            truck_load = []  # Packages currently on truck

            # Determine which packages are ready at departure
            for customer_id in route:
                customer = customers[customers["id"] == customer_id].iloc[0]
                release_date = customer["release_date"]

                if release_date <= 0:  # Ready at start
                    truck_load.append(customer_id)

            # Packages needing drone resupply
            packages_needing_resupply = [cid for cid in route if cid not in truck_load]

            print(
                f"[DEBUG] {truck_id}: {len(truck_load)} ready, {len(packages_needing_resupply)} need resupply"
            )

            # Schedule drone resupply trips
            drone_trips = self._schedule_drone_trips(
                truck_id,
                packages_needing_resupply,
                truck_routes[truck_idx],
                customers,
                depot,
                distance_matrix,
                num_drones,
                drone_available_times,
                drone_speed,
                drone_capacity,
            )

            resupply_operations.extend(drone_trips)

            # Simulate truck journey
            for customer_id in route:
                customer = customers[customers["id"] == customer_id].iloc[0]

                # Travel to customer
                travel_time = (
                    distance_matrix[current_location][customer_id] / truck_speed
                ) * 60
                arrival_time = current_time + travel_time

                # Check if package is available
                waiting_time = 0.0
                package_ready = customer_id in truck_load

                if not package_ready:
                    # Find drone delivery for this package
                    drone_delivery = next(
                        (
                            trip
                            for trip in drone_trips
                            if customer_id in trip.get("packages", [])
                            and trip["meeting_customer_id"] == customer_id
                        ),
                        None,
                    )

                    if drone_delivery:
                        drone_arrival = drone_delivery["arrival_time"]
                        if drone_arrival > arrival_time:
                            waiting_time = drone_arrival - arrival_time
                            arrival_time = drone_arrival

                # Service customer
                service_time = customer.get("service_time", 5)
                departure_time = arrival_time + service_time

                schedule.append(
                    {
                        "vehicle_id": truck_id,
                        "customer_id": f"C{customer_id}",
                        "action": "Service",
                        "start_time": arrival_time,
                        "end_time": departure_time,
                        "service_time": service_time,
                        "waiting_time": waiting_time,
                    }
                )

                current_time = departure_time
                current_location = customer_id

            # Return to depot
            travel_time = (distance_matrix[current_location][0] / truck_speed) * 60
            return_time = current_time + travel_time

            schedule.append(
                {
                    "vehicle_id": truck_id,
                    "customer_id": "Depot",
                    "action": "Return",
                    "start_time": current_time,
                    "end_time": return_time,
                    "service_time": 0,
                }
            )

        # Add drone schedules
        for trip in resupply_operations:
            if not trip.get("packages"):
                continue

            schedule.append(
                {
                    "vehicle_id": trip["drone_id"],
                    "customer_id": f"C{trip['meeting_customer_id']}",
                    "action": "Resupply",
                    "start_time": trip["departure_time"],
                    "end_time": trip["arrival_time"],
                    "service_time": 0,
                }
            )

        return schedule, resupply_operations

    def _schedule_drone_trips(
        self,
        truck_id: str,
        packages_needing_resupply: List[int],
        truck_route: List[int],
        customers: pd.DataFrame,
        depot: Dict,
        distance_matrix: np.ndarray,
        num_drones: int,
        drone_available_times: List[float],
        drone_speed: float,
        drone_capacity: int,
    ) -> List[Dict]:
        """Schedule drone trips to resupply truck"""
        trips = []

        if not packages_needing_resupply:
            return trips

        # Group packages by meeting point (where on route to deliver)
        package_groups = []
        current_group = []
        current_weight = 0

        for pkg_id in packages_needing_resupply:
            customer = customers[customers["id"] == pkg_id].iloc[0]
            pkg_weight = customer["demand"]
            release_date = customer["release_date"]

            if current_weight + pkg_weight <= drone_capacity and len(current_group) < 3:
                current_group.append(pkg_id)
                current_weight += pkg_weight
            else:
                if current_group:
                    package_groups.append(current_group)
                current_group = [pkg_id]
                current_weight = pkg_weight

        if current_group:
            package_groups.append(current_group)

        # Schedule each group
        for group_idx, group in enumerate(package_groups):
            # Meeting point: deliver to first customer in group
            meeting_customer_id = group[0]
            meeting_customer = customers[customers["id"] == meeting_customer_id].iloc[0]

            # Find available drone
            drone_idx = drone_available_times.index(min(drone_available_times))
            drone_id = f"Drone {drone_idx + 1}"

            # Drone can depart after latest release date in group
            max_release = max(
                customers[customers["id"] == pid].iloc[0]["release_date"]
                for pid in group
            )
            earliest_departure = max(drone_available_times[drone_idx], max_release)

            # Flight time to meeting point
            flight_time = (distance_matrix[0][meeting_customer_id] / drone_speed) * 60
            arrival_time = earliest_departure + flight_time
            return_time = arrival_time + flight_time

            # Update drone availability
            drone_available_times[drone_idx] = return_time

            # Record trip
            total_weight = sum(
                customers[customers["id"] == pid].iloc[0]["demand"] for pid in group
            )

            trips.append(
                {
                    "drone_id": drone_id,
                    "truck_id": truck_id,
                    "meeting_customer_id": meeting_customer_id,
                    "packages": group,
                    "total_weight": total_weight,
                    "departure_time": earliest_departure,
                    "arrival_time": arrival_time,
                    "return_time": return_time,
                    "distance": distance_matrix[0][meeting_customer_id] * 2,
                }
            )

        return trips

    def _calculate_total_distance_p3(
        self, truck_routes: List[List[int]], distance_matrix: np.ndarray
    ) -> float:
        """Calculate total distance for truck routes"""
        total = 0
        for route in truck_routes:
            if not route:
                continue
            # Depot to first customer
            total += distance_matrix[0][route[0]]
            # Between customers
            for i in range(len(route) - 1):
                total += distance_matrix[route[i]][route[i + 1]]
            # Last customer to depot
            total += distance_matrix[route[-1]][0]
        return total

    # ============= Other methods (Problem 1 & 2) remain the same =============
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
        base_makespan = max([task["end_time"] for task in schedule]) if schedule else 0
        base_distance = self._calculate_total_distance(routes, distance_matrix)
        base_cost = self._calculate_cost(routes, distance_matrix, vehicle_config)

        convergence_history = []

        for iteration in range(0, max_iterations + 1, max(1, max_iterations // 50)):
            num_candidates = 5

            for _ in range(num_candidates):
                trade_off_factor = np.random.uniform(0, 1)
                improvement_rate = iteration / max_iterations
                makespan_factor = 1.0 - (
                    0.25 * improvement_rate * (1 - trade_off_factor)
                )
                candidate_makespan = base_makespan * makespan_factor

                cost_factor = 1.0 - (0.25 * improvement_rate * trade_off_factor)
                candidate_cost = base_cost * cost_factor

                candidate_makespan *= 1 + np.random.uniform(-0.03, 0.03)
                candidate_cost *= 1 + np.random.uniform(-0.03, 0.03)

                candidate_makespan = max(base_makespan * 0.7, candidate_makespan)
                candidate_cost = max(base_cost * 0.7, candidate_cost)

                self.pareto_tracker.add_solution(
                    candidate_makespan,
                    candidate_cost,
                    solution_data={"iteration": iteration},
                )

            current_front = self.pareto_tracker.get_pareto_front()
            if current_front:
                weights = (0.5, 0.5)
                makespans, costs = zip(*current_front)
                max_ms = max(makespans)
                max_cost = max(costs)

                scores = [
                    weights[0] * (ms / max_ms) + weights[1] * (c / max_cost)
                    for ms, c in current_front
                ]
                best_weighted_fitness = min(scores) * max_ms

                convergence_history.append((iteration, best_weighted_fitness))

        pareto_front = self.pareto_tracker.get_pareto_front()
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
        """Generate dummy routes for Problems 1 & 2"""
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
            routes[f"Truck {i + 1}"] = customer_ids[idx:end_idx]
            idx = end_idx

        for i in range(num_drones):
            end_idx = min(idx + customers_per_vehicle, len(customer_ids))
            routes[f"Drone {i + 1}"] = customer_ids[idx:end_idx]
            idx = end_idx

        if idx < len(customer_ids) and num_trucks > 0:
            routes["Truck 1"].extend(customer_ids[idx:])

        return routes

    def _generate_dummy_schedule(
        self, routes, customers, depot, distance_matrix, vehicle_config
    ) -> List[Dict]:
        """Generate dummy schedule for Problems 1 & 2"""
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
