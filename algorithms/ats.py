import time
import copy
import random
import numpy as np
import math
import json
import os

from typing import List, Tuple, Any

# ===================== CONFIGURABLE PARAMETERS =====================
# Reward parameters for adaptive mechanism
GAMMA_1 = 0.5  # γ₁: Reward for improving best solution
GAMMA_2 = 0.3  # γ₂: Reward for improving current solution
GAMMA_3 = 0.1  # γ₃: Reward for accepting worse solution
GAMMA_4 = 0.0  # γ₄: Penalty (not used in basic version)

# Iteration parameters
VARIABLE_MAX_ITER = 2  # η: Variable max iteration (multiplier for segment length)
FIXED_MAX_ITER = 100  # LOOP: Fixed max iteration (not used in adaptive version)
NUM_SEGMENTS = 5  # SEG: Number of restart segments
DELTA = 0.3  # δ: Learning rate for weight adaptation

# Tabu search parameters
EPSILON = -1e-5  # Tolerance for improvement detection
CC = 1  # Number of potential solutions to explore


# Problem data
class ProblemData:
    """Store problem instance data."""

    def __init__(self):
        self.number_of_cities = 0
        self.number_of_trucks = 2
        self.number_of_drones = 2
        self.truck_speed = 0.5
        self.drone_speed = 1.0
        self.drone_capacity = 8
        self.drone_limit_time = 90
        self.unloading_time = 5

        # Data arrays
        self.city_coords = []
        self.city_demand = []
        self.release_date = []

        # Distance matrices
        self.manhattan_matrix = None
        self.euclid_matrix = None

    def read_data(self, filepath: str):
        """Read problem data from file."""
        with open(filepath, "r") as f:
            lines = f.readlines()

        # Parse header
        for line in lines[:7]:
            parts = line.strip().split()
            if len(parts) >= 2:
                key, value = parts[0], parts[1]
                if key == "number_truck":
                    self.number_of_trucks = int(value)
                elif key == "number_drone":
                    self.number_of_drones = int(value)
                elif key == "truck_speed":
                    self.truck_speed = float(value)
                elif key == "drone_speed":
                    self.drone_speed = float(value)
                elif key == "M_d":
                    self.drone_capacity = int(value)
                elif key == "L_d":
                    self.drone_limit_time = int(value)

        # Parse city data
        self.city_coords = []
        self.city_demand = []
        self.release_date = []

        for line in lines[8:]:  # Skip header lines
            parts = line.strip().split()
            if len(parts) >= 4:
                x, y = float(parts[0]), float(parts[1])
                demand = int(parts[2])
                release = int(parts[3])

                self.city_coords.append([x, y])
                self.city_demand.append(demand)
                self.release_date.append(release)

        self.number_of_cities = len(self.city_coords)

        # Build distance matrices
        self._build_distance_matrices()

    def _build_distance_matrices(self):
        """Build Manhattan and Euclidean distance matrices."""
        n = self.number_of_cities
        self.manhattan_matrix = np.zeros((n, n))
        self.euclid_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                dx = abs(self.city_coords[i][0] - self.city_coords[j][0])
                dy = abs(self.city_coords[i][1] - self.city_coords[j][1])

                # Manhattan distance / truck speed
                self.manhattan_matrix[i][j] = (dx + dy) / self.truck_speed

                # Euclidean distance / drone speed
                euclid_dist = math.sqrt(dx**2 + dy**2)
                self.euclid_matrix[i][j] = euclid_dist / self.drone_speed


# Solution
class Solution:
    """Represents a solution with truck routes and drone assignments."""

    def __init__(self, truck_routes: List, drone_queue: List):
        """
        truck_routes: List of trucks, each containing list of [city, packages]
        drone_queue: List of drone trips, each containing [[city, packages_list]]
        """
        self.truck_routes = truck_routes
        self.drone_queue = drone_queue

    def copy(self):
        """Create deep copy of solution."""
        return Solution(
            copy.deepcopy(self.truck_routes), copy.deepcopy(self.drone_queue)
        )

    def to_dict(self):
        """Convert to dictionary format (compatible with original code)."""
        return [self.truck_routes, self.drone_queue]


class SolutionEvaluator:
    """Evaluate solution quality."""

    def __init__(self, data: ProblemData):
        self.data = data

    def evaluate(self, solution: Solution) -> Tuple[float, List[float]]:
        """
        Calculate solution fitness (makespan).
        Returns: (total_time, truck_times)
        """
        if self.data.manhattan_matrix is None:
            raise ValueError("manhattan_matrix is not initialized")

        if self.data.euclid_matrix is None:
            raise ValueError("euclid_matrix is not initialized")

        truck_times = [0.0] * self.data.number_of_trucks

        # Simulate truck movements
        for truck_idx, route in enumerate(solution.truck_routes):
            current_city = 0  # Depot
            current_time = 0.0

            for node in route[1:]:  # Skip depot
                city = node[0]
                packages = node[1]

                # Travel time
                travel_time = self.data.manhattan_matrix[current_city][city]

                # Consider release dates
                max_release = 0
                if packages:
                    max_release = max([self.data.release_date[p] for p in packages])

                arrival_time = max(current_time + travel_time, max_release)
                current_time = arrival_time
                current_city = city

            # Return to depot
            current_time += self.data.manhattan_matrix[current_city][0]
            truck_times[truck_idx] = current_time

        # Simulate drone operations
        drone_available_time = [0.0] * self.data.number_of_drones

        for trip in solution.drone_queue:
            # Get earliest available drone
            drone_idx = np.argmin(drone_available_time)
            start_time = drone_available_time[drone_idx]

            # Calculate trip time
            cities = [t[0] for t in trip]
            packages = []
            for t in trip:
                packages.extend(t[1])

            # Release date constraint
            max_release = max([self.data.release_date[p] for p in packages])
            start_time = max(start_time, max_release)

            # Flight time
            flight_time = self.data.euclid_matrix[0][cities[0]]
            for i in range(len(cities) - 1):
                flight_time += self.data.euclid_matrix[cities[i]][cities[i + 1]]
            flight_time += self.data.euclid_matrix[cities[-1]][0]

            # Unloading time
            flight_time += len(cities) * self.data.unloading_time

            # Update drone availability and truck times
            end_time = start_time + flight_time
            drone_available_time[drone_idx] = end_time

            # Update truck times based on drone delivery points
            for city in cities:
                truck_idx = self._find_truck_for_city(solution, city)
                if truck_idx >= 0:
                    truck_times[truck_idx] = max(truck_times[truck_idx], end_time)

        total_time = max(truck_times + list(drone_available_time))
        return total_time, truck_times

    def _find_truck_for_city(self, solution: Solution, city: int) -> int:
        """Find which truck serves a city."""
        for truck_idx, route in enumerate(solution.truck_routes):
            for node in route:
                if node[0] == city:
                    return truck_idx
        return -1


class InitialSolutionGenerator:
    """Generate initial solution using greedy heuristic."""

    def __init__(self, data: ProblemData):
        self.data = data

    def generate(self) -> Solution:
        """Generate initial solution by release date ordering."""
        # Create list of (city, release_date)
        cities = [
            (i, self.data.release_date[i]) for i in range(1, self.data.number_of_cities)
        ]
        cities.sort(key=lambda x: x[1])

        # Assign cities to trucks round-robin
        truck_routes = [[[0, []]] for _ in range(self.data.number_of_trucks)]

        for idx, (city, _) in enumerate(cities):
            truck_idx = idx % self.data.number_of_trucks
            truck_routes[truck_idx].append([city, []])

        # Initially assign all packages to be picked up at depot
        for truck_idx in range(self.data.number_of_trucks):
            packages_for_truck = []
            for node in truck_routes[truck_idx][1:]:
                city = node[0]
                if self.data.release_date[city] == 0:
                    packages_for_truck.append(city)
            truck_routes[truck_idx][0][1] = packages_for_truck

        # Empty drone queue initially
        drone_queue = []

        return Solution(truck_routes, drone_queue)


class NeighborhoodOperator:
    """Base class for neighborhood operators."""

    def __init__(self, data: ProblemData, evaluator: SolutionEvaluator):
        self.data = data
        self.evaluator = evaluator

    def generate_neighbors(
        self, solution: Solution, tabu_list: List, best_fitness: float
    ) -> List[Tuple[Solution, float, Any]]:
        """Generate neighbor solutions. Returns list of (solution, fitness, move)."""
        raise NotImplementedError


class OneOptOperator(NeighborhoodOperator):
    """Relocate one city to a different position."""

    def generate_neighbors(
        self, solution: Solution, tabu_list: List, best_fitness: float
    ) -> List[Tuple[Solution, float, Any]]:
        neighbors = []

        for truck_i in range(len(solution.truck_routes)):
            route_i = solution.truck_routes[truck_i]

            for pos_i in range(1, len(route_i)):  # Skip depot
                city = route_i[pos_i][0]

                # Try inserting into other positions
                for truck_j in range(len(solution.truck_routes)):
                    route_j = solution.truck_routes[truck_j]

                    for pos_j in range(len(route_j)):
                        if truck_i == truck_j and abs(pos_i - pos_j) <= 1:
                            continue

                        # Check tabu status
                        if tabu_list[city] > 0:
                            continue

                        # Create neighbor
                        new_sol = solution.copy()
                        node = new_sol.truck_routes[truck_i].pop(pos_i)
                        new_sol.truck_routes[truck_j].insert(pos_j, node)

                        # Evaluate
                        fitness, _ = self.evaluator.evaluate(new_sol)
                        neighbors.append((new_sol, fitness, city))

        return neighbors[:20]  # Limit neighborhood size


class SwapOperator(NeighborhoodOperator):
    """Swap two cities."""

    def generate_neighbors(
        self, solution: Solution, tabu_list: List, best_fitness: float
    ) -> List[Tuple[Solution, float, Any]]:
        neighbors = []

        for truck_i in range(len(solution.truck_routes)):
            route_i = solution.truck_routes[truck_i]

            for pos_i in range(1, len(route_i)):
                city_i = route_i[pos_i][0]

                for truck_j in range(len(solution.truck_routes)):
                    route_j = solution.truck_routes[truck_j]

                    for pos_j in range(1, len(route_j)):
                        if truck_i == truck_j and pos_i >= pos_j:
                            continue

                        city_j = route_j[pos_j][0]

                        # Check tabu
                        if tabu_list[city_i] > 0 or tabu_list[city_j] > 0:
                            continue

                        # Create neighbor
                        new_sol = solution.copy()
                        new_sol.truck_routes[truck_i][pos_i][0] = city_j
                        new_sol.truck_routes[truck_j][pos_j][0] = city_i

                        fitness, _ = self.evaluator.evaluate(new_sol)
                        neighbors.append((new_sol, fitness, (city_i, city_j)))

        return neighbors[:20]


# ===================== ADAPTIVE TABU SEARCH =====================
class AdaptiveTabuSearch:
    """
    Adaptive Tabu Search for Vehicle Routing Problem with Drone Resupply.

    Parameters:
    -----------
    gamma: tuple of (γ₁, γ₂, γ₃, γ₄)
        Reward parameters for adaptive mechanism
    eta: int (η)
        Variable max iteration multiplier
    num_segments: int (SEG)
        Number of restart segments
    delta: float (δ)
        Learning rate for weight adaptation
    """

    def __init__(
        self,
        gamma=(GAMMA_1, GAMMA_2, GAMMA_3, GAMMA_4),
        eta=VARIABLE_MAX_ITER,
        num_segments=NUM_SEGMENTS,
        delta=DELTA,
    ):
        self.gamma_1, self.gamma_2, self.gamma_3, self.gamma_4 = gamma
        self.eta = eta
        self.num_segments = num_segments
        self.delta = delta
        self.epsilon = EPSILON

        self.data = None
        self.evaluator = None
        self.operators = []
        self.num_operations = 0

    def _initialize_operators(self):
        """Generate initial solution using greedy heuristic."""
        assert self.data is not None, "ProblemData must be initialized"
        assert self.evaluator is not None, "SolutionEvaluator must be initialized"

        self.operators = [
            ("one_opt", OneOptOperator(self.data, self.evaluator)),
            ("swap", SwapOperator(self.data, self.evaluator)),
        ]
        self.num_operators = len(self.operators)

    def _initialize_solution(self):
        """Generate initial solution."""
        assert self.data is not None, "ProblemData must be initialized"
        generator = InitialSolutionGenerator(self.data)
        return generator.generate()

    def _solution_to_dict(self, solution):
        """Convert solution to serializable dictionary format."""
        return {
            "truck_routes": [
                [[node[0], node[1].copy()] for node in route]
                for route in solution.truck_routes
            ],
            "drone_queue": [
                [[node[0], node[1].copy()] for node in trip]
                for trip in solution.drone_queue
            ],
        }

    def _solution_to_readable(self, solution):
        """Convert solution to human-readable format."""
        readable = {"trucks": [], "drones": []}

        # Truck routes
        for truck_idx, route in enumerate(solution.truck_routes):
            truck_info = {
                "truck_id": truck_idx,
                "route": [node[0] for node in route],  # Just city IDs
                "packages_at_nodes": {
                    node[0]: node[1]
                    for node in route
                    if node[1]  # Only nodes with packages
                },
            }
            readable["trucks"].append(truck_info)

        # Drone trips
        for trip_idx, trip in enumerate(solution.drone_queue):
            trip_info = {
                "trip_id": trip_idx,
                "stops": [{"city": node[0], "packages": node[1]} for node in trip],
            }
            readable["drones"].append(trip_info)

        return readable

    def _calculate_segment_length(self):
        """Calculate iteration length for each segment based on problem size."""
        assert self.data is not None, "ProblemData must be initialized"

        return (
            int(self.data.number_of_cities / math.log10(self.data.number_of_cities))
            * self.eta
        )

    def _initialize_tabu_structures(self):
        """Initialize tabu tenure and structures."""
        assert self.data is not None, "ProblemData must be initialized"

        tabu_tenure = random.uniform(
            2 * math.log(self.data.number_of_cities), self.data.number_of_cities
        )

        # Tạo tabu structure cho mỗi operator (dùng operator index làm key)
        structures = {}
        tenures = {}
        iteration_counts = {}

        for idx in range(self.num_operators):
            structures[idx] = [-tabu_tenure] * self.data.number_of_cities
            tenures[idx] = tabu_tenure
            iteration_counts[idx] = 0

        return tenures, structures, iteration_counts

    def _roulette_wheel_selection(self, weights):
        """Select neighborhood operator using roulette wheel selection."""
        total = sum(weights)
        if total == 0:
            return random.randint(0, len(weights) - 1)

        probs = [w / total for w in weights]
        return np.random.choice(len(weights), p=probs)

    def _explore_neighborhood(
        self,
        solution,
        operator_idx,
        tabu_structures,
        tabu_tenures,
        iteration_counts,
        best_fitness,
    ):
        """
        Explore neighborhood using selected operator.

        Returns:
        --------
        tuple: (best_neighbor, best_fitness_neighbor, move_info, is_improved)
        """

        assert self.data is not None, "ProblemData must be initialized"

        name, operator = self.operators[operator_idx]

        tabu_list = tabu_structures.get(
            operator_idx, [-1000] * self.data.number_of_cities
        )

        neighbors = operator.generate_neighbors(solution, tabu_list, best_fitness)

        if not neighbors:
            return None, float("inf"), None, False

        # Select best non-tabu or aspiration solution
        best_neighbor = None
        best_fitness_neighbor = float("inf")
        best_move = None
        is_improved = False

        for neighbor_sol, fitness, move in neighbors:
            if fitness < best_fitness - self.epsilon:
                # Aspiration criterion: accept if improves global best
                best_neighbor = neighbor_sol
                best_fitness_neighbor = fitness
                best_move = move
                is_improved = True
                break
            elif fitness < best_fitness_neighbor:
                best_neighbor = neighbor_sol
                best_fitness_neighbor = fitness
                best_move = move

        return best_neighbor, best_fitness_neighbor, best_move, is_improved

    def _update_tabu_structure(
        self, tabu_structures, operator_idx, move, iteration_count
    ):
        """Update tabu structure with new move."""
        assert self.data is not None, "ProblemData must be initialized"

        if move is None:
            return

        if isinstance(move, (list, tuple)):
            # Multiple cities (e.g., swap)
            for city in move:
                if isinstance(city, int) and 0 <= city < self.data.number_of_cities:
                    tabu_structures[operator_idx][city] = iteration_count
        else:
            # Single city (e.g., one-opt)
            if isinstance(move, int) and 0 <= move < self.data.number_of_cities:
                tabu_structures[operator_idx][move] = iteration_count

    def _adaptive_weight_update(self, weights, scores, usage_counts):
        """Update operator weights using adaptive mechanism."""
        new_weights = []
        for i in range(len(weights)):
            if usage_counts[i] == 0:
                new_weights.append(weights[i])
            else:
                avg_score = scores[i] / usage_counts[i]
                new_weights.append(
                    (1 - self.delta) * weights[i] + self.delta * avg_score
                )

        return new_weights

    def _diversification(self, solution):
        """Apply diversification by swapping array segments (optimized version)."""

        assert self.evaluator is not None, "SolutionEvaluator must be initialized"

        best_sol = solution
        best_fit, _ = self.evaluator.evaluate(solution)

        max_neighbors = 20  # Limit number of neighbors to evaluate
        neighbors_evaluated = 0

        for truck_idx in range(len(solution.truck_routes)):
            if neighbors_evaluated >= max_neighbors:
                break

            route = solution.truck_routes[truck_idx]
            length = len(route) - 1

            if length < 4:  # Need at least 4 cities for meaningful swap
                continue

            middle = length // 2

            # Select random segment size
            seg_sizes = list(range(2, min(middle + 1, 5)))  # Max segment size 4
            if not seg_sizes:
                continue

            ran = random.choice(seg_sizes)

            # Try a limited number of random swaps
            attempts = 0
            max_attempts = min(10, max_neighbors - neighbors_evaluated)

            while attempts < max_attempts:
                # Random positions for first segment
                i = random.randint(1, middle - ran + 2) if middle - ran + 2 > 1 else 1
                j = min(i + ran - 1, middle)

                # Random positions for second segment
                k = (
                    random.randint(middle + 1, length - ran + 2)
                    if length - ran + 2 > middle + 1
                    else middle + 1
                )
                l = min(k + ran - 1, length)

                if i >= j or k >= l:
                    attempts += 1
                    continue

                new_sol = solution.copy()

                # Collect affected cities
                affected_cities = set()
                for x in range(i, j + 1):
                    affected_cities.add(route[x][0])
                for x in range(k, l + 1):
                    affected_cities.add(route[x][0])

                # Clear packages from affected nodes
                for node in new_sol.truck_routes[truck_idx]:
                    if node[0] in affected_cities:
                        node[1] = []

                # Swap segments
                new_sol.truck_routes[truck_idx] = (
                    new_sol.truck_routes[truck_idx][:i]
                    + new_sol.truck_routes[truck_idx][k : l + 1]
                    + new_sol.truck_routes[truck_idx][j + 1 : k]
                    + new_sol.truck_routes[truck_idx][i : j + 1]
                    + new_sol.truck_routes[truck_idx][l + 1 :]
                )

                # Remove affected packages from drone queue
                new_sol.drone_queue = [
                    [
                        [node[0], [p for p in node[1] if p not in affected_cities]]
                        for node in trip
                        if node[1] or node[0] not in affected_cities
                    ]
                    for trip in new_sol.drone_queue
                ]
                # Clean empty trips
                new_sol.drone_queue = [trip for trip in new_sol.drone_queue if trip]

                # Evaluate
                fitness, _ = self.evaluator.evaluate(new_sol)

                if fitness < best_fit:
                    best_sol = new_sol
                    best_fit = fitness

                attempts += 1
                neighbors_evaluated += 1

        return best_sol

    def solve(self, data_file, max_time=14400, verbose=True):
        """
        Main solving procedure.

        Parameters:
        -----------
        data_file: str
            Path to problem data file
        max_time: int
            Maximum runtime in seconds
        verbose: bool
            Print progress information

        Returns:
        --------
        dict: Solution results including best solution and fitness
        """
        start_time = time.time()

        # Load problem data
        self.data = ProblemData()
        self.data.read_data(data_file)

        self.evaluator = SolutionEvaluator(self.data)
        self._initialize_operators()

        # Initialize
        p_best = self._initialize_solution()  # Global best solution
        f_best_result = self.evaluator.evaluate(p_best)  # Global best fitness
        f_best = (
            float(f_best_result[0])
            if isinstance(f_best_result, tuple)
            else float(f_best_result)
        )
        p = p_best  # Current best solution

        segment_length = self._calculate_segment_length()

        # Operator weights initialization
        weights = [1.0 / self.num_operators] * self.num_operators

        # Statistics
        results = {
            "best_fitness": f_best,
            "best_solution": p_best,
            "fitness_history": [f_best],
            "time_history": [0],
            "solution_history": [self._solution_to_dict(p_best)],  # THÊM DÒNG NÀY
            "segment_results": [],
        }

        T = 0  # Segment counter
        no_improve_segments = 0

        if verbose:
            print(f"\nInitial solution fitness: {f_best:.2f}")
            print(f"Segment length: {segment_length} iterations")
            print("=" * 60)

        # Main loop: SEG segments
        while T < self.num_segments:
            if time.time() - start_time > max_time:
                if verbose:
                    print(f"\nTime limit reached: {max_time}s")
                break

            segment_start_time = time.time()

            segment_start_fitness = f_best

            if verbose:
                print(f"\n--- Segment {T + 1}/{self.num_segments} ---")

            # Initialize tabu structures for this segment
            tabu_tenures, tabu_structures = {}, {}
            iteration_counts = {}

            # Initialize tabu structures for this segment
            tabu_tenures, tabu_structures, iteration_counts = (
                self._initialize_tabu_structures()
            )

            # Operator statistics for this segment
            scores = [0.0] * self.num_operators
            usage_counts = [0] * self.num_operators

            p_current = copy.deepcopy(p)
            f_current = f_best
            prev_f = f_best

            i = 0
            restart_count = 0

            # Segment loop
            while i < segment_length:
                if time.time() - start_time > max_time:
                    break

                # Select neighborhood operator
                # Select neighborhood operator
                operator_idx = self._roulette_wheel_selection(weights)
                name, _ = self.operators[operator_idx]

                # Explore neighborhood
                p_neighbor, f_neighbor, move, globally_improved = (
                    self._explore_neighborhood(
                        p_current,
                        operator_idx,
                        tabu_structures,
                        tabu_tenures,
                        iteration_counts,
                        f_best,
                    )
                )

                if p_neighbor is None:
                    if verbose and i % 10 == 0:
                        print(f"  Iter {i}: No valid neighbor (operator: {name})")
                    i += 1
                    continue

                # Update iteration count for this operator
                iteration_counts[operator_idx] += 1

                # Update tabu structure
                self._update_tabu_structure(
                    tabu_structures, operator_idx, move, iteration_counts[operator_idx]
                )

                # Accept move
                p_current = p_neighbor
                f_current = f_neighbor

                # Update statistics
                usage_counts[operator_idx] += 1

                # Calculate reward based on improvement type
                if globally_improved:
                    # Case 1: New global best
                    scores[operator_idx] += self.gamma_1
                    p_best = copy.deepcopy(p_neighbor)
                    f_best = f_neighbor
                    p = p_best
                    i = 0  # Reset segment counter (restart)
                    restart_count += 1

                    if verbose:
                        print(
                            f"  Iter {i}: NEW BEST = {f_best:.2f} (operator: {name}) ***"
                        )
                elif f_neighbor < f_current + self.epsilon:
                    # Case 2: Improves current solution
                    scores[operator_idx] += self.gamma_2
                    if verbose and i % 20 == 0:
                        print(
                            f"  Iter {i}: Improved = {f_neighbor:.2f} (operator: {name})"
                        )
                else:
                    # Case 3: Worse solution accepted
                    scores[operator_idx] += self.gamma_3

                # Update weights adaptively
                weights = self._adaptive_weight_update(weights, scores, usage_counts)

                # Record history
                if i % 10 == 0:
                    results["fitness_history"].append(f_best)
                    results["time_history"].append(time.time() - start_time)
                    results["solution_history"].append(
                        self._solution_to_dict(p_best)
                    )  # THÊM DÒNG NÀY

                i += 1

            # End of segment
            segment_time = time.time() - segment_start_time

            # Replace:
            segment_improvement = segment_start_fitness - f_best

            # By:
            # Ensure both are float values for comparison
            # start_fit = (
            #    segment_start_fitness
            #    if isinstance(segment_start_fitness, (int, float))
            #    else segment_start_fitness[0]
            # )
            # end_fit = f_best if isinstance(f_best, (int, float)) else f_best[0]
            # segment_improvement = start_fit - end_fit

            results["segment_results"].append(
                {
                    "segment": T,
                    "fitness": f_best,
                    "improvement": segment_improvement,
                    "time": segment_time,
                    "restarts": restart_count,
                    "operator_usage": usage_counts,
                    "operator_weights": weights.copy(),
                    "best_solution": self._solution_to_dict(p_best),
                }
            )

            if verbose:
                print(f"\nSegment {T + 1} completed:")
                print(f"  Best fitness: {f_best:.2f}")
                print(f"  Improvement: {segment_improvement:.2f}")
                print(f"  Time: {segment_time:.1f}s")
                print(f"  Restarts: {restart_count}")
                print(f"  Operator usage: {usage_counts}")
                print(f"  Operator weights: {[f'{w:.3f}' for w in weights]}")

            # Diversification if no improvement
            if segment_improvement < self.epsilon:
                no_improve_segments += 1
                if no_improve_segments >= 2:
                    if verbose:
                        print("  Applying diversification...")
                    p = self._diversification(p)
                    no_improve_segments = 0
            else:
                no_improve_segments = 0

            T += 1

        # Final results
        total_time = time.time() - start_time
        results["best_fitness"] = f_best
        results["best_solution"] = self._solution_to_dict(p_best)
        results["best_solution_object"] = p_best
        results["best_solution_readable"] = self._solution_to_readable(p_best)
        results["total_time"] = total_time
        results["segments_completed"] = T

        if verbose:
            print("\n" + "=" * 60)
            print("FINAL RESULTS:")
            print(f"  Best fitness: {f_best:.2f}")
            print(f"  Total time: {total_time:.1f}s")
            print(f"  Segments completed: {T}/{self.num_segments}")
            print("=" * 60)

        return results


# ===================== MAIN EXECUTION =====================


def main():
    """Run Adaptive Tabu Search on test data."""

    # Configuration
    dataset = os.getenv("DATA_SET", "RC101_3.dat")
    num_cities = int(os.getenv("NUMBER_OF_CITIES", 10))

    # Adaptive parameters
    gamma = (
        float(os.getenv("GAMMA_1", GAMMA_1)),
        float(os.getenv("GAMMA_2", GAMMA_2)),
        float(os.getenv("GAMMA_3", GAMMA_3)),
        float(os.getenv("GAMMA_4", GAMMA_4)),
    )
    eta = int(os.getenv("ETA", VARIABLE_MAX_ITER))
    num_segments = int(os.getenv("NUM_SEGMENTS", NUM_SEGMENTS))
    delta = float(os.getenv("DELTA", DELTA))

    # Data path
    data_path = f"../data/3/{num_cities}/{dataset}"

    print("=" * 60)
    print("ADAPTIVE TABU SEARCH")
    print("=" * 60)
    print(f"Dataset: {dataset}")
    print("Parameters:")
    print(f"  γ₁ (best improvement): {gamma[0]}")
    print(f"  γ₂ (current improvement): {gamma[1]}")
    print(f"  γ₃ (worse accepted): {gamma[2]}")
    print(f"  η (variable max iter): {eta}")
    print(f"  SEG (num segments): {num_segments}")
    print(f"  δ (learning rate): {delta}")
    print("=" * 60)

    # Initialize solver
    solver = AdaptiveTabuSearch(
        gamma=gamma, eta=eta, num_segments=num_segments, delta=delta
    )

    # Solve
    results = solver.solve(data_path, max_time=14400, verbose=True)

    # Save results
    output_file = f"results_ATS_{dataset}.json"
    with open(output_file, "w") as f:
        # Prepare results for saving
        results_save = results.copy()

        # Remove non-serializable object if exists
        if "best_solution_object" in results_save:
            del results_save["best_solution_object"]

        # best_solution is already in dict format from _solution_to_dict()
        # No need to convert again

        json.dump(results_save, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Optionally save readable format separately
    readable_file = f"results_ATS_{dataset}_readable.json"
    with open(readable_file, "w") as f:
        readable_results = {
            "best_fitness": results["best_fitness"],
            "total_time": results["total_time"],
            "solution": results.get(
                "best_solution_readable",
                solver._solution_to_readable(
                    Solution(
                        results["best_solution"]["truck_routes"],
                        results["best_solution"]["drone_queue"],
                    )
                ),
            ),
        }
        json.dump(readable_results, f, indent=2)

    print(f"Readable results saved to: {readable_file}")

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
