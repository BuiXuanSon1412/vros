# utils/solver.py (refactored)
from algorithms.tabu_search import TabuSearch
from algorithms.ts_multilevel import TabuSearchMultiLevel
from algorithms.nsga2_ts import NSGAII_TS
from algorithms.moead import MOEAD
from algorithms.ats import AdaptiveTabuSearch


class AlgorithmFactory:
    """Factory to create algorithm instances"""

    ALGORITHM_MAP = {
        1: {
            "Tabu Search": TabuSearch,
            "Tabu Search Multi-Level": TabuSearchMultiLevel,
        },
        2: {
            "Tabu Search": TabuSearch,  # Can reuse
            "MOEA/D": MOEAD,
            "NSGAII": NSGAII_TS,
            "HNSGAII-TS": NSGAII_TS,
        },
        3: {
            "Adaptive Tabu Search": AdaptiveTabuSearch,
        },
    }

    @staticmethod
    def create(problem_type: int, algorithm_name: str):
        """Create algorithm instance"""
        algo_class = AlgorithmFactory.ALGORITHM_MAP[problem_type].get(algorithm_name)
        if not algo_class:
            raise ValueError(
                f"Algorithm {algorithm_name} not found for Problem {problem_type}"
            )
        return algo_class()


class Solver:
    """Main solver orchestrator"""

    def __init__(self, problem_type: int, algorithm_name: str):
        self.problem_type = problem_type
        self.algorithm_name = algorithm_name
        self.algorithm = AlgorithmFactory.create(problem_type, algorithm_name)

    def solve(
        self, customers, depot, distance_matrix, vehicle_config, algorithm_params
    ):
        """Solve problem using selected algorithm"""

        # Call the actual algorithm
        solution = self.algorithm.solve(
            customers, depot, distance_matrix, vehicle_config, algorithm_params
        )

        # Add metadata
        solution["algorithm"] = self.algorithm_name
        solution["problem_type"] = self.problem_type

        return solution
