# utils/solver.py (refactored)
from algorithms.hnsgaii_ts.main import HNSGAIITSSolver
from algorithms.ada_ts.ats import ATSSolver


class AlgorithmFactory:
    """Factory to create algorithm instances"""

    ALGORITHM_MAP = {
        1: {},
        2: {
            "HNSGAII-TS": HNSGAIITSSolver,
        },
        3: {
            "Adaptive Tabu Search": ATSSolver,
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

    def solve(self, customers, depot, vehicle_config, algorithm_params):
        """Solve problem using selected algorithm"""

        # Call the actual algorithm
        solution = self.algorithm.solve(
            customers, depot, vehicle_config, algorithm_params
        )

        # Add metadata
        solution["algorithm"] = self.algorithm_name
        solution["problem_type"] = self.problem_type

        return solution
