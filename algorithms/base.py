# algorithms/base_algorithm.py
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class BaseAlgorithm(ABC):
    """Base class for all optimization algorithms"""

    def __init__(self, problem_type: int):
        self.problem_type = problem_type
        self.convergence_history = []

    @abstractmethod
    def solve(
        self,
        customers: pd.DataFrame,
        depot: Dict,
        distance_matrix: np.ndarray,
        vehicle_config: Dict,
        algorithm_params: Dict,
    ) -> Dict:
        """
        Must return dictionary with:
        - routes: Dict[str, List[int]]
        - schedule: List[Dict]
        - makespan: float
        - cost: float
        - total_distance: float
        - convergence_history: List[Tuple[int, float]]
        - computation_time: float
        """
        pass

    def _validate_solution(self, solution: Dict) -> bool:
        """Validate solution has required fields"""
        required = ["routes", "schedule", "makespan", "cost", "total_distance"]
        return all(k in solution for k in required)
