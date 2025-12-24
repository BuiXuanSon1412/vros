from dataclasses import dataclass, field
from typing import List


@dataclass
class Customer:
    x: float
    y: float
    demand: float
    only_by_truck: float
    truck_service_time: float
    drone_service_time: float


@dataclass
class Individual:
    route: List[int] = field(default_factory=list)
    fitness1: float = 0.0
    fitness2: float = 0.0
    crowding_distance: float = 0.0
    tabu_search: int = 0
    local_search: int = 0


@dataclass
class MoveRecord:
    # Initializing lists of size 2 to mimic C++ fixed-size arrays
    start: List[int] = field(default_factory=lambda: [0, 0])
    end: int = 0
    back: List[int] = field(default_factory=lambda: [0, 0])
    front: List[int] = field(default_factory=lambda: [0, 0])


@dataclass
class TabuRecord:
    indi: Individual = field(default_factory=Individual)
    record: MoveRecord = field(default_factory=MoveRecord)


@dataclass
class ParetoRecord:
    indi: Individual = field(default_factory=Individual)
    norm_fit1: float = 0.0
    norm_fit2: float = 0.0
    criteria: float = 0.0
    ranking: int = 0


@dataclass
class DistanceRanking:
    node: int
    ranking: float


@dataclass
class NodeDistance:
    node: int
    distance: float
