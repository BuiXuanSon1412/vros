# py_impl/run_multlevel.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import os
import copy

from .config import Config
from .input import Input
from .solution import Solution
from .multilevel import MultiLevel


@dataclass
class StopEvent:
    customer: int
    arrival: float
    depart: float


@dataclass
class TripSchedule:
    vehicle_type: str  # "DRONE" | "TECHNICIAN"
    vehicle_id: int
    trip_id: int
    start_time: float
    end_time: float
    stops: List[StopEvent]


@dataclass
class MultiLevelJSONResult:
    dataset: str
    best_score: float
    solution: Dict[str, Any]
    schedules: List[TripSchedule]
    customer_arrivals: Dict[int, Dict[str, Any]]
    convergence: List[float]


def _solution_to_dict(sol: Solution) -> Dict[str, Any]:
    return {
        "droneTripList": sol.droneTripList,
        "techTripList": sol.techTripList,
        "alpha1": float(getattr(sol, "alpha1", 0.0)),
        "alpha2": float(getattr(sol, "alpha2", 0.0)),
        "cz": float(getattr(sol, "cz", 0.0)),
        "dz": float(getattr(sol, "dz", 0.0)),
        "score": float(sol.getScore()),
    }


def _build_schedules_and_arrivals(
    sol: Solution,
    inp: Input,
) -> Tuple[List[TripSchedule], Dict[int, Dict[str, Any]]]:
    schedules: List[TripSchedule] = []
    arrivals: Dict[int, Dict[str, Any]] = {}

    # Technician
    for te_id, route in enumerate(sol.techTripList):
        if not route:
            continue
        t = 0.0
        prev = 0
        stops: List[StopEvent] = []
        for cus in route:
            t += inp.techTimes[prev][cus]
            arr = t
            stops.append(StopEvent(customer=cus, arrival=arr, depart=arr))
            if cus not in arrivals or arr < float(arrivals[cus]["arrival"]):
                arrivals[cus] = {
                    "vehicle_type": "TECHNICIAN",
                    "vehicle_id": te_id,
                    "trip_id": 0,
                    "arrival": arr,
                }
            prev = cus
        end_time = t + inp.techTimes[prev][0]
        schedules.append(
            TripSchedule(
                vehicle_type="TECHNICIAN",
                vehicle_id=te_id,
                trip_id=0,
                start_time=0.0,
                end_time=end_time,
                stops=stops,
            )
        )

    # Drones (trip nối tiếp)
    for d_id, drone_trips in enumerate(sol.droneTripList):
        current_start = 0.0
        for trip_id, trip in enumerate(drone_trips):
            if not trip:
                continue
            t = current_start
            prev = 0
            stops: List[StopEvent] = []
            for cus in trip:
                t += inp.droneTimes[prev][cus]
                arr = t
                stops.append(StopEvent(customer=cus, arrival=arr, depart=arr))
                if cus not in arrivals or arr < float(arrivals[cus]["arrival"]):
                    arrivals[cus] = {
                        "vehicle_type": "DRONE",
                        "vehicle_id": d_id,
                        "trip_id": trip_id,
                        "arrival": arr,
                    }
                prev = cus
            end_time = t + inp.droneTimes[prev][0]
            schedules.append(
                TripSchedule(
                    vehicle_type="DRONE",
                    vehicle_id=d_id,
                    trip_id=trip_id,
                    start_time=current_start,
                    end_time=end_time,
                    stops=stops,
                )
            )
            current_start = end_time

    arrivals = {c: v for c, v in arrivals.items() if 1 <= c <= inp.numCus}
    return schedules, arrivals


def run_multilevel_to_json(
    cfg: Config,
    data_path: Optional[str] = None,
    input_obj: Optional[Input] = None,
    indent: int = 2,
    return_dict: bool = False,
) -> Union[str, Dict[str, Any]]:
    """
    Chạy Multi-Level và TRẢ VỀ JSON (không ghi file).

    JSON gồm:
    - dataset
    - best_score
    - solution
    - schedules
    - customer_arrivals
    - convergence (list hội tụ theo iteration của Tabu ở mọi stage)
    """

    # 1) Build input
    if input_obj is None:
        if not data_path:
            data_path = os.path.join(cfg.ws, cfg.dataPath.strip("/\\"), cfg.dataName)
        input_obj = Input(
            droneVelocity=cfg.droneVelocity,
            techVelocity=cfg.techVelocity,
            limitationFlightTime=cfg.droneLimitationFlightTime,
            path=data_path,
        )

    # backup input gốc vì multilevel sẽ mutate matrices/numCus
    input_backup = copy.deepcopy(input_obj)

    # 2) Run multilevel
    ml = MultiLevel(cfg, input_obj)
    best_score, best_sol, convergence = ml.run()

    # 3) restore input gốc để schedule/arrival đúng theo dataset gốc
    best_sol.input = input_backup

    schedules, arrivals = _build_schedules_and_arrivals(best_sol, input_backup)

    result = MultiLevelJSONResult(
        dataset=getattr(input_backup, "dataSet", ""),
        best_score=float(best_score),
        solution=_solution_to_dict(best_sol),
        schedules=schedules,
        customer_arrivals=arrivals,
        convergence=[float(x) for x in convergence],
    )

    payload = asdict(result)
    payload["customer_arrivals"] = {str(k): v for k, v in payload["customer_arrivals"].items()}

    if return_dict:
        return payload
    return json.dumps(payload, ensure_ascii=False, indent=indent)
