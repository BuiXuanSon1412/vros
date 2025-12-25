from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import os

from py_impl.config import Config
from py_impl.input import Input
from py_impl.solution import Solution
from py_impl.tabu import TabuSearch

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
class TabuJSONResult:
    dataset: str
    best_score: float
    solution: Dict[str, Any]
    schedules: List[TripSchedule]
    customer_arrivals: Dict[int, Dict[str, Any]]
    convergence: List[float]


def _build_schedules_and_arrivals(
    sol: Solution,
    inp: Input,
) -> Tuple[List[TripSchedule], Dict[int, Dict[str, Any]]]:
    schedules: List[TripSchedule] = []
    arrivals: Dict[int, Dict[str, Any]] = {}

    # --- Technician: 1 route / tech ---
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

    # --- Drones: nhiều trip nối tiếp ---
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

    # chỉ giữ customer 1..numCus
    arrivals = {c: v for c, v in arrivals.items() if 1 <= c <= inp.numCus}
    return schedules, arrivals


class _TabuSearchWithHistory(TabuSearch):

    def run(self, solution: Solution) -> Tuple[float, Solution, List[List[int]], List[float]]:
        solution.input = self.input

        for i in range(len(solution.droneTripList)):
            if not solution.droneTripList[i]:
                solution.droneTripList[i].append([])

        current_sol = solution.copy()
        current_sol.alpha1 = self.alpha1
        current_sol.alpha2 = self.alpha2

        best_feasible_sol = None
        best_feasible_score = 999999.0

        if current_sol.check_feasible():
            best_feasible_sol = current_sol.copy()
            best_feasible_score = best_feasible_sol.getScore()

        not_improve_iter = 0
        act_order_cycle = -1

        matrix = [[0] * (self.input.numCus + 1) for _ in range(self.input.numCus + 1)]

        convergence: List[float] = []
        best_so_far = best_feasible_score if best_feasible_sol else current_sol.getScore()

        import random

        for _it in range(self.config.tabuMaxIter):
            act_order_cycle = (act_order_cycle + 1) % 5
            act_ord = (act_order_cycle + 1) if self.config.isCycle else random.randint(1, 5)

            target_feasible = best_feasible_sol if best_feasible_sol else current_sol

            s_neighbor = None
            if act_ord == 1:
                s_neighbor = current_sol.relocate(self.tabu_lists[1], target_feasible, "INTER")
            elif act_ord == 2:
                s_neighbor = current_sol.exchange(self.tabu_lists[2], target_feasible, "INTER")
            elif act_ord == 3:
                s_neighbor = current_sol.orOpt(self.tabu_lists[3], target_feasible, "INTER", 1)
            elif act_ord == 4:
                s_neighbor = current_sol.crossExchange(self.tabu_lists[4], target_feasible, "INTER", 1, 1)
            elif act_ord == 5:
                s_neighbor = current_sol.twoOpt(self.tabu_lists[5], target_feasible, "INTER")

            if s_neighbor:
                move_state = s_neighbor.ext.get("state", "")
                if move_state:
                    self.tabu_lists[act_ord].append(move_state)
                    if len(self.tabu_lists[act_ord]) > self.tabuDuration:
                        self.tabu_lists[act_ord].pop(0)

                current_sol = s_neighbor
                self.updatePenalty(current_sol.dz, current_sol.cz)
                current_sol.alpha1 = self.alpha1
                current_sol.alpha2 = self.alpha2

                # update best feasible
                if current_sol.check_feasible():
                    if (best_feasible_sol is None) or (current_sol.getScore() < best_feasible_score - self.config.tabuEpsilon):
                        best_feasible_sol = current_sol.copy()
                        best_feasible_score = best_feasible_sol.getScore()
                        not_improve_iter = 0
                    else:
                        not_improve_iter += 1
                else:
                    not_improve_iter += 1
            else:
                not_improve_iter += 1

            # convergence best-so-far
            if best_feasible_sol:
                best_so_far = min(best_so_far, best_feasible_sol.getScore())
            else:
                best_so_far = min(best_so_far, current_sol.getScore())
            convergence.append(best_so_far)

            if not_improve_iter > self.config.tabuNotImproveIter:
                break

        if best_feasible_sol:
            best_feasible_sol.refactorSolution()
            best_feasible_sol = self.runPostOptimization(best_feasible_sol)
            final_score = best_feasible_sol.getScore()
            if convergence:
                convergence[-1] = min(convergence[-1], final_score)
            return final_score, best_feasible_sol, matrix, convergence

        return current_sol.getScore(), current_sol, matrix, convergence


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

def run_tabu_to_json(
    cfg: Config,
    data_path: Optional[str] = None,
    input_obj: Optional[Input] = None,
    init_solution: Optional[Solution] = None,
    indent: int = 2,
    return_dict: bool = False,
) -> Union[str, Dict[str, Any]]:
    """
    Chạy Tabu Search và TRẢ VỀ KẾT QUẢ DƯỚI DẠNG JSON (KHÔNG GHI FILE).

    JSON gồm:
    - dataset
    - best_score
    - solution (droneTripList, techTripList, alpha1/alpha2, cz/dz, score)
    - schedules (theo từng trip)
    - customer_arrivals (arrival time theo customer)
    - convergence (best score so far theo iteration)

    Tham số:
    - return_dict = False → trả về chuỗi JSON
    - return_dict = True  → trả về dict Python (đã JSON-serializable)
    """

    # =========================
    # 1. Build Input
    # =========================
    if input_obj is None:
        if not data_path:
            data_path = os.path.join(cfg.ws, cfg.dataPath.strip("/\\"), cfg.dataName)
        input_obj = Input(
            droneVelocity=cfg.droneVelocity,
            techVelocity=cfg.techVelocity,
            limitationFlightTime=cfg.droneLimitationFlightTime,
            path=data_path,
        )

    # =========================
    # 2. Run Tabu Search
    # =========================
    tabu = _TabuSearchWithHistory(cfg, input_obj)

    if init_solution is None:
        init_solution = (
            tabu.initSolution.copy()
            if getattr(tabu, "initSolution", None)
            else Solution(cfg, input_obj, tabu.alpha1, tabu.alpha2)
        )

    best_score, best_sol, _matrix, convergence = tabu.run(init_solution)

    # =========================
    # 3. Build schedules & arrivals
    # =========================
    schedules, arrivals = _build_schedules_and_arrivals(best_sol, input_obj)

    # =========================
    # 4. Build JSON payload
    # =========================
    result = TabuJSONResult(
        dataset=getattr(input_obj, "dataSet", ""),
        best_score=float(best_score),
        solution=_solution_to_dict(best_sol),
        schedules=schedules,
        customer_arrivals=arrivals,
        convergence=[float(x) for x in convergence],
    )

    payload = asdict(result)

    # JSON chuẩn: key phải là string
    payload["customer_arrivals"] = {
        str(k): v for k, v in payload["customer_arrivals"].items()
    }

    # =========================
    # 5. Return
    # =========================
    if return_dict:
        return payload

    return json.dumps(payload, ensure_ascii=False, indent=indent)