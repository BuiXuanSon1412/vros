# py_impl/multilevel.py
from __future__ import annotations

import copy
import random
from typing import List, Tuple, Dict, Any, Optional

from .solution import Solution
from .tabu import TabuSearch


class TabuSearchWithHistory(TabuSearch):
    """
    Kế thừa TabuSearch để ghi lịch sử hội tụ (best-so-far) theo iteration.
    Return: (best_score, best_solution, matrix, convergence)
    """

    def run(self, solution: Solution) -> Tuple[float, Solution, List[List[int]], List[float]]:
        # Thiết lập input cho solution
        solution.input = self.input

        # đảm bảo cấu trúc droneTripList
        for i in range(len(solution.droneTripList)):
            if not solution.droneTripList[i]:
                solution.droneTripList[i].append([])

        current_sol = solution.copy()
        current_sol.alpha1 = self.alpha1
        current_sol.alpha2 = self.alpha2

        best_feasible_sol: Optional[Solution] = None
        best_feasible_score = float("inf")

        # lưu best feasible nếu có
        if current_sol.check_feasible():
            best_feasible_sol = current_sol.copy()
            best_feasible_score = best_feasible_sol.getScore()

        not_improve_iter = 0
        act_order_cycle = -1

        # matrix dummy như code cũ
        matrix = [[0] * (self.input.numCus + 1) for _ in range(self.input.numCus + 1)]

        # hội tụ: best-so-far (ưu tiên feasible, nếu chưa có feasible thì best current score)
        convergence: List[float] = []
        best_so_far = best_feasible_score if best_feasible_sol else current_sol.getScore()

        for _it in range(self.config.tabuMaxIter):
            act_order_cycle = (act_order_cycle + 1) % 5
            act_ord = (act_order_cycle + 1) if self.config.isCycle else random.randint(1, 5)

            target_feasible = best_feasible_sol if best_feasible_sol else current_sol

            # --- Neighborhood Selection ---
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
                # cập nhật tabu list
                move_state = s_neighbor.ext.get("state", "")
                if move_state:
                    self.tabu_lists[act_ord].append(move_state)
                    if len(self.tabu_lists[act_ord]) > self.tabuDuration:
                        self.tabu_lists[act_ord].pop(0)

                current_sol = s_neighbor

                # update penalty & alpha
                self.updatePenalty(current_sol.dz, current_sol.cz)
                current_sol.alpha1 = self.alpha1
                current_sol.alpha2 = self.alpha2

                # cập nhật best feasible
                if current_sol.check_feasible():
                    sc = current_sol.getScore()
                    if sc < best_feasible_score - self.config.tabuEpsilon:
                        best_feasible_sol = current_sol.copy()
                        best_feasible_score = sc
                        not_improve_iter = 0
                    else:
                        not_improve_iter += 1
                else:
                    not_improve_iter += 1
            else:
                not_improve_iter += 1

            # update convergence
            if best_feasible_sol is not None:
                best_so_far = min(best_so_far, best_feasible_score)
            else:
                best_so_far = min(best_so_far, current_sol.getScore())
            convergence.append(float(best_so_far))

            if not_improve_iter > self.config.tabuNotImproveIter:
                break

        # finalize
        if best_feasible_sol is not None:
            best_feasible_sol.refactorSolution()
            best_feasible_sol = self.runPostOptimization(best_feasible_sol)
            final_score = best_feasible_sol.getScore()
            if convergence:
                convergence[-1] = min(convergence[-1], float(final_score))
            return float(final_score), best_feasible_sol, matrix, convergence

        # nếu không có feasible
        final_score = current_sol.getScore()
        if convergence:
            convergence[-1] = min(convergence[-1], float(final_score))
        return float(final_score), current_sol, matrix, convergence


class MultiLevel:
    def __init__(self, config, input_obj):
        self.config = config
        self.input = input_obj

    def convertMatrix(self, currentMatrix, matrixReBefore, mapping: Dict[int, List[int]]):
        size = len(mapping)  # mapping bao gồm node 0
        newMatrix = [[0.0] * (size + 1) for _ in range(size + 1)]
        matrixRe = [[0.0] * (size + 1) for _ in range(size + 1)]

        for i in range(size):
            for j in range(size):
                nodes_i = mapping[i]
                nodes_j = mapping[j]

                matrixRe[i][j] = matrixReBefore[nodes_i[-1]][nodes_j[0]]

                dist_sum = 0.0
                combined = nodes_i + nodes_j
                for k in range(len(combined) - 1):
                    dist_sum += currentMatrix[combined[k]][combined[k + 1]]
                newMatrix[i][j] = dist_sum

        # depot cuối trùng depot đầu
        for i in range(len(newMatrix)):
            newMatrix[i][-1] = newMatrix[i][0]
            newMatrix[-1][i] = newMatrix[0][i]

        return newMatrix, matrixRe

    def mergeSol(
        self,
        solution: Solution,
        numCus: int,
        matrixRe: List[List[float]],
        c1: List[bool],
    ) -> Tuple[int, Solution, Dict[int, List[int]]]:

        # 1) collect edges in current solution
        edgeSol = set()
        for drone in solution.droneTripList:
            for trip in drone:
                for k in range(len(trip) - 1):
                    edgeSol.add((trip[k], trip[k + 1]))

        for tech in solution.techTripList:
            for k in range(len(tech) - 1):
                edgeSol.add((tech[k], tech[k + 1]))

        # 2) chọn k% cặp gần nhất (không dùng node trùng)
        numUpdate = int(numCus * self.config.percent_select)
        update_pairs = []
        used_nodes = set()

        all_possible_pairs = []
        for i in range(1, numCus + 1):
            for j in range(1, numCus + 1):
                if i == j:
                    continue
                if c1[i] == c1[j] and (i, j) not in edgeSol:
                    all_possible_pairs.append((i, j, matrixRe[i][j]))
        all_possible_pairs.sort(key=lambda x: x[2])

        for i, j, _dist in all_possible_pairs:
            if len(update_pairs) >= numUpdate:
                break
            if i not in used_nodes and j not in used_nodes:
                update_pairs.append((i, j))
                used_nodes.add(i)
                used_nodes.add(j)

        # 3) build mapping: (0)->[0], pairs -> cluster, remaining -> singleton
        new_nodes_list = []
        nodes_in_pairs = set()
        for u in update_pairs:
            new_nodes_list.append([u[0], u[1]])
            nodes_in_pairs.add(u[0])
            nodes_in_pairs.add(u[1])

        for i in range(1, numCus + 1):
            if i not in nodes_in_pairs:
                new_nodes_list.append([i])

        res_mapping: Dict[int, List[int]] = {0: [0]}
        for idx, cluster in enumerate(new_nodes_list):
            res_mapping[idx + 1] = cluster

        # reverse map old -> super
        reverse_map: Dict[int, int] = {}
        for super_node, members in res_mapping.items():
            for m in members:
                reverse_map[m] = super_node

        # 4) remap solution ids -> super ids
        new_sol = solution.copy()

        for d in range(len(new_sol.droneTripList)):
            for t in range(len(new_sol.droneTripList[d])):
                old_trip = new_sol.droneTripList[d][t]
                new_trip = []
                for node in old_trip:
                    new_id = reverse_map[node]
                    if not new_trip or new_trip[-1] != new_id:
                        new_trip.append(new_id)
                new_sol.droneTripList[d][t] = new_trip

        for t in range(len(new_sol.techTripList)):
            old_route = new_sol.techTripList[t]
            new_route = []
            for node in old_route:
                new_id = reverse_map[node]
                if not new_route or new_route[-1] != new_id:
                    new_route.append(new_id)
            new_sol.techTripList[t] = new_route

        return len(update_pairs), new_sol, res_mapping

    def mergeProcess(self):
        map_levels = []
        dist_matrix_levels = []
        c1_levels = []
        convergence: List[float] = []

        # init solution từ tabu
        tabu0 = TabuSearchWithHistory(self.config, self.input)
        current_sol = tabu0.initSolution.copy() if getattr(tabu0, "initSolution", None) else Solution(self.config, self.input)

        # current matrices
        curr_dist_matrix = copy.deepcopy(self.input.distances)
        curr_matrix_re = copy.deepcopy(curr_dist_matrix)
        curr_c1 = copy.deepcopy(self.input.cusOnlyServedByTech)

        dist_matrix_levels.append(curr_dist_matrix)
        c1_levels.append(curr_c1)

        curr_num_cus = self.input.numCus

        for _level in range(self.config.num_level):
            # 1) optimize at current level (with history)
            tabu = TabuSearchWithHistory(self.config, self.input)
            _, optimized_sol, _, hist = tabu.run(current_sol)
            convergence.extend(hist)

            # 2) merge nodes
            num_merged, merged_sol, mapping = self.mergeSol(optimized_sol, curr_num_cus, curr_matrix_re, curr_c1)
            if num_merged == 0:
                current_sol = optimized_sol
                break

            map_levels.append(mapping)

            # 3) build next level matrix
            curr_dist_matrix, curr_matrix_re = self.convertMatrix(curr_dist_matrix, curr_matrix_re, mapping)
            dist_matrix_levels.append(curr_dist_matrix)

            # 4) update c1 for super nodes
            new_c1 = [False] * (len(mapping) + 1)
            for super_id, members in mapping.items():
                if any(curr_c1[m] for m in members):
                    new_c1[super_id] = True
            c1_levels.append(new_c1)

            # 5) mutate input for next level
            curr_num_cus = len(mapping) - 1
            self.input.numCus = curr_num_cus
            self.input.distances = curr_dist_matrix
            self.input.cusOnlyServedByTech = new_c1
            self.input.droneTimes = [[d / self.config.droneVelocity for d in row] for row in curr_dist_matrix]
            self.input.techTimes = [[d / self.config.techVelocity for d in row] for row in curr_dist_matrix]

            current_sol = merged_sol

        return current_sol, map_levels, dist_matrix_levels, c1_levels, convergence

    def splitProcess(self, solution: Solution, map_levels, dist_levels, c1_levels):
        curr_sol = solution
        convergence: List[float] = []

        for i in range(len(map_levels) - 1, -1, -1):
            mapping = map_levels[i]

            # 1) expand super nodes -> original nodes
            new_drone_list = []
            for drone in curr_sol.droneTripList:
                new_drone = []
                for trip in drone:
                    new_trip = []
                    for node in trip:
                        new_trip.extend(mapping[node])
                    new_drone.append(new_trip)
                new_drone_list.append(new_drone)

            new_tech_list = []
            for tech in curr_sol.techTripList:
                new_route = []
                for node in tech:
                    new_route.extend(mapping[node])
                new_tech_list.append(new_route)

            curr_sol.droneTripList = new_drone_list
            curr_sol.techTripList = new_tech_list

            # 2) restore input for lower level
            self.input.distances = dist_levels[i]
            self.input.cusOnlyServedByTech = c1_levels[i]
            self.input.numCus = len(dist_levels[i]) - 2
            self.input.droneTimes = [[d / self.config.droneVelocity for d in row] for row in dist_levels[i]]
            self.input.techTimes = [[d / self.config.techVelocity for d in row] for row in dist_levels[i]]

            # 3) refine by tabu (with history)
            tabu = TabuSearchWithHistory(self.config, self.input)
            _, curr_sol, _, hist = tabu.run(curr_sol)
            convergence.extend(hist)

        return curr_sol, convergence

    def run(self):
        print("Starting Multi-Level Process...")

        merged_sol, map_lv, dist_lv, c1_lv, conv_merge = self.mergeProcess()
        final_sol, conv_split = self.splitProcess(merged_sol, map_lv, dist_lv, c1_lv)

        convergence = conv_merge + conv_split
        return float(final_sol.getScore()), final_sol, convergence
