import time
import random
from typing import List, Tuple, Dict
from .solution import Solution
from .config import Config
from .input import Input

class TabuSearch:
    def __init__(self, config: Config, input_obj: Input):
        self.config = config
        self.input = input_obj
        self.tabu_lists: Dict[int, List[str]] = {
            1: [], # MOVE_10 (Relocate)
            2: [], # MOVE_11 (Exchange)
            3: [], # MOVE_20 (OrOpt)
            4: [], # MOVE_21 (CrossExchange)
            5: []  # TWO_OPT
        }
        self.alpha1 = config.tabuAlpha1
        self.alpha2 = config.tabuAlpha2
        self.tabuDuration = config.tabuDuration

        # Khởi tạo Solution ban đầu ngay khi tạo object TabuSearch (Giống C++)
        # Lưu ý: Hàm Solution.initSolution phải được định nghĩa là @staticmethod
        init = Solution.initSolution(self.config, self.input, "MIX", self.alpha1, self.alpha2)
        self.initSolution = init if init else None

    def updatePenalty(self, dz: float, cz: float):
        eps = abs(self.config.tabuEpsilon)
        beta_factor = 1 + self.config.tabuBeta
        
        # Cập nhật Alpha2 (Drone flight)
        if abs(dz) > eps:
            self.alpha2 *= beta_factor
        else:
            self.alpha2 /= beta_factor

        # Cập nhật Alpha1 (Waiting time)
        if abs(cz) > eps:
            self.alpha1 *= beta_factor
        else:
            self.alpha1 /= beta_factor

    def run(self, solution: Solution) -> Tuple[float, Solution, List[List[int]]]:
        # Thiết lập input cho solution
        solution.input = self.input
        
        # Đảm bảo mỗi drone có ít nhất 1 trip trống nếu droneTripList rỗng (giống C++)
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
        
        # Matrix rỗng giống C++ (không dùng trong logic nhưng trả về cho đủ tuple)
        matrix = [[0] * (self.input.numCus + 1) for _ in range(self.input.numCus + 1)]

        for it in range(self.config.tabuMaxIter):
            act_order_cycle = (act_order_cycle + 1) % 5
            act_ord = (act_order_cycle + 1) if self.config.isCycle else random.randint(1, 5)

            # Lưu lại feasible tốt nhất trước khi thực hiện move
            best_feasible_before = best_feasible_sol.copy() if best_feasible_sol else None
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
                # Cập nhật Tabu List
                move_state = s_neighbor.ext.get("state", "")
                if move_state:
                    self.tabu_lists[act_ord].append(move_state)
                    if len(self.tabu_lists[act_ord]) > self.tabuDuration:
                        self.tabu_lists[act_ord].pop(0)

                current_sol = s_neighbor
                self.updatePenalty(current_sol.dz, current_sol.cz)
                current_sol.alpha1 = self.alpha1
                current_sol.alpha2 = self.alpha2
                
                # Check if we found a new best feasible
                if best_feasible_sol:
                    current_feasible_score = best_feasible_sol.getScore()
                    if current_feasible_score < best_feasible_score - self.config.tabuEpsilon:
                        best_feasible_score = current_feasible_score
                        # Trong C++: bestSolution = bestFeasibleBefore;
                        # Ở đây chúng ta gán best_feasible_sol đã được update bên trong hàm move
                        not_improve_iter = 0
                    else:
                        not_improve_iter += 1
                else:
                    if current_sol.check_feasible():
                        best_feasible_sol = current_sol.copy()
                        best_feasible_score = best_feasible_sol.getScore()

            if not_improve_iter > self.config.tabuNotImproveIter:
                break

        if best_feasible_sol:
            best_feasible_sol.refactorSolution()
            # Post Optimization logic (Ejection, Inter, Intra)
            best_feasible_sol = self.runPostOptimization(best_feasible_sol)
            return best_feasible_sol.getScore(), best_feasible_sol, matrix
        
        return current_sol.getScore(), current_sol, matrix

    def runPostOptimization(self, solution: Solution) -> Solution:
        sol = solution.copy()
        # Chạy chuỗi tối ưu hóa (giống C++)
        sol = self.runEjection(sol)
        sol = self.runInterRoute(sol)
        sol = self.runIntraRoute(sol)
        sol.refactorSolution()
        return sol

    # Các hàm static bên dưới chuyển từ C++
    @staticmethod
    def runEjection(solution: Solution) -> Solution:
        curr = solution.copy()
        while True:
            old_score = curr.getScore()
            s_new = curr.ejectionNeighborhoodAdd(curr)
            if s_new and s_new.getScore() < old_score - 1e-3:
                curr = s_new
            else: break
        return curr

    @staticmethod
    def runInterRoute(solution: Solution) -> Solution:
        curr = solution.copy()
        types = ["RELOCATE", "EXCHANGE", "OR_OPT", "TWO_OPT", "CROSS"]
        while True:
            improved = False
            old_score = curr.getScore()
            random.shuffle(types)
            for t in types:
                s_new = None
                if t == "RELOCATE": s_new = curr.relocate([], curr, "INTER")
                elif t == "EXCHANGE": s_new = curr.exchange([], curr, "INTER")
                # ... các loại khác
                if s_new and s_new.getScore() < old_score - 1e-3:
                    curr = s_new
                    improved = True
                    break
            if not improved: break
        return curr

    @staticmethod
    def runIntraRoute(solution: Solution) -> Solution:
        # Tương tự InterRoute nhưng tham số type là "INTRA"
        return solution