import math
import copy
import json
from typing import List, Optional, Tuple
import copy
import json
from typing import List, Dict, Any

class Solution:
    def __init__(self, config=None, input_obj=None, alpha1=1.0, alpha2=1.0):
        self.config = config
        self.input = input_obj
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        
        # droneTripList[drone_idx][trip_idx][cus_idx]
        num_drone = getattr(config, 'numDrone', 0)
        self.droneTripList: List[List[List[int]]] = [[] for _ in range(num_drone)]
        
        # techTripList[tech_idx][cus_idx]
        num_tech = getattr(config, 'numTech', 0)
        self.techTripList: List[List[int]] = [[] for _ in range(num_tech)]
        
        self.ext: Dict[str, Any] = {}
        self.dz = 0.0 # Vi phạm drone flight time
        self.cz = 0.0 # Vi phạm customer waiting time

    def copy(self):
        """Tạo bản sao độc lập của lời giải hiện tại."""
        new_sol = Solution(self.config, self.input, self.alpha1, self.alpha2)
        # Sao chép sâu danh sách trips để tránh lỗi tham chiếu
        new_sol.droneTripList = [[list(trip) for trip in drone] for drone in self.droneTripList]
        new_sol.techTripList = [list(tech) for tech in self.techTripList]
        new_sol.ext = copy.deepcopy(self.ext)
        new_sol.dz = self.dz
        new_sol.cz = self.cz
        return new_sol

    def getScore(self) -> float:
        """Tính toán Makespan và các Penalty vi phạm ràng buộc."""
        self.dz = 0.0
        self.cz = 0.0
        num_cus = self.input.numCus
        tech_complete = [0.0] * len(self.techTripList)
        cus_finish = [0.0] * (num_cus + 2)
        
        # 1. Tính cho Technician
        all_tech_time = 0.0
        for i, route in enumerate(self.techTripList):
            if not route: continue
            curr = self.input.techTimes[0][route[0]]
            cus_finish[route[0]] = curr
            for j in range(len(route) - 1):
                curr += self.input.techTimes[route[j]][route[j+1]]
                cus_finish[route[j+1]] = curr
            tech_complete[i] = curr + self.input.techTimes[route[-1]][0]
            all_tech_time = max(all_tech_time, tech_complete[i])

        # 2. Tính cho Drone
        all_drone_time = 0.0
        # drone_trip_finish_times[drone_idx][trip_idx]
        drone_trip_finish = []
        for i, drone in enumerate(self.droneTripList):
            drone_trip_finish.append([0.0] * len(drone))
            total_d = 0.0
            for j, trip in enumerate(drone):
                if not trip: continue
                curr = self.input.droneTimes[0][trip[0]]
                cus_finish[trip[0]] = curr
                for k in range(len(trip) - 1):
                    curr += self.input.droneTimes[trip[k]][trip[k+1]]
                    cus_finish[trip[k+1]] = curr
                drone_trip_finish[i][j] = curr + self.input.droneTimes[trip[-1]][0]
                total_d += drone_trip_finish[i][j]
            all_drone_time = max(all_drone_time, total_d)

        makespan = max(all_tech_time, all_drone_time)
        
        # 3. Tính Penalty
        # cz cho Tech
        for i, route in enumerate(self.techTripList):
            for c in route:
                self.cz += max(0.0, tech_complete[i] - cus_finish[c] - self.config.sampleLimitationWaitingTime)
        
        # cz và dz cho Drone
        for i, drone in enumerate(self.droneTripList):
            for j, trip in enumerate(drone):
                if not trip: continue
                t_finish = drone_trip_finish[i][j]
                self.dz += max(0.0, t_finish - self.config.droneLimitationFlightTime)
                for c in trip:
                    self.cz += max(0.0, t_finish - cus_finish[c] - self.config.sampleLimitationWaitingTime)

        return makespan + self.alpha1 * self.cz + self.alpha2 * self.dz
    
    def getScoreATrip(self, tripIndex: int, vehicle_type: str) -> List[float]:
        """
        Tính toán điểm số cho một phương tiện cụ thể.
        Trả về danh sách: [Thời gian hoàn thành, Vi phạm FlightTime, Vi phạm WaitingTime]
        """
        cus_complete_time = [0.0] * (self.input.numCus + 2)
        all_tech_time = 0.0
        all_drone_time = 0.0
        dzt = 0.0 # Drone violation time
        czt = 0.0 # Customer waiting violation time

        if vehicle_type == "DRONE":
            total_time = 0.0
            # drone_trip_finish_times cho từng trip của drone này
            drone_trip_finish_times = [0.0] * len(self.droneTripList[tripIndex])
            
            for j, trip in enumerate(self.droneTripList[tripIndex]):
                if not trip: continue
                
                curr_t = self.input.droneTimes[0][trip[0]]
                cus_complete_time[trip[0]] = curr_t
                
                for k in range(len(trip) - 1):
                    curr_t += self.input.droneTimes[trip[k]][trip[k+1]]
                    cus_complete_time[trip[k+1]] = curr_t
                
                finish_t = curr_t + self.input.droneTimes[trip[-1]][0]
                drone_trip_finish_times[j] = finish_t
                total_time += finish_t
                
                # Tính vi phạm cho từng trip của drone
                dzt += max(0.0, finish_t - self.config.droneLimitationFlightTime)
                for cus in trip:
                    czt += max(0.0, finish_t - cus_complete_time[cus] - self.config.sampleLimitationWaitingTime)
            
            return [total_time, dzt, czt]

        else: # vehicle_type == "TECHNICIAN"
            route = self.techTripList[tripIndex]
            if not route:
                return [0.0, 0.0, 0.0]
                
            curr = self.input.techTimes[0][route[0]]
            cus_complete_time[route[0]] = curr
            
            for j in range(len(route) - 1):
                curr += self.input.techTimes[route[j]][route[j+1]]
                cus_complete_time[route[j+1]] = curr
                
            finish_t = curr + self.input.techTimes[route[-1]][0]
            
            # Tính vi phạm cho tuyến của tech
            for cus in route:
                czt += max(0.0, finish_t - cus_complete_time[cus] - self.config.sampleLimitationWaitingTime)
            
            return [finish_t, 0.0, czt] # Tech không có giới hạn flight time (dz=0)

    def check_feasible(self) -> bool:
        self.getScore()
        return self.cz <= getattr(self.config, 'tabuEpsilon', 1e-3) and self.dz <= getattr(self.config, 'tabuEpsilon', 1e-3)

    def toString(self):
        return f"{json.dumps(self.droneTripList)}::{json.dumps(self.techTripList)}"

    # --- Initialization Methods ---
    @staticmethod
    def initSolution(config, input_obj, init_type, alpha1, alpha2):
        best = None
        best_score = float('inf')
        
        # Logic for ANGLE/DISTANCE
        for reverse in [False, True]:
            curr = Solution(config, input_obj, alpha1, alpha2)
            curr.initByDistance(reverse)
            score = curr.getScore()
            if score < best_score:
                best = curr
                best_score = score
        return best

    def initByDistance(self, reverse: bool):
        order_dist = []
        for i in range(self.input.numCus + 1):
            dists = self.input.distances[i]
            indices = sorted([idx for idx in range(len(dists)) if idx != i and idx <= self.input.numCus], 
                             key=lambda k: dists[k], reverse=reverse)
            order_dist.append(indices)

        visited = [False] * (self.input.numCus + 1)
        visited[0], num_v, i = True, 0, 0
        total_v = self.config.numDrone + self.config.numTech
        travel_time = [0.0] * total_v

        while num_v < self.input.numCus:
            if i < self.config.numDrone:
                if not self.droneTripList[i]: self.droneTripList[i].append([])
                last = self.droneTripList[i][-1][-1] if self.droneTripList[i][-1] else 0
                next_c = next((c for c in order_dist[last] if not visited[c] and not self.input.cusOnlyServedByTech[c]), -1)
                if next_c > 0:
                    # Trip breaking logic
                    self.droneTripList[i][-1].append(next_c)
                    visited[next_c] = True
                    num_v += 1
            else:
                idx = i - self.config.numDrone
                last = self.techTripList[idx][-1] if self.techTripList[idx] else 0
                next_c = next((c for c in order_dist[last] if not visited[c]), -1)
                if next_c > 0:
                    self.techTripList[idx].append(next_c)
                    visited[next_c] = True
                    num_v += 1
            i = (i + 1) % total_v

    # --- Heuristic Moves ---
    def checkTabuCondition(self, tabuList, *vals) -> bool:
        if not tabuList: return True
        s = "-".join(map(str, vals))
        s_rev = "-".join(map(str, reversed(vals)))
        return s not in tabuList and s_rev not in tabuList

    def relocate(self, tabuList: List[str], bestFeasible: 'Solution', route_type: str):
        best_sol = self.copy()
        best_f_score = bestFeasible.getScore()
        current_best_score = float('inf')
        is_improved = False
        eps = getattr(self.config, 'tabuEpsilon', 1e-3)

        def try_update(new_s: 'Solution', customer_id: int):
            nonlocal is_improved, current_best_score, best_sol
            new_score = new_s.getScore()
            cust_str = str(customer_id)
            if new_s.check_feasible() and (new_score - best_f_score < eps):
                is_improved = True
                bestFeasible.droneTripList = [[list(t) for t in d] for d in new_s.droneTripList]
                bestFeasible.techTripList = [list(t) for t in new_s.techTripList]
                best_sol = new_s.copy()
                best_sol.ext["state"] = cust_str
                current_best_score = new_score
            elif not is_improved and (new_score - current_best_score < eps):
                if self.checkTabuCondition(tabuList, cust_str):
                    best_sol = new_s.copy()
                    best_sol.ext["state"] = cust_str
                    current_best_score = new_score

        # --- 1. DI CHUYỂN TỪ DRONE TRIP ---
        for d1, drone in enumerate(self.droneTripList):
            for t1, trip in enumerate(drone):
                # Tạo bản sao danh sách khách hàng để tránh lỗi index khi loop
                customers_in_trip = list(trip) 
                for x1, customer in enumerate(customers_in_trip):
                    # Thử chèn vào Drone khác
                    for d2 in range(len(self.droneTripList)):
                        for t2 in range(len(self.droneTripList[d2])):
                            for y_idx in range(len(self.droneTripList[d2][t2]) + 1):
                                if d1 == d2 and t1 == t2 and (y_idx == x1 or y_idx == x1 + 1): continue
                                s = self.copy()
                                if len(s.droneTripList[d1][t1]) > x1: # Kiểm tra an toàn
                                    cus = s.droneTripList[d1][t1].pop(x1)
                                    s.droneTripList[d2][t2].insert(y_idx, cus)
                                    try_update(s, customer)

                    # Thử chèn vào Tech
                    for te2 in range(len(self.techTripList)):
                        for y_idx in range(len(self.techTripList[te2]) + 1):
                            s = self.copy()
                            if len(s.droneTripList[d1][t1]) > x1: # Kiểm tra an toàn
                                cus = s.droneTripList[d1][t1].pop(x1)
                                s.techTripList[te2].insert(y_idx, cus)
                                try_update(s, customer)

        # --- 2. DI CHUYỂN TỪ TECH TRIP ---
        for te1, route in enumerate(self.techTripList):
            # Tạo bản sao để duyệt an toàn
            customers_in_route = list(route)
            for x1, customer in enumerate(customers_in_route):
                
                # A. Chèn vào Tech khác
                for te2 in range(len(self.techTripList)):
                    for y_idx in range(len(self.techTripList[te2]) + 1):
                        if te1 == te2 and (y_idx == x1 or y_idx == x1 + 1): continue
                        
                        s = self.copy()
                        if len(s.techTripList[te1]) > x1: # SỬA LỖI POP TẠI ĐÂY
                            cus = s.techTripList[te1].pop(x1)
                            s.techTripList[te2].insert(y_idx, cus)
                            try_update(s, customer)

                # B. Chèn vào Drone Trip
                if not self.input.cusOnlyServedByTech[customer]:
                    for d2 in range(len(self.droneTripList)):
                        for t2 in range(len(self.droneTripList[d2])):
                            for y_idx in range(len(self.droneTripList[d2][t2]) + 1):
                                s = self.copy()
                                if len(s.techTripList[te1]) > x1:
                                    cus = s.techTripList[te1].pop(x1)
                                    s.droneTripList[d2][t2].insert(y_idx, cus)
                                    try_update(s, customer)

        if best_sol.getScore() > self.getScore() + eps: return self
        return best_sol


    def exchange(self, tabuList: List[str], bestFeasible: 'Solution', route_type: str):
            best_sol = self.copy()
            best_f_score = bestFeasible.getScore()
            current_best_score = float('inf')
            is_improved = False
            eps = getattr(self.config, 'tabuEpsilon', 1e-9)

            # HÀM TRỢ GIÚP: Cập nhật kết quả tốt nhất dựa trên logic hoán đổi
            def try_update_swap(new_s: 'Solution', cus_i: int, cus_j: int):
                nonlocal is_improved, current_best_score, best_sol
                new_score = new_s.getScore()
                
                # Trạng thái Tabu cho exchange thường là cặp ID khách hàng hoán đổi
                val1, val2 = str(cus_i), str(cus_j)

                # Logic Method 1: Aspiration (Tốt hơn cả best feasible hiện tại)
                if new_s.check_feasible() and (new_score - best_f_score < eps):
                    is_improved = True
                    # Cập nhật kết quả khả thi tốt nhất toàn cục
                    bestFeasible.droneTripList = [[list(t) for t in d] for d in new_s.droneTripList]
                    bestFeasible.techTripList = [list(t) for t in new_s.techTripList]
                    
                    best_sol = new_s.copy()
                    best_sol.ext["state"] = f"{val1}-{val2}"
                    current_best_score = new_score
                    return True

                # Logic Method 2: Check Tabu condition
                elif not is_improved and (new_score - current_best_score < eps):
                    if self.checkTabuCondition(tabuList, val1, val2):
                        best_sol = new_s.copy()
                        best_sol.ext["state"] = f"{val1}-{val2}"
                        current_best_score = new_score
                return False

            # --- 1. SWAP DRONE VỚI DRONE ---
            for d1 in range(len(self.droneTripList)):
                for t1 in range(len(self.droneTripList[d1])):
                    for x1 in range(len(self.droneTripList[d1][t1])):
                        
                        # Duyệt các drone trip khác (hoặc cùng drone khác trip/vị trí)
                        for d2 in range(d1, len(self.droneTripList)):
                            for t2 in range(len(self.droneTripList[d2])):
                                # Nếu cùng drone trip, chỉ swap với các khách hàng phía sau để tránh lặp
                                start_y = (x1 + 1) if (d1 == d2 and t1 == t2) else 0
                                for y1 in range(start_y, len(self.droneTripList[d2][t2])):
                                    
                                    s = self.copy()
                                    # Hoán đổi giá trị
                                    val_i = s.droneTripList[d1][t1][x1]
                                    val_j = s.droneTripList[d2][t2][y1]
                                    s.droneTripList[d1][t1][x1], s.droneTripList[d2][t2][y1] = val_j, val_i
                                    try_update_swap(s, val_i, val_j)

            # --- 2. SWAP DRONE VỚI TECH ---
            for d1 in range(len(self.droneTripList)):
                for t1 in range(len(self.droneTripList[d1])):
                    for x1 in range(len(self.droneTripList[d1][t1])):
                        for te2 in range(len(self.techTripList)):
                            for y1 in range(len(self.techTripList[te2])):
                                
                                val_drone = self.droneTripList[d1][t1][x1]
                                val_tech = self.techTripList[te2][y1]
                                
                                # Drone chỉ nhận được khách hàng không bị giới hạn "Only Tech"
                                if not self.input.cusOnlyServedByTech[val_tech]:
                                    s = self.copy()
                                    s.droneTripList[d1][t1][x1] = val_tech
                                    s.techTripList[te2][y1] = val_drone
                                    try_update_swap(s, val_drone, val_tech)

            # --- 3. SWAP TECH VỚI TECH ---
            for te1 in range(len(self.techTripList)):
                for x1 in range(len(self.techTripList[te1])):
                    for te2 in range(te1, len(self.techTripList)):
                        start_y = (x1 + 1) if (te1 == te2) else 0
                        for y1 in range(start_y, len(self.techTripList[te2])):
                            
                            s = self.copy()
                            val_i = s.techTripList[te1][x1]
                            val_j = s.techTripList[te2][y1]
                            s.techTripList[te1][x1], s.techTripList[te2][y1] = val_j, val_i
                            try_update_swap(s, val_i, val_j)

            # Nếu không tìm được hàng xóm nào tốt hơn giải pháp hiện tại
            if best_sol.getScore() > self.getScore() + eps:
                return self

            return best_sol

    def twoOpt(self, tabuList: List[str], bestFeasible: 'Solution', route_type: str):
        best_sol = self.copy()
        best_f_score = bestFeasible.getScore()
        current_best_score = float('inf')
        is_improved = False
        eps = getattr(self.config, 'tabuEpsilon', 1e-9)

        # Hàm trợ giúp cập nhật kết quả theo cơ chế Tabu
        def try_update_2opt(new_s: 'Solution', val1: int, val2: int):
            nonlocal is_improved, current_best_score, best_sol
            new_score = new_s.getScore()
            s_val1, s_val2 = str(val1), str(val2)

            # Logic Method 1: Aspiration (Tốt hơn best feasible hiện tại)
            if new_s.check_feasible() and (new_score - best_f_score < eps):
                is_improved = True
                bestFeasible.droneTripList = [[list(t) for t in d] for d in new_s.droneTripList]
                bestFeasible.techTripList = [list(t) for t in new_s.techTripList]
                
                best_sol = new_s.copy()
                best_sol.ext["state"] = f"{s_val1}-{s_val2}"
                current_best_score = new_score
                return True

            # Logic Method 2: Check Tabu
            elif not is_improved and (new_score - current_best_score < eps):
                if self.checkTabuCondition(tabuList, s_val1, s_val2):
                    best_sol = new_s.copy()
                    best_sol.ext["state"] = f"{s_val1}-{s_val2}"
                    current_best_score = new_score
            return False

        # --- 1. TWO-OPT CHO DRONE (Inter & Intra Drone Trips) ---
        for d1 in range(len(self.droneTripList)):
            for t1 in range(len(self.droneTripList[d1])):
                # xIndex từ -1 để bao gồm cả việc đảo từ Depot (0)
                for x_idx in range(-1, len(self.droneTripList[d1][t1])):
                    
                    # A. Giữa Drone và Tech (Liên phương tiện)
                    for te2 in range(len(self.techTripList)):
                        for y_idx in range(-1, len(self.techTripList[te2])):
                            s = self.copy()
                            # Tách và nối lộ trình (Logic 2-Opt inter-route)
                            route_a = s.droneTripList[d1][t1]
                            route_b = s.techTripList[te2]
                            
                            # Tạo lộ trình mới bằng cách tráo đổi phần đuôi
                            new_a = route_a[:x_idx+1] + route_b[y_idx+1:]
                            new_b = route_b[:y_idx+1] + route_a[x_idx+1:]
                            
                            # Drone chỉ nhận các khách hàng mà nó có thể phục vụ
                            if all(not self.input.cusOnlyServedByTech[c] for c in new_a):
                                s.droneTripList[d1][t1] = new_a
                                s.techTripList[te2] = new_b
                                val_x = route_a[x_idx] if x_idx >= 0 else 0
                                val_y = route_b[y_idx] if y_idx >= 0 else 0
                                try_update_2opt(s, val_x, val_y)

                    # B. Giữa Drone và Drone (Trong nội bộ nhóm Drone)
                    for d2 in range(d1, len(self.droneTripList)):
                        for t2 in range(len(self.droneTripList[d2])):
                            if d1 == d2 and t1 == t2:
                                # Intra-route 2-Opt (Đảo ngược một đoạn trong cùng 1 trip)
                                for i in range(len(self.droneTripList[d1][t1])):
                                    for j in range(i + 1, len(self.droneTripList[d1][t1])):
                                        s = self.copy()
                                        seg = s.droneTripList[d1][t1][i:j+1]
                                        s.droneTripList[d1][t1][i:j+1] = seg[::-1] # Đảo ngược đoạn
                                        try_update_2opt(s, s.droneTripList[d1][t1][i], s.droneTripList[d1][t1][j])
                            else:
                                # Inter-route 2-Opt giữa 2 Drone Trips
                                s = self.copy()
                                r1, r2 = s.droneTripList[d1][t1], s.droneTripList[d2][t2]
                                new_r1 = r1[:x_idx+1] + r2[y_idx+1:]
                                new_r2 = r2[:y_idx+1] + r1[x_idx+1:]
                                s.droneTripList[d1][t1], s.droneTripList[d2][t2] = new_r1, new_r2
                                try_update_2opt(s, r1[x_idx] if x_idx >= 0 else 0, r2[y_idx] if y_idx >= 0 else 0)

        # --- 2. TWO-OPT CHO TECH (Intra Tech Routes) ---
        for te1 in range(len(self.techTripList)):
            for i in range(len(self.techTripList[te1])):
                for j in range(i + 1, len(self.techTripList[te1])):
                    s = self.copy()
                    seg = s.techTripList[te1][i:j+1]
                    s.techTripList[te1][i:j+1] = seg[::-1]
                    try_update_2opt(s, s.techTripList[te1][i], s.techTripList[te1][j])

        # Trả về kết quả tốt nhất tìm được hoặc bản thân nếu không có cải thiện
        if best_sol.getScore() > self.getScore() + eps:
            return self
        return best_sol
    
    def orOpt(self, tabuList: List[str], bestFeasible: 'Solution', route_type: str, dis: int):
        best_sol = self.copy()
        best_f_score = bestFeasible.getScore()
        current_best_score = float('inf')
        is_improved = False
        eps = getattr(self.config, 'tabuEpsilon', 1e-3)

        def try_update_or(new_s: 'Solution', v1: int, v2: int):
            nonlocal is_improved, current_best_score, best_sol
            new_score = new_s.getScore()
            s1, s2 = str(v1), str(v2)
            if new_s.check_feasible() and (new_score - best_f_score < eps):
                is_improved = True
                bestFeasible.droneTripList = [[list(t) for t in d] for d in new_s.droneTripList]
                bestFeasible.techTripList = [list(t) for t in new_s.techTripList]
                best_sol = new_s.copy()
                best_sol.ext["state"] = f"{s1}-{s2}"
                current_best_score = new_score
            elif not is_improved and (new_score - current_best_score < eps):
                if self.checkTabuCondition(tabuList, s1, s2):
                    best_sol = new_s.copy()
                    best_sol.ext["state"] = f"{s1}-{s2}"
                    current_best_score = new_score

        # Logic Or-Opt: Di chuyển một đoạn (dis + 1 khách hàng)
        # Di chuyển từ Drone
        for d1 in range(len(self.droneTripList)):
            for t1 in range(len(self.droneTripList[d1])):
                if len(self.droneTripList[d1][t1]) <= dis: continue
                for x_idx in range(len(self.droneTripList[d1][t1]) - dis):
                    # Tách đoạn
                    s_base = self.copy()
                    segment = [s_base.droneTripList[d1][t1].pop(x_idx) for _ in range(dis + 1)]
                    
                    # Thử chèn vào Drone khác
                    for d2 in range(len(self.droneTripList)):
                        for t2 in range(len(self.droneTripList[d2])):
                            for y_idx in range(len(s_base.droneTripList[d2][t2]) + 1):
                                s = s_base.copy()
                                for i, val in enumerate(segment):
                                    s.droneTripList[d2][t2].insert(y_idx + i, val)
                                try_update_or(s, segment[0], segment[-1])
                    
                    # Thử chèn vào Tech
                    for te2 in range(len(self.techTripList)):
                        for y_idx in range(len(s_base.techTripList[te2]) + 1):
                            s = s_base.copy()
                            for i, val in enumerate(segment):
                                s.techTripList[te2].insert(y_idx + i, val)
                            try_update_or(s, segment[0], segment[-1])
        return best_sol

    def crossExchange(self, tabuList: List[str], bestFeasible: 'Solution', route_type: str, dis1: int, dis2: int):
        best_sol = self.copy()
        best_f_score = bestFeasible.getScore()
        current_best_score = float('inf')
        is_improved = False
        eps = getattr(self.config, 'tabuEpsilon', 1e-3)

        # Hàm trợ giúp cập nhật kết quả theo logic Tabu & Aspiration
        def try_update_cross(new_s: 'Solution', v1_start: int, v1_end: int, v2_start: int, v2_end: int):
            nonlocal is_improved, current_best_score, best_sol
            new_score = new_s.getScore()
            # Mã hóa trạng thái Tabu bằng các node biên của 2 đoạn hoán đổi
            state = f"{v1_start}-{v1_end}-{v2_start}-{v2_end}"

            if new_s.check_feasible() and (new_score - best_f_score < eps):
                is_improved = True
                bestFeasible.droneTripList = [[list(t) for t in d] for d in new_s.droneTripList]
                bestFeasible.techTripList = [list(t) for t in new_s.techTripList]
                best_sol = new_s.copy()
                best_sol.ext["state"] = state
                current_best_score = new_score
            elif not is_improved and (new_score - current_best_score < eps):
                if self.checkTabuCondition(tabuList, state):
                    best_sol = new_s.copy()
                    best_sol.ext["state"] = state
                    current_best_score = new_score

        # --- 1. CROSS-EXCHANGE GIỮA DRONE VÀ TECH ---
        for d_idx, drone in enumerate(self.droneTripList):
            for t_idx, trip in enumerate(drone):
                # Kiểm tra độ dài trip có đủ để lấy đoạn dis1 không
                if len(trip) <= dis1: continue
                for x in range(len(trip) - dis1):
                    for te_idx, route in enumerate(self.techTripList):
                        if len(route) <= dis2: continue
                        for y in range(len(route) - dis2):
                            # Tạo bản sao và trích xuất 2 đoạn segment
                            s = self.copy()
                            seg1 = trip[x : x + dis1 + 1]
                            seg2 = route[y : y + dis2 + 1]

                            # Drone chỉ nhận segment từ Tech nếu Drone bay được tất cả các node đó
                            if all(not self.input.cusOnlyServedByTech[c] for c in seg2):
                                # Thay thế đoạn trong Drone bằng seg2
                                s.droneTripList[d_idx][t_idx][x : x + dis1 + 1] = seg2
                                # Thay thế đoạn trong Tech bằng seg1
                                s.techTripList[te_idx][y : y + dis2 + 1] = seg1
                                try_update_cross(s, seg1[0], seg1[-1], seg2[0], seg2[-1])

        # --- 2. CROSS-EXCHANGE GIỮA DRONE VÀ DRONE ---
        for d1 in range(len(self.droneTripList)):
            for t1 in range(len(self.droneTripList[d1])):
                if len(self.droneTripList[d1][t1]) <= dis1: continue
                for x in range(len(self.droneTripList[d1][t1]) - dis1):
                    for d2 in range(d1, len(self.droneTripList)):
                        for t2 in range(len(self.droneTripList[d2])):
                            if d1 == d2 and t1 == t2: continue # Chỉ xét liên tuyến (inter-route)
                            if len(self.droneTripList[d2][t2]) <= dis2: continue
                            
                            for y in range(len(self.droneTripList[d2][t2]) - dis2):
                                s = self.copy()
                                seg1 = self.droneTripList[d1][t1][x : x + dis1 + 1]
                                seg2 = self.droneTripList[d2][t2][y : y + dis2 + 1]
                                
                                s.droneTripList[d1][t1][x : x + dis1 + 1] = seg2
                                s.droneTripList[d2][t2][y : y + dis2 + 1] = seg1
                                try_update_cross(s, seg1[0], seg1[-1], seg2[0], seg2[-1])

        # --- 3. CROSS-EXCHANGE GIỮA TECH VÀ TECH ---
        for te1 in range(len(self.techTripList)):
            if len(self.techTripList[te1]) <= dis1: continue
            for x in range(len(self.techTripList[te1]) - dis1):
                for te2 in range(te1 + 1, len(self.techTripList)):
                    if len(self.techTripList[te2]) <= dis2: continue
                    for y in range(len(self.techTripList[te2]) - dis2):
                        s = self.copy()
                        seg1 = self.techTripList[te1][x : x + dis1 + 1]
                        seg2 = self.techTripList[te2][y : y + dis2 + 1]
                        
                        s.techTripList[te1][x : x + dis1 + 1] = seg2
                        s.techTripList[te2][y : y + dis2 + 1] = seg1
                        try_update_cross(s, seg1[0], seg1[-1], seg2[0], seg2[-1])

        return best_sol
    

    def refactorSolution(self):
        """Loại bỏ các chuyến đi rỗng trong danh sách droneTripList."""
        for i in range(len(self.droneTripList)):
            # Chỉ giữ lại các trip có ít nhất 1 khách hàng
            self.droneTripList[i] = [trip for trip in self.droneTripList[i] if len(trip) > 0]
            # Đảm bảo drone luôn có ít nhất 1 trip trống nếu danh sách hoàn toàn rỗng
            if not self.droneTripList[i]:
                self.droneTripList[i].append([])

    def ejectionNeighborhoodAdd(self, bestFeasible: 'Solution'):
        best_sol = self.copy()
        best_f_score = bestFeasible.getScore()
        
        # Các biến lưu vết cho quá trình đệ quy
        self.bestShiftSequence = []
        self.bestGain = 0.0
        
        # 1. Thử trục xuất từ Drone Trips
        for d_idx in range(len(self.droneTripList)):
            for t_idx in range(len(self.droneTripList[d_idx])):
                for x_idx in range(len(self.droneTripList[d_idx][t_idx])):
                    s = self.copy()
                    self.ejection(s, [d_idx, t_idx, x_idx], "DRONE", [], 0.0, 0)
                    
                    new_score = s.getScore()
                    if s.check_feasible() and (new_score < best_f_score - getattr(self.config, 'tabuEpsilon', 1e-3)):
                        bestFeasible.droneTripList = [[list(t) for t in drone] for drone in s.droneTripList]
                        bestFeasible.techTripList = [list(tech) for tech in s.techTripList]
                        best_f_score = new_score

        # 2. Thử trục xuất từ Tech Trips
        for te_idx in range(len(self.techTripList)):
            for x_idx in range(len(self.techTripList[te_idx])):
                s = self.copy()
                self.ejection(s, [te_idx, x_idx], "TECHNICIAN", [], 0.0, 0)
                
                new_score = s.getScore()
                if s.check_feasible() and (new_score < best_f_score - getattr(self.config, 'tabuEpsilon', 1e-3)):
                    bestFeasible.droneTripList = [[list(t) for t in drone] for drone in s.droneTripList]
                    bestFeasible.techTripList = [list(tech) for tech in s.techTripList]
                    best_f_score = new_score

        # Cập nhật kết quả cuối cùng từ best feasible
        best_sol.droneTripList = [[list(t) for t in drone] for drone in bestFeasible.droneTripList]
        best_sol.techTripList = [list(tech) for tech in bestFeasible.techTripList]
        
        return best_sol

    def ejection(self, solution, xIndex, vehicle_type, customerX, gain, level):
        """Hàm đệ quy thực hiện chuỗi trục xuất (Ejection Chain)."""
        if level >= getattr(self.config, 'maxEjectionLevel', 2):
            return

        # Lấy Makespan hiện tại của các xe để xác định xe "tệ nhất"
        drone_scores = [solution.getScoreATrip(i, "DRONE")[0] for i in range(len(solution.droneTripList))]
        tech_scores = [solution.getScoreATrip(i, "TECHNICIAN")[0] for i in range(len(solution.techTripList))]
        
        max_drone = max(drone_scores) if drone_scores else 0
        max_tech = max(tech_scores) if tech_scores else 0
        f_score = max(max_drone, max_tech)

        # Chỉ thực hiện trục xuất trên phương tiện đang chiếm Makespan (đỉnh nhọn của lộ trình)
        if vehicle_type == "DRONE" and drone_scores[xIndex[0]] < max_drone: return
        if vehicle_type == "TECHNICIAN" and tech_scores[xIndex[0]] < max_tech: return

        # Thực hiện trục xuất customer x ra khỏi lộ trình hiện tại
        if vehicle_type == "DRONE":
            x = solution.droneTripList[xIndex[0]][xIndex[1]].pop(xIndex[2])
            # Tính toán Gain (độ giảm thời gian) sau khi rút x
            # (Giả định getScoreATrip cập nhật lại thời gian thực tế)
            new_max_drone = max([solution.getScoreATrip(i, "DRONE")[0] for i in range(len(solution.droneTripList))], default=0)
            c_score = max(max_tech, new_max_drone)
            g = f_score - c_score
            gain += g

            # Thử chèn x vào các Drone Trips khác
            for d2 in range(len(solution.droneTripList)):
                for t2 in range(len(solution.droneTripList[d2])):
                    if d2 == xIndex[0] and t2 == xIndex[1]: continue
                    for cus_idx in range(len(solution.droneTripList[d2][t2]) + 1):
                        s_temp = solution.copy()
                        s_temp.droneTripList[d2][t2].insert(cus_idx, x)
                        
                        score_trip = s_temp.getScoreATrip(d2, "DRONE")
                        # Nếu chèn vào mà không vi phạm (cz, dz == 0)
                        if score_trip[1] == 0 and score_trip[2] == 0:
                            if gain > self.bestGain:
                                self.bestGain = gain
                                # Lưu lại trạng thái tốt nhất
                        elif level + 1 < self.config.maxEjectionLevel:
                            # Đệ quy: Trục xuất tiếp một khách hàng ở xe vừa bị chèn vào
                            for next_x_idx in range(len(s_temp.droneTripList[d2][t2])):
                                self.ejection(s_temp, [d2, t2, next_x_idx], "DRONE", customerX, gain, level + 1)
            
            # Khôi phục lại x để thử các nhánh khác (Backtracking)
            solution.droneTripList[xIndex[0]][xIndex[1]].insert(xIndex[2], x)

        else: # Tương tự cho TECHNICIAN
            x = solution.techTripList[xIndex[0]].pop(xIndex[1])
            # ... Logic tương tự Drone nhưng chèn vào Tech ...
            solution.techTripList[xIndex[0]].insert(xIndex[1], x)

    def toString(self):
        return f"{json.dumps(self.droneTripList)}::{json.dumps(self.techTripList)}"