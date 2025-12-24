import math
import copy
from typing import List, Tuple, Dict, Any
from .solution import Solution
from .tabu import TabuSearch


class MultiLevel:
    def __init__(self, config, input_obj):
        self.config = config
        self.input = input_obj
        self.mapLevel = []
        self.distanceMatrixLevel = []
        self.c1Level = []

    def convertMatrix(
        self, currentMatrix, matrixReBefore, mapping: Dict[int, List[int]]
    ):
        size = len(mapping)  # mapping bao gồm cả node 0 (depot)
        # Khởi tạo ma trận mới kích thước (size+1) x (size+1) để dành chỗ cho Depot cuối nếu cần
        newMatrix = [[0.0] * (size + 1) for _ in range(size + 1)]
        matrixRe = [[0.0] * (size + 1) for _ in range(size + 1)]

        for i in range(size):
            for j in range(size):
                # Lấy danh sách các node gốc trong cụm i và cụm j
                nodes_i = mapping[i]
                nodes_j = mapping[j]

                # MatrixRe: khoảng cách thực tế giữa node cuối của cụm i và node đầu của cụm j
                matrixRe[i][j] = matrixReBefore[nodes_i[-1]][nodes_j[0]]

                # newMatrix: Tổng chi phí di chuyển xuyên suốt các node trong cụm i
                # cộng với chi phí kết nối sang cụm j
                dist_sum = 0.0
                combined = nodes_i + nodes_j
                for k in range(len(combined) - 1):
                    dist_sum += currentMatrix[combined[k]][combined[k + 1]]
                newMatrix[i][j] = dist_sum

        # Logic gán Depot cuối trùng với Depot đầu (numCus + 1)
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
        # 1. Thu thập các cạnh đang tồn tại trong giải pháp hiện tại
        edgeSol = set()
        for drone in solution.droneTripList:
            for trip in drone:
                for k in range(len(trip) - 1):
                    edgeSol.add((trip[k], trip[k + 1]))

        for tech in solution.techTripList:
            for k in range(len(tech) - 1):
                edgeSol.add((tech[k], tech[k + 1]))

        # 2. Chọn k% cạnh ngắn nhất để thực hiện gộp (Matching)
        numUpdate = int(numCus * self.config.percent_select)
        update_pairs = []
        used_nodes = set()

        # Tìm các cặp (i, j) có MatrixRe nhỏ nhất mà không vi phạm ràng buộc
        all_possible_pairs = []
        for i in range(1, numCus + 1):
            for j in range(1, numCus + 1):
                if i == j:
                    continue
                # Ràng buộc: cùng loại dịch vụ (C1) và không nằm trong Solution hiện tại
                if c1[i] == c1[j] and (i, j) not in edgeSol:
                    all_possible_pairs.append((i, j, matrixRe[i][j]))

        all_possible_pairs.sort(key=lambda x: x[2])  # Sắp xếp theo khoảng cách tăng dần

        for i, j, dist in all_possible_pairs:
            if len(update_pairs) >= numUpdate:
                break
            if i not in used_nodes and j not in used_nodes:
                update_pairs.append((i, j))
                used_nodes.add(i)
                used_nodes.add(j)

        # 3. Gom cụm (BeMerge) và tạo Mapping
        # Logic re-indexing phức tạp từ C++: gom các node liên tiếp thành siêu node
        new_mapping = {0: [0]}
        # Giả sử mỗi cặp trong update_pairs trở thành 1 cụm mới
        new_nodes_list = []
        nodes_in_pairs = set()
        for u in update_pairs:
            new_nodes_list.append([u[0], u[1]])
            nodes_in_pairs.add(u[0])
            nodes_in_pairs.add(u[1])

        # Những node không được gộp thì đứng một mình
        for i in range(1, numCus + 1):
            if i not in nodes_in_pairs:
                new_nodes_list.append([i])

        # Đánh số lại siêu node từ 1 đến N_mới
        res_mapping = {0: [0]}
        for idx, cluster in enumerate(new_nodes_list):
            res_mapping[idx + 1] = cluster

        # 4. Cập nhật Solution với các chỉ số siêu node mới
        new_sol = solution.copy()
        # Hàm map ngược: node cũ -> siêu node mới
        reverse_map = {}
        for super_node, members in res_mapping.items():
            for m in members:
                reverse_map[m] = super_node

        # Thay thế ID cũ bằng ID siêu node trong Drone và Tech
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
        score_history = []

        tabu_search = TabuSearch(self.config, self.input)
        current_sol = tabu_search.initSolution  # Lấy giải pháp khởi tạo từ Tabu

        curr_dist_matrix = copy.deepcopy(self.input.distances)
        curr_matrix_re = copy.deepcopy(curr_dist_matrix)
        curr_c1 = copy.deepcopy(self.input.cusOnlyServedByTech)

        dist_matrix_levels.append(curr_dist_matrix)
        c1_levels.append(curr_c1)

        curr_num_cus = self.input.numCus

        for level in range(self.config.num_level):
            # 1. Chạy Tabu Search để tối ưu Level hiện tại
            tabu = TabuSearch(self.config, self.input)
            assert current_sol, ""
            _, optimized_sol, _ = tabu.run(current_sol)

            # 2. Merge các node lại
            num_merged, merged_sol, mapping = self.mergeSol(
                optimized_sol, curr_num_cus, curr_matrix_re, curr_c1
            )

            if num_merged == 0:
                break  # Không còn gì để gộp

            map_levels.append(mapping)

            # 3. Tạo ma trận mới cho level tiếp theo
            curr_dist_matrix, curr_matrix_re = self.convertMatrix(
                curr_dist_matrix, curr_matrix_re, mapping
            )
            dist_matrix_levels.append(curr_dist_matrix)

            # 4. Cập nhật ràng buộc C1 cho siêu node (siêu node bị cấm drone nếu có bất kỳ thành viên nào bị cấm)
            new_c1 = [False] * (len(mapping) + 1)
            for super_id, members in mapping.items():
                if any(curr_c1[m] for m in members):
                    new_c1[super_id] = True
            c1_levels.append(new_c1)

            # 5. Cập nhật input giả để Tabu chạy được ở vòng lặp sau
            curr_num_cus = len(mapping) - 1
            self.input.numCus = curr_num_cus
            self.input.distances = curr_dist_matrix
            self.input.cusOnlyServedByTech = new_c1
            # Cập nhật droneTimes/techTimes dựa trên vận tốc
            self.input.droneTimes = [
                [d / self.config.droneVelocity for d in row] for row in curr_dist_matrix
            ]
            self.input.techTimes = [
                [d / self.config.techVelocity for d in row] for row in curr_dist_matrix
            ]

            current_sol = merged_sol
            score_history.append(current_sol.getScore())

        return current_sol, map_levels, dist_matrix_levels, c1_levels

    def splitProcess(self, solution: Solution, map_levels, dist_levels, c1_levels):
        curr_sol = solution
        # Đi ngược từ level cao nhất về level 0
        for i in range(len(map_levels) - 1, -1, -1):
            mapping = map_levels[i]
            # 1. Tách các siêu node thành các node gốc
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

            # 2. Khôi phục dữ liệu Input của level thấp hơn
            self.input.distances = dist_levels[i]
            self.input.cusOnlyServedByTech = c1_levels[i]
            self.input.numCus = len(dist_levels[i]) - 2
            self.input.droneTimes = [
                [d / self.config.droneVelocity for d in row] for row in dist_levels[i]
            ]
            self.input.techTimes = [
                [d / self.config.techVelocity for d in row] for row in dist_levels[i]
            ]

            # 3. Chạy Tabu Search để tinh chỉnh lộ trình sau khi tách
            tabu = TabuSearch(self.config, self.input)
            _, curr_sol, _ = tabu.run(curr_sol)

        return curr_sol

    def run(self):
        # Sơ đồ: Merge -> Tabu -> Split -> Tabu
        print("Starting Multi-Level Process...")
        # Giai đoạn Coarsening (Merge)
        merged_sol, map_lv, dist_lv, c1_lv = self.mergeProcess()

        assert merged_sol, ""
        # Giai đoạn Refinement (Split)
        final_sol = self.splitProcess(merged_sol, map_lv, dist_lv, c1_lv)

        return final_sol.getScore(), final_sol

