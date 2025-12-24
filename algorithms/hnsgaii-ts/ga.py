import random
from typing import List
import parameters as params
from fitness import calculate_fitness, TRUCK_HOUR
from route_check import feasible_drone_route
from structs import Individual, NodeDistance, DistanceRanking
from repair import repair_route
# --- CÁC HÀM TIỆN ÍCH CƠ BẢN ---


def compute_linear_1(obj1: float, obj2: float) -> float:
    """Tính toán hàm mục tiêu tuyến tính."""
    return (obj1 - params.min_obj1) / params.obj1_norm + (
        obj2 - params.min_obj2
    ) / params.obj2_norm


def cal_truck(cur: int, des: int, time_use: float) -> float:
    """Tính toán thời gian di chuyển của xe tải dựa trên vận tốc thay đổi theo giờ."""
    distance = params.M[cur][des]
    hour = int(time_use / 3600)

    while (
        distance
        - ((hour + 1) * 3600 - time_use) * params.V_MAX_TRUCK * TRUCK_HOUR[hour % 12]
        > 0
    ):
        distance -= (
            ((hour + 1) * 3600 - time_use) * params.V_MAX_TRUCK * TRUCK_HOUR[hour % 12]
        )
        time_use = (hour + 1) * 3600
        hour = int(time_use / 3600)

    time_use += (
        distance / (params.V_MAX_TRUCK * TRUCK_HOUR[hour % 12])
        + params.TRUCK_SERVICE_TIME
    )
    return time_use


def cal_drone(cur: int, des: int) -> float:
    """Tính toán thời gian di chuyển của drone."""
    distance = params.M[cur][des]
    time_val = distance / params.CRUISE_SPEED
    return (
        params.TAKEOFF_TIME + params.LANDING_TIME + time_val
    ) + params.DRONE_SERVICE_TIME


# --- LOGIC XẾP HẠNG KHOẢNG CÁCH ---


def find_rank(cur: int, target: int, temp: List[int]) -> float:
    """Tìm thứ hạng khoảng cách của node target so với node cur."""
    nds = []
    for i in range(len(temp)):
        if i != cur:
            nd = NodeDistance(node=temp[i], distance=params.M[temp[i]][temp[cur]])
            nds.append(nd)

    # Sắp xếp nds theo distance tăng dần (sử dụng logic compareNodeDistance)
    nds.sort(key=lambda x: x.distance)

    found = 0
    for i in range(len(nds)):
        if nds[i].node == temp[target]:
            found = i
            break
    return float(found)


def distance_ranking(cur: int, remaining: List[int]) -> List[int]:
    """Xếp hạng các node còn lại dựa trên trung bình cộng thứ hạng hai chiều."""
    drs = []
    temp = remaining + [cur]
    cur_idx_in_temp = len(temp) - 1

    for i in range(len(remaining)):
        dr = DistanceRanking(
            node=remaining[i],
            ranking=(
                find_rank(i, cur_idx_in_temp, temp)
                + find_rank(cur_idx_in_temp, i, temp)
            )
            / 2.0,
        )
        drs.append(dr)

    drs.sort(key=lambda x: x.ranking)
    return [dr.node for dr in drs]


def check_droneable(group: List[int]) -> int:
    """Đếm số lượng node có thể phục vụ bằng drone trong nhóm."""
    return sum(1 for node in group if params.customers[node].only_by_truck == 0)


def mintime(times: List[float]) -> int:
    """Trả về chỉ số của phần tử nhỏ nhất trong danh sách thời gian."""
    return times.index(min(times))


# --- CÁC HÀM SINH LỜI GIẢI (SOL GENERATORS) ---


def generate_sol() -> List[int]:
    """Bản triển khai rename từ generatesol trong gaBase.h."""
    sol = []
    drone_pool = []
    truck_pool = []
    trucks_route = [[] for _ in range(params.NUM_TRUCKS)]
    drones_route = [[] for _ in range(params.NUM_DRONES)]

    for i in range(1, params.NUM_CUS + 1):
        if params.customers[i].only_by_truck == 0 and random.random() < 0.7:
            drone_pool.append(i)
        else:
            truck_pool.append(i)

    for node in truck_pool:
        trucks_route[random.randint(0, params.NUM_TRUCKS - 1)].append(node)

    for i in range(params.NUM_TRUCKS):
        random.shuffle(trucks_route[i])
        sol.extend(trucks_route[i])
        if i == params.NUM_TRUCKS - 1:
            sol.append(0)
        else:
            sol.append(params.NUM_CUS + i + 1)

    for node in drone_pool:
        drones_route[random.randint(0, params.NUM_DRONES - 1)].append(node)

    for i in range(params.NUM_DRONES):
        random.shuffle(drones_route[i])
        dronetrfin = 0
        for node in drones_route[i]:
            sol.append(node)
            if dronetrfin < params.drone_max_tracks - 1:
                if random.randint(0, 1) == 1:
                    sol.append(
                        params.NUM_CUS
                        + params.NUM_TRUCKS
                        + dronetrfin
                        + i * params.drone_max_tracks
                    )
                    dronetrfin += 1
        while dronetrfin < params.drone_max_tracks - 1:
            sol.append(
                params.NUM_CUS
                + params.NUM_TRUCKS
                + dronetrfin
                + i * params.drone_max_tracks
            )
            dronetrfin += 1
        if i < params.NUM_DRONES - 1:
            sol.append(
                params.NUM_CUS
                + params.NUM_TRUCKS
                + params.drone_max_tracks * (i + 1)
                - 1
            )

    return repair_route(sol)


def generate_sol_2() -> List[int]:
    """Triển khai đầy đủ generateSol2 từ gaBase.h."""
    group = [[] for _ in range(params.NUM_TRUCKS)]
    check = False
    while not check:
        for i in range(params.NUM_TRUCKS):
            group[i].clear()
        check = True
        for i in range(1, params.NUM_CUS + 1):
            group[random.randint(0, params.NUM_TRUCKS - 1)].append(i)
        for i in range(params.NUM_TRUCKS):
            if len(group[i]) < 2:
                check = False

    trucks = [[] for _ in range(params.NUM_TRUCKS)]
    drones = [[] for _ in range(params.NUM_DRONES)]

    for i in range(params.NUM_TRUCKS):
        droneable = check_droneable(group[i])
        truck_route = []
        drone_routes = []
        drone_route = []

        tr_rand = random.randrange(len(group[i]))
        truck_route.append(group[i][tr_rand])
        if params.customers[truck_route[-1]].only_by_truck == 0:
            droneable -= 1
        group[i].pop(tr_rand)

        drone_time = 0.0
        if droneable != 0:
            while True:
                dr_rand = random.randrange(len(group[i]))
                if params.customers[group[i][dr_rand]].only_by_truck == 0:
                    break
            drone_route.append(group[i][dr_rand])
            group[i].pop(dr_rand)
            drone_time = cal_drone(0, drone_route[-1])
            droneable -= 1
        else:
            drone_time = 1e18  # mvalue

        truck_time = cal_truck(0, truck_route[0], 0.0)

        while group[i]:
            if drone_time < truck_time:
                cur = drone_route[-1] if drone_route else 0
                sel = distance_ranking(cur, group[i])
                fin = False
                for node in sel:
                    if params.customers[node].only_by_truck == 0:
                        drone_route.append(node)
                        if feasible_drone_route(drone_route) == 0:
                            fin = True
                            break
                        else:
                            drone_route.pop()
                if fin:
                    group[i].remove(drone_route[-1])
                    drone_time += cal_drone(cur, drone_route[-1])
                    droneable -= 1
                else:
                    drone_routes.append(list(drone_route))
                    if cur != 0:
                        drone_time += cal_drone(cur, 0) - params.DRONE_SERVICE_TIME
                    drone_route.clear()
                if droneable == 0 or len(drone_routes) == params.drone_max_tracks:
                    drone_time = 1e18
            else:
                truck_sel = distance_ranking(truck_route[-1], group[i])[0]
                truck_time = cal_truck(truck_route[-1], truck_sel, truck_time)
                truck_route.append(truck_sel)
                if params.customers[truck_route[-1]].only_by_truck == 0:
                    droneable -= 1
                group[i].remove(truck_sel)
                if droneable == 0:
                    drone_time = 1e18

        if drone_route:
            drone_routes.append(drone_route)
        trucks[i] = truck_route
        if i < params.NUM_DRONES:
            drones[i] = drone_routes

    # Assembly logic
    tour = []
    for i in range(params.NUM_TRUCKS):
        tour.extend(trucks[i])
        if i < params.NUM_TRUCKS - 1:
            tour.append(params.NUM_CUS + i + 1)
    tour.append(0)
    for i in range(params.NUM_DRONES):
        for j in range(params.drone_max_tracks):
            if j < len(drones[i]):
                tour.extend(drones[i][j])
            tour.append(
                params.NUM_CUS + params.NUM_TRUCKS + j + i * params.drone_max_tracks
            )
    tour.pop()
    return tour


def generate_sol_3() -> List[int]:
    """Triển khai generateSol3 từ gaBase.h."""
    group = [[] for _ in range(params.NUM_TRUCKS)]
    check = False
    while not check:
        for i in range(params.NUM_TRUCKS):
            group[i].clear()
        check = True
        for i in range(1, params.NUM_CUS + 1):
            group[random.randint(0, params.NUM_TRUCKS - 1)].append(i)
        for i in range(params.NUM_TRUCKS):
            if len(group[i]) < 2:
                check = False

    trucks = [[] for _ in range(params.NUM_TRUCKS)]
    drone_times = [0.0] * params.NUM_DRONES
    drones = [[] for _ in range(params.NUM_DRONES)]

    for i in range(params.NUM_TRUCKS):
        chosen_drone = i if i < params.NUM_DRONES else mintime(drone_times)
        droneable = check_droneable(group[i])
        truck_route = []
        drone_route = []

        if i >= params.NUM_DRONES and drones[chosen_drone]:
            drone_route = drones[chosen_drone].pop()

        tr_rand = random.randrange(len(group[i]))
        truck_route.append(group[i][tr_rand])
        if params.customers[truck_route[-1]].only_by_truck == 0:
            droneable -= 1
        group[i].pop(tr_rand)

        drone_time = drone_times[chosen_drone]
        if droneable != 0 and i < params.NUM_DRONES:
            while True:
                dr_rand = random.randrange(len(group[i]))
                if params.customers[group[i][dr_rand]].only_by_truck == 0:
                    break
            drone_route.append(group[i][dr_rand])
            group[i].pop(dr_rand)
            drone_time = cal_drone(0, drone_route[-1])
            droneable -= 1

        truck_time = cal_truck(0, truck_route[0], 0.0)
        while group[i]:
            if (
                drone_time < truck_time
                and droneable != 0
                and len(drones[chosen_drone]) < params.drone_max_tracks
            ):
                cur = drone_route[-1] if drone_route else 0
                sel = distance_ranking(cur, group[i])
                fin = False
                for node in sel:
                    if params.customers[node].only_by_truck == 0:
                        drone_route.append(node)
                        if feasible_drone_route(drone_route) == 0:
                            fin = True
                            break
                        else:
                            drone_route.pop()
                if fin:
                    group[i].remove(drone_route[-1])
                    drone_time += cal_drone(cur, drone_route[-1])
                    droneable -= 1
                else:
                    if drone_route:
                        drones[chosen_drone].append(list(drone_route))
                    if cur != 0:
                        drone_time += cal_drone(cur, 0) - params.DRONE_SERVICE_TIME
                    drone_route.clear()
            else:
                truck_sel = distance_ranking(truck_route[-1], group[i])[0]
                truck_time = cal_truck(truck_route[-1], truck_sel, truck_time)
                truck_route.append(truck_sel)
                if params.customers[truck_route[-1]].only_by_truck == 0:
                    droneable -= 1
                group[i].remove(truck_sel)

        if drone_route:
            drones[chosen_drone].append(list(drone_route))
            drone_time += cal_drone(drone_route[-1], 0) - params.DRONE_SERVICE_TIME
        if drones[chosen_drone]:
            drone_time = (
                drone_time
                - cal_drone(drones[chosen_drone][-1][-1], 0)
                + params.DRONE_SERVICE_TIME
            )
        trucks[i] = truck_route
        drone_times[chosen_drone] = drone_time

    # Assembly logic
    tour = []
    for i in range(params.NUM_TRUCKS):
        tour.extend(trucks[i])
        if i < params.NUM_TRUCKS - 1:
            tour.append(params.NUM_CUS + i + 1)
    tour.append(0)
    for i in range(params.NUM_DRONES):
        for j in range(params.drone_max_tracks):
            if j < len(drones[i]):
                tour.extend(drones[i][j])
            tour.append(
                params.NUM_CUS + params.NUM_TRUCKS + j + i * params.drone_max_tracks
            )
    tour.pop()
    return tour


# --- QUẢN LÝ QUẦN THỂ ---


def select_population(size: int) -> List[Individual]:
    """Khởi tạo quần thể."""
    population = []
    for _ in range(size):
        route = generate_sol_3()
        f1, f2 = calculate_fitness(route)
        ind = Individual(route=route, fitness1=f1, fitness2=f2)
        ind.crowding_distance = 0
        ind.tabu_search = 0
        ind.local_search = 0
        population.append(ind)
    return population


def tournament_selection(population: List[Individual]) -> Individual:
    """Lựa chọn cá thể qua đấu loại."""
    random_indices = set()
    while len(random_indices) < 4:
        random_indices.add(random.randint(0, len(population) - 1))

    best_val = 0.0
    best_sol_idx = 0
    for idx in random_indices:
        score = compute_linear_1(population[idx].fitness1, population[idx].fitness2)
        if score < best_val or best_val == 0:
            best_val = score
            best_sol_idx = idx
    return population[best_sol_idx]


def check_dif_sol(route1: List[int], route2: List[int]) -> bool:
    """Kiểm tra sự khác biệt giữa hai lộ trình."""
    total_node = len(route1)
    for i in range(1, params.NUM_CUS + 1):
        try:
            po1 = route1.index(i)
            po2 = route2.index(i)
            if po1 < total_node - 1 and po2 < total_node - 1:
                if route1[po1 + 1] != route2[po2 + 1] and (
                    route1[po1 + 1] <= params.NUM_CUS
                    or route2[po2 + 1] <= params.NUM_CUS
                ):
                    return True
        except ValueError:
            continue
    return False
