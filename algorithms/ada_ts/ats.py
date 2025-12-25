import Data
import Function
import Neighborhood
import time
import random
import Neighborhood11
import Neighborhood10
import glob
import os
import numpy as np
import math
import json
import numpy
from Data import (
    calculate_angle,
    calculate_standard_deviation,
    euclid_distance,
    manhattan_distance,
)

global LOOP
global tabu_tenure
global best_sol
global best_fitness
global Tabu_Structure
global current_neighborhood
global LOOP_IMPROVED
global SET_LAST_10
global BEST

# Set up chỉ số -------------------------------------------------------------------
ITE = 1
epsilon = (-1) * 0.00001
# 15:   120,    20:    150
# BREAKLOOP = Data.number_of_cities * 8
LOOP_IMPROVED = 0
SET_LAST_10 = []
BEST = []
#
number_of_cities = int(os.getenv("NUMBER_OF_CITIES", 20))
delta = Data.delta
alpha = Data.alpha
theta = Data.theta
data_set = str(os.getenv("DATA_SET", "RC101_2.dat"))
solution_pack_len = 0
TIME_LIMIT = 14000
SEGMENT = int(os.getenv("SEGMENT", 12))
ite = int(os.getenv("ITERATION", 1))

progress_log = []


def roulette_wheel_selection(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    probabilities = [score / total_fitness for score in fitness_scores]
    selected_index = np.random.choice(len(population), p=probabilities)
    return population[selected_index]


def Tabu_search(
    init_solution, tabu_tenure, CC, first_time, Data1, index_consider_elite_set
):
    solution_pack_len = 5
    solution_pack = []

    current_fitness, current_truck_time, current_sum_fitness = Function.fitness(
        init_solution
    )
    best_sol = init_solution
    best_fitness = current_fitness
    sol_chosen_to_break = init_solution
    fit_of_sol_chosen_to_break = current_fitness

    lennn = [0] * 6
    lenght_i = [0] * 6
    i = 0

    Result_print = []
    # LOOP = BREAKLOOP * AA
    # print(Data.standard_deviation)
    global current_neighborhood
    global LOOP_IMPROVED
    LOOP_IMPROVED = 0
    global use_optimize_truck_route
    use_optimize_truck_route = False

    Data1 = [
        [
            "act",
            "fitness",
            "change1",
            "change2",
            "solution",
            "tabu structue",
            "tabu structure1",
        ]
    ]
    LOOP = min(int(Data.number_of_cities * math.log(Data.number_of_cities)), 100)

    # BREAKLOOP = Data.number_of_cities
    SEGMENT = 5
    END_SEGMENT = int(Data.number_of_cities / math.log10(Data.number_of_cities))

    T = 0
    nei_set = [0, 1, 2, 3, 4]
    weight = [1 / len(nei_set)] * len(nei_set)
    current_sol = init_solution

    while T < SEGMENT:
        tabu_tenure = tabu_tenure1 = tabu_tenure3 = tabu_tenure2 = random.uniform(
            2 * math.log(Data.number_of_cities), Data.number_of_cities
        )
        Tabu_Structure = [(tabu_tenure + 1) * (-1)] * Data.number_of_cities
        Tabu_Structure1 = [(tabu_tenure + 1) * (-1)] * Data.number_of_cities
        Tabu_Structure2 = [(tabu_tenure + 1) * (-1)] * Data.number_of_cities
        Tabu_Structure3 = [(tabu_tenure + 1) * (-1)] * Data.number_of_cities
        factor = 0.3  # 0.3 0.6
        score = [0.0] * len(nei_set)
        used = [0] * len(nei_set)
        prev_f = best_fitness
        prev_fitness = current_fitness

        LOOP_IMPROVED = 0
        lennn = [0] * 6
        lenght_i = [0] * 6
        i = 0
        while i < END_SEGMENT:
            current_neighborhood = []
            choose = roulette_wheel_selection(nei_set, weight)
            if choose == 0:
                current_neighborhood1, solution_pack = (
                    Neighborhood.Neighborhood_combine_truck_and_drone_neighborhood_with_tabu_list_with_package(
                        name_of_truck_neiborhood=Neighborhood10.Neighborhood_one_opt_standard,
                        solution=current_sol,
                        number_of_potial_solution=CC,
                        number_of_loop_drone=2,
                        tabu_list=Tabu_Structure,
                        tabu_tenure=tabu_tenure,
                        index_of_loop=lenght_i[1],
                        best_fitness=best_fitness,
                        kind_of_tabu_structure=1,
                        need_truck_time=False,
                        solution_pack=solution_pack,
                        solution_pack_len=solution_pack_len,
                        use_solution_pack=first_time,
                        index_consider_elite_set=index_consider_elite_set,
                    )
                )
                current_neighborhood.append([1, current_neighborhood1])
            elif choose == 2:
                current_neighborhood5, solution_pack = (
                    Neighborhood.Neighborhood_combine_truck_and_drone_neighborhood_with_tabu_list_with_package(
                        name_of_truck_neiborhood=Neighborhood11.Neighborhood_two_opt_tue,
                        solution=current_sol,
                        number_of_potial_solution=CC,
                        number_of_loop_drone=2,
                        tabu_list=Tabu_Structure3,
                        tabu_tenure=tabu_tenure3,
                        index_of_loop=lenght_i[5],
                        best_fitness=best_fitness,
                        kind_of_tabu_structure=5,
                        need_truck_time=False,
                        solution_pack=solution_pack,
                        solution_pack_len=solution_pack_len,
                        use_solution_pack=first_time,
                        index_consider_elite_set=index_consider_elite_set,
                    )
                )
                current_neighborhood.append([5, current_neighborhood5])
            elif choose == 3:
                current_neighborhood4, solution_pack = (
                    Neighborhood.Neighborhood_combine_truck_and_drone_neighborhood_with_tabu_list_with_package(
                        name_of_truck_neiborhood=Neighborhood11.Neighborhood_move_2_1,
                        solution=current_sol,
                        number_of_potial_solution=CC,
                        number_of_loop_drone=2,
                        tabu_list=Tabu_Structure2,
                        tabu_tenure=tabu_tenure2,
                        index_of_loop=lenght_i[4],
                        best_fitness=best_fitness,
                        kind_of_tabu_structure=4,
                        need_truck_time=False,
                        solution_pack=solution_pack,
                        solution_pack_len=solution_pack_len,
                        use_solution_pack=first_time,
                        index_consider_elite_set=index_consider_elite_set,
                    )
                )
                current_neighborhood.append([4, current_neighborhood4])
            else:
                current_neighborhood3, solution_pack = (
                    Neighborhood.Neighborhood_combine_truck_and_drone_neighborhood_with_tabu_list_with_package(
                        name_of_truck_neiborhood=Neighborhood11.Neighborhood_move_1_1_ver2,
                        solution=current_sol,
                        number_of_potial_solution=CC,
                        number_of_loop_drone=2,
                        tabu_list=Tabu_Structure1,
                        tabu_tenure=tabu_tenure1,
                        index_of_loop=lenght_i[3],
                        best_fitness=best_fitness,
                        kind_of_tabu_structure=3,
                        need_truck_time=False,
                        solution_pack=solution_pack,
                        solution_pack_len=solution_pack_len,
                        use_solution_pack=first_time,
                        index_consider_elite_set=index_consider_elite_set,
                    )
                )
                current_neighborhood.append([3, current_neighborhood3])

            flag = False
            index = [0] * len(current_neighborhood)
            min_nei = [100000] * len(current_neighborhood)
            min_sum = [1000000000] * len(current_neighborhood)
            # print(current_neighborhood)
            for j in range(len(current_neighborhood)):
                if current_neighborhood[j][0] in [1, 2]:
                    for k in range(len(current_neighborhood[j][1])):
                        cfnode = current_neighborhood[j][1][k][1][0]
                        if cfnode - best_fitness < epsilon:
                            min_nei[j] = cfnode
                            index[j] = k
                            best_fitness = cfnode
                            best_sol = current_neighborhood[j][1][k][0]
                            LOOP_IMPROVED = i
                            flag = True

                        elif (
                            cfnode - min_nei[j] < epsilon
                            and Tabu_Structure[current_neighborhood[j][1][k][2]]
                            + tabu_tenure
                            <= lenght_i[1]
                        ):
                            min_nei[j] = cfnode
                            index[j] = k
                            min_sum[j] = current_neighborhood[j][1][k][1][2]

                        elif (
                            min_nei[j] - epsilon > cfnode
                            and Tabu_Structure[current_neighborhood[j][1][k][2]]
                            + tabu_tenure
                            <= lenght_i[1]
                        ):
                            if min_sum[j] > current_neighborhood[j][1][k][1][2]:
                                min_nei[j] = cfnode
                                index[j] = k
                                min_sum[j] = current_neighborhood[j][1][k][1][2]
                elif current_neighborhood[j][0] == 3:
                    for k in range(len(current_neighborhood[j][1])):
                        cfnode = current_neighborhood[j][1][k][1][0]
                        if cfnode - best_fitness < epsilon:
                            min_nei[j] = cfnode
                            index[j] = k
                            best_fitness = cfnode
                            best_sol = current_neighborhood[j][1][k][0]
                            LOOP_IMPROVED = i
                            flag = True

                        elif (
                            cfnode - min_nei[j] < epsilon
                            and Tabu_Structure1[current_neighborhood[j][1][k][2][0]]
                            + tabu_tenure1
                            <= lenght_i[3]
                            or Tabu_Structure1[current_neighborhood[j][1][k][2][1]]
                            + tabu_tenure1
                            <= lenght_i[3]
                        ):
                            min_nei[j] = cfnode
                            index[j] = k
                            min_sum[j] = current_neighborhood[j][1][k][1][2]

                        elif (
                            cfnode < min_nei[j] - epsilon
                            and Tabu_Structure1[current_neighborhood[j][1][k][2][0]]
                            + tabu_tenure1
                            <= lenght_i[3]
                            or Tabu_Structure1[current_neighborhood[j][1][k][2][1]]
                            + tabu_tenure1
                            <= lenght_i[3]
                        ):
                            if min_sum[j] > current_neighborhood[j][1][k][1][2]:
                                min_nei[j] = cfnode
                                index[j] = k
                                min_sum[j] = current_neighborhood[j][1][k][1][2]
                elif current_neighborhood[j][0] == 4:
                    for k in range(len(current_neighborhood[j][1])):
                        cfnode = current_neighborhood[j][1][k][1][0]
                        if cfnode - best_fitness < epsilon:
                            min_nei[j] = cfnode
                            index[j] = k
                            best_fitness = cfnode
                            best_sol = current_neighborhood[j][1][k][0]
                            LOOP_IMPROVED = i
                            flag = True

                        elif (
                            cfnode - min_nei[j] < epsilon
                            and Tabu_Structure2[current_neighborhood[j][1][k][2][0]]
                            + tabu_tenure2
                            <= lenght_i[4]
                            or Tabu_Structure2[current_neighborhood[j][1][k][2][1]]
                            + tabu_tenure2
                            <= lenght_i[4]
                            or Tabu_Structure2[current_neighborhood[j][1][k][2][2]]
                            + tabu_tenure2
                            <= lenght_i[4]
                        ):
                            min_nei[j] = cfnode
                            index[j] = k
                            min_sum[j] = current_neighborhood[j][1][k][1][2]

                        elif (
                            cfnode < min_nei[j] - epsilon
                            and Tabu_Structure2[current_neighborhood[j][1][k][2][0]]
                            + tabu_tenure2
                            <= lenght_i[4]
                            or Tabu_Structure2[current_neighborhood[j][1][k][2][1]]
                            + tabu_tenure2
                            <= lenght_i[4]
                            or Tabu_Structure2[current_neighborhood[j][1][k][2][2]]
                            + tabu_tenure2
                            <= lenght_i[4]
                        ):
                            if min_sum[j] > current_neighborhood[j][1][k][1][2]:
                                min_nei[j] = cfnode
                                index[j] = k
                                min_sum[j] = current_neighborhood[j][1][k][1][2]
                elif current_neighborhood[j][0] == 5:
                    for k in range(len(current_neighborhood[j][1])):
                        cfnode = current_neighborhood[j][1][k][1][0]
                        if cfnode - best_fitness < epsilon:
                            min_nei[j] = cfnode
                            index[j] = k
                            best_fitness = cfnode
                            best_sol = current_neighborhood[j][1][k][0]
                            LOOP_IMPROVED = i
                            flag = True

                        elif (
                            cfnode - min_nei[j] < epsilon
                            and Tabu_Structure3[current_neighborhood[j][1][k][2][0]]
                            + tabu_tenure3
                            <= lenght_i[5]
                            or Tabu_Structure3[current_neighborhood[j][1][k][2][1]]
                            + tabu_tenure3
                            <= lenght_i[5]
                        ):
                            min_nei[j] = cfnode
                            index[j] = k
                            min_sum[j] = current_neighborhood[j][1][k][1][2]

                        elif (
                            cfnode < min_nei[j] - epsilon
                            and Tabu_Structure3[current_neighborhood[j][1][k][2][0]]
                            + tabu_tenure3
                            <= lenght_i[5]
                            or Tabu_Structure3[current_neighborhood[j][1][k][2][1]]
                            + tabu_tenure3
                            <= lenght_i[5]
                        ):
                            if min_sum[j] > current_neighborhood[j][1][k][1][2]:
                                min_nei[j] = cfnode
                                index[j] = k
                                min_sum[j] = current_neighborhood[j][1][k][1][2]
                else:
                    for k in range(len(current_neighborhood[j][1])):
                        cfnode = current_neighborhood[j][1][k][1][0]
                        if cfnode - best_fitness < epsilon:
                            min_nei[j] = cfnode
                            index[j] = k
                            best_fitness = cfnode
                            best_sol = current_neighborhood[j][1][k][0]
                            LOOP_IMPROVED = i
                            flag = True

                        elif cfnode - min_nei[j] < epsilon:
                            min_nei[j] = cfnode
                            index[j] = k
                            min_sum[j] = current_neighborhood[j][1][k][1][2]

                        elif cfnode < min_nei[j] - epsilon:
                            if min_sum[j] > current_neighborhood[j][1][k][1][2]:
                                min_nei[j] = cfnode
                                index[j] = k
                                min_sum[j] = current_neighborhood[j][1][k][1][2]
                progress_log.append(
                    {
                        "iteration": T,
                        "sub_iteration": i,
                        "fitness": best_fitness,
                        "time": time.time(),
                    }
                )

            index_best_nei = 0
            best_fit_in_cur_loop = min_nei[0]

            # for j in range(len(min_nei)):
            #     print(min_nei[j])
            #     print(current_neighborhood[j][1][index[j]][0])
            #     print("-------")

            for j in range(1, len(min_nei)):
                if min_nei[j] < best_fit_in_cur_loop:
                    index_best_nei = j
                    best_fit_in_cur_loop = min_nei[j]

            if current_neighborhood[index_best_nei][0] in [1, 2]:
                lenght_i[1] += 1

            if current_neighborhood[index_best_nei][0] == 3:
                lenght_i[3] += 1

            if current_neighborhood[index_best_nei][0] == 4:
                lenght_i[4] += 1

            if current_neighborhood[index_best_nei][0] == 5:
                lenght_i[5] += 1

            # print(current_neighborhood[index_best_nei][0])
            # print(len(current_neighborhood[index_best_nei][1]))
            # print(current_neighborhood[index_best_nei][1])
            # print(lenght_i[1], " then ", Tabu_Structure)
            # print(lenght_i[3], " then ", Tabu_Structure1)
            # print(lenght_i[4], " then ", Tabu_Structure2)
            # print(lenght_i[5], " then ", Tabu_Structure3)

            if len(current_neighborhood[index_best_nei][1]) == 0:
                # print("hahhaa")
                continue

            # print(index[index_best_nei])
            current_sol = current_neighborhood[index_best_nei][1][
                index[index_best_nei]
            ][0]
            current_fitness = current_neighborhood[index_best_nei][1][
                index[index_best_nei]
            ][1][0]
            current_truck_time = current_neighborhood[index_best_nei][1][
                index[index_best_nei]
            ][1][1]
            current_sum_fitness = current_neighborhood[index_best_nei][1][
                index[index_best_nei]
            ][1][2]
            # print(current_fitness, current_sol)
            Data1.append(current_fitness)
            Data1.append(current_sol)
            # SET_LAST_10.append([current_sol, [current_fitness, current_truck_time]])
            # if len(SET_LAST_10) > 10:
            #     SET_LAST_10.pop(0)

            if current_neighborhood[index_best_nei][0] in [1, 2]:
                Tabu_Structure[
                    current_neighborhood[index_best_nei][1][index[index_best_nei]][2]
                ] = lenght_i[1] - 1
                lennn[current_neighborhood[index_best_nei][0]] += 1

            if current_neighborhood[index_best_nei][0] == 3:
                Tabu_Structure1[
                    current_neighborhood[index_best_nei][1][index[index_best_nei]][2][0]
                ] = lenght_i[3] - 1
                Tabu_Structure1[
                    current_neighborhood[index_best_nei][1][index[index_best_nei]][2][1]
                ] = lenght_i[3] - 1
                lennn[current_neighborhood[index_best_nei][0]] += 1

            if current_neighborhood[index_best_nei][0] == 4:
                Tabu_Structure2[
                    current_neighborhood[index_best_nei][1][index[index_best_nei]][2][0]
                ] = lenght_i[4] - 1
                Tabu_Structure2[
                    current_neighborhood[index_best_nei][1][index[index_best_nei]][2][1]
                ] = lenght_i[4] - 1
                Tabu_Structure2[
                    current_neighborhood[index_best_nei][1][index[index_best_nei]][2][2]
                ] = lenght_i[4] - 1
                lennn[current_neighborhood[index_best_nei][0]] += 1

            if current_neighborhood[index_best_nei][0] == 5:
                Tabu_Structure3[
                    current_neighborhood[index_best_nei][1][index[index_best_nei]][2][0]
                ] = lenght_i[5] - 1
                Tabu_Structure3[
                    current_neighborhood[index_best_nei][1][index[index_best_nei]][2][1]
                ] = lenght_i[5] - 1
                lennn[current_neighborhood[index_best_nei][0]] += 1

            if fit_of_sol_chosen_to_break > current_fitness:
                sol_chosen_to_break = current_sol
                fit_of_sol_chosen_to_break = current_fitness
                LOOP_IMPROVED = i

            if current_neighborhood[index_best_nei][0] in [1, 2]:
                temp = [
                    current_neighborhood[index_best_nei][0],
                    current_fitness,
                    current_neighborhood[index_best_nei][1][index[index_best_nei]][2],
                    -1,
                    current_sol,
                    Tabu_Structure,
                    Tabu_Structure1,
                ]
            elif current_neighborhood[index_best_nei][0] in [3]:
                temp = [
                    current_neighborhood[index_best_nei][0],
                    current_fitness,
                    current_neighborhood[index_best_nei][1][index[index_best_nei]][2][
                        0
                    ],
                    current_neighborhood[index_best_nei][1][index[index_best_nei]][2][
                        1
                    ],
                    current_sol,
                    Tabu_Structure,
                    Tabu_Structure1,
                ]
            else:
                temp = [
                    current_neighborhood[index_best_nei][0],
                    current_fitness,
                    -1,
                    -1,
                    current_sol,
                ]
            Data1.append(temp)

            used[choose] += 1
            if flag:
                score[choose] += 0.5
            elif current_fitness - prev_fitness < epsilon:
                score[choose] += 0.3
            else:
                score[choose] += 0.1

            for j in range(len(nei_set)):
                if used[j] == 0:
                    continue
                else:
                    weight[j] = (1 - factor) * weight[j] + factor * score[j] / used[j]
            if flag:
                i = 0
            else:
                i += 1
        print("-------", T, "--------")
        print(f"Best fitness: {best_fitness}")
        # print(T, best_sol, "\n", best_fitness)
        # print(used, score, sum(used))

        if best_fitness - prev_f < epsilon:
            T = 0
        else:
            T += 1

    return best_sol, best_fitness, Result_print, solution_pack, Data1


def Tabu_search_for_CVRP(CC):
    Data1 = []
    list_init = []

    start_time = time.time()
    current_sol5 = Function.initial_solution7()
    list_init.append(current_sol5)

    list_fitness_init = []
    fitness5 = Function.fitness(current_sol5)

    list_fitness_init.append(fitness5)

    current_fitness = list_fitness_init[0][0]
    current_sol = list_init[0]

    for i in range(1, len(list_fitness_init)):
        if current_fitness > list_fitness_init[i][0]:
            current_sol = list_init[i]
            current_fitness = list_fitness_init[i][0]

    # Initial solution thay ở đây ------------->
    # current_sol = check     # Để dòng này làm comment để tìm initial solution theo tham lam
    # <------------- Initial solution thay ở đây

    # print(best_sol)
    # print(best_fitness)
    # print(Function.Check_if_feasible(best_sol))
    solution_pack_len = 5
    best_sol, best_fitness, result_print, solution_pack, Data1 = Tabu_search(
        init_solution=current_sol,
        tabu_tenure=Data.number_of_cities - 1,
        CC=CC,
        first_time=True,
        Data1=Data1,
        index_consider_elite_set=0,
    )
    for pi in range(solution_pack_len):
        # print(
        #    "+++++++++++++++++++++++++",
        #    len(solution_pack),
        #    "+++++++++++++++++++++++++",
        # )
        # for iiii in range(len(solution_pack)):
        #    print(solution_pack[iiii][0])
        #    print(solution_pack[iiii][1][0])
        #    print("$$$$$$$$$$$$$$")
        if pi < len(solution_pack):
            current_neighborhood5, _ = Neighborhood.swap_two_array(solution_pack[pi][0])
            best_sol_in_brnei = current_neighborhood5[0][0]
            best_fitness_in_brnei = current_neighborhood5[0][1][0]
            for i in range(1, len(current_neighborhood5)):
                cfnode = current_neighborhood5[i][1][0]
                if cfnode - best_fitness_in_brnei < epsilon:
                    best_sol_in_brnei = current_neighborhood5[i][0]
                    best_fitness_in_brnei = cfnode
            temp = ["break", "break", "break", "break", "break", "break", "break"]
            Data1.append(temp)
            best_sol1, best_fitness1, result_print1, solution_pack1, Data1 = (
                Tabu_search(
                    init_solution=best_sol_in_brnei,
                    tabu_tenure=Data.number_of_cities - 1,
                    CC=CC,
                    first_time=False,
                    Data1=Data1,
                    index_consider_elite_set=pi + 1,
                )
            )
            print("-----------------", pi, "------------------------")
            print(f"Solution: {best_sol1}")
            print(f"Fitness: {best_fitness1}")
            if best_fitness1 - best_fitness < epsilon:
                best_sol = best_sol1
                best_fitness = best_fitness1

        end_time = time.time()
        # if end_time - start_time > 3000:
        #     break

    return best_fitness, best_sol


# ===============================
# CONFIGURATION
# ===============================
ITE = 1
DATA_ROOT = "./data"
RESULT_ROOT = "./result"


# ===============================
# MAIN EXPERIMENT FUNCTION
# ===============================
def run_experiment(number_of_cities: int):
    data_folder = os.path.join(DATA_ROOT, str(number_of_cities))
    result_folder = os.path.join(RESULT_ROOT, str(number_of_cities))

    os.makedirs(result_folder, exist_ok=True)

    data_files = glob.glob(os.path.join(data_folder, "*.dat"))
    if not data_files:
        raise RuntimeError(f"No data files found in {data_folder}")

    for data_file in data_files[:1]:
        filename = os.path.splitext(os.path.basename(data_file))[0]
        print(f"[INFO] Processing {filename}")

        Data.read_data_random(data_file)

        run_results = []
        best_overall_fitness = float("inf")
        best_overall_solution = None
        total_runtime = 0.0

        for run_id in range(ITE):
            start_time = time.time()

            best_fitness, best_solution = Tabu_search_for_CVRP(1)

            runtime = time.time() - start_time
            total_runtime += runtime

            feasible = Function.Check_if_feasible(best_solution)

            run_results.append(
                {
                    "run_id": run_id,
                    "fitness": best_fitness,
                    "runtime_sec": runtime,
                    "feasible": feasible,
                }
            )

            if best_fitness < best_overall_fitness:
                best_overall_fitness = best_fitness
                best_overall_solution = best_solution

        print(best_overall_solution)

        output = {
            "instance": filename,
            "number_of_cities": number_of_cities,
            "ITE": ITE,
            "average_runtime_sec": total_runtime / ITE,
            "best_fitness": best_overall_fitness,
            "best_solution": Function.schedule(best_overall_solution),
            "runs": run_results,
        }

        output_path = os.path.join(result_folder, f"{filename}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

        print(f"[DONE] Saved result → {output_path}")


class ATSSolver:
    def __init__(self, problem_type, algorithm_name):
        self.problem_type = problem_type
        self.algorithm_name = algorithm_name

    def load(self, customers, depot, vehicle_config, algorithm_params):
        Data.city = [[depot["x"], depot["y"]]]

        for customer in customers:
            Data.city.append([customer["x"], customer["y"]])
            Data.city_demand.append(customer["demand"])
            Data.release_date.append(customer["release_date"])

        Data.number_of_cities = len(Data.city)

        Data.truck_speed = vehicle_config["truck_speed"]
        Data.drone_speed = vehicle_config["drone_speed"]

        Data.manhattan_move_matrix = [
            [0.0] * Data.number_of_cities for _ in range(Data.number_of_cities)
        ]
        Data.euclid_flight_matrix = [
            [0.0] * Data.number_of_cities for _ in range(Data.number_of_cities)
        ]

        Data.value_tan_of_city = [0.0] * Data.number_of_cities

        for i in range(Data.number_of_cities):
            for j in range(Data.number_of_cities):
                Data.euclid_flight_matrix[i][j] = (
                    euclid_distance(Data.city[i], Data.city[j]) / Data.drone_speed
                )
        Data.euclid_flight_matrix = numpy.array(Data.euclid_flight_matrix)
        for i in range(Data.number_of_cities):
            for j in range(Data.number_of_cities):
                Data.manhattan_move_matrix[i][j] = (
                    Data.manhattan_distance(Data.city[i], Data.city[j])
                    / Data.truck_speed
                )
        Data.manhattan_move_matrix = numpy.array(Data.manhattan_move_matrix)
        for i in range(1, number_of_cities):
            Data.value_tan_of_city[i] = calculate_angle(Data.city[0], Data.city[i])

    def solve(self, customers, depot, vehicle_config, algorithm_params):
        self.load(customers, depot, vehicle_config, algorithm_params)

        start_time = time.time()

        best_fitness, best_solution = Tabu_search_for_CVRP(1)

        print(best_solution)
        runtime = time.time() - start_time

        feasible = Function.Check_if_feasible(best_solution)

        result = {
            "fitness": best_fitness,
            "solution": Function.schedule(best_solution),
            "runtime_sec": runtime,
            "feasible": feasible,
            "progress": progress_log,
        }

        return result


# ===============================
# ENTRY POINT
# ===============================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Tabu Search experiments")
    parser.add_argument(
        "--cities", type=int, required=True, help="Number of cities (e.g., 20, 50, 100)"
    )

    args = parser.parse_args()

    run_experiment(args.cities)
