from typing import List
from . import parameters as params
from algorithms.hnsgaii_ts.structs import Individual
from algorithms.hnsgaii_ts.domination_check import check_domination


def fast_non_dominated_sort(population: List[Individual]) -> List[List[int]]:
    pop_size = len(population)
    fronts = [[] for _ in range(pop_size)]
    domination_count = [0] * pop_size
    dominates_list = [[] for _ in range(pop_size)]

    for i in range(pop_size):
        for j in range(i + 1, pop_size):
            if check_domination(population[i], population[j]):
                dominates_list[i].append(j)
                domination_count[j] += 1
            elif check_domination(population[j], population[i]):
                dominates_list[j].append(i)
                domination_count[i] += 1

    # Assign fronts iteratively
    check_status = [-1] * pop_size
    for i in range(pop_size):
        for j in range(pop_size):
            if domination_count[j] == 0 and check_status[j] == -1:
                fronts[i].append(j)
                check_status[j] = 0

        for idx in fronts[i]:
            for dominated_idx in dominates_list[idx]:
                domination_count[dominated_idx] -= 1

    return [f for f in fronts if f]  # Remove empty fronts


def longest_trips(route: List[int]) -> List[List[int]]:
    """Helper for duplicate checking: extracts the longest sequences of customers."""
    trips = []
    temp = []
    longest = 0
    for node in route:
        if 0 < node <= params.NUM_CUS:
            temp.append(node)
        else:
            if len(temp) > longest:
                longest = len(temp)
                trips = [list(temp)]
            elif len(temp) == longest and longest > 0:
                trips.append(list(temp))
            temp = []
    return trips


def check_sol_duplicate(route1: List[int], route2: List[int]) -> bool:
    """Heuristic check to see if two routes are functionally identical."""
    t1 = longest_trips(route1)
    t2 = longest_trips(route2)
    if len(t1) != len(t2):
        return False

    matches = 0
    for trip_a in t1:
        for trip_b in t2:
            if trip_a == trip_b:
                matches += 1
                break
    return matches == len(t1)


def get_crowding_and_clean(
    population: List[Individual], front_indices: List[int]
) -> List[Individual]:
    temp = [population[i] for i in front_indices]
    if not temp:
        return []

    # 1. Duplicate Removal
    temp.sort(key=lambda x: x.fitness1)
    unique_temp = []
    if temp:
        unique_temp.append(temp[0])
        for i in range(1, len(temp)):
            is_dup = False
            if abs(temp[i].fitness1 - temp[i - 1].fitness1) < 1e-6:
                if check_sol_duplicate(temp[i].route, temp[i - 1].route):
                    is_dup = True
            if not is_dup:
                unique_temp.append(temp[i])

    # 2. Crowding Distance Calculation
    size = len(unique_temp)
    if size == 0:
        return []
    if size <= 2:
        for ind in unique_temp:
            ind.crowding_distance = params.M_VALUE
        return unique_temp

    # Initialize
    for ind in unique_temp:
        ind.crowding_distance = 0.0

    # Sort by Obj 1
    unique_temp.sort(key=lambda x: x.fitness1)
    unique_temp[0].crowding_distance = params.M_VALUE
    unique_temp[-1].crowding_distance = params.M_VALUE
    f1_range = unique_temp[-1].fitness1 - unique_temp[0].fitness1
    if f1_range > 0:
        for i in range(1, size - 1):
            unique_temp[i].crowding_distance += (
                unique_temp[i + 1].fitness1 - unique_temp[i - 1].fitness1
            ) / f1_range

    # Sort by Obj 2
    unique_temp.sort(key=lambda x: x.fitness2)
    unique_temp[0].crowding_distance = params.M_VALUE
    unique_temp[-1].crowding_distance = params.M_VALUE
    f2_range = unique_temp[-1].fitness2 - unique_temp[0].fitness2
    if f2_range > 0:
        for i in range(1, size - 1):
            if unique_temp[i].crowding_distance != params.M_VALUE:
                unique_temp[i].crowding_distance += (
                    unique_temp[i + 1].fitness2 - unique_temp[i - 1].fitness2
                ) / f2_range

    # Sort by descending distance (best for diversity)
    unique_temp.sort(key=lambda x: x.crowding_distance, reverse=True)
    return unique_temp


def select_new_population(
    population: List[Individual], target_pop_size: int
) -> List[Individual]:
    new_population = []
    fronts = fast_non_dominated_sort(population)

    for front in fronts:
        cleaned_front = get_crowding_and_clean(population, front)
        for ind in cleaned_front:
            if len(new_population) < target_pop_size:
                new_population.append(ind)
            else:
                break
        if len(new_population) >= target_pop_size:
            break

    # Update global normalization parameters for the next generation
    if new_population:
        global min_obj1, max_obj1, min_obj2, max_obj2, obj1_norm, obj2_norm
        min_obj1 = min(ind.fitness1 for ind in new_population)
        max_obj1 = max(ind.fitness1 for ind in new_population)
        min_obj2 = min(ind.fitness2 for ind in new_population)
        max_obj2 = max(ind.fitness2 for ind in new_population)
        obj1_norm = max(1.0, max_obj1 - min_obj1)
        obj2_norm = max(1.0, max_obj2 - min_obj2)

    return new_population
