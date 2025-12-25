import random
from typing import List

from algorithms.hnsgaii_ts.structs import Individual
from . import parameters as params
from algorithms.hnsgaii_ts.repair import repair_position, repair_route


def crossover_selection() -> int:
    """Matches crossoverSelection() in crossover.h"""
    r = random.random()
    cumulative_probability = 0.0

    # Iterate through current proportions
    for i, prop in enumerate(params.crossover_proportion):
        cumulative_probability += prop
        if r <= cumulative_probability:
            return i

    # In case of numerical issues or empty sum, return a random index
    return random.randint(0, 2)


# --- Crossover Operators ---


def pmx_crossover(parent1: Individual, parent2: Individual) -> Individual:
    """Partially Mapped Crossover (PMX) matching crossover.h"""
    size = len(parent1.route)
    start, end = sorted(random.sample(range(size), 2))

    offspring_route = [-1] * size
    visited = [0] * (max(max(parent1.route), max(parent2.route)) + 1)

    # 1. Copy the selected segment from parent 1
    for i in range(start, end + 1):
        offspring_route[i] = parent1.route[i]
        visited[parent1.route[i]] = 1

    # 2. Fill remaining positions using PMX mapping logic
    for i in range(size):
        if i < start or i > end:
            value = parent2.route[i]
            if visited[value] == 0:
                offspring_route[i] = value
            else:
                # Trace the mapping
                # Find where p2's value is in p1's segment
                idx_in_p1 = parent1.route.index(value)
                while start <= idx_in_p1 <= end:
                    value = parent2.route[idx_in_p1]
                    idx_in_p1 = parent1.route.index(value)
                offspring_route[i] = value
            visited[offspring_route[i]] = 1

    return Individual(route=repair_all(offspring_route))


def ox_crossover(parent1: Individual, parent2: Individual) -> Individual:
    """Order Crossover (OX) matching crossover.h"""
    size = len(parent1.route)
    start, end = sorted(random.sample(range(size), 2))

    offspring_route = [-1] * size
    visited = [0] * (max(max(parent1.route), max(parent2.route)) + 1)

    # Copy selected segment from parent 1
    for i in range(start, end + 1):
        offspring_route[i] = parent1.route[i]
        visited[parent1.route[i]] = 1

    # Fill remaining from parent 2 in circular order
    remain = []
    # Check parent2 from start to end
    for i in range(start, size):
        if not visited[parent2.route[i]]:
            remain.append(parent2.route[i])
    # Check parent2 from 0 to start
    for i in range(0, start):
        if not visited[parent2.route[i]]:
            remain.append(parent2.route[i])

    t = 0
    if end == size - 1:
        for x in range(start):
            offspring_route[x] = remain[t]
            t += 1
    else:
        for x in range(end + 1, size):
            offspring_route[x] = remain[t]
            t += 1
        for x in range(start):
            offspring_route[x] = remain[t]
            t += 1

    return Individual(route=repair_all(offspring_route))


def pos_crossover(parent1: Individual, parent2: Individual) -> Individual:
    """Position-based Crossover (POS) matching crossover.h"""
    size = len(parent1.route)
    offspring_route = [-1] * size
    visited = [0] * (max(max(parent1.route), max(parent2.route)) + 1)

    num_positions = random.randint(1, size)
    selected_positions = random.sample(range(size), num_positions)

    # Copy selected positions from first parent
    for pos in selected_positions:
        offspring_route[pos] = parent1.route[pos]
        visited[parent1.route[pos]] = 1

    # Fill empty slots from parent 2
    current_pos_p2 = 0
    for i in range(size):
        if offspring_route[i] == -1:
            while visited[parent2.route[current_pos_p2]] == 1:
                current_pos_p2 += 1
            offspring_route[i] = parent2.route[current_pos_p2]
            visited[parent2.route[current_pos_p2]] = 1
            current_pos_p2 += 1

    return Individual(route=repair_all(offspring_route))


# --- Utility and Management ---


def repair_all(route: List[int]) -> List[int]:
    """Applies repair logic as seen in crossover.h"""
    route = repair_position(route)
    route = repair_route(route)
    return route


def crossover(indi1: Individual, indi2: Individual, selection: int) -> Individual:
    """Main crossover switch from crossover.h"""
    if selection == 0:
        return pmx_crossover(indi1, indi2)
    elif selection == 1:
        return ox_crossover(indi1, indi2)
    elif selection == 2:
        return pos_crossover(indi1, indi2)
    return pmx_crossover(indi1, indi2)


def update_crossover_proportion(nRe, nPe):
    """
    Matches updateCrossoverProportion logic in crossover.h
    nRe and nPe are expected to be 2D lists [CROSSOVER_MOD][3]
    """
    Re = [0] * 3
    Pe = [0] * 3
    stra = [0.0] * 3
    sum_stra = 0.0
    remain = 1.0

    # Sum up rewards and penalties over the crossoverMod period
    for i in range(len(nRe)):  # crossoverMod
        for j in range(3):
            Re[j] += nRe[i][j]
            Pe[j] += nPe[i][j]

    for i in range(3):
        if Re[i] == 0 and Pe[i] == 0:
            remain -= params.crossover_proportion[i]
            stra[i] = 0.0
        else:
            stra[i] = float(Re[i]) / float(Pe[i] + Re[i])
        sum_stra += stra[i]

    if sum_stra > 0:
        for i in range(3):
            if Re[i] > 0 or Pe[i] > 0:
                params.crossover_proportion[i] = (stra[i] * remain) / sum_stra
