import time
import random
from typing import List
from algorithms.hnsgaii_ts.structs import Individual
from algorithms.hnsgaii_ts.hash import calculate_hash
from algorithms.hnsgaii_ts.crossover import crossover, crossover_selection
from algorithms.hnsgaii_ts.ga import tournament_selection
from algorithms.hnsgaii_ts.nsgaii_utils import (
    select_new_population,
    fast_non_dominated_sort,
)
from algorithms.hnsgaii_ts.domination_check import check_domination
from algorithms.hnsgaii_ts.repair import repair_route, repair_position
from algorithms.hnsgaii_ts.output import output
from algorithms.hnsgaii_ts.pareto_update import (
    in_pareto,
    update_pareto,
    dominated_in_pareto,
)
from algorithms.hnsgaii_ts.fitness import calculate_fitness
from algorithms.hnsgaii_ts.tabu_search import tabu_search
from . import parameters as params


def nsgaii(defaultpop: List[Individual]) -> List[Individual]:
    start_time = time.time()
    end_iter = 0
    pareto: List[Individual] = []

    nochangeStreak = 0
    tbtime = 0
    lastUpdate = 0.0

    # C++ uses nRe[100][3]
    nRe = [[0] * 3 for _ in range(100)]
    nPe = [[0] * 3 for _ in range(100)]

    # Initialize population: sort(defaultpop.begin(), defaultpop.end(), comparefit1)
    defaultpop.sort(key=lambda x: x.fitness1)
    population = [ind for ind in defaultpop]

    for i in range(params.MAX_GENERATIONS):
        # Reset adaptive tracking every CROSSOVER_MOD generations
        if i % params.CROSSOVER_MOD == 0:
            for row in range(min(100, params.CROSSOVER_MOD)):
                for col in range(3):
                    nRe[row][col] = 0
                    nPe[row][col] = 0

        improveInpareto = 0
        hashRecord = set()
        newPopulation: List[Individual] = []

        # 1. Add current population to the pool
        for ind in population:
            ind.crowding_distance = 0  # Match: population[j].crowdingdistance=0
            hash_val = calculate_hash(ind.route)
            newPopulation.append(ind)
            hashRecord.add(hash_val)

        # 2. Crossover and Mutation Loop
        offpopu = 0
        while offpopu < len(population):
            crossoverAlgo = crossover_selection()
            nReward = 0
            nPenalty = 0

            # FIX: Initialize variables to avoid Unbound Errors
            offspring = None
            parent1 = None
            parent2 = None
            crossoverPerformed = False

            while not crossoverPerformed:
                parent1 = tournament_selection(population)
                # Ensure parents are different (C++: while abs diff < 1e-3)
                while True:
                    parent2 = tournament_selection(population)
                    if (
                        abs(parent1.fitness1 - parent2.fitness1) >= 1e-3
                        or abs(parent1.fitness2 - parent2.fitness2) >= 1e-3
                    ):
                        break

                if random.random() < params.CROSSOVER_RATE:
                    crossoverPerformed = True
                    offspring = crossover(parent1, parent2, crossoverAlgo)

            assert offspring is not None, ""
            # Mutation
            if random.random() < params.MUTATION_RATE:
                idx1 = random.randint(0, params.total_node - 1)
                idx2 = random.randint(0, params.total_node - 1)
                offspring.route[idx1], offspring.route[idx2] = (
                    offspring.route[idx2],
                    offspring.route[idx1],
                )
                offspring.route = repair_position(offspring.route)
                offspring.route = repair_route(offspring.route)

            # Fitness calculation
            fit = calculate_fitness(offspring.route)
            offspring.fitness1, offspring.fitness2 = fit[0], fit[1]
            offspring.tabu_search = 0

            hash_off = calculate_hash(offspring.route)
            if hash_off not in hashRecord:
                hashRecord.add(hash_off)
                newPopulation.append(offspring)
                offpopu += 1

                assert parent1 and parent2, ""
                # 3. Adaptive Scoring Logic (requires crossoverPerformed check)
                if crossoverPerformed:
                    if check_domination(parent1, parent2):
                        if check_domination(parent1, offspring):
                            nPenalty += 1
                        else:
                            nReward += 1
                    elif check_domination(parent2, parent1):
                        if check_domination(parent2, offspring):
                            nPenalty += 1
                        else:
                            nReward += 1
                    else:
                        if not check_domination(
                            parent1, offspring
                        ) and not check_domination(parent2, offspring):
                            nReward += 1
                        else:
                            nPenalty += 1

            nRe[i % params.CROSSOVER_MOD][crossoverAlgo] += nReward
            nPe[i % params.CROSSOVER_MOD][crossoverAlgo] += nPenalty

        # 4. Periodic Tabu Search Optimization
        if nochangeStreak > 0 and nochangeStreak % 30 == 0:
            tbtime += 1
            for f in range(len(pareto)):
                if (time.time() - start_time) >= params.time_limit:
                    break
                if params.max_tabu_iter > 0:
                    tabu_results = tabu_search(pareto[f].route, params.max_tabu_iter)
                    for res in tabu_results:
                        # C++ doesn't check hash here, just appends
                        newPopulation.append(res)

        # 5. Selection
        population = select_new_population(newPopulation, params.POPULATION_SIZE)

        # C++ logic: sort by fitness1 and fitness2 to find best objects (internal logic)
        population.sort(key=lambda x: x.fitness1)

        # Update Global Pareto Front
        fronts = fast_non_dominated_sort(population)
        current_front_indices = fronts[0]

        for idx in current_front_indices:
            candidate = population[idx]
            if not in_pareto(candidate, pareto):
                if not dominated_in_pareto(candidate, pareto):
                    improveInpareto = 1
                update_pareto(candidate, pareto)

        if improveInpareto > 0:
            nochangeStreak = 0
            lastUpdate = time.time() - start_time
        else:
            nochangeStreak += 1

        # 6. Re-inject Pareto members (Matches the C++ "if(paretonum.size()!=pareto.size())" logic)
        current_front_hashes = {
            calculate_hash(population[idx].route) for idx in current_front_indices
        }
        for p_ind in pareto:
            if calculate_hash(p_ind.route) not in current_front_hashes:
                population.append(p_ind)

        # Time Limit Check
        if (time.time() - start_time) >= params.time_limit:
            end_iter = i + 1
            break
        if i == params.MAX_GENERATIONS - 1:
            end_iter = params.MAX_GENERATIONS

    # Final Output
    pareto.sort(key=lambda x: x.fitness1)
    total_duration = time.time() - start_time
    output(
        pareto, total_duration, end_iter, tbtime, lastUpdate, end_iter - nochangeStreak
    )

    return pareto
