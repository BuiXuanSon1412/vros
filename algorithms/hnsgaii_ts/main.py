import sys
import random
import time
from algorithms.hnsgaii_ts.input import input_time_limit, input_data
from algorithms.hnsgaii_ts.distance import init_matrix
from algorithms.hnsgaii_ts.ga import select_population
from algorithms.hnsgaii_ts.nsgaii import nsgaii
from . import parameters as params


class HNSGAIITSSolver:
    def __init__(self):
        pass

    def solve(self):
        pass


def main():
    # Setup random seed

    random.seed(time.time())

    if len(sys.argv) < 4:
        print(
            "Usage: python main.py <instance_name> <params.POPULATION_SIZE> <max_tabu_iter>"
        )
        return

    instance = sys.argv[1]

    # Instance parsing logic (mirrors stoi(token))
    token = instance.split(".")[0]
    num = int(token)

    if num == 20:
        params.NUM_CUS, params.NUM_TRUCKS, params.NUM_DRONES = 20, 2, 2
    elif num == 50:
        params.NUM_CUS, params.NUM_TRUCKS, params.NUM_DRONES = 50, 3, 3
    elif num == 100:
        params.NUM_CUS, params.NUM_TRUCKS, params.NUM_DRONES = 100, 4, 4
    elif num == 200:
        params.NUM_CUS, params.NUM_TRUCKS, params.NUM_DRONES = 200, 10, 4
    else:
        # Default fallback or custom logic
        params.NUM_CUS, params.NUM_TRUCKS, params.NUM_DRONES = num, 2, 2

    # Set Time Limit based on instance
    input_time_limit(instance)

    params.input_file = f"./data/random_data/{instance}.txt"
    params.MAX_GENERATIONS = 100000
    params.POPULATION_SIZE = int(sys.argv[2])
    params.max_tabu_iter = int(sys.argv[3])
    print(f"input file: {params.input_file}")
    # Load data and initialize distance matrix
    input_data()

    # Calculate track and node constraints
    # Equivalent to: (int)(canuseddrone/params.NUM_DRONES)+1
    params.drone_max_tracks = (params.can_used_drone // params.NUM_DRONES) + 1
    params.total_node = (
        params.NUM_CUS
        + params.NUM_TRUCKS
        + (params.drone_max_tracks * params.NUM_DRONES)
        - 1
    )

    init_matrix()

    # Loop for 10 test runs
    for test_count_int in range(1, 11):
        # Re-seed for each run
        random.seed(time.time())

        # Initialize population
        population = select_population(params.POPULATION_SIZE)

        # Reset optimization params
        params.CROSSOVER_MOD = 50
        params.crossover_proportion = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]

        params.output_filename = "./result/result.json"

        # Calculate normalization bounds from initial population
        if population:
            params.min_obj1 = population[0].fitness1
            params.max_obj1 = population[0].fitness1
            params.min_obj2 = population[0].fitness2
            params.max_obj2 = population[0].fitness2

            for ind in population:
                if ind.fitness1 < params.min_obj1:
                    params.min_obj1 = ind.fitness1
                if ind.fitness2 < params.min_obj2:
                    params.min_obj2 = ind.fitness2
                if ind.fitness1 > params.max_obj1:
                    params.max_obj1 = ind.fitness1
                # Note: Correcting a likely typo in your C++: if(population[j].fitness2 > maxobj1)
                if ind.fitness2 > params.max_obj2:
                    params.max_obj2 = ind.fitness2

            params.obj1_norm = params.max_obj1 - params.min_obj1
            params.obj2_norm = params.max_obj2 - params.min_obj2
            print(f"obj1_norm: {params.obj1_norm}")
            print(f"obj2_norm: {params.obj2_norm}")
        # Create a copy of the population
        population_copy = [ind for ind in population]

        # Execute NSGA-II
        # This will internally call your tabu_search2 if integrated
        pareto_front = nsgaii(population_copy)

        print(f"Run {test_count_int} complete. Pareto size: {len(pareto_front)}")


if __name__ == "__main__":
    main()
