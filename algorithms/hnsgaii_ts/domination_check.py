from algorithms.hnsgaii_ts.structs import Individual


def check_domination(dominator: Individual, dominated: Individual) -> bool:
    """
    Checks if 'dominator' dominates 'dominated' in a multi-objective context.

    A solution A dominates B if:
    1. A is not worse than B in all objectives.
    2. A is strictly better than B in at least one objective.
    """
    ch = False
    epsilon = 1e-3  # Floating point tolerance

    # Check Objective 1 (Fitness1)
    # If dominator is strictly worse, it cannot dominate
    if dominator.fitness1 - dominated.fitness1 > epsilon:
        return False
    # If dominated is strictly worse, we've met the "strictly better" condition
    elif dominated.fitness1 - dominator.fitness1 > epsilon:
        ch = True

    # Check Objective 2 (Fitness2)
    if dominator.fitness2 - dominated.fitness2 > epsilon:
        return False
    elif dominated.fitness2 - dominator.fitness2 > epsilon:
        ch = True

    return ch
