# Using standard functions (Mirroring C++ logic)
def compare_fit1(indi1, indi2):
    """Sort by Fitness 1 (Ascending)"""
    return indi1.fitness1 < indi2.fitness1


def compare_fit2(indi1, indi2):
    """Sort by Fitness 2 (Ascending)"""
    return indi1.fitness2 < indi2.fitness2


def compare_distance(indi1, indi2):
    """Sort by Crowding Distance (Descending - higher is better)"""
    return indi1.crowding_distance > indi2.crowding_distance


def compare_criteria(rec1, rec2):
    """Sort ParetoRecord by criteria (Ascending)"""
    return rec1.criteria < rec2.criteria


def compare_ranking(rec1, rec2):
    """Sort ParetoRecord by ranking (Ascending)"""
    return rec1.ranking < rec2.ranking


def compare_node_distance(nd1, nd2):
    """Sort NodeDistance by distance (Ascending)"""
    return nd1.distance < nd2.distance


def compare_distance_ranking(dr1, dr2):
    """Sort DistanceRanking by ranking (Ascending)"""
    return dr1.ranking < dr2.ranking
