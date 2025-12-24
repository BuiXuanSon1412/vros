import random
import time
from typing import List, Tuple

import parameters as params

from structs import MoveRecord, Individual, TabuRecord
from localsearch_utils import track_result
from route_check import check_route
from fitness import calculate_fitness
from localsearch_utils import split_tracks
from pareto_update import update_pareto, in_pareto
from nsgaii_utils import fast_non_dominated_sort
from output import output


def in_tabu_list(tabulist: List[MoveRecord], move: MoveRecord, move_type: int) -> bool:
    """Checks if a move is currently restricted by the Tabu list."""
    if not tabulist:
        return False
    for item in tabulist:
        # Check if the primary nodes and destination match
        if (
            move.start[0] == item.start[0]
            and move.start[1] == item.start[1]
            and move.end == item.end
        ):
            # Simple check for odd move types
            if move_type % 2 == 1:
                return True
            # Reciprocal link check for even move types (0 and 2)
            else:
                if (
                    move.back[1] == item.back[0]
                    and move.front[1] == item.front[0]
                    and move.back[0] == item.back[1]
                    and move.front[0] == item.front[1]
                ):
                    return True
    return False


# --- Tabu Swap Delta Calculations ---


def swapTabu20(
    Tracks: List[List[int]], iTrack: int, position: int, kTrack: int, swapPo: int
) -> Tuple[float, float]:
    tempTracki = list(Tracks[iTrack])
    tempTrackk = list(Tracks[kTrack])
    temp = tempTracki[position]
    temp2 = tempTracki[position + 1]

    if iTrack > kTrack:
        tempTrackk.insert(swapPo, temp2)
        tempTrackk.insert(swapPo, temp)
        del tempTracki[position : position + 2]
    elif iTrack < kTrack:
        tempTrackk.insert(swapPo + 1, temp2)
        tempTrackk.insert(swapPo + 1, temp)
        del tempTracki[position : position + 2]
    else:
        if swapPo > position:
            tempTracki.insert(swapPo + 1, temp2)
            tempTracki.insert(swapPo + 1, temp)
            del tempTracki[position : position + 2]
        else:
            del tempTracki[position : position + 2]
            tempTracki.insert(swapPo, temp2)
            tempTracki.insert(swapPo, temp)

    iResult = track_result(tempTracki, iTrack)
    kResult = track_result(tempTrackk, kTrack)

    if iResult[0] == 0 or kResult[0] == 0:
        return (0.0, 0.0)

    if iTrack != kTrack:
        return (max(iResult[0], kResult[0]), iResult[1] + kResult[1])
    else:
        return (iResult[0], iResult[1])


def swapTabu21(
    Tracks: List[List[int]], iTrack: int, position: int, kTrack: int, swapPo: int
) -> Tuple[float, float]:
    tempTracki = list(Tracks[iTrack])
    tempTrackk = list(Tracks[kTrack])
    temp = tempTracki[position]
    temp2 = tempTracki[position + 1]
    temp3 = tempTrackk[swapPo]

    if iTrack != kTrack:
        tempTracki[position] = temp3
        tempTrackk[swapPo] = temp
        tempTrackk.insert(swapPo + 1, temp2)
        del tempTracki[position + 1]
    else:
        tempTracki[position] = temp3
        tempTracki[swapPo] = temp
        if swapPo > position:
            tempTracki.insert(swapPo + 1, temp2)
            del tempTracki[position + 1]
        else:
            del tempTracki[position + 1]
            tempTracki.insert(swapPo + 1, temp2)

    iResult = track_result(tempTracki, iTrack)
    kResult = track_result(tempTrackk, kTrack)

    if iResult[0] == 0 or kResult[0] == 0:
        return (0.0, 0.0)

    if iTrack != kTrack:
        return (max(iResult[0], kResult[0]), iResult[1] + kResult[1])
    else:
        return iResult


def swapTabu10(
    Tracks: List[List[int]], iTrack: int, position: int, kTrack: int, swapPo: int
) -> Tuple[float, float]:
    tempTracki = list(Tracks[iTrack])
    tempTrackk = list(Tracks[kTrack])
    temp = tempTracki[position]

    if iTrack > kTrack:
        tempTrackk.insert(swapPo, temp)
        del tempTracki[position]
    elif iTrack < kTrack:
        tempTrackk.insert(swapPo + 1, temp)
        del tempTracki[position]
    else:
        if swapPo > position:
            tempTracki.insert(swapPo + 1, temp)
            del tempTracki[position]
        else:
            del tempTracki[position]
            tempTracki.insert(swapPo, temp)

    iResult = track_result(tempTracki, iTrack)
    kResult = track_result(tempTrackk, kTrack)

    if iResult[0] == 0 or kResult[0] == 0:
        return (0.0, 0.0)

    if iTrack != kTrack:
        return (max(iResult[0], kResult[0]), iResult[1] + kResult[1])
    else:
        return iResult


def swapTabu11(
    Tracks: List[List[int]], iTrack: int, position: int, kTrack: int, swapPo: int
) -> Tuple[float, float]:
    tempTracki = list(Tracks[iTrack])
    tempTrackk = list(Tracks[kTrack])
    temp = tempTracki[position]
    temp2 = tempTrackk[swapPo]

    tempTracki[position] = temp2
    tempTrackk[swapPo] = temp

    iResult = track_result(tempTracki, iTrack)
    kResult = track_result(tempTrackk, kTrack)

    if iResult[0] == 0 or kResult[0] == 0:
        return (0.0, 0.0)

    if iTrack != kTrack:
        return (max(iResult[0], kResult[0]), iResult[1] + kResult[1])
    else:
        return iResult


def swapTabu(
    Tracks: List[List[int]],
    iTrack: int,
    position: int,
    kTrack: int,
    swapPo: int,
    move_type: int,
) -> Tuple[float, float]:
    if move_type == 0:
        return swapTabu20(Tracks, iTrack, position, kTrack, swapPo)
    elif move_type == 1:
        return swapTabu21(Tracks, iTrack, position, kTrack, swapPo)
    elif move_type == 2:
        return swapTabu10(Tracks, iTrack, position, kTrack, swapPo)
    else:
        return swapTabu11(Tracks, iTrack, position, kTrack, swapPo)


# --- Feasibility Check Functions ---


def checkswaptabu20(
    temp: List[int],
    Tracks: List[List[int]],
    endpoint: List[int],
    Tracknum: int,
    position: int,
    swapTrack: int,
    swapPo: int,
) -> bool:
    if position == len(Tracks[Tracknum]) - 1:
        return False
    if swapTrack >= params.NUM_TRUCKS and (
        params.customers[Tracks[Tracknum][position]].OnlyByTruck == 1
        or params.customers[Tracks[Tracknum][position + 1]].OnlyByTruck == 1
    ):
        return False
    if (
        Tracks[Tracknum][position] > params.NUM_CUS
        or Tracks[Tracknum][position + 1] > params.NUM_CUS
    ):
        return False

    s1, s2 = (
        temp[endpoint[Tracknum] + 1 + position],
        temp[endpoint[Tracknum] + 1 + position + 1],
    )
    pos1, pos2 = endpoint[Tracknum] + 1 + position, endpoint[swapTrack] + 1 + swapPo

    if pos2 == pos1 or pos2 == pos1 + 1:
        return False

    if pos2 > pos1 + 1:
        temp.insert(endpoint[swapTrack] + swapPo + 1, s2)
        temp.insert(endpoint[swapTrack] + swapPo + 1, s1)
        del temp[
            endpoint[Tracknum] + 1 + position : endpoint[Tracknum] + 1 + position + 2
        ]
    else:
        del temp[
            endpoint[Tracknum] + 1 + position : endpoint[Tracknum] + 1 + position + 2
        ]
        temp.insert(endpoint[swapTrack] + swapPo + 1, s2)
        temp.insert(endpoint[swapTrack] + swapPo + 1, s1)

    return check_route(temp)


def checkswaptabu21(
    temp: List[int],
    Tracks: List[List[int]],
    endpoint: List[int],
    Tracknum: int,
    position: int,
    swapTrack: int,
    swapPo: int,
) -> bool:
    if position == len(Tracks[Tracknum]) - 1:
        return False
    if swapTrack >= params.NUM_TRUCKS and (
        params.customers[Tracks[Tracknum][position]].OnlyByTruck == 1
        or params.customers[Tracks[Tracknum][position + 1]].OnlyByTruck == 1
    ):
        return False
    if (
        Tracknum >= params.NUM_TRUCKS
        and params.customers[Tracks[swapTrack][swapPo]].OnlyByTruck == 1
    ):
        return False
    if (
        Tracks[Tracknum][position] > params.NUM_CUS
        or Tracks[Tracknum][position + 1] > params.NUM_CUS
        or Tracks[swapTrack][swapPo] > params.NUM_CUS
    ):
        return False

    s = temp[endpoint[Tracknum] + 1 + position + 1]
    pos1, pos2 = endpoint[Tracknum] + 1 + position, endpoint[swapTrack] + 1 + swapPo

    if pos1 == pos2 or pos1 + 1 == pos2:
        return False

    temp[pos1], temp[pos2] = temp[pos2], temp[pos1]
    if pos1 > pos2:
        del temp[pos1 + 1]
        temp.insert(pos2 + 1, s)
    else:
        temp.insert(pos2 + 1, s)
        del temp[pos1 + 1]

    return check_route(temp)


def checkswaptabu10(
    temp: List[int],
    Tracks: List[List[int]],
    endpoint: List[int],
    Tracknum: int,
    position: int,
    swapTrack: int,
    swapPo: int,
) -> bool:
    if Tracks[Tracknum][position] > params.NUM_CUS:
        return False
    if (
        swapTrack >= params.NUM_TRUCKS
        and params.customers[Tracks[Tracknum][position]].OnlyByTruck == 1
    ):
        return False

    pos1, pos2 = endpoint[Tracknum] + 1 + position, endpoint[swapTrack] + 1 + swapPo
    s = temp[pos1]

    if pos2 == pos1:
        return False
    if pos2 > pos1:
        temp.insert(pos2 + 1, s)
        del temp[pos1]
    else:
        del temp[pos1]
        temp.insert(pos2, s)

    return check_route(temp)


def checkswaptabu11(
    temp: List[int],
    Tracks: List[List[int]],
    endpoint: List[int],
    Tracknum: int,
    position: int,
    swapTrack: int,
    swapPo: int,
) -> bool:
    if (
        swapTrack >= params.NUM_TRUCKS
        and params.customers[Tracks[Tracknum][position]].OnlyByTruck == 1
    ):
        return False
    if (
        Tracknum >= params.NUM_TRUCKS
        and params.customers[Tracks[swapTrack][swapPo]].OnlyByTruck == 1
    ):
        return False
    if (
        Tracks[Tracknum][position] > params.NUM_CUS
        or Tracks[swapTrack][swapPo] > params.NUM_CUS
    ):
        return False

    pos1, pos2 = endpoint[Tracknum] + 1 + position, endpoint[swapTrack] + 1 + swapPo
    temp[pos1], temp[pos2] = temp[pos2], temp[pos1]

    return check_route(temp)


def checkswaptabu(
    temp: List[int],
    Tracks: List[List[int]],
    endpoint: List[int],
    Tracknum: int,
    position: int,
    swapTrack: int,
    swapPo: int,
    move_type: int,
) -> bool:
    if not Tracks[Tracknum]:
        return False
    temp_copy = list(temp)
    if move_type == 0:
        return checkswaptabu20(
            temp_copy, Tracks, endpoint, Tracknum, position, swapTrack, swapPo
        )
    elif move_type == 1:
        return checkswaptabu21(
            temp_copy, Tracks, endpoint, Tracknum, position, swapTrack, swapPo
        )
    elif move_type == 2:
        return checkswaptabu10(
            temp_copy, Tracks, endpoint, Tracknum, position, swapTrack, swapPo
        )
    else:
        return checkswaptabu11(
            temp_copy, Tracks, endpoint, Tracknum, position, swapTrack, swapPo
        )


# --- Move Application Functions ---


def swap20(
    temp: List[int],
    endpoint: List[int],
    Tracknum: int,
    position: int,
    swapTrack: int,
    swapPo: int,
) -> List[int]:
    res = list(temp)
    s1, s2 = (
        res[endpoint[Tracknum] + 1 + position],
        res[endpoint[Tracknum] + 1 + position + 1],
    )
    pos1, pos2 = endpoint[Tracknum] + 1 + position, endpoint[swapTrack] + 1 + swapPo

    if pos2 > pos1 + 1:
        res.insert(pos2 + 1, s2)
        res.insert(pos2 + 1, s1)
        del res[pos1 : pos1 + 2]
    else:
        del res[pos1 : pos1 + 2]
        res.insert(pos2, s2)
        res.insert(pos2, s1)
    return res


def swap21(
    temp: List[int],
    endpoint: List[int],
    Tracknum: int,
    position: int,
    swapTrack: int,
    swapPo: int,
) -> List[int]:
    res = list(temp)
    s = res[endpoint[Tracknum] + 1 + position + 1]
    pos1, pos2 = endpoint[Tracknum] + 1 + position, endpoint[swapTrack] + 1 + swapPo
    res[pos1], res[pos2] = res[pos2], res[pos1]
    if pos1 > pos2:
        del res[pos1 + 1]
        res.insert(pos2 + 1, s)
    else:
        res.insert(pos2 + 1, s)
        del res[pos1 + 1]
    return res


def swap10TB(
    temp: List[int],
    endpoint: List[int],
    Tracknum: int,
    position: int,
    swapTrack: int,
    swapPo: int,
) -> List[int]:
    res = list(temp)
    pos1, pos2 = endpoint[Tracknum] + 1 + position, endpoint[swapTrack] + 1 + swapPo
    s = res[pos1]
    if pos2 > pos1:
        res.insert(pos2 + 1, s)
        del res[pos1]
    else:
        del res[pos1]
        res.insert(pos2, s)
    return res


def swap11TB(
    temp: List[int],
    endpoint: List[int],
    Tracknum: int,
    position: int,
    swapTrack: int,
    swapPo: int,
) -> List[int]:
    res = list(temp)
    pos1, pos2 = endpoint[Tracknum] + 1 + position, endpoint[swapTrack] + 1 + swapPo
    res[pos1], res[pos2] = res[pos2], res[pos1]
    return res


# --- Main Search Logic ---


def tabu_search(initial_solution: List[int], max_iterations: int) -> List["Individual"]:
    current_solution = list(initial_solution)
    tabu_list = [[] for _ in range(4)]
    Tabupareto = []

    initial_fit = calculate_fitness(initial_solution)
    best_fit1, best_fit2 = initial_fit[0], initial_fit[1]

    with open(params.output_tblog, "a") as outputFile:
        for iter_num in range(max_iterations):
            outputFile.write(
                f"Iteration {iter_num}:\nCurrent solution: {' '.join(map(str, current_solution))}\n"
            )
            start_time = time.perf_counter()

            neighbors = []
            taburecord = []
            move_type = random.randint(0, 3)
            fitbefore = calculate_fitness(current_solution)

            tracks_data = split_tracks(current_solution)
            Tracks, endpoint = tracks_data[0], tracks_data[1]

            obj1List, obj2List = [], []
            for j in range(params.NUM_TRUCKS + params.NUM_DRONES):
                res = track_result(Tracks[j], j)
                obj1List.append(res[0])
                obj2List.append(res[1])

            # Neighborhood Exploration
            for i in range(params.NUM_DRONES + params.NUM_TRUCKS):
                for j in range(len(Tracks[i])):
                    for k in range(params.NUM_DRONES + params.NUM_TRUCKS):
                        for l in range(len(Tracks[k])):
                            if checkswaptabu(
                                current_solution,
                                Tracks,
                                endpoint,
                                i,
                                j,
                                k,
                                l,
                                move_type,
                            ):
                                fitchange = swapTabu(Tracks, i, j, k, l, move_type)
                                if fitchange[0] != 0 and fitchange[1] != 0:
                                    # Calculate Delta Objective 2
                                    if i != k:
                                        fitafter2 = (
                                            fitbefore[1]
                                            + fitchange[1]
                                            - obj2List[i]
                                            - obj2List[k]
                                        )
                                    else:
                                        fitafter2 = (
                                            fitbefore[1] + fitchange[1] - obj2List[i]
                                        )

                                    # Calculate Delta Objective 1
                                    obj1new = [
                                        obj1List[t]
                                        for t in range(len(obj1List))
                                        if t != i and t != k
                                    ]
                                    obj1new.append(fitchange[0])
                                    fitafter1 = max(obj1new)

                                    # Improvement / Aspiration Criteria
                                    if (fitbefore[0] - fitafter1 > 1e-3) or (
                                        fitbefore[1] - fitafter2 > 1e-3
                                    ):
                                        new_temp = list(current_solution)
                                        if move_type == 0:
                                            new_temp = swap20(
                                                new_temp, endpoint, i, j, k, l
                                            )
                                        elif move_type == 1:
                                            new_temp = swap21(
                                                new_temp, endpoint, i, j, k, l
                                            )
                                        elif move_type == 2:
                                            new_temp = swap10TB(
                                                new_temp, endpoint, i, j, k, l
                                            )
                                        else:
                                            new_temp = swap11TB(
                                                new_temp, endpoint, i, j, k, l
                                            )

                                        newsol = Individual()
                                        newsol.route = new_temp
                                        newsol.fitness1, newsol.fitness2 = (
                                            fitafter1,
                                            fitafter2,
                                        )
                                        neighbors.append(newsol)

                                        # Update Move Record for Tabu logic
                                        newrecord = TabuRecord()
                                        newrecord.indi = newsol
                                        newrecord.record.start[0] = Tracks[i][j]
                                        newrecord.record.start[1] = (
                                            Tracks[i][j + 1]
                                            if (
                                                move_type < 2 and j + 1 < len(Tracks[i])
                                            )
                                            else -1
                                        )
                                        newrecord.record.end = Tracks[k][l]

                                        p1, p2 = (
                                            endpoint[i] + 1 + j,
                                            endpoint[k] + 1 + l,
                                        )
                                        if move_type in [0, 2]:
                                            newrecord.record.front[0] = (
                                                current_solution[p1 - 1]
                                                if p1 > 0
                                                else -1
                                            )
                                            offset = 2 if move_type == 0 else 1
                                            newrecord.record.back[0] = (
                                                current_solution[p1 + offset]
                                                if p1 + offset < params.total_node
                                                else -1
                                            )

                                            if p1 > p2:
                                                newrecord.record.back[1] = (
                                                    current_solution[p2]
                                                )
                                                newrecord.record.front[1] = (
                                                    current_solution[p2 - 1]
                                                    if p2 > 0
                                                    else -1
                                                )
                                            else:
                                                newrecord.record.front[1] = (
                                                    current_solution[p2]
                                                )
                                                newrecord.record.back[1] = (
                                                    current_solution[p2 + 1]
                                                    if p2 < params.total_node - 1
                                                    else -1
                                                )
                                        taburecord.append(newrecord)

            # Pareto Update and Selection
            if not neighbors:
                continue

            front = fast_non_dominated_sort(neighbors)
            for idx in front[0]:
                if not in_pareto(neighbors[idx], Tabupareto):
                    update_pareto(neighbors[idx], Tabupareto)

            check_found, cnt = False, 0
            while not check_found and cnt < 20:
                check_found = True
                sel = random.randint(0, len(front[0]) - 1)
                best_neighbor = neighbors[front[0][sel]]
                m_record = taburecord[front[0][sel]].record

                # If improvement exists or not tabu
                if (best_fit1 - best_neighbor.fitness1 > 1e-3) and (
                    best_fit2 - best_neighbor.fitness2 > 1e-3
                ):
                    current_solution = list(best_neighbor.route)
                    best_fit1, best_fit2 = (
                        best_neighbor.fitness1,
                        best_neighbor.fitness2,
                    )
                    if not in_tabu_list(tabu_list[move_type], m_record, move_type):
                        tabu_list[move_type].append(m_record)
                else:
                    if not in_tabu_list(tabu_list[move_type], m_record, move_type):
                        tabu_list[move_type].append(m_record)
                        current_solution = list(best_neighbor.route)
                    else:
                        check_found, cnt = False, cnt + 1

            if len(tabu_list[move_type]) > 5:
                tabu_list[move_type].pop(0)

            duration = (time.perf_counter() - start_time) * 1000
            outputFile.write(f"Type: {move_type}\nTime: {duration:.4f}ms\n")

    return Tabupareto


def tabu_search2(
    initial_solution: List[int], max_iterations: int
) -> List["Individual"]:
    # Time management setup
    start_tb = time.time()
    tb_time = 0.0
    end_iter = 0

    current_solution = list(initial_solution)
    # tabu_list[4] for 4 movement types
    tabu_list: List[List[MoveRecord]] = [[] for _ in range(4)]
    tabu_pareto: List[Individual] = []

    # Fitness initialization
    initial_fitness = calculate_fitness(initial_solution)
    best_fit1, best_fit2 = initial_fitness[0], initial_fitness[1]

    no_improve_count = 0

    # Open log file in append mode
    with open(params.output_tblog, "a") as output_file:
        for iter_idx in range(max_iterations):
            output_file.write(f"Iteration {iter_idx}:\n")
            output_file.write(
                f"Current solution: {' '.join(map(str, current_solution))}\n"
            )

            # Start timer for the current iteration (equivalent to high_resolution_clock)
            iter_start_time = time.perf_counter()

            neighbors: List[Individual] = []
            tabu_records: List[TabuRecord] = []

            # Randomly select move type (0-3)
            move_type = random.randint(0, 3)

            fit_before = calculate_fitness(current_solution)
            tracks_data = split_tracks(current_solution)
            tracks, endpoint = tracks_data[0], tracks_data[1]

            # Pre-calculate objective lists
            obj1_list = []
            obj2_list = []
            for j in range(params.NUM_TRUCKS + params.NUM_DRONES):
                ori_track_res = track_result(tracks[j], j)
                obj1_list.append(ori_track_res[0])
                obj2_list.append(ori_track_res[1])

            # Neighborhood Search
            for i in range(params.NUM_DRONES + params.NUM_TRUCKS):
                for j in range(len(tracks[i])):
                    for k in range(params.NUM_DRONES + params.NUM_TRUCKS):
                        for l in range(len(tracks[k])):
                            temp_route = list(current_solution)

                            if checkswaptabu(
                                temp_route, tracks, endpoint, i, j, k, l, move_type
                            ):
                                fit_change = swapTabu(tracks, i, j, k, l, move_type)

                                if fit_change[0] != 0 and fit_change[1] != 0:
                                    # Calculate Objective 2 (Sum)
                                    if i != k:
                                        fit_after_2 = (
                                            fit_before[1]
                                            + fit_change[1]
                                            - obj2_list[i]
                                            - obj2_list[k]
                                        )
                                    else:
                                        fit_after_2 = (
                                            fit_before[1] + fit_change[1] - obj2_list[i]
                                        )

                                    # Calculate Objective 1 (Max)
                                    obj1_new = [
                                        obj1_list[t]
                                        for t in range(len(obj1_list))
                                        if t != i and t != k
                                    ]
                                    obj1_new.append(fit_change[0])
                                    fit_after_1 = max(obj1_new)

                                    # Improvement / Aspiration Criteria (Epsilon 1e-3)
                                    if (fit_before[0] - fit_after_1 > 1e-3) or (
                                        fit_before[1] - fit_after_2 > 1e-3
                                    ):
                                        # Apply the move to get the new route
                                        if move_type == 0:
                                            modified_route = swap20(
                                                temp_route, endpoint, i, j, k, l
                                            )
                                        elif move_type == 1:
                                            modified_route = swap21(
                                                temp_route, endpoint, i, j, k, l
                                            )
                                        elif move_type == 2:
                                            modified_route = swap10TB(
                                                temp_route, endpoint, i, j, k, l
                                            )
                                        else:
                                            modified_route = swap11TB(
                                                temp_route, endpoint, i, j, k, l
                                            )

                                        new_sol = Individual()
                                        new_sol.route = modified_route
                                        new_sol.fitness1 = fit_after_1
                                        new_sol.fitness2 = fit_after_2
                                        new_sol.tabu_search = 0
                                        new_sol.local_search = 0
                                        neighbors.append(new_sol)

                                        # Create MoveRecord and TabuRecord
                                        new_record = TabuRecord()
                                        new_record.indi = new_sol
                                        new_record.record.start[0] = tracks[i][j]

                                        if move_type < 2:
                                            new_record.record.start[1] = (
                                                tracks[i][j + 1]
                                                if (j + 1) < len(tracks[i])
                                                else -1
                                            )
                                        else:
                                            new_record.record.start[1] = -1

                                        new_record.record.end = tracks[k][l]

                                        # Record context for reverse-move checks
                                        pos1, pos2 = (
                                            endpoint[i] + 1 + j,
                                            endpoint[k] + 1 + l,
                                        )
                                        if move_type in [0, 2]:
                                            new_record.record.front[0] = (
                                                current_solution[pos1 - 1]
                                                if pos1 > 0
                                                else -1
                                            )

                                            offset = 2 if move_type == 0 else 1
                                            new_record.record.back[0] = (
                                                current_solution[pos1 + offset]
                                                if (pos1 + offset) < params.total_node
                                                else -1
                                            )

                                            if pos1 > pos2:
                                                new_record.record.back[1] = (
                                                    current_solution[pos2]
                                                )
                                                new_record.record.front[1] = (
                                                    current_solution[pos2 - 1]
                                                    if pos2 > 0
                                                    else -1
                                                )
                                            else:
                                                new_record.record.front[1] = (
                                                    current_solution[pos2]
                                                )
                                                new_record.record.back[1] = (
                                                    current_solution[pos2 + 1]
                                                    if (pos2 + 1) < params.total_node
                                                    else -1
                                                )

                                        tabu_records.append(new_record)

            # Sort neighbors using NSGA-II fast non-dominated sort
            fronts = fast_non_dominated_sort(neighbors)

            if not neighbors:
                no_improve_count += 1
            else:
                update_occurred = False
                for idx in fronts[0]:
                    if not in_pareto(neighbors[idx], tabu_pareto):
                        update_occurred = True
                        update_pareto(neighbors[idx], tabu_pareto)

                if update_occurred:
                    no_improve_count = 0
                    tb_time = time.time() - start_tb
                else:
                    no_improve_count += 1

                # Selection logic (try up to 20 times to find a non-tabu or improving move)
                check_found = False
                cnt = 0
                while not check_found and cnt < 20:
                    check_found = True
                    select_idx = random.randint(0, len(fronts[0]) - 1)
                    neighbor_idx = fronts[0][select_idx]
                    best_neighbor = neighbors[neighbor_idx]
                    m_record = tabu_records[neighbor_idx].record

                    # Strict Improvement check
                    if (
                        best_fit1 - best_neighbor.fitness1 > 1e-3
                        and best_fit2 - best_neighbor.fitness2 > 1e-3
                    ):
                        current_solution = list(best_neighbor.route)
                        best_fit1, best_fit2 = (
                            best_neighbor.fitness1,
                            best_neighbor.fitness2,
                        )

                        if not in_tabu_list(tabu_list[move_type], m_record, move_type):
                            tabu_list[move_type].append(m_record)
                    else:
                        # Non-improving move: check Tabu list
                        if not in_tabu_list(tabu_list[move_type], m_record, move_type):
                            tabu_list[move_type].append(m_record)
                            current_solution = list(best_neighbor.route)
                        else:
                            check_found = False
                            cnt += 1

                    if check_found:
                        output_file.write(
                            f"New Solution: {' '.join(map(str, current_solution))}\n"
                        )
                        output_file.write(f"Move: {m_record.start[0]} {m_record.end}\n")

                # Maintain Tabu list tenure (Limit 5)
                if len(tabu_list[move_type]) > 5:
                    tabu_list[move_type].pop(0)

            # Finalize iteration logging
            iter_duration = (time.perf_counter() - iter_start_time) * 1000  # ms
            output_file.write(f"Type: {move_type}\nTime: {iter_duration:.4f}ms\n")

            # Global time limit check
            if (time.time() - start_tb) > params.time_limit:
                end_iter = iter_idx + 1
                break

            if iter_idx == max_iterations - 1:
                end_iter = max_iterations

    # Final summary output
    output(
        tabu_pareto,
        time.time() - start_tb,
        end_iter,
        0,
        tb_time,
        end_iter - no_improve_count,
    )
    return tabu_pareto
