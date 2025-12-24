from localsearch_utils import split_tracks, track_result, compute_linear
from fitness import calculate_fitness
from route_check import feasible_route
from pareto_update import in_pareto, update_pareto
import parameters as params
from structs import Individual


def check1(i, k, i_track, k_track, temp_route, ban_val):
    if temp_route[i] == 0 or temp_route[i] > params.NUM_CUS:
        return False
    if temp_route[k] == 0 or temp_route[k] > params.NUM_CUS:
        return False
    if ban_val == 1:
        return False
    if (
        i_track >= params.NUM_TRUCKS
        and params.customers[temp_route[k]].only_by_truck == 1
    ):
        return False
    if (
        k_track >= params.NUM_TRUCKS
        and params.customers[temp_route[i]].only_by_truck == 1
    ):
        return False
    return True


def check2(i, k_track, temp_route, ban_val):
    if temp_route[i] == 0 or temp_route[i] > params.NUM_CUS:
        return False
    if (
        k_track >= params.NUM_TRUCKS
        and params.customers[temp_route[i]].only_by_truck == 1
    ):
        return False
    if ban_val == 1:
        return False
    return True


def swap11(track_i, track_k, i_track_id, k_track_id, idx_i, idx_k):
    temp_i = list(track_i)
    temp_k = list(track_k)
    if i_track_id == k_track_id:
        temp_i[idx_i], temp_i[idx_k] = temp_i[idx_k], temp_i[idx_i]
        return track_result(temp_i, i_track_id)

    temp_i[idx_i], temp_k[idx_k] = temp_k[idx_k], temp_i[idx_i]
    res_i = track_result(temp_i, i_track_id)
    res_k = track_result(temp_k, k_track_id)

    if res_i[0] == 0 or res_k[0] == 0:
        return (0.0, 0.0)
    return max(res_i[0], res_k[0]), res_i[1] + res_k[1]


def swap10(track_i, track_k, i_track_id, k_track_id, idx_i, idx_k):
    temp_i = list(track_i)
    temp_k = list(track_k)

    val_i = temp_i[idx_i]
    if i_track_id == k_track_id:
        if idx_k > idx_i:
            temp_i.insert(idx_k + 1, val_i)
            temp_i.pop(idx_i)
        else:
            temp_i.insert(idx_k, val_i)
            temp_i.pop(idx_i + 1)
        return track_result(temp_i, i_track_id)

    if i_track_id < k_track_id:
        temp_k.insert(idx_k + 1, val_i)
        temp_i.pop(idx_i)
    else:
        temp_k.insert(idx_k, val_i)
        temp_i.pop(idx_i)

    res_i = track_result(temp_i, i_track_id)
    res_k = track_result(temp_k, k_track_id)
    if res_i[0] == 0 or res_k[0] == 0:
        return (0.0, 0.0)
    return max(res_i[0], res_k[0]), res_i[1] + res_k[1]


def local_searcher1(tour, choice, search_type, pareto):
    improved = True
    count = 0
    best_improve = list(tour)
    copy_tour = list(tour)
    ban_list = [[0 for _ in range(1000)] for _ in range(1000)]
    imove, kmove = -1, -1

    while improved and count < 100:
        count += 1
        improved = False
        fit_before = calculate_fitness(copy_tour)
        tracks, endpoints = split_tracks(copy_tour)

        obj1_list, obj2_list = [], []
        for j in range(params.NUM_TRUCKS + params.NUM_DRONES):
            res = track_result(tracks[j], j)
            obj1_list.append(res[0])
            obj2_list.append(res[1])

        if search_type == 1:
            fit_best = compute_linear(fit_before[0], fit_before[1])
        elif search_type == 0:
            fit_best = fit_before[1]
        else:
            fit_best = fit_before[0]

        fit_init = fit_best

        for i in range(params.total_node - 1, -1, -1):
            newc = 0
            for k in range(params.total_node - 1, -1, -1):
                if i == k:
                    continue

                temp_route_for_feasibility = list(copy_tour)
                i_track = next(idx for idx, end in enumerate(endpoints) if i < end) - 1
                k_track = next(idx for idx, end in enumerate(endpoints) if k < end) - 1

                fit_after = [0.0, 0.0]
                fit_change = (0.0, 0.0)
                error = 0

                # --- NEIGHBORHOOD OPERATIONS ---
                if choice in [50, 150]:  # Swap
                    if check1(i, k, i_track, k_track, copy_tour, ban_list[i][k]):
                        fit_change = swap11(
                            tracks[i_track],
                            tracks[k_track],
                            i_track,
                            k_track,
                            i - endpoints[i_track] - 1,
                            k - endpoints[k_track] - 1,
                        )
                        if fit_change[0] != 0:
                            if i_track == k_track:
                                fit_after[1] = (
                                    fit_before[1] + fit_change[1] - obj2_list[i_track]
                                )
                                obj1_new = [
                                    obj1_list[j]
                                    for j in range(len(obj1_list))
                                    if j != i_track
                                ]
                                obj1_new.append(fit_change[0])
                                fit_after[0] = max(obj1_new)
                            else:
                                fit_after[1] = (
                                    fit_before[1]
                                    + fit_change[1]
                                    - obj2_list[i_track]
                                    - obj2_list[k_track]
                                )
                                obj1_new = [
                                    obj1_list[j]
                                    for j in range(len(obj1_list))
                                    if j != i_track and j != k_track
                                ]
                                obj1_new.append(fit_change[0])
                                fit_after[0] = max(obj1_new)
                        else:
                            error = 1
                    else:
                        error = 1
                else:  # Relocate
                    if check2(i, k_track, copy_tour, ban_list[i][k]):
                        if k == endpoints[k_track + 1]:
                            if i < k:
                                if check2(i, k_track + 1, copy_tour, ban_list[i][k]):
                                    fit_change = swap10(
                                        tracks[i_track],
                                        tracks[k_track + 1],
                                        i_track,
                                        k_track + 1,
                                        i - endpoints[i_track] - 1,
                                        -1,
                                    )
                                else:
                                    fit_change = (0.0, 0.0)
                            else:
                                fit_change = swap10(
                                    tracks[i_track],
                                    tracks[k_track],
                                    i_track,
                                    k_track,
                                    i - endpoints[i_track] - 1,
                                    len(tracks[k_track]),
                                )
                        else:
                            fit_change = swap10(
                                tracks[i_track],
                                tracks[k_track],
                                i_track,
                                k_track,
                                i - endpoints[i_track] - 1,
                                k - endpoints[k_track] - 1,
                            )

                        if fit_change[0] != 0:
                            if i_track == k_track and k != endpoints[k_track + 1]:
                                fit_after[1] = (
                                    fit_before[1] + fit_change[1] - obj2_list[i_track]
                                )
                                obj1_new = [
                                    obj1_list[j]
                                    for j in range(len(obj1_list))
                                    if j != i_track
                                ]
                                obj1_new.append(fit_change[0])
                                fit_after[0] = max(obj1_new)
                            elif (
                                i_track == k_track and k == endpoints[k_track + 1]
                            ) or (i_track < k_track and k == endpoints[k_track + 1]):
                                fit_after[1] = (
                                    fit_before[1]
                                    + fit_change[1]
                                    - obj2_list[i_track]
                                    - obj2_list[k_track + 1]
                                )
                                obj1_new = [
                                    obj1_list[j]
                                    for j in range(len(obj1_list))
                                    if j != i_track and j != k_track + 1
                                ]
                                obj1_new.append(fit_change[0])
                                fit_after[0] = max(obj1_new)
                            else:
                                fit_after[1] = (
                                    fit_before[1]
                                    + fit_change[1]
                                    - obj2_list[i_track]
                                    - obj2_list[k_track]
                                )
                                obj1_new = [
                                    obj1_list[j]
                                    for j in range(len(obj1_list))
                                    if j != i_track and j != k_track
                                ]
                                obj1_new.append(fit_change[0])
                                fit_after[0] = max(obj1_new)
                        else:
                            error = 1
                    else:
                        error = 1

                # --- Acceptance ---
                fit_temp = float("inf")  # Default high value if error occurs
                if error == 0:
                    # Construct temp route to check full constraints
                    if choice in [50, 150]:
                        temp_route_for_feasibility[i], temp_route_for_feasibility[k] = (
                            temp_route_for_feasibility[k],
                            temp_route_for_feasibility[i],
                        )
                    else:
                        val_i = temp_route_for_feasibility.pop(i)
                        temp_route_for_feasibility.insert(k, val_i)

                    if search_type == 1:
                        fit_temp = compute_linear(fit_after[0], fit_after[1])
                    elif search_type == 0:
                        fit_temp = fit_after[1]
                    else:
                        fit_temp = fit_after[0]

                    if (fit_best - fit_temp > 1e-3) and feasible_route(
                        temp_route_for_feasibility
                    ):
                        improved = True
                        best_improve = list(temp_route_for_feasibility)
                        imove, kmove = i, k
                        newc = 1
                        fit_best = fit_temp

                        newsol = Individual()
                        newsol.route = list(temp_route_for_feasibility)
                        newsol.fitness1, newsol.fitness2 = fit_after[0], fit_after[1]
                        if not in_pareto(newsol, pareto):
                            update_pareto(newsol, pareto)

                if fit_init < fit_temp or error == 1:
                    ban_list[i][k] = 1

            copy_tour = list(best_improve)

            if newc == 1:
                # Resetting ban list for indices affected by the move
                for s in range(params.total_node):
                    ban_list[imove][s] = ban_list[kmove][s] = 0
                    ban_list[s][imove] = ban_list[s][kmove] = 0
                    if imove > 0:
                        ban_list[imove - 1][s] = ban_list[s][imove - 1] = 0
                    if imove < params.total_node - 1:
                        ban_list[imove + 1][s] = ban_list[s][imove + 1] = 0
                    if kmove > 0:
                        ban_list[kmove - 1][s] = ban_list[s][kmove - 1] = 0
                    if kmove < params.total_node - 1:
                        ban_list[kmove + 1][s] = ban_list[s][kmove + 1] = 0

    return best_improve
