from domination_check import check_domination


def dominated_in_pareto(indi, pareto_list):
    """
    Kiểm tra xem cá thể có bị chi phối bởi bất kỳ ai trong tập Pareto không.
    Tương ứng với dominatedinpareto trong paretoUpdate.h.
    """
    if not pareto_list:
        return False
    for p in pareto_list:
        if check_domination(p, indi):
            return True
    return False


def dominated_in_tabu_pareto(indi, tabu_pareto):
    """
    Kiểm tra xem cá thể có bị chi phối bởi bất kỳ ai trong tập Tabu Pareto không.
    Tương ứng với dominatedinTabupareto trong paretoUpdate.h.
    """
    if not tabu_pareto:
        return False
    for p in tabu_pareto:
        if check_domination(p, indi):
            return True
    return False


def update_pareto(indi, pareto_list):
    """
    Cập nhật tập lưu trữ Pareto với một cá thể mới.
    1. Nếu bị chi phối bởi tập cũ thì bỏ qua.
    2. Nếu không, xóa tất cả các cá thể trong tập cũ bị cá thể mới chi phối.
    3. Thêm cá thể mới vào tập.
    Tương ứng với updatepareto trong paretoUpdate.h.
    """
    if not dominated_in_pareto(indi, pareto_list):
        # Loại bỏ những cá thể bị 'indi' chi phối (thay thế logic sort/erase trong C++)
        pareto_list[:] = [p for p in pareto_list if not check_domination(indi, p)]
        # Thêm cá thể mới
        pareto_list.append(indi)


def update_tabu_pareto(indi, tabu_pareto):
    """
    Cập nhật tập lưu trữ Tabu Pareto với một cá thể mới.
    Tương ứng với updateTabupareto trong paretoUpdate.h.
    """
    if not dominated_in_tabu_pareto(indi, tabu_pareto):
        # Loại bỏ những cá thể bị chi phối bởi indi
        tabu_pareto[:] = [p for p in tabu_pareto if not check_domination(indi, p)]
        # Thêm cá thể mới
        tabu_pareto.append(indi)


def in_pareto(indi, pareto_list):
    """
    Kiểm tra xem một cá thể có cùng giá trị fitness đã tồn tại trong tập Pareto chưa.
    Sử dụng sai số epsilon 1e-4 như trong C++.
    Tương ứng với inpareto trong paretoUpdate.h.
    """
    epsilon = 1e-4
    for p in pareto_list:
        if (
            abs(indi.fitness1 - p.fitness1) < epsilon
            and abs(indi.fitness2 - p.fitness2) < epsilon
        ):
            return True
    return False


def in_tabu_pareto(indi, tabu_pareto):
    """
    Kiểm tra xem cá thể đã tồn tại trong tập Tabu Pareto chưa.
    Tương ứng với inTabupareto trong paretoUpdate.h.
    """
    epsilon = 1e-4
    for p in tabu_pareto:
        if (
            abs(indi.fitness1 - p.fitness1) < epsilon
            and abs(indi.fitness2 - p.fitness2) < epsilon
        ):
            return True
    return False
