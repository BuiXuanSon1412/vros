import os
import sys
import json
import time
from typing import Tuple

from py_impl.config import Config
from py_impl.input import Input
from py_impl.solution import Solution
from py_impl.tabu import TabuSearch
from py_impl.multilevel import MultiLevel

def get_vehicle_config(num_cus: int) -> Tuple[int, int]:
    """Cấu hình số lượng xe dựa trên quy mô khách hàng."""
    mapping = {
        6: (1, 1), 10: (1, 1), 12: (0, 4), 20: (0, 4),
        50: (0, 6), 100: (0, 8), 150: (0, 10), 200: (0, 12)
    }
    return mapping.get(num_cus, (1, 4)) # Mặc định 1 drone, 4 tech nếu không khớp

def create_initial_solution(cfg: Config, inp: Input) -> Solution:
    """Khởi tạo lời giải ban đầu bằng Round-Robin (Greedy)."""
    init_sol = Solution(cfg, inp)
    total_v = max(1, cfg.numDrone + cfg.numTech)
    
    for idx in range(1, inp.numCus + 1):
        v_idx = (idx - 1) % total_v
        if v_idx < cfg.numDrone:
            # Gán cho Drone: Đảm bảo drone có ít nhất 1 trip
            if not init_sol.droneTripList[v_idx]:
                init_sol.droneTripList[v_idx].append([])
            init_sol.droneTripList[v_idx][0].append(idx)
        else:
            # Gán cho Technician
            tech_idx = v_idx - cfg.numDrone
            init_sol.techTripList[tech_idx].append(idx)
            
    return init_sol

def main():
    # 1. Khởi tạo Config
    cfg = Config()
    
    # 2. Lấy file dữ liệu từ đối số dòng lệnh (mặc định là 50.10.1.txt)
    data_filename = sys.argv[1] if len(sys.argv) > 1 else "50.10.1.txt"
    full_path = os.path.join(cfg.ws, cfg.dataPath.strip('/'), data_filename)

    if not os.path.exists(full_path):
        print(f"Lỗi: Không tìm thấy file dữ liệu tại {full_path}")
        return

    # 3. Thiết lập Input và cấu hình xe
    try:
        num_cus_from_name = int(data_filename.split('.')[0])
        cfg.numDrone, cfg.numTech = get_vehicle_config(num_cus_from_name)
    except ValueError:
        print("Cảnh báo: Không định dạng được số khách từ tên file, dùng cấu hình mặc định.")

    inp = Input(cfg.droneVelocity, cfg.techVelocity, cfg.droneLimitationFlightTime, full_path)

    print("-" * 60)
    print(f"DATASET: {data_filename} | N={inp.numCus} | D={cfg.numDrone} | T={cfg.numTech}")
    print("-" * 60)

    # 4. Khởi tạo lời giải ban đầu
    init_sol = create_initial_solution(cfg, inp)
    print(f"Initial Score: {init_sol.getScore():.4f}")

    # 5. Chạy TABU SEARCH (Single-Level)
    # ===== THUẬT TOÁN 1: TABU SEARCH THUẦN (SINGLE-LEVEL) =====
    print("[1] Running Pure Tabu Search...")
    start_t = time.time()
    tabu_engine = TabuSearch(cfg, inp)
    
    # Cập nhật dòng này: nhận 3 tham số trả về (score, solution, matrix)
    tabu_score, tabu_best_sol, _ = tabu_engine.run(init_sol) 
    
    tabu_time = round(time.time() - start_t, 2)
    print(f"    => Tabu Result: {tabu_score:.4f} | Time: {tabu_time}s")
    print("Drone trip: ", tabu_best_sol.droneTripList)
    print("Tech trip: ", tabu_best_sol.techTripList)

    # 6. Chạy MULTI-LEVEL TABU SEARCH
    print("\n[2] Running Multi-Level Tabu Search...")
    ml_start = time.time()
    ml_engine = MultiLevel(cfg, inp)
    # run() bên trong MultiLevel sẽ thực hiện Merge -> Tabu -> Split -> Tabu
    ml_score, ml_best_sol = ml_engine.run() 
    ml_duration = time.time() - ml_start
    print(f"    Multi-Level Score: {ml_score:.4f} | Time: {ml_duration:.2f}s")
    print("Drone trip: ", ml_best_sol.droneTripList)
    print("Tech trip: ", ml_best_sol.techTripList)


    # 7. Lưu kết quả vào JSON
    comparison_results = {
        "dataset": data_filename,
        "parameters": {
            "drones": cfg.numDrone,
            "techs": cfg.numTech,
            "max_iter": cfg.tabuMaxIter
        },
        "tabu_search": {"score": tabu_score, "time": tabu_time},
        "multi_level": {"score": ml_score, "time": ml_duration}
    }

    results_dir = cfg.MultiLevelresultFolder
    os.makedirs(results_dir, exist_ok=True)
    out_file = os.path.join(results_dir, f"Comparison_{data_filename.replace('.txt', '.json')}")
    
    with open(out_file, 'w') as f:
        json.dump(comparison_results, f, indent=4)
    
    print("\n" + "=" * 60)
    print(f"KẾT QUẢ ĐÃ LƯU TẠI: {out_file}")
    print("=" * 60)

if __name__ == '__main__':
    main()