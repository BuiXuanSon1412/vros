import math
import os
from typing import List

class Input:
    def __init__(self, droneVelocity: float, techVelocity: float, limitationFlightTime: float, path: str):
        self.droneVelocity = droneVelocity
        self.techVelocity = techVelocity
        self.limitationFlightTime = limitationFlightTime
        self.numCus = 0
        self.coordinates = []
        self.technician_flags = []
        self.distances = []
        self.droneTimes = []
        self.techTimes = []
        self.cusOnlyServedByTech = []
        self.dataSet = ""
        
        self._load_from_file(path)
        self._calculate_matrices()
        self._check_constraints()

    def _load_from_file(self, path: str):
        print(f"\nLoading data from: {path}")
        if not os.path.exists(path):
            print("Cannot open data file ...")
            return

        with open(path, 'r') as f:
            lines = f.readlines()

        if not lines:
            return

        # Dòng 1: Lấy số lượng khách hàng (bỏ 10 ký tự đầu "Customers ")
        first_line = lines[0].strip()
        self.numCus = int(first_line[10:])

        # Khởi tạo depot tại tọa độ (0,0)
        self.coordinates.append([0.0, 0.0])
        self.technician_flags.append(0)

        # Đọc các dòng tiếp theo: x, y, z (z thường là cờ kỹ thuật viên)
        for line in lines[2:]:  # Bắt đầu từ dòng thứ 3 theo logic C++
            parts = line.split()
            if len(parts) >= 3:
                x, y, z = float(parts[0]), float(parts[1]), int(float(parts[2]))
                self.coordinates.append([x, y])
                self.technician_flags.append(z)

        # Thêm depot cuối cùng để khớp với cấu trúc numCus + 2 trong C++
        self.coordinates.append([0.0, 0.0])
        
        # Lấy tên dataset từ đường dẫn
        self.dataSet = os.path.splitext(os.path.basename(path))[0]

    def _calculate_matrices(self):
        n = len(self.coordinates)
        # Khởi tạo ma trận rỗng
        self.distances = [[0.0] * n for _ in range(n)]
        self.droneTimes = [[0.0] * n for _ in range(n)]
        self.techTimes = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                # Tính khoảng cách Euclidean
                d = math.hypot(self.coordinates[i][0] - self.coordinates[j][0],
                               self.coordinates[i][1] - self.coordinates[j][1])
                
                self.distances[i][j] = d
                self.droneTimes[i][j] = d / self.droneVelocity
                self.techTimes[i][j] = d / self.techVelocity

    def _check_constraints(self):
        # Khởi tạo danh sách cờ: numCus + 1 phần tử
        self.cusOnlyServedByTech = [False] * (self.numCus + 1)
        
        # Logic 1: Dựa trên thời gian bay của Drone từ Depot đến khách hàng
        for i in range(1, self.numCus + 1):
            if self.droneTimes[0][i] > self.limitationFlightTime:
                self.cusOnlyServedByTech[i] = True
        
        # Logic 2: Dựa trên dữ liệu cột Z trong file (nếu có)
        # Giữ nguyên phần comment từ C++: nếu technician[i] == 1 thì chỉ tech phục vụ
        for i in range(1, self.numCus + 1):
            if i < len(self.technician_flags) and self.technician_flags[i] == 1:
                self.cusOnlyServedByTech[i] = True

    def __str__(self):
        return f"Dataset: {self.dataSet}, Customers: {self.numCus}"