# 🚚 Vehicle Routing Optimization System

Hệ thống lập lịch giao vận tối ưu cho xe tải và máy bay không người lái (UAV).

## 📋 Tổng quan

Hệ thống hỗ trợ 3 bài toán:

1. **Bài toán 1**: Min-timespan parallel technician-and-drone scheduling
   - Thuật toán: Tabu Search, Tabu Search Multilevel
   
2. **Bài toán 2**: Bi-objective Medical Sampling Service
   - Thuật toán: NSGA-II + TS, MOEA/D
   
3. **Bài toán 3**: Resupply with release date
   - Thuật toán: Tabu Search

## 🚀 Cài đặt

### 1. Clone repository

```bash
git clone <your-repo-url>
cd vehicle-routing-system
```

### 2. Tạo virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

## 🎯 Chạy ứng dụng

```bash
streamlit run app.py
```

Ứng dụng sẽ mở tại: `http://localhost:8501`

## 📁 Cấu trúc thư mục

```
vehicle-routing-system/
├── app.py                      # Main Streamlit app
├── requirements.txt            # Python dependencies
├── README.md                   # Tài liệu này
│
├── config/
│   ├── __init__.py
│   └── default_config.py       # Cấu hình mặc định
│
├── algorithms/                 # Các thuật toán
│   ├── __init__.py
│   ├── tabu_search.py          # [TODO] Bài toán 1 - TS
│   ├── ts_multilevel.py        # [TODO] Bài toán 1 - TS Multilevel
│   ├── nsga2_ts.py             # [TODO] Bài toán 2 - NSGA-II+TS
│   ├── moead.py                # [TODO] Bài toán 2 - MOEA/D
│   └── ts_resupply.py          # [TODO] Bài toán 3 - TS Resupply
│
├── utils/
│   ├── __init__.py
│   ├── data_generator.py       # Tạo dữ liệu mẫu
│   ├── visualizer.py           # Visualization functions
│   └── solver.py               # Wrapper cho thuật toán (hiện tại dùng dummy)
│
└── data/
    └── sample_data/            # Dữ liệu mẫu
```

## 🎨 Tính năng

### ✅ Đã hoàn thành

- [x] Giao diện Streamlit với layout 2 cột (visualization + config)
- [x] Cấu hình phương tiện (xe tải, drone)
- [x] Cấu hình khách hàng và tạo dữ liệu ngẫu nhiên
- [x] Visualization routes trên bản đồ 2D
- [x] Biểu đồ hội tụ thuật toán
- [x] Gantt chart cho timeline
- [x] Pareto front cho bài toán đa mục tiêu
- [x] So sánh nhiều thuật toán
- [x] Export dữ liệu

### 🔨 Cần hoàn thiện

- [ ] Tích hợp thuật toán thật vào `algorithms/`
- [ ] Import dữ liệu từ file
- [ ] Thêm validation cho input
- [ ] Thêm map thực với Folium/OpenStreetMap
- [ ] Export kết quả ra PDF/Excel
- [ ] Thêm real-time monitoring cho thuật toán
- [ ] Unit tests

## 🔧 Hướng dẫn tích hợp thuật toán thật

Hiện tại hệ thống dùng `DummySolver` để giả lập. Để tích hợp thuật toán thật:

### Bước 1: Tạo class thuật toán

Tạo file trong `algorithms/`, ví dụ `tabu_search.py`:

```python
class TabuSearch:
    def __init__(self, **params):
        self.max_iterations = params.get('max_iterations', 1000)
        # ... các tham số khác
    
    def solve(self, customers, depot, distance_matrix, vehicle_config):
        """
        Giải bài toán
        
        Returns:
            dict: {
                'routes': Dict[vehicle_id, List[customer_ids]],
                'schedule': List[Dict],
                'makespan': float,
                'cost': float,
                'convergence_history': List[Tuple[int, float]]
            }
        """
        # Code thuật toán của bạn
        pass
```

### Bước 2: Cập nhật solver.py

Sửa `utils/solver.py`:

```python
from algorithms.tabu_search import TabuSearch

class RealSolver:
    def __init__(self, problem_type, algorithm):
        self.problem_type = problem_type
        self.algorithm = algorithm
        
        # Map algorithm name to class
        if algorithm == "Tabu Search":
            self.solver = TabuSearch()
        # ... thêm các thuật toán khác
    
    def solve(self, customers, depot, distance_matrix, vehicle_config, algorithm_params):
        return self.solver.solve(customers, depot, distance_matrix, vehicle_config)
```

### Bước 3: Thay DummySolver bằng RealSolver trong app.py

```python
# Thay dòng này:
solver = DummySolver(problem_type, selected_algorithm)

# Bằng:
solver = RealSolver(problem_type, selected_algorithm)
```

## 📊 Format dữ liệu

### Input

```python
customers = pd.DataFrame({
    'id': [1, 2, 3, ...],
    'x': [10.5, 20.3, ...],  # tọa độ X (km)
    'y': [15.2, 25.1, ...],  # tọa độ Y (km)
    'demand': [5, 8, ...],   # nhu cầu (kg)
    'service_time': [10, 15, ...],  # thời gian phục vụ (phút)
    'priority': [1, 2, ...], # mức ưu tiên
    'time_window_start': [0, 60, ...],  # bắt đầu time window
    'time_window_end': [120, 180, ...]  # kết thúc time window
})

depot = {
    'id': 0,
    'x': 25.0,
    'y': 25.0
}

vehicle_config = {
    'truck': {
        'count': 2,
        'capacity': 100,
        'speed': 40,
        'cost_per_km': 5000
    },
    'drone': {
        'count': 3,
        'capacity': 5,
        'speed': 60,
        'energy_limit': 30,
        'cost_per_km': 2000
    }
}
```

### Output

```python
result = {
    'routes': {
        'Truck_1': [1, 3, 5, 7],
        'Truck_2': [2, 4, 6],
        'Drone_1': [8, 9, 10]
    },
    'schedule': [
        {
            'vehicle_id': 'Truck_1',
            'customer_id': 'C1',
            'start_time': 10.5,
            'end_time': 20.5,
            'service_time': 10
        },
        # ...
    ],
    'makespan': 180.5,  # phút
    'cost': 250000,     # VND
    'total_distance': 150.3,  # km
    'convergence_history': [(0, 200), (10, 190), ...],
    'pareto_front': [(180, 250000), (185, 240000), ...]  # chỉ cho bài toán 2
}
```

## 🎓 Tham khảo

- Bài toán 1: [Bai1TS.pdf, Bai1TSMultilevel.pdf]
- Bài toán 2: [Bai2NSGAII.pdf]
- Bài toán 3: [Bai3TS.pdf]

## 📝 License

MIT License

## 👥 Contributors

- Your Name

## 📞 Liên hệ

- Email: your.email@example.com
- GitHub: your-github-username
