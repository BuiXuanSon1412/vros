from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    # --- Tabu Search Parameters ---
    tabuMaxIter: int = 100
    tabuNumRunPerDataSet: int = 5
    tabuNotImproveIter: int = 200
    tabuAlpha1: float = 1.0
    tabuAlpha2: float = 1.0
    tabuBeta: float = 0.5
    tabuEpsilon: float = 1e-3
    tabu_size: int = 5
    tabuDuration: int = 5

    # --- Ejection Chain Parameters ---
    use_ejection: bool = True
    maxEjectionLevel: int = 2
    addEjectionType: int = 1
    ejectionIte: int = 1

    # --- Multi-Level Parameters ---
    num_level: int = 3
    percent_match: float = 0.2
    percent_select: float = 0.1 # Giữ từ bản Python cũ nếu cần

    # --- Vehicle & Constraints Parameters ---
    droneVelocity: float = 0.83
    techVelocity: float = 0.58
    numDrone: int = 2
    numTech: int = 1
    droneLimitationFlightTime: float = 120.0
    sampleLimitationWaitingTime: float = 60.0
    
    # --- Operational Parameters ---
    overwrite: bool = True
    run_type: int = 1
    isCycle: int = 1
    use_inter: bool = True
    use_intra: bool = True
    NumRunPerDataSet: int = 5

    # --- File Paths & Data Info ---
    ws: str = "D:/Research/MultiLevel/DASTS2_VERSION9_C"
    TaburesultFolder: str = "D:/Research/MultiLevel/DASTS2_VERSION9_C/tabu_result"
    MultiLevelresultFolder: str = "D:/Research/MultiLevel/DASTS2_VERSION9_C/multilevel_result"
    dataPath: str = "/data"
    dataName: str = "6.5.1.txt"
    multiData: bool = True
    dataType: str = "6"