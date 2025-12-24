"""Example runner for simplified Multilevel + Tabu implementation.

Usage: python run_py_impl.py [data_file]
"""
import sys
from py_impl.config import Config
from py_impl.input import Input
from py_impl.multilevel import MultiLevel

def run_test(data_file=None):
    if data_file:
        print(f"Loading data from {data_file}...")
        inp = Input.from_file(data_file)
        print(f"Loaded {inp.numCus} customers.")
    else:
        coords = [(0, 0), (1, 2), (2, 1), (3, 4), (4, 2), (5, 3), (3, 0)]
        inp = Input(coords=coords)
        print(f"Using synthetic instance with {inp.numCus} customers.")
    
    cfg = Config()
    cfg.num_level = 2
    cfg.numDrone = max(1, inp.numCus // 4)
    cfg.numTech = max(1, inp.numCus // 3)
    cfg.tabuMaxIter = 50  # smaller for demo
    cfg.tabuDuration = 5
    
    print(f"Config: {cfg.numDrone} drones, {cfg.numTech} techs, tabuMaxIter={cfg.tabuMaxIter}")
    ml = MultiLevel(cfg, inp)
    score, sol, score_list = ml.run()
    
    print(f"\nBest score: {score}")
    print(f"Score list: {score_list}")
    print(f"Solution drone routes: {sol.droneTripList}")
    print(f"Solution tech routes: {sol.techTripList}")
    return score

if __name__ == '__main__':
    data_file = sys.argv[1] if len(sys.argv) > 1 else None
    run_test(data_file)
