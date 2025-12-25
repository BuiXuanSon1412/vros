1. problem 1 (rename PTDS-DDSS): 
- system: depot coordinate, number of technicians/drones, technician/drone speed, flight endurance time limit, sample waiting limitation time (60)
- dataset (inputs reflected after upload file): number of customers, coordinate range, coordinate distribution, demand range, demand distribution, only-technician distribution 
- algorithm (one algorithm, no selection): maximum iteration, maximum iteration without improvement, two penalty parameters for total violation of limited drone’s flight endurance and maximum waiting time (denoted by alpha1, alpha 2), penalty factor (denoted by beta)

2. problem 2 (rename MSSVTDE):
- system: depot coordinate, number of technicians/drones, technician speed (2 factors: baseline speed (35), congestion factor range(0.4, 0.9)), drone speed (takeoff speed, cruise speed, landing speed), truck/drone capacity, flight endurance time limit, sample waiting limitation time
- dataset (inputs reflected after upload): number of patients, coordinate range, coordinate distribution, demand range, demand distribution, only-technician distribution 
- algorithm (one algorithm named (HNSGAII-TS)): crossover rate (initially 0.9), mutation rate (initially 0.05), number of generations, population size, number of tabu search iteration, 

3. problem 3 (rename VRP-MRDR):
- system: depot coordinate, number of truck/drones, technician speed (), drone speed (60), drone capacity (2, 4, 8) flight endurance time limit, drone capacity limit, sample waiting limitation time
- dataset (inputs reflected after upload): number of customers, coordinate range, coordinate distribution, demand range, demand distribution, release date range (release time factor - β (0.5, 1, 1.5, 2.0, 2.5, 3.0) and d_TSP (optimal tour length)), release date distribution
- algorithm (ATS): score factor (γ1, γ2, γ3, γ4), variable maximum iteration (η), fixed maximum iteration (LOOP), number of segment (SEG)


