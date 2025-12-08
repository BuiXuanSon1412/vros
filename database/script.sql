-- ============================================================================
-- Vehicle Routing Optimization System (VROS) - Database Schema
-- ============================================================================
-- Purpose: Store simulation configurations and results for all three problems
-- Database: PostgreSQL (can be adapted for MySQL/SQLite)
-- ============================================================================

-- Drop existing tables (in correct order due to foreign keys)
DROP TABLE IF EXISTS resupply_operations CASCADE;
DROP TABLE IF EXISTS schedule_entries CASCADE;
DROP TABLE IF EXISTS route_details CASCADE;
DROP TABLE IF EXISTS convergence_history CASCADE;
DROP TABLE IF EXISTS pareto_solutions CASCADE;
DROP TABLE IF EXISTS simulation_results CASCADE;
DROP TABLE IF EXISTS customers CASCADE;
DROP TABLE IF EXISTS simulations CASCADE;

-- ============================================================================
-- 1. SIMULATIONS TABLE (Main configuration)
-- ============================================================================
CREATE TABLE simulations (
    simulation_id SERIAL PRIMARY KEY,
    
    -- Basic info
    simulation_name VARCHAR(200) NOT NULL,
    problem_type INTEGER NOT NULL CHECK (problem_type IN (1, 2, 3)),
    algorithm_name VARCHAR(100) NOT NULL,
    description TEXT,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    
    -- System Configuration (JSON format)
    depot_config JSONB NOT NULL, -- {x, y, name}
    vehicle_config JSONB NOT NULL, -- {num_trucks, num_drones, speeds, capacities, etc}
    constraint_config JSONB, -- {flight_endurance, waiting_limit, etc}
    
    -- Algorithm Parameters (JSON format)
    algorithm_params JSONB NOT NULL, -- {max_iteration, crossover_rate, etc}
    
    -- Dataset metadata
    num_customers INTEGER NOT NULL,
    dataset_source VARCHAR(100), -- 'uploaded' or 'generated'
    dataset_filename VARCHAR(255),
    
    -- Indexes for fast querying
    CONSTRAINT unique_simulation_name UNIQUE (simulation_name, created_at)
);

CREATE INDEX idx_simulations_problem_type ON simulations(problem_type);
CREATE INDEX idx_simulations_status ON simulations(status);
CREATE INDEX idx_simulations_created_at ON simulations(created_at DESC);
CREATE INDEX idx_simulations_algorithm ON simulations(algorithm_name);

-- ============================================================================
-- 2. CUSTOMERS TABLE
-- ============================================================================
CREATE TABLE customers (
    customer_id SERIAL PRIMARY KEY,
    simulation_id INTEGER NOT NULL REFERENCES simulations(simulation_id) ON DELETE CASCADE,
    
    -- Customer identification
    customer_index INTEGER NOT NULL, -- 1, 2, 3... (order in dataset)
    
    -- Location
    coordinate_x DECIMAL(10, 4) NOT NULL,
    coordinate_y DECIMAL(10, 4) NOT NULL,
    
    -- Attributes
    demand DECIMAL(10, 4) NOT NULL,
    service_time INTEGER, -- in minutes
    priority INTEGER DEFAULT 1,
    
    -- Time windows
    time_window_start INTEGER DEFAULT 0,
    time_window_end INTEGER DEFAULT 480,
    
    -- Problem-specific attributes
    only_staff BOOLEAN DEFAULT FALSE, -- Problem 2: only serviceable by staff
    service_time_truck INTEGER, -- Problem 2
    service_time_drone INTEGER, -- Problem 2
    release_date INTEGER, -- Problem 3: when package becomes available
    
    -- Indexes
    CONSTRAINT unique_customer_per_simulation UNIQUE (simulation_id, customer_index)
);

CREATE INDEX idx_customers_simulation ON customers(simulation_id);

-- ============================================================================
-- 3. SIMULATION RESULTS TABLE
-- ============================================================================
CREATE TABLE simulation_results (
    result_id SERIAL PRIMARY KEY,
    simulation_id INTEGER NOT NULL UNIQUE REFERENCES simulations(simulation_id) ON DELETE CASCADE,
    
    -- Primary Objectives
    makespan DECIMAL(10, 2) NOT NULL, -- Total completion time (minutes)
    total_cost DECIMAL(12, 2) NOT NULL, -- Total operational cost ($)
    total_distance DECIMAL(10, 2) NOT NULL, -- Total distance traveled (km)
    
    -- Distance breakdown
    truck_distance DECIMAL(10, 2),
    drone_distance DECIMAL(10, 2),
    
    -- Cost breakdown
    truck_cost DECIMAL(12, 2),
    drone_cost DECIMAL(12, 2),
    
    -- Performance metrics
    computation_time DECIMAL(10, 3) NOT NULL, -- seconds
    num_iterations INTEGER,
    
    -- Vehicle utilization
    num_trucks_used INTEGER,
    num_drones_used INTEGER,
    avg_vehicle_utilization DECIMAL(5, 2), -- percentage
    
    -- Constraint violations
    flight_endurance_violations INTEGER DEFAULT 0,
    waiting_time_violations INTEGER DEFAULT 0,
    capacity_violations INTEGER DEFAULT 0,
    
    -- Problem-specific metrics
    num_resupplies INTEGER, -- Problem 3: number of drone resupply operations
    packages_delivered_by_drone INTEGER, -- Problem 3
    total_waiting_time DECIMAL(10, 2), -- Problem 3
    
    -- Multi-objective (Problem 2)
    pareto_rank INTEGER, -- 1 = non-dominated
    is_pareto_optimal BOOLEAN DEFAULT FALSE,
    num_pareto_solutions INTEGER,
    hypervolume DECIMAL(10, 4), -- Quality indicator
    
    -- Additional info
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_results_simulation ON simulation_results(simulation_id);
CREATE INDEX idx_results_makespan ON simulation_results(makespan);
CREATE INDEX idx_results_cost ON simulation_results(total_cost);

-- ============================================================================
-- 4. ROUTE DETAILS TABLE
-- ============================================================================
CREATE TABLE route_details (
    route_id SERIAL PRIMARY KEY,
    simulation_id INTEGER NOT NULL REFERENCES simulations(simulation_id) ON DELETE CASCADE,
    
    -- Vehicle identification
    vehicle_id VARCHAR(50) NOT NULL, -- 'Truck_1', 'Drone_2', etc
    vehicle_type VARCHAR(20) NOT NULL CHECK (vehicle_type IN ('truck', 'drone')),
    
    -- Route sequence (stored as array of customer indices)
    route_sequence INTEGER[] NOT NULL, -- [1, 5, 3, 7] = customer order
    
    -- Route metrics
    route_distance DECIMAL(10, 2),
    route_duration DECIMAL(10, 2), -- minutes
    num_customers_served INTEGER,
    total_demand_served DECIMAL(10, 2),
    
    -- Route visualization data (JSON)
    route_coordinates JSONB, -- [{x, y, customer_id}, ...]
    
    CONSTRAINT unique_vehicle_per_simulation UNIQUE (simulation_id, vehicle_id)
);

CREATE INDEX idx_routes_simulation ON route_details(simulation_id);
CREATE INDEX idx_routes_vehicle_type ON route_details(vehicle_type);

-- ============================================================================
-- 5. SCHEDULE ENTRIES TABLE
-- ============================================================================
CREATE TABLE schedule_entries (
    schedule_id SERIAL PRIMARY KEY,
    simulation_id INTEGER NOT NULL REFERENCES simulations(simulation_id) ON DELETE CASCADE,
    
    -- Task identification
    vehicle_id VARCHAR(50) NOT NULL,
    customer_id VARCHAR(50) NOT NULL, -- 'C1', 'C2', 'Depot', etc
    action VARCHAR(50) DEFAULT 'Service', -- 'Service', 'Return', 'Resupply', 'Wait'
    
    -- Timing
    start_time DECIMAL(10, 2) NOT NULL, -- minutes from start
    end_time DECIMAL(10, 2) NOT NULL,
    service_time DECIMAL(10, 2) DEFAULT 0,
    waiting_time DECIMAL(10, 2) DEFAULT 0,
    
    -- Sequence order
    sequence_order INTEGER, -- Order within vehicle's schedule
    
    -- Additional info
    notes TEXT
);

CREATE INDEX idx_schedule_simulation ON schedule_entries(simulation_id);
CREATE INDEX idx_schedule_vehicle ON schedule_entries(vehicle_id);
CREATE INDEX idx_schedule_time ON schedule_entries(start_time, end_time);

-- ============================================================================
-- 6. CONVERGENCE HISTORY TABLE
-- ============================================================================
CREATE TABLE convergence_history (
    convergence_id SERIAL PRIMARY KEY,
    simulation_id INTEGER NOT NULL REFERENCES simulations(simulation_id) ON DELETE CASCADE,
    
    -- Iteration data
    iteration INTEGER NOT NULL,
    fitness_value DECIMAL(10, 4) NOT NULL, -- Objective function value
    
    -- Multi-objective tracking (Problem 2)
    objective1_value DECIMAL(10, 4), -- Makespan
    objective2_value DECIMAL(10, 4), -- Cost
    
    -- Additional metrics
    population_diversity DECIMAL(5, 4), -- For genetic algorithms
    best_solution_found BOOLEAN DEFAULT FALSE,
    
    -- Timestamp
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT unique_iteration_per_simulation UNIQUE (simulation_id, iteration)
);

CREATE INDEX idx_convergence_simulation ON convergence_history(simulation_id);
CREATE INDEX idx_convergence_iteration ON convergence_history(iteration);

-- ============================================================================
-- 7. PARETO SOLUTIONS TABLE (Problem 2 - Multi-objective)
-- ============================================================================
CREATE TABLE pareto_solutions (
    pareto_id SERIAL PRIMARY KEY,
    simulation_id INTEGER NOT NULL REFERENCES simulations(simulation_id) ON DELETE CASCADE,
    
    -- Objectives
    objective1_makespan DECIMAL(10, 2) NOT NULL,
    objective2_cost DECIMAL(12, 2) NOT NULL,
    
    -- Solution rank
    pareto_rank INTEGER DEFAULT 1,
    dominance_count INTEGER DEFAULT 0, -- How many solutions dominate this
    
    -- Solution data (JSON) - can store full route if needed
    solution_data JSONB,
    
    -- Weighted score
    weighted_score DECIMAL(10, 4),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_pareto_simulation ON pareto_solutions(simulation_id);
CREATE INDEX idx_pareto_rank ON pareto_solutions(pareto_rank);
CREATE INDEX idx_pareto_objectives ON pareto_solutions(objective1_makespan, objective2_cost);

-- ============================================================================
-- 8. RESUPPLY OPERATIONS TABLE (Problem 3 - Drone Resupply)
-- ============================================================================
CREATE TABLE resupply_operations (
    resupply_id SERIAL PRIMARY KEY,
    simulation_id INTEGER NOT NULL REFERENCES simulations(simulation_id) ON DELETE CASCADE,
    
    -- Vehicles involved
    drone_id VARCHAR(50) NOT NULL,
    truck_id VARCHAR(50) NOT NULL,
    
    -- Meeting point
    meeting_customer_id INTEGER NOT NULL, -- Customer index where meeting occurs
    meeting_location JSONB, -- {x, y}
    
    -- Packages resupplied
    packages INTEGER[] NOT NULL, -- [customer_ids]
    num_packages INTEGER NOT NULL,
    total_weight DECIMAL(10, 2) NOT NULL,
    
    -- Timing
    departure_time DECIMAL(10, 2) NOT NULL, -- Drone leaves depot
    arrival_time DECIMAL(10, 2) NOT NULL, -- Drone arrives at truck
    return_time DECIMAL(10, 2) NOT NULL, -- Drone returns to depot
    
    -- Distance
    distance DECIMAL(10, 2) NOT NULL, -- km (round trip)
    
    -- Status
    is_loaded BOOLEAN DEFAULT TRUE, -- Whether drone carried packages
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_resupply_simulation ON resupply_operations(simulation_id);
CREATE INDEX idx_resupply_drone ON resupply_operations(drone_id);
CREATE INDEX idx_resupply_truck ON resupply_operations(truck_id);

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- View: Summary of all simulations with results
CREATE VIEW simulation_summary AS
SELECT 
    s.simulation_id,
    s.simulation_name,
    s.problem_type,
    s.algorithm_name,
    s.status,
    s.num_customers,
    s.created_at,
    s.completed_at,
    r.makespan,
    r.total_cost,
    r.total_distance,
    r.computation_time
FROM simulations s
LEFT JOIN simulation_results r ON s.simulation_id = r.simulation_id;

-- View: Best results per problem type
CREATE VIEW best_results_by_problem AS
SELECT 
    s.problem_type,
    s.algorithm_name,
    MIN(r.makespan) as best_makespan,
    MIN(r.total_cost) as best_cost,
    MIN(r.total_distance) as best_distance,
    AVG(r.computation_time) as avg_computation_time,
    COUNT(*) as num_runs
FROM simulations s
JOIN simulation_results r ON s.simulation_id = r.simulation_id
WHERE s.status = 'completed'
GROUP BY s.problem_type, s.algorithm_name;

-- View: Recent simulations (last 30 days)
CREATE VIEW recent_simulations AS
SELECT 
    s.simulation_id,
    s.simulation_name,
    s.problem_type,
    s.algorithm_name,
    s.status,
    s.created_at,
    r.makespan,
    r.total_cost
FROM simulations s
LEFT JOIN simulation_results r ON s.simulation_id = r.simulation_id
WHERE s.created_at >= CURRENT_DATE - INTERVAL '30 days'
ORDER BY s.created_at DESC;

-- ============================================================================
-- SAMPLE DATA (Optional - for testing)
-- ============================================================================

-- No sample data needed without user management

-- ============================================================================
-- USEFUL QUERIES (Comments)
-- ============================================================================

-- Query 1: Get all simulations
-- SELECT * FROM simulation_summary ORDER BY created_at DESC;

-- Query 2: Get convergence data for a simulation
-- SELECT iteration, fitness_value FROM convergence_history 
-- WHERE simulation_id = 1 ORDER BY iteration;

-- Query 3: Get Pareto front for Problem 2
-- SELECT objective1_makespan, objective2_cost FROM pareto_solutions
-- WHERE simulation_id = 1 AND pareto_rank = 1 ORDER BY objective1_makespan;

-- Query 4: Get route details for a simulation
-- SELECT vehicle_id, route_sequence, route_distance 
-- FROM route_details WHERE simulation_id = 1;

-- Query 5: Compare algorithms for same problem type
-- SELECT algorithm_name, AVG(makespan) as avg_makespan, AVG(total_cost) as avg_cost
-- FROM simulations s JOIN simulation_results r ON s.simulation_id = r.simulation_id
-- WHERE problem_type = 1 GROUP BY algorithm_name;

-- Query 6: Get resupply operations for Problem 3
-- SELECT drone_id, truck_id, meeting_customer_id, num_packages, total_weight
-- FROM resupply_operations WHERE simulation_id = 1 ORDER BY departure_time;

-- ============================================================================
-- MAINTENANCE QUERIES
-- ============================================================================

-- Clean up old pending simulations (> 7 days)
-- DELETE FROM simulations WHERE status = 'pending' 
-- AND created_at < CURRENT_DATE - INTERVAL '7 days';

-- Archive old completed simulations (> 1 year)
-- CREATE TABLE simulations_archive AS 
-- SELECT * FROM simulations WHERE completed_at < CURRENT_DATE - INTERVAL '1 year';

-- ============================================================================
-- END OF SCHEMA
-- ============================================================================
