-- ============================================================================
-- Vehicle Routing Optimization System (VROS) - Simple Database Schema
-- ============================================================================
-- Purpose: Store simulation configurations and results
-- Database: PostgreSQL/MySQL/SQLite compatible
-- ============================================================================

-- Drop existing tables
DROP TABLE IF EXISTS simulations CASCADE;

-- ============================================================================
-- SIMULATIONS TABLE (All-in-one)
-- ============================================================================
CREATE TABLE simulations (
    -- Primary Key
    simulation_id SERIAL PRIMARY KEY,
    
    -- Basic Info
    simulation_name VARCHAR(200) NOT NULL,
    problem_type INTEGER NOT NULL CHECK (problem_type IN (1, 2, 3)),
    algorithm_name VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Configuration (JSON)
    config JSONB NOT NULL, -- All system + algorithm parameters
    
    -- Input Data (JSON)
    customers JSONB NOT NULL, -- Array of customer objects
    
    -- Results (JSON)
    results JSONB, -- All results: routes, schedule, metrics, convergence
    
    -- Quick Access Metrics (for fast queries)
    makespan DECIMAL(10, 2),
    total_cost DECIMAL(12, 2),
    total_distance DECIMAL(10, 2),
    computation_time DECIMAL(10, 3),
    
    -- Status
    status VARCHAR(20) DEFAULT 'completed' CHECK (status IN ('pending', 'completed', 'failed'))
);

-- Indexes
CREATE INDEX idx_simulations_problem ON simulations(problem_type);
CREATE INDEX idx_simulations_algorithm ON simulations(algorithm_name);
CREATE INDEX idx_simulations_date ON simulations(created_at DESC);
CREATE INDEX idx_simulations_makespan ON simulations(makespan);

-- ============================================================================
-- SAMPLE DATA STRUCTURE (as comments)
-- ============================================================================

/*
-- Example INSERT:

INSERT INTO simulations (
    simulation_name,
    problem_type,
    algorithm_name,
    config,
    customers,
    results,
    makespan,
    total_cost,
    total_distance,
    computation_time,
    status
) VALUES (
    'Test Run 1',
    1,
    'Tabu Search',
    '{
        "depot": {"x": 0, "y": 0},
        "num_trucks": 2,
        "num_drones": 2,
        "truck_speed": 35,
        "drone_speed": 60,
        "max_iteration": 1000
    }'::jsonb,
    '[
        {"id": 1, "x": 10, "y": 20, "demand": 5},
        {"id": 2, "x": 30, "y": 40, "demand": 3}
    ]'::jsonb,
    '{
        "routes": {"Truck_1": [1, 2]},
        "schedule": [...],
        "convergence_history": [[0, 150], [1, 145], ...]
    }'::jsonb,
    145.5,
    5000.00,
    50.3,
    2.5,
    'completed'
);

-- Example QUERY - Get all simulations:
SELECT simulation_id, simulation_name, algorithm_name, makespan, created_at 
FROM simulations 
ORDER BY created_at DESC;

-- Example QUERY - Compare algorithms:
SELECT algorithm_name, AVG(makespan) as avg_makespan, AVG(total_cost) as avg_cost
FROM simulations 
WHERE problem_type = 1 
GROUP BY algorithm_name;

-- Example QUERY - Get full results:
SELECT results FROM simulations WHERE simulation_id = 1;
*/

-- ============================================================================
-- END OF SCHEMA
-- ============================================================================
