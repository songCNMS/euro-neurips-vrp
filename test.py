from dataclasses import replace
from tracemalloc import start
import tools
from solver import solve_static_vrptw
from cvrptw_utility import *




import argparse
import time
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", help="Instance to solve")
    args = parser.parse_args()
    instance = tools.read_vrplib(args.instance)
    solutions = list(solve_static_vrptw(instance, time_limit=120, seed=7, tmp_dir="tmp"))
    solution, total_cost = solutions[-1]
    
    start_time = time.time()
    ep = 0
    while time.time() - start_time <= 14*3600:
        ep += 1
        route_idx = np.random.randint(len(solution))
        node_idx = np.random.randint(len(solution[route_idx]))
        node = solution[route_idx][node_idx]
        # route_idxs = select_close_routes(node, 3, solution, instance)
        # solution, cost_improvement = multi_routes_optimization(route_idxs, solution, instance)
        solution, cost_improvement = ruin_and_recreation(node, solution, instance)

    tools.validate_static_solution(instance, solution)
    new_total_cost = tools.compute_solution_driving_time(instance, solution)
    print(f"ori cost: {total_cost}, new cost: {new_total_cost}")