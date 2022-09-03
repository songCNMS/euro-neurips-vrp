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
        # swap_candidate_list = []
        # for route_idx1 in range(len(solution)):
        #     for route_idx2 in range(len(solution)):
        #         for node_idx1 in range(2*len(solution[route_idx1])+1):
        #             for node_idx2 in range(len(solution[route_idx2])):
        #                 cost_reduction, route1, route2 = swap(route_idx1, node_idx1, route_idx2, node_idx2, solution, instance)
        #                 print(cost_reduction)
        #                 if cost_reduction > 0:
        #                     swap_candidate_list.append((cost_reduction, (route_idx1, node_idx1, route_idx2, node_idx2)))
        # swap_candidate_list = sorted(swap_candidate_list, key=lambda x: x[0], reverse=True)
        # print(swap_candidate_list[:10])
        # if len(swap_candidate_list) <= 0: break
        # route_idx1, node_idx1, route_idx2, node_idx2 = swap_candidate_list[0][1]
        # cost_reduction, route1, route2 = swap(route_idx1, node_idx1, route_idx2, node_idx2, solution, instance)
        # if cost_reduction > 0:
        #     print("improve {ep}:", cost_reduction)
        #     solution[route_idx1] = route1
        #     solution[route_idx2] = route2
        route_idx = np.random.randint(len(solution))
        node_idx = np.random.randint(len(solution[route_idx]))
        node = solution[route_idx][node_idx]
        # route_idxs = select_close_routes(node, 3, solution, instance)
        # solution, cost_improvement = multi_routes_optimization(route_idxs, solution, instance)
        solution, cost_improvement = ruin_and_recreation(node, solution, instance)

    tools.validate_static_solution(instance, solution)
    new_total_cost = tools.compute_solution_driving_time(instance, solution)
    print(f"ori cost: {total_cost}, new cost: {new_total_cost}")