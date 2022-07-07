from dataclasses import replace
from platform import node
from select import select
import numpy as np
import pandas as pd
from pulp import *
import sys
import os
import pickle
from cvrptw import read_input_cvrptw
from tsp import get_tsp_solution
from cvrptw_single_route import path_selection, cvrptw_one_vehicle, add_path
import tools
from solver import solve_static_vrptw


depot = "Customer_0"


def construct_solution_from_seed(seed_customers, all_customers, truck_capacity,
                                 demands, service_time, 
                                 earliest_start, latest_end,
                                 distance_matrix):
    # num_routes = np.sum(list(demands.values())) // truck_capacity
    num_routes = len(seed_customers)
    cur_routes = {}
    selected_customers = []
    for i in range(1, 1+num_routes):
        cur_routes[f"PATH_NAME_{i}"] = [seed_customers[i-1]]
        selected_customers.append(seed_customers[i-1])
    for c in all_customers:
        if c in selected_customers: continue
        selected_customers.append(c)
        route_cost_list = []
        for route_name, route in cur_routes.items():
            if demands[c] + np.sum([demands[_c] for _c in route]) < truck_capacity:
                min_cost, min_pos = route_insertion_cost(route, c, service_time, 
                                                         earliest_start, latest_end,
                                                         distance_matrix)
                if min_pos is not None: route_cost_list.append((route_name, (min_cost, min_pos)))
        if len(route_cost_list) <= 0:
            num_routes += 1
            cur_routes[f"PATH_NAME_{num_routes}"] = [c]
        else:
            route_cost_list = sorted(route_cost_list, key=lambda x: x[1][0])
            route_name, insert_pos = route_cost_list[0][0], route_cost_list[0][1][1]
            cur_routes[route_name] = cur_routes[route_name][:insert_pos] + [c] + cur_routes[route_name][insert_pos:]
    return cur_routes

def cvrptw_one_route(selected_customers, truck_capacity,
                     demands, service_time, 
                     earliest_start, latest_end,
                     distance_matrix):
    min_cost = float("inf")
    best_route = None
    for c in selected_customers:
        route = [c]
        cur_demand = demands[c]
        cost = distance_matrix[depot][c]
        for _c in selected_customers:
            if _c in route: continue
            if demands[_c] + cur_demand < truck_capacity:
                _cost, _pos = route_insertion_cost(route, _c, service_time, 
                                                   earliest_start, latest_end,
                                                   distance_matrix)
                if _pos is not None:
                    cur_demand += demands[_c]
                    route = route[:_pos] + [_c] + route[_pos:]
                    cost += _cost
        if cost < min_cost:
            min_cost = cost
            best_route = route[:]
    return best_route

def select_candidate_points(routes, distance_matrix, all_customers, only_short_routes=False):
    if only_short_routes:
        route_list = sorted([(r, len(r)) for r in routes.keys()], key=lambda x: x[1])
        route_list = [x[0] for x in route_list]
        route_key = np.random.choice(route_list[:min(5, len(route_list))])
    else: route_key = np.random.choice(list(routes.keys()))
    route = routes[route_key]
    if len(route) <= 2: 
        M = route[:]
        prev_node = next_node = depot
    else:
        node_idx = np.random.randint(0, len(route)-1)
        M = route[node_idx:node_idx+2]
        prev_node = (depot if node_idx == 0 else route[node_idx-1])
        next_node = (depot if node_idx == len(route)-2 else route[node_idx+2])
    dist = [(c, distance_matrix[prev_node][c]+distance_matrix[c][next_node]) for c in all_customers if c not in route]
    dist = sorted(dist, key=lambda x: x[1])
    M.extend([dist[i][0] for i in range(min(4, len(dist)))])
    return M

def is_valid_pos(route, pos, customer, service_time, earliest_start, latest_end):
    new_route = route[:pos] + [customer] + route[pos:]
    return time_window_check(new_route, service_time, earliest_start, latest_end)


def route_validity_check(cur_routes, nb_customers, truck_capacity, demands, service_time, earliest_start, latest_end):
    num_customers = 0
    for route in cur_routes.values():
        if np.sum([demands[c] for c in route]) > truck_capacity or (not time_window_check(route, service_time, earliest_start, latest_end)):
            return False
        num_customers += len(route)
    return num_customers == nb_customers

def time_window_check(route, service_time, earliest_start, latest_end):
    cur_time = 0.0
    for r in [depot] + route + [depot]:
        if cur_time > latest_end[r]: return False
        cur_time = max(cur_time, earliest_start[r]) + service_time[r]
    return True

def route_insertion_cost(route, customer, service_time, 
                         earliest_start, latest_end,
                         distance_matrix):
    route_len = len(route)
    min_cost = float("inf")
    min_pos = None
    for i in range(route_len+1):
        if is_valid_pos(route, i, customer, service_time, earliest_start, latest_end):
            if i == 0:
                old_cost = distance_matrix[depot][route[0]]
                new_cost = distance_matrix[depot][customer] + distance_matrix[customer][route[0]]
            elif i == route_len:
                old_cost = distance_matrix[route[-1]][depot]
                new_cost = distance_matrix[customer][depot] + distance_matrix[route[-1]][customer]
            else:
                old_cost = distance_matrix[route[i-1]][route[i]]
                new_cost = distance_matrix[route[i-1]][customer] + distance_matrix[customer][route[i]]
            if new_cost - old_cost < min_cost: 
                min_cost = new_cost - old_cost
                min_pos = i 
    return min_cost, min_pos


def compute_route_cost(routes, distance_matrix):
    total_cost = 0.0
    for route in routes.values():
        total_cost += distance_matrix[depot][route[0]]
        for i in range(len(route)-1):
            total_cost += distance_matrix[route[i]][route[i+1]]
        total_cost += distance_matrix[route[-1]][depot]
    return total_cost


def heuristic_improvement(cur_routes, all_customers, truck_capacity, demands, service_time, 
                          earliest_start, latest_end,
                          distance_matrix, only_short_routes=False):
    ori_total_cost = compute_route_cost(cur_routes, distance_matrix)
    customers = select_candidate_points(cur_routes, distance_matrix, all_customers, only_short_routes=only_short_routes)
    routes_before_insert = {}
    # print("ori routes: ", cur_routes)
    for route_name, route in cur_routes.items():
        new_route = [c for c in route if c not in customers]
        if len(new_route) > 0: routes_before_insert[route_name] = new_route
    # print("selected customers: ", customers)
    # print("after routes: ", routes_before_insert)
    total_cost_before_insert = compute_route_cost(routes_before_insert, distance_matrix)
    customer_to_route_dict = {}
    for c in customers:
        route_cost_list = sorted([(route_name, route_insertion_cost(route, c, service_time, 
                                  earliest_start, latest_end,
                                  distance_matrix))
                                  for route_name, route in routes_before_insert.items()], key=lambda x: x[1][0])
        customer_to_route_dict[c] = [x for x in route_cost_list[:min(2, len(route_cost_list))] if (x[1][1] is not None)]
    
    min_total_cost_increase = float("inf")
    new_routes_after_insertion = None
    for i in range(2**(len(customers))):
        idx_list = [(i//(2**j))%2 for j in range(len(customers))]
        if np.any([idx+1>len(customer_to_route_dict[c]) for idx, c in zip(idx_list, customers)]): continue
        customer_on_route = {}
        for idx, c in zip(idx_list, customers):
            route_name = customer_to_route_dict[c][idx][0]
            if route_name not in customer_on_route:
                customer_on_route[route_name] = [c]
            else: customer_on_route[route_name].append(c)
        valid_insertion = True
        total_cost_increase = 0.0
        routes_after_insertion = {}
        for route_name, customers_to_insert in customer_on_route.items():
            if not valid_insertion: break
            new_route = routes_before_insert[route_name][:]
            for c in customers_to_insert:
                if np.sum([demands[_c] for _c in new_route]) + demands[c] > truck_capacity: 
                    valid_insertion = False
                    break
                min_cost, min_pos = route_insertion_cost(new_route, c, service_time, 
                                                         earliest_start, latest_end,
                                                         distance_matrix)
                if min_pos is None:
                    valid_insertion = False
                    break
                else: 
                    new_route = new_route[:min_pos] + [c] + new_route[min_pos:]
                    total_cost_increase += min_cost
            routes_after_insertion[route_name] = new_route
        
        if valid_insertion and total_cost_increase < min_total_cost_increase:
            min_total_cost_increase = total_cost_increase
            new_routes_after_insertion = {route_name: route[:] for route_name, route in routes_after_insertion.items()}
    if min_total_cost_increase + total_cost_before_insert < ori_total_cost:
        new_routes = {}
        for route_name, route in routes_before_insert.items():
            if route_name in new_routes_after_insertion: new_routes[route_name] = new_routes_after_insertion[route_name]
            else: new_routes[route_name] = route
        ori_total_cost = min_total_cost_increase + total_cost_before_insert
    else: new_routes = cur_routes
    return new_routes, ori_total_cost

def get_problem_dict(nb_customers,
                     demands, service_time, 
                     earliest_start, latest_end, max_horizon, 
                     distance_warehouses, distance_matrix):
    distance_matrix_dict = {}
    demands_dict = {}
    service_time_dict = {}
    earliest_start_dict = {}
    latest_end_dict = {}
    all_customers = [f"Customer_{i}" for i in range(1, 1+nb_customers)]
    for i, customer1 in enumerate([depot] + all_customers):
        if i == 0:
            demands_dict[customer1] = 0
            service_time_dict[customer1] = 0
            earliest_start_dict[customer1] = 0
            latest_end_dict[customer1] = max_horizon
        else:
            demands_dict[customer1] = demands[i-1]
            service_time_dict[customer1] = service_time[i-1]
            earliest_start_dict[customer1] = earliest_start[i-1]
            latest_end_dict[customer1] = latest_end[i-1]
        distance_matrix_dict[customer1] = {}
        for j, customer2 in enumerate([depot] + all_customers):
            if i == 0 and j == 0: distance_matrix_dict[customer1][customer2] = 0.0
            elif i == 0 and j > 0: distance_matrix_dict[customer1][customer2] = distance_warehouses[j-1]
            elif i > 0 and j == 0: distance_matrix_dict[customer1][customer2] = distance_warehouses[i-1]
            else: distance_matrix_dict[customer1][customer2] = distance_matrix[i-1][j-1]
    return all_customers, demands_dict, service_time_dict, earliest_start_dict, latest_end_dict, distance_matrix_dict

def generate_init_solution(nb_customers, truck_capacity,
                           demands, service_time, 
                           earliest_start, latest_end, max_horizon, 
                           distance_warehouses, distance_matrix):
    min_path_frag_len, max_path_frag_len = 20, 50
    print("solving a non-constraint tsp problem")
    tsp_solution = get_tsp_solution(nb_customers, distance_warehouses, distance_matrix)
    all_customers, demands_dict, \
        service_time_dict, earliest_start_dict, latest_end_dict,\
            distance_matrix_dict = get_problem_dict(nb_customers, demands, service_time,
                                                    earliest_start, latest_end, max_horizon,
                                                    distance_warehouses, distance_matrix)
    paths_dict = {}
    paths_cost_dict = {}
    paths_customers_dict = {}
    for i in range(nb_customers):
        path_name = f"PATH_{i}"
        customer = f"Customer_{i+1}"
        paths_dict[path_name] = [customer]
        paths_cost_dict[path_name] = distance_warehouses[i]*2
        for j in range(nb_customers):
            paths_customers_dict[path_name, f"Customer_{j+1}"] = 0
        paths_customers_dict[path_name, customer] = 1
                                                                                        
    # initialize path from tsp
    num_selected_customers = 0
    for i in range(20):
        if i == 0: _tsp_solution = tsp_solution[:]
        else:
            idx = np.random.randint(1, nb_customers-1)
            _tsp_solution = tsp_solution[idx:] + tsp_solution[:idx]
        while len(_tsp_solution) > 0:
            path_frag_len = np.random.randint(min_path_frag_len, max_path_frag_len)
            selected_customers = _tsp_solution[:min(path_frag_len, len(_tsp_solution))]
            _, route, _ = cvrptw_one_vehicle(selected_customers, truck_capacity, 
                                             distance_matrix, distance_warehouses, 
                                             demands, service_time,
                                             earliest_start, latest_end, 
                                             max_horizon, solver_type="PULP_CBC_CMD")
            
            _selected_customers = [f"Customer_{c+1}" for c in route]
            if np.sum([demands_dict[c] for c in _selected_customers]) <= truck_capacity and time_window_check(_selected_customers, service_time_dict, earliest_start_dict, latest_end_dict):
                paths_dict, paths_cost_dict, paths_customers_dict = add_path(route, paths_dict, paths_cost_dict, paths_customers_dict, nb_customers, distance_warehouses, distance_matrix)
            else:
                route = cvrptw_one_route(_selected_customers, truck_capacity,
                                        demands_dict, service_time_dict, 
                                        earliest_start_dict, latest_end_dict,
                                        distance_matrix_dict)
                route = [int(c.split("_")[1])-1 for c in route]
                paths_dict, paths_cost_dict, paths_customers_dict = add_path(route, paths_dict, paths_cost_dict, paths_customers_dict, nb_customers, distance_warehouses, distance_matrix)
            
            _selected_customers = [f"Customer_{c+1}" for c in selected_customers]
            route = cvrptw_one_route(_selected_customers, truck_capacity,
                                    demands_dict, service_time_dict, 
                                    earliest_start_dict, latest_end_dict,
                                    distance_matrix_dict)
            route = [int(c.split("_")[1])-1 for c in route]
            paths_dict, paths_cost_dict, paths_customers_dict = add_path(route, paths_dict, paths_cost_dict, paths_customers_dict, nb_customers, distance_warehouses, distance_matrix)
            num_selected_customers += len(route)
            for c in route: _tsp_solution.remove(c)
    total_cost, _, cur_routes = path_selection(nb_customers, 
                                               paths_dict,
                                               paths_cost_dict,
                                               paths_customers_dict,
                                               solver_type='PULP_CBC_CMD',
                                               binary_model=True)
    
    return cur_routes, total_cost, paths_dict, paths_cost_dict, paths_customers_dict
    
def one_round_heuristics(exp_round, round_res_dict, nb_customers, truck_capacity, demands, service_time,
                         earliest_start, latest_end, max_horizon,
                         distance_warehouses, distance_matrix):
    np.random.seed(exp_round)
    num_episodes = 100000
    early_stop_rounds = 1000
    all_customers, demands_dict, service_time_dict,\
        earliest_start_dict, latest_end_dict, distance_matrix_dict \
                    = get_problem_dict(nb_customers, demands, service_time,
                                       earliest_start, latest_end, max_horizon,
                                       distance_warehouses, distance_matrix)
    cur_routes, total_cost, _, _, _ = \
                            generate_init_solution(nb_customers, truck_capacity, demands, service_time,
                                                    earliest_start, latest_end, max_horizon,
                                                    distance_warehouses, distance_matrix)
    
    
    cost_list = []
    print("Master model total cost: ", total_cost)
    cost_list.append(total_cost)
    # fine tuning using dual-variable
    for i in range(num_episodes):
        cur_routes, total_cost = heuristic_improvement(cur_routes, all_customers, truck_capacity, 
                                                       demands_dict, service_time_dict, 
                                                       earliest_start_dict, latest_end_dict,
                                                       distance_matrix_dict)
        print(f"Round {exp_round}, Fine tune {i}, total cost: {total_cost}")
        cost_list.append(total_cost)
        if len(cost_list) > early_stop_rounds and np.min(cost_list[-early_stop_rounds:]) >= np.min(cost_list[:-early_stop_rounds]):
            break
    round_res_dict[exp_round] = (total_cost, cur_routes)
    return

import time

def main(problem_file, round_res_dict, m_process, solo=True):
    # dir_name = os.path.dirname(problem_file)
    # file_name = os.path.splitext(os.path.basename(problem_file))[0]
    if solo:
        (nb_customers, nb_trucks, truck_capacity, distance_matrix, distance_warehouses, demands, service_time,
                earliest_start, latest_end, max_horizon, warehouse_x, warehouse_y, customers_x, customers_y) = read_input_cvrptw(problem_file)
    else:
        problem = tools.read_vrplib(problem_file)
        nb_customers = len(problem['is_depot']) - 1
        truck_capacity = problem['capacity']
        _time_windows = problem['time_windows']
        _duration_matrix = problem['duration_matrix']
        _service_times = problem['service_times']
        _demands = problem['demands']
        distance_matrix = np.zeros((nb_customers, nb_customers))
        distance_warehouses = np.zeros(nb_customers)
        earliest_start, latest_end, service_time, demands, max_horizon = [], [], [], [], _time_windows[0][1]
        for i in range(nb_customers):
            distance_warehouses[i] = _duration_matrix[0][i+1]
            earliest_start.append(_time_windows[i+1][0])
            latest_end.append(_time_windows[i+1][1])
            service_time.append(_service_times[i+1])
            demands.append(_demands[i+1])
            for j in range(nb_customers):
                distance_matrix[i][j] = _duration_matrix[i+1][j+1]
    
    if m_process:
        num_rounds = max(64, mp.cpu_count())
        procs = []
        for exp_round in range(num_rounds):
            print("start round ", exp_round)
            proc = mp.Process(target=one_round_heuristics, args=(exp_round, round_res_dict, nb_customers, truck_capacity, demands, service_time,
                                                                earliest_start, latest_end, max_horizon,
                                                                distance_warehouses, distance_matrix,))
            procs.append(proc)
            proc.start()
        for proc in procs:
            proc.join()
    else:
        one_round_heuristics(1, round_res_dict, nb_customers, 
                             truck_capacity, demands, service_time,
                             earliest_start, latest_end, max_horizon,
                             distance_warehouses, distance_matrix)
    round_cost_list = sorted([(round, val[0]) for round, val in round_res_dict.items()], key=lambda x: x[1])
    print(round_cost_list)
    best_round = round_cost_list[0][0]
    if solo: total_cost = round(round_res_dict[best_round][0]/100,2)
    else: total_cost = round(round_res_dict[best_round][0],2)
    return len(round_res_dict[best_round][1]), total_cost
    

import multiprocessing as mp
from datetime import datetime
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--problem", type=str)
parser.add_argument("--instance", type=str)
parser.add_argument("--retrain", action="store_true")
parser.add_argument("--opt", action="store_true")
parser.add_argument("--batch", action="store_true")
parser.add_argument("--mp", action="store_true")
parser.add_argument("--remote", action="store_true")
args = parser.parse_args()

if args.remote: 
    data_dir = os.getenv("AMLT_DATA_DIR", "cvrp_benchmarks/")
    output_dir = os.environ['AMLT_OUTPUT_DIR']
else:
    data_dir = "./"
    output_dir = "./"

if __name__ == '__main__':
    sota_res = pd.read_csv("sota_res.csv")
    sota_res_dict = {row["problem"]: (row["distance"], row["vehicle"]) for _, row in sota_res.iterrows()}
    manager = mp.Manager()
    round_res_dict = manager.dict()
    is_solo = (args.instance != "ortec")
    if args.batch:
        result_list = []
        dir_name = os.path.dirname(f"{data_dir}/cvrp_benchmarks/homberger_{args.instance}_customer_instances/")
        problem_list = sorted(os.listdir(dir_name))
        res_file_name = f"{output_dir}/heuristic_res_{args.instance}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        for problem in problem_list:
            problem_file = os.path.join(dir_name, problem)
            if str.lower(os.path.splitext(os.path.basename(problem_file))[1]) != '.txt': continue
            if "path" in problem: continue
            print(problem_file)
            problem_name =  str.lower(os.path.splitext(os.path.basename(problem_file))[0])
            total_path_num, total_cost = main(problem_file, round_res_dict, args.mp, is_solo)
            sota = sota_res_dict.get(problem_name, (1, 1))
            result_list.append([problem, total_path_num, total_cost, sota[1], sota[0]])
            res_df = pd.DataFrame(data=result_list, columns=['problem', 'vehicles', 'total_cost', 'sota_vehicles', 'sota_cost'])
            res_df.loc[:, "gap"] = (res_df["total_cost"] - res_df["sota_cost"])/res_df["sota_cost"]
            res_df.to_csv(res_file_name, index=False)
        print(res_df.head())
    else:
        problem_file = f"{data_dir}/cvrp_benchmarks/homberger_{args.instance}_customer_instances/{args.problem}"
        dir_name = os.path.dirname(problem_file)
        problem_name = str.lower(os.path.splitext(os.path.basename(problem_file))[0])
        sota = sota_res_dict.get(problem_name, (1, 1))
        total_path_num, total_cost = main(problem_file, round_res_dict, args.mp, is_solo)
        print(args.problem, (total_path_num, sota[1]), (total_cost, sota[0]), (total_cost-sota[0])/sota[0])
