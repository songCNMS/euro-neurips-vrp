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
from cvrptw_heuristic import heuristic_improvement, get_problem_dict, generate_init_solution, route_validity_check


depot = "Customer_0"

def construct_solution_from_ge_solver(instance, seed=1):
    solutions = list(solve_static_vrptw(instance, time_limit=120, seed=seed))
    assert len(solutions) >= 1, "failed to init"
    return solutions[-1]

def one_round_heuristics(exp_round, res_dict, problem, nb_customers, 
                         truck_capacity, demands, service_time,
                         earliest_start, latest_end, max_horizon,
                         distance_warehouses, distance_matrix):
    all_customers, demands_dict, \
        service_time_dict, earliest_start_dict, latest_end_dict,\
            distance_matrix_dict = get_problem_dict(nb_customers, demands, service_time,
                                                    earliest_start, latest_end, max_horizon,
                                                    distance_warehouses, distance_matrix)
    if exp_round % 2 == 0:
        np.random.seed(exp_round)
        cur_routes, total_cost, _, _, _ = \
                                generate_init_solution(nb_customers, truck_capacity, demands, service_time,
                                                        earliest_start, latest_end, max_horizon,
                                                        distance_warehouses, distance_matrix)
    else:
        init_routes, total_cost = construct_solution_from_ge_solver(problem, seed=exp_round)
        cur_routes = {}
        for i, route in enumerate(init_routes):
            path_name = f"PATH{i}"
            cur_routes[path_name] = [f"Customer_{c}" for c in route]
    print(f"Round: {exp_round}, init cost {total_cost}")
    num_episodes = 100000
    early_stop_rounds = 200
    cost_list = []
    for i in range(num_episodes):
        cur_routes, total_cost = heuristic_improvement(cur_routes, all_customers, truck_capacity, 
                                                       demands_dict, service_time_dict, 
                                                       earliest_start_dict, latest_end_dict,
                                                       distance_matrix_dict)
        print(f"Round: {exp_round}, fine tune {i}, total cost: {total_cost}")
        cost_list.append(total_cost)
        if len(cost_list) > early_stop_rounds and np.min(cost_list[-early_stop_rounds:]) >= np.min(cost_list[:-early_stop_rounds]):
            break
    assert route_validity_check(cur_routes, nb_customers, truck_capacity, demands_dict, service_time_dict, earliest_start_dict, latest_end_dict), f"wrong routes: {cur_routes}"
    res_dict[exp_round] = (total_cost, cur_routes)
    return

import time

def main(problem_file, round_res_dict, m_process, solo=True):
    if solo:
        (nb_customers, nb_trucks, truck_capacity, 
            distance_matrix, distance_warehouses, demands, service_time,
                earliest_start, latest_end, max_horizon, 
                    warehouse_x, warehouse_y, customers_x, customers_y) = read_input_cvrptw(problem_file)
        problem = tools.read_solomon(problem_file)
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
        num_rounds = max(16, mp.cpu_count())
        procs = []
        for exp_round in range(num_rounds):
            print("start round ", exp_round)
            proc = mp.Process(target=one_round_heuristics, args=(exp_round, round_res_dict, problem, nb_customers, 
                                                                 truck_capacity, demands, service_time,
                                                                 earliest_start, latest_end, max_horizon,
                                                                 distance_warehouses, distance_matrix,))
            procs.append(proc)
            proc.start()
        for proc in procs:
            proc.join()
    else:
        one_round_heuristics(1, round_res_dict, problem, nb_customers, 
                             truck_capacity, demands, service_time,
                             earliest_start, latest_end, max_horizon,
                             distance_warehouses, distance_matrix)
    
    round_cost_list = sorted([(round, val[0]) for round, val in round_res_dict.items()], key=lambda x: x[1])
    print(round_cost_list)
    best_round = round_cost_list[0][0]
    return len(round_res_dict[best_round][1]), round(round_res_dict[best_round][0],2)
    

import multiprocessing as mp
from datetime import datetime
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--problem", type=str)
parser.add_argument("--instance", type=str)
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
        problem_list = os.listdir(dir_name)
        res_file_name = f"{output_dir}/res_{args.instance}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
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
