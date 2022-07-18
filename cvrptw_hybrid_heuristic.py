from dataclasses import replace
from platform import node
from select import select
import numpy as np
import pandas as pd
from pulp import *
import sys
import os
import pickle
from cvrptw import read_input_cvrptw, compute_cost_from_routes
import tools
from solver import solve_static_vrptw
from cvrptw_heuristic import heuristic_improvement, get_problem_dict, generate_init_solution, route_validity_check
from candidate_predictor_features import depot


def construct_solution_from_ge_solver(instance, seed=1, tmp_dir='tmp', time_limit=240):
    print("solving using ges method")
    solutions = list(solve_static_vrptw(instance, time_limit=time_limit, seed=seed, tmp_dir=tmp_dir))
    # assert len(solutions) >= 1, "failed to init"
    if len(solutions) >= 1: return solutions[-1]
    else: return None, None

def one_round_heuristics(exp_round, res_dict, problem, nb_customers, 
                         truck_capacity, demands, service_time,
                         earliest_start, latest_end, max_horizon,
                         distance_warehouses, distance_matrix):
    all_customers, demands_dict, \
        service_time_dict, earliest_start_dict, latest_end_dict,\
            distance_matrix_dict = get_problem_dict(nb_customers, demands, service_time,
                                                    earliest_start, latest_end, max_horizon,
                                                    distance_warehouses, distance_matrix)
    
    # if exp_round % 2 == 1: init_routes, total_cost = construct_solution_from_ge_solver(problem, seed=exp_round, tmp_dir=f'tmp/tmp_{exp_round}')
    # else: init_routes, total_cost = None, None
    init_routes, total_cost = construct_solution_from_ge_solver(problem, seed=exp_round, tmp_dir=f'tmp/tmp_{exp_round}')
    num_episodes = 100000
    early_stop_rounds = 200
    if init_routes is None:
        np.random.seed(exp_round)
        cur_routes, total_cost, _, _, _ = \
                                generate_init_solution(nb_customers, truck_capacity, demands, service_time,
                                                       earliest_start, latest_end, max_horizon,
                                                       distance_warehouses, distance_matrix)
    else:
        cur_routes = {}
        for i, route in enumerate(init_routes):
            path_name = f"PATH{i}"
            cur_routes[path_name] = [f"Customer_{c}" for c in route]
    print(f"Round: {exp_round}, init cost {total_cost}")
    cost_list = []
    for i in range(num_episodes):
        cur_routes, total_cost, _ = heuristic_improvement(cur_routes, all_customers, truck_capacity, 
                                                          demands_dict, service_time_dict, 
                                                          earliest_start_dict, latest_end_dict,
                                                          distance_matrix_dict)
        print(f"Round: {exp_round}, fine tune {i}, total cost: {total_cost}")
        cost_list.append(total_cost)
        res_dict[exp_round] = (total_cost, cur_routes)
        if len(cost_list) > early_stop_rounds and np.min(cost_list[-early_stop_rounds:]) >= np.min(cost_list[:-early_stop_rounds]):
            break
    if route_validity_check(cur_routes, nb_customers, truck_capacity, demands_dict, service_time_dict, earliest_start_dict, latest_end_dict):
        res_dict[exp_round] = (total_cost, cur_routes)
    else: res_dict[exp_round] = (float("inf"), cur_routes)
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
        TIMEOUT = (12 * nb_customers)
        start = time.time()
        while time.time() - start <= TIMEOUT:
            if not any(p.is_alive() for p in procs):
                break
            time.sleep(120)
        else:
            # We only enter this if we didn't 'break' above.
            print("timed out, killing all processes")
            for p in procs:
                p.terminate()
        for proc in procs:
            proc.join()
    else:
        one_round_heuristics(1, round_res_dict, problem, nb_customers, 
                             truck_capacity, demands, service_time,
                             earliest_start, latest_end, max_horizon,
                             distance_warehouses, distance_matrix)
    
    round_cost_list = sorted([(round, val[0]) for round, val in round_res_dict.items()], key=lambda x: x[1])
    print(round_cost_list)
    if len(round_cost_list) > 0:
        best_round = round_cost_list[0][0]
        best_routes = round_res_dict[best_round][1]
        if solo:
            routes = []
            for _, route in best_routes.items():
                routes.append([int(c.split('_')[1]) for c in route])
            total_cost = compute_cost_from_routes(routes, problem['coords'])
        else: total_cost = round(round_res_dict[best_round][0],2)
        return len(best_routes), total_cost
    else: return -1, -1

import multiprocessing as mp
from datetime import datetime
import matplotlib.pyplot as plt
import argparse


if __name__ == '__main__':
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
    sota_res = pd.read_csv("sota_res.csv")
    sota_res_dict = {row["problem"]: (row["distance"], row["vehicle"]) for _, row in sota_res.iterrows()}
    manager = mp.Manager()
    round_res_dict = manager.dict()
    is_solo = (args.instance != "ortec")
    if args.batch:
        result_list = []
        dir_name = os.path.dirname(f"{data_dir}/cvrp_benchmarks/homberger_{args.instance}_customer_instances/")
        problem_list = sorted(os.listdir(dir_name))
        res_file_name = f"{output_dir}/hybrid_res_{args.instance}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
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
