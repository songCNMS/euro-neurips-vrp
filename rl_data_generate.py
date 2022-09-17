import os
import pandas as pd
import numpy as np
import math
import tools
from datetime import datetime
import time
import multiprocessing as mp
from datetime import datetime
import matplotlib.pyplot as plt
import argparse
from solver import solve_static_vrptw


def construct_solution_from_ge_solver(instance, seed=1, tmp_dir='tmp', time_limit=240):
    print("solving using ges method")
    solutions = list(solve_static_vrptw(instance, time_limit=time_limit, seed=seed, tmp_dir=tmp_dir))
    # assert len(solutions) >= 1, "failed to init"
    if len(solutions) >= 1: return solutions[-1]
    else: return None, None

            
def get_features(problem_file, exp_round, output_dir, solo):
    problem_name = str.lower(os.path.splitext(os.path.basename(problem_file))[0])
    if solo:
        problem = tools.read_solomon(problem_file)
    else:
        problem = tools.read_vrplib(problem_file)
    
    num_customers = len(problem["demands"])
    # epoch_tlim = (5*60 if num_customers <= 300 else (10*60 if num_customers <= 500 else 15*60))
    epoch_tlim = 30
    init_routes, total_cost = construct_solution_from_ge_solver(problem, seed=exp_round, tmp_dir=f'tmp/tmp_{problem_name}_{exp_round}', time_limit=epoch_tlim)
    cur_routes = {}
    for i, route in enumerate(init_routes):
        path_name = f"PATH{i}"
        cur_routes[path_name] = [f"Customer_{c}" for c in route]
    os.makedirs(f"{output_dir}/cvrp_benchmarks/RL_train_data/", exist_ok=True)
    solution_file_name = f"{output_dir}/cvrp_benchmarks/RL_train_data/{problem_name}.npy"
    np.save(solution_file_name, np.array(init_routes))
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str)
    parser.add_argument("--instance", type=str)
    parser.add_argument("--mp", action="store_true")
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--data", action="store_true")
    args = parser.parse_args()

    if args.remote: 
        data_dir = os.getenv("AMLT_DATA_DIR", "cvrp_benchmarks/")
        output_dir = os.environ['AMLT_OUTPUT_DIR']
    else:
        data_dir = "./"
        output_dir = "./"

    # instance_list = ["200", "400", "600", "800", "1000", "ortec"]
    instance_list = [args.instance]
    max_exp_round = 1
    all_experiments_list = []
    for exp_round in range(1, max_exp_round+1):
        for instance in instance_list:
            is_solo = (instance != 'ortec')
            dir_name = os.path.dirname(f"{data_dir}/cvrp_benchmarks/homberger_{instance}_customer_instances/")
            problem_list = os.listdir(dir_name)
            for problem in problem_list:
                problem_file = os.path.join(dir_name, problem)
                all_experiments_list.append((problem_file, exp_round, is_solo))
    if args.data and args.mp:
        procs = []
        for problem_file, exp_round, is_solo in all_experiments_list:
            proc = mp.Process(target=get_features, args=(problem_file, exp_round, output_dir, is_solo))
            procs.append(proc)
            proc.start()
        TIMEOUT = 7200
        start = time.time()
        while time.time() - start <= TIMEOUT:
            if not any(p.is_alive() for p in procs):
                break
            time.sleep(720)
        else:
            # We only enter this if we didn't 'break' above.
            print("timed out, killing all processes")
            for p in procs:
                p.terminate()
        for proc in procs:
            proc.join()
    elif args.data:
        for problem_file, exp_round, is_solo in all_experiments_list:
            proc = get_features(problem_file, exp_round, output_dir, is_solo)