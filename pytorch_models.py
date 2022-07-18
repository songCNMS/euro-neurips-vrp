import torch as pt
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import numpy as np
import math
from candidate_predictor_features import extract_features_from_candidates, feature_dim, selected_nodes_num
from cvrptw import read_input_cvrptw
import tools
from cvrptw_hybrid_heuristic import construct_solution_from_ge_solver
from cvrptw_heuristic import heuristic_improvement, get_problem_dict, generate_init_solution
from cvrptw_utility import map_node_to_route, VRPTWDataset, device,  acc_mrse_compute, MLP_Model



def eval_model(model, lossfunc, dataset):
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)
    total_loss = 0.0
    with pt.no_grad():
        iteration = 0
        for candidates, customers, labels in dataloader:
            outputs = model(candidates.to(device), customers.to(device)).squeeze(axis=1)
            loss = lossfunc(outputs, labels.to(device))
            total_loss += loss.item()
            iteration += 1
    return total_loss/iteration


def train_model(model, optimizer, lossfunc, dataset, output_dir, eval_dataset=None):
    train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    train_loss_list,  eval_loss_list = [], []
    for epoch in range(2000):
        iteration = 0
        train_loss = 0.0
        for candidates, customers, labels in train_dataloader:
            optimizer.zero_grad()
            outputs = model(candidates.to(device), customers.to(device)).squeeze(axis=1)
            loss = lossfunc(outputs, labels.to(device))
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if (iteration+1) % 100 == 0:
                acc = acc_mrse_compute(outputs, labels)
                print(f"Epoch {epoch}, iter {iteration}:", acc)
                print(outputs[:10], labels[:10])
            iteration += 1
        train_loss = train_loss/iteration
        train_loss_list.append(train_loss)
        if eval_dataset is not None: eval_loss = eval_model(model, lossfunc, eval_dataset)
        else: eval_loss = eval_model(model, lossfunc, dataset)
        eval_loss_list.append(eval_loss)
        wandb.log({"train_loss": train_loss, "eval_loss": eval_loss})
        if np.min(eval_loss_list) == eval_loss_list[-1]:
            pt.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, f'{output_dir}/model.pt')
        plt.plot(train_loss_list, label='train loss')
        plt.plot(eval_loss_list, label='eval loss')
        plt.ylabel('train and eval loss')
        plt.legend()
        # if os.path.exists("loss.png"): os.remove("loss.png")
        plt.savefig(f"{output_dir}/loss.png")
        plt.close()
    return model, optimizer
                
                
def get_features(problem_file, exp_round, output_dir, solo=True):
    problem_name =  str.lower(os.path.splitext(os.path.basename(problem_file))[0])
    file_name = f"{output_dir}/predict_data/{problem_name}_{exp_round}"
    if os.path.exists(file_name+".candidates"):
        candidate_features = np.load(file_name+".candidates")
        customer_features = np.load(file_name+".customers")
        cost_improvements = np.load(file_name+".cost")
        return candidate_features, customer_features, cost_improvements
        
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
    
    max_distance = max(np.max(distance_matrix), np.max(distance_warehouses))
    all_customers, demands_dict, \
        service_time_dict, earliest_start_dict, latest_end_dict,\
            distance_matrix_dict = get_problem_dict(nb_customers, demands, service_time,
                                                    earliest_start, latest_end, max_horizon,
                                                    distance_warehouses, distance_matrix)
    
    num_episodes = 1000
    early_stop_rounds = 10
    if exp_round % 2 == 1: init_routes, total_cost = construct_solution_from_ge_solver(problem, seed=exp_round, tmp_dir=f'tmp/tmp_{problem_name}_{exp_round}', time_limit=600)
    else: init_routes, total_cost = None, None
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
    
    cost_list = [total_cost]
    candidate_features_list = []
    customer_features_list = []
    cost_improvement_list = []
    for i in range(num_episodes):
        cur_routes, total_cost, candidates = heuristic_improvement(cur_routes, all_customers, truck_capacity, 
                                                                   demands_dict, service_time_dict, 
                                                                   earliest_start_dict, latest_end_dict,
                                                                   distance_matrix_dict)
        cost_list.append(total_cost)
        cost_improvement_list.append((cost_list[-2]-cost_list[-1])/max_distance)
        node_to_route_dict = map_node_to_route(cur_routes)
        candidates_feature, other_customers_features = \
                            extract_features_from_candidates(candidates, node_to_route_dict,
                                                             cur_routes, truck_capacity, all_customers,
                                                             demands_dict, service_time_dict,
                                                             earliest_start_dict, latest_end_dict,
                                                             distance_matrix_dict, max_distance)
        candidate_features_list.append(candidates_feature)
        customer_features_list.append(other_customers_features)
        if len(cost_list) > early_stop_rounds and np.min(cost_list[-early_stop_rounds:]) >= np.min(cost_list[:-early_stop_rounds]):
            break
    candidate_features = np.concatenate(candidate_features_list, axis=0)
    customer_features = np.concatenate(customer_features_list, axis=0)
    cost_improvements = np.array(cost_improvement_list)
    os.makedirs(f"{output_dir}/predict_data", exist_ok=True)
    np.save(file_name+".candidates", candidate_features, allow_pickle=False)
    np.save(file_name+".customers", customer_features, allow_pickle=False)
    np.save(file_name+".cost", cost_improvements, allow_pickle=False)
    return candidate_features, customer_features, cost_improvements

def get_local_features(folder_name, eval=False):
    if os.path.exists(f"{folder_name}/all_data.candidates.npy"):
        candidates = np.load(f"{folder_name}/all_data.candidates.npy")
        customers = np.load(f"{folder_name}/all_data.customers.npy")
        costs = np.load(f"{folder_name}/all_data.cost.npy")
        return candidates, customers, costs
    all_files = os.listdir(folder_name)
    problem_list = [f.split('.')[0] for f in all_files if f.find("cost") >= 0]
    candidates_list = []
    customers_list = []
    cost_improvements_list = []
    for problem in problem_list:
        print(f"loading {problem}")
        exp_round = int(problem.split('_')[-1])
        if exp_round <= 10: continue
        if eval and exp_round < 45: continue
        elif (not eval) and exp_round >= 45: continue
        problem_file = os.path.join(folder_name, problem)
        candidates_list.append(np.load(problem_file + ".candidates.npy"))
        tmp = np.load(problem_file+".customers.npy")
        if len(tmp.shape) == 1: tmp = tmp.reshape(-1, feature_dim*selected_nodes_num)
        customers_list.append(tmp)
        cost_improvements_list.append(np.load(problem_file+".cost.npy"))
    costs = np.concatenate(cost_improvements_list, axis=0)
    mean_cost, std_cost = np.mean(costs), np.std(costs)
    costs = np.array([(x-mean_cost)/std_cost for x in costs])
    candidates = np.concatenate(candidates_list, axis=0)
    customers = np.concatenate(customers_list, axis=0)
    np.save(f"{folder_name}/all_data.candidates", candidates, allow_pickle=False)
    np.save(f"{folder_name}/all_data.customers", customers, allow_pickle=False)
    np.save(f"{folder_name}/all_data.cost", costs, allow_pickle=False)
    return candidates, customers, costs

from datetime import datetime
import time
import multiprocessing as mp
from datetime import datetime
import matplotlib.pyplot as plt
import argparse
import wandb
import os
os.environ["WANDB_API_KEY"] = "116a4f287fd4fbaa6f790a50d2dd7f97ceae4a03"
wandb.login()

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
    instance_list = ["ortec"]
    max_exp_round = 50
    all_experiments_list = []
    for exp_round in range(1, max_exp_round+1):
        for instance in instance_list:
            is_solo = (instance != 'ortec')
            dir_name = os.path.dirname(f"{data_dir}/cvrp_benchmarks/homberger_{instance}_customer_instances/")
            problem_list = os.listdir(dir_name)
            for problem in problem_list:
                problem_file = os.path.join(dir_name, problem)
                all_experiments_list.append((problem_file, exp_round, is_solo))
    # candidate_features, customer_features, cost_improvements = get_features(problem_file, is_solo)
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
            proc = get_features(problem_file, exp_round, is_solo)
    else:
        exp_name = datetime.now().strftime("%m%d-%H%M")
        wandb.init(dir=f"{output_dir}/wandb", project="VRPTW", config={}, name=exp_name)
        input_dim = feature_dim*(selected_nodes_num+2)
        # loss func and optim
        model = MLP_Model().to(device)
        # optimizer = pt.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
        optimizer = pt.optim.SGD(model.parameters(), lr=0.001)
        lossfunc = pt.nn.MSELoss().to(device)
        # data_folder = "amlt/vrptw_feature/vrptw_ortec/predict_data/"
        data_folder = f"{data_dir}/cvrp_benchmarks/predict_data/"
        candidate_features, customer_features, cost_improvements = get_local_features(data_folder)
        candidate_features_eval, customer_features_eval, cost_improvements_eval = get_local_features(data_folder, eval=True)
        vrptw_dataset = VRPTWDataset(candidate_features, customer_features, cost_improvements)
        vrptw_dataset_eval = VRPTWDataset(candidate_features_eval, customer_features_eval, cost_improvements_eval)
        train_model(model, optimizer, lossfunc, vrptw_dataset, output_dir, eval_dataset=vrptw_dataset_eval)