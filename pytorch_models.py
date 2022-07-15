from cmath import exp
import torch as pt
import torchvision as ptv
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
route_output_dim = 128
max_num_route = 30
max_num_nodes_per_route = 2
device = "cuda:0"

class Route_Model(pt.nn.Module):
    def __init__(self):
        super(Route_Model, self).__init__()
        self.rnn = pt.nn.GRU(feature_dim, route_output_dim, 8)
    
    def forward(self, x):
        x = x.reshape(max_num_nodes_per_route, -1, feature_dim)
        h = pt.torch.zeros(8, x.size(1), route_output_dim).to(device)
        return self.rnn(x, h)


class MLP_Model(pt.nn.Module):
    def __init__(self):
        super(MLP_Model, self).__init__()
        self.route_model = Route_Model()
        self.candidate_model = pt.nn.Linear(2*feature_dim, 64)
        self.fc1 = pt.nn.Linear(route_output_dim+64, 256)
        self.fc2 = pt.nn.Linear(256, 128)
        self.fc3 = pt.nn.Linear(128, 1)
        self.dropout = pt.nn.Dropout(p=0.2)
        pt.nn.init.xavier_uniform_(self.fc1.weight)
        pt.nn.init.xavier_uniform_(self.fc2.weight)
        pt.nn.init.xavier_uniform_(self.fc3.weight)
        
    def forward(self, candidates, customers):
        x_c = self.candidate_model(candidates)
        route_rnn_output_list = []
        for i in range(max_num_route):
            x_r = customers[:, i, :]
            x_r, _ = self.route_model(x_r)
            x_r = x_r[-1, :, :]
            route_rnn_output_list.append(x_r)
        x_r = pt.torch.stack(route_rnn_output_list, dim=1).mean(axis=1)
        x = pt.torch.concat((x_r, x_c), axis=1)
        dout = pt.tanh(self.dropout(self.fc1(x)))
        dout = pt.tanh(self.fc2(dout))
        return self.fc3(dout)
    

class VRPTWDataset(Dataset):
    def __init__(self, candidate_features, customer_features, costs):
        self.labels = costs.astype(np.float32)
        num_samples = len(costs)
        self.candidates = candidate_features.reshape(num_samples, -1).astype(np.float32)
        self.customers = np.zeros((num_samples, max_num_route, max_num_nodes_per_route*feature_dim))
        customer_features = customer_features.reshape(num_samples, -1)
        for i in range(num_samples):
            j = -1 # route idx
            k = 0 # node idx on current route
            all_routes_features = customer_features[i, :]
            for l in range(selected_nodes_num):
                if np.sum(all_routes_features[l*feature_dim:]==0): break
                elif np.sum(all_routes_features[l:feature_dim:(l+1)*feature_dim]) == 0:
                    j += 1
                    k = 0
                elif k >= max_num_nodes_per_route-1: break
                else:
                    self.customers[i, j, feature_dim*k:feature_dim*(k+1)] = all_routes_features[l:feature_dim:(l+1)*feature_dim]
                    k += 1
        self.customers = self.customers.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.candidates[idx, :], self.customers[idx, :, :], self.labels[idx]
    
    
def acc_mrse_compute(pred, label):
    pred = pred.cpu().data.numpy()
    label = label.cpu().data.numpy()
    return math.sqrt(np.mean((pred-label)**2))


def eval_model(model, lossfunc, dataset):
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    total_loss = 0.0
    with pt.no_grad():
        iteration = 0
        for candidates, customers, labels in dataloader:
            outputs = model(candidates.to(device), customers.to(device)).squeeze(axis=1)
            loss = lossfunc(outputs, labels.to(device))
            total_loss += loss.item()
            iteration += 1
    return total_loss/iteration


def train_model(model, optimizer, lossfunc, dataset, eval_dataset=None):
    train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
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
            }, 'model.pt')
        plt.plot(train_loss_list, label='train loss')
        plt.plot(eval_loss_list, label='eval loss')
        plt.ylabel('train and eval loss')
        plt.legend()
        # if os.path.exists("loss.png"): os.remove("loss.png")
        plt.savefig("loss.png")
        plt.close()
    return model, optimizer
                
def map_node_to_route(cur_routes):
    node_to_route_dict = {}
    for route_name, route in cur_routes.items():
        for c in route: node_to_route_dict[c] = route_name
    return node_to_route_dict
                
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
    if os.path.exists("predict_data/all_data.candidates.npy"):
        candidates = np.load("predict_data/all_data.candidates.npy")
        customers = np.load("predict_data/all_data.customers.npy")
        costs = np.load("predict_data/all_data.cost.npy")
        return candidates, customers, costs
    all_files = os.listdir(folder_name)
    problem_list = [f.split('.')[0] for f in all_files if f.find("cost") >= 0]
    candidates_list = []
    customers_list = []
    cost_improvements_list = []
    for problem in problem_list:
        print(f"loading {problem}")
        exp_round = int(problem.split('_')[-1])
        if eval and exp_round < 13: continue
        elif (not eval) and exp_round >= 13: continue
        problem_file = os.path.join(folder_name, problem)
        candidates_list.append(np.load(problem_file + ".candidates.npy"))
        customers_list.append(np.load(problem_file+".customers.npy"))
        cost_improvements_list.append(np.load(problem_file+".cost.npy"))
    costs = np.concatenate(cost_improvements_list, axis=0)
    mean_cost, std_cost = np.mean(costs), np.std(costs)
    costs = np.array([(x-mean_cost)/std_cost for x in costs])
    candidates = np.concatenate(candidates_list, axis=0)
    customers = np.concatenate(customers_list, axis=0)
    np.save("predict_data/all_data.candidates", candidates, allow_pickle=False)
    np.save("predict_data/all_data.customers", customers, allow_pickle=False)
    np.save("predict_data/all_data.cost", costs, allow_pickle=False)
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
    max_exp_round = 2
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
        exp_name = datetime.now().date().strftime("%m%d-%H%M")
        wandb.init(project="VRPTW", config={}, name=exp_name)
        input_dim = feature_dim*(selected_nodes_num+2)
        # loss func and optim
        model = MLP_Model().to(device)
        # optimizer = pt.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
        optimizer = pt.optim.SGD(model.parameters(), lr=0.001)
        lossfunc = pt.nn.MSELoss().to(device)
        # data_folder = "amlt/vrptw_feature/vrptw_ortec/predict_data/"
        data_folder = f"{output_dir}/predict_data/"
        candidate_features, customer_features, cost_improvements = get_local_features(data_folder)
        candidate_features_eval, customer_features_eval, cost_improvements_eval = get_local_features(data_folder, eval=True)
        vrptw_dataset = VRPTWDataset(candidate_features, customer_features, cost_improvements)
        vrptw_dataset_eval = VRPTWDataset(candidate_features_eval, customer_features_eval, cost_improvements_eval)
        train_model(model, optimizer, lossfunc, vrptw_dataset, eval_dataset=vrptw_dataset_eval)