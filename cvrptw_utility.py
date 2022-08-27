from re import I
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import math
from nn_builder.pytorch.NN import NN
import tools
from pulp import *
import elkai
import sys
import lkh


route_output_dim = 128
max_num_route = 48
max_num_nodes_per_route = 24
node_embedding_dim = 32
depot = 0
feature_dim = 4+node_embedding_dim # (service_time, earlieast_time, latest_time, demand) + node_embedding
selected_nodes_num = 900


def extract_features_for_nodes(node, route, truck_capacity, 
                               demands, service_time,
                               earliest_start, latest_end,
                               distance_matrix, max_distance, 
                               node_embeddings):
    node_feature = np.zeros(feature_dim)
    # if node == depot: return node_feature
    max_duration = latest_end[depot]
    max_service_time = np.max(list(service_time.values()))
    # node_remaining_demand = truck_capacity
    # node_arrival_time = 0
    # next_node = pre_node = None
    # for i, c in enumerate(route):
    #     node_remaining_demand -= demands[c]
    #     node_arrival_time = max(node_arrival_time, earliest_start[c]) + service_time[c]
    #     if c == node: 
    #         next_node = (route[i+1] if i < len(route)-1 else depot)
    #         pre_node = (route[i-1] if i > 0 else depot)
    #         break
    # x_max = np.max([abs(c[0]) for c in coordinations.values()])
    # y_max = np.max([abs(c[1]) for c in coordinations.values()])
    # node_feature[0] = distance_matrix[pre_node][node] / max_distance
    # node_feature[1] = distance_matrix[node][next_node] / max_distance
    node_feature[0] = service_time[node] / max_service_time
    node_feature[1] = earliest_start[node] / max_duration
    node_feature[2] = latest_end[node] / max_duration
    # node_feature[5] = node_arrival_time / max_duration
    node_feature[3] = demands[node] / truck_capacity
    node_feature[4:] = node_embeddings[node]
    # node_feature[7] = node_remaining_demand / truck_capacity
    # node_feature[4] = distance_matrix[node][depot] / max_distance
    # node_feature[5] = coordinations[node][0] / x_max
    # node_feature[6] = coordinations[node][1] / y_max
    return node_feature

def get_candidate_feateures(candidates, node_to_route_dict,
                            cur_routes, truck_capacity,
                            demands, service_time,
                            earliest_start, latest_end,
                            distance_matrix, max_distance, coordinations):
    candidates_feature = np.zeros((2, feature_dim))
    candidates_feature[0, :] = extract_features_for_nodes(candidates[0], node_to_route_dict[candidates[0]],
                                                    truck_capacity, demands, service_time,
                                                    earliest_start, latest_end,
                                                    distance_matrix, max_distance, coordinations)
    if len(candidates) > 1:
        candidates_feature[1, :] = extract_features_for_nodes(candidates[1], node_to_route_dict[candidates[1]],
                                                        truck_capacity, demands, service_time,
                                                        earliest_start, latest_end,
                                                        distance_matrix, max_distance, coordinations)
    return candidates_feature

def get_customers_features(node_to_route_dict,
                           cur_routes, truck_capacity,
                           demands, service_time,
                           earliest_start, latest_end,
                           distance_matrix, max_distance, coordinations):
    customers_features = np.zeros((1, selected_nodes_num*feature_dim))
    i = 0
    for _, route in cur_routes.items():
        i += 1
        for c in route:
            customers_features[0, i*feature_dim:(i+1)*feature_dim]\
                                        = extract_features_for_nodes(c, node_to_route_dict, cur_routes,
                                                                    truck_capacity, demands, service_time,
                                                                    earliest_start, latest_end,
                                                                    distance_matrix, max_distance, coordinations)
            i += 1
    return customers_features

def extract_features_from_candidates(candidates, node_to_route_dict,
                                     cur_routes, truck_capacity, 
                                     demands, service_time,
                                     earliest_start, latest_end,
                                     distance_matrix, max_distance, coordinations):
    candidates_feature = get_candidate_feateures(candidates, node_to_route_dict,
                                                cur_routes, truck_capacity, 
                                                demands, service_time,
                                                earliest_start, latest_end,
                                                distance_matrix, max_distance, coordinations)
    customers_features = get_customers_features(node_to_route_dict,
                                                cur_routes, truck_capacity, 
                                                demands, service_time,
                                                earliest_start, latest_end,
                                                distance_matrix, max_distance, coordinations)
    return  candidates_feature, customers_features

def map_node_to_route(cur_routes):
    node_to_route_dict = {}
    for route_name, route in cur_routes.items():
        for c in route: node_to_route_dict[c] = route_name
    return node_to_route_dict


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

def time_window_check_partial_route(route, cur_time, service_time, distance_matrix, earliest_start, latest_end):
    prev_node = None
    for r in route + [depot]:
        if prev_node is not None: cur_time += distance_matrix[prev_node][r]
        if cur_time > latest_end[r]:
            return False
        cur_time = max(cur_time, earliest_start[r]) + service_time[r]
        prev_node = r
    return True


def route_insertion_cost(route, customer, instance):
    route_len = len(route)
    min_cost = float("inf")
    min_pos = None
    if not isinstance(customer, list): customer = [customer]
    if instance["demands"][route+customer].sum() > instance["capacity"]: return min_cost, min_pos
    earliest_start = instance["time_windows"][:, 0]
    latest_end = instance["time_windows"][:, 1]
    service_time = instance["service_times"]
    distance_matrix = instance["duration_matrix"]
    cur_time = earliest_start[depot] + service_time[depot]
    prev_node = depot
    for i in range(route_len+1):
        new_partial_route = customer + route[i:]
        # if cur_time + distance_matrix[prev_node][customer[0]] > latest_end[customer[0]]: break
        if time_window_check_partial_route(new_partial_route, cur_time+distance_matrix[prev_node][customer[0]], service_time, distance_matrix, earliest_start, latest_end):
            if i == 0:
                old_cost = distance_matrix[depot][route[0]]
                new_cost = distance_matrix[depot][customer[0]] + distance_matrix[customer[-1]][route[0]]
            elif i == route_len:
                old_cost = distance_matrix[route[-1]][depot]
                new_cost = distance_matrix[route[-1]][customer[0]] + distance_matrix[customer[-1]][depot]
            else:
                old_cost = distance_matrix[route[i-1]][route[i]]
                new_cost = distance_matrix[route[i-1]][customer[0]] + distance_matrix[customer[-1]][route[i]]
            cur_cost = new_cost-old_cost
            if cur_cost < min_cost: 
                min_cost = cur_cost
                min_pos = i
        if i < route_len:
            cur_time += distance_matrix[prev_node][route[i]]
            cur_time = max(cur_time, earliest_start[route[i]]) + service_time[route[i]]
            prev_node = route[i]
    return min_cost, min_pos


def compute_route_cost(routes, distance_matrix):
    total_cost = 0.0
    for route in routes.values():
        total_cost += distance_matrix[depot][route[0]]
        for i in range(len(route)-1):
            total_cost += distance_matrix[route[i]][route[i+1]]
        total_cost += distance_matrix[route[-1]][depot]
    return total_cost


def best_insertion_cost(cur_solution, customers, instance):
    routes_before_insert = []
    for route in cur_solution:
        new_route = [c for c in route if c not in customers]
        if len(new_route) > 0: routes_before_insert.append(new_route)
    for c in customers:
        route_cost_list = sorted([(i, route_insertion_cost(route, c, instance))
                                  for i, route in enumerate(routes_before_insert)], key=lambda x: x[1][0])
        route_idx, (min_cost, min_pos) = route_cost_list[0]
        if min_pos is not None: routes_before_insert[route_idx] = routes_before_insert[route_idx][:min_pos] + [c] + routes_before_insert[route_idx][min_pos:]
        else: return cur_solution
    return routes_before_insert
     

def heuristic_improvement_with_candidates(cur_solution, customers, instance):
    num_neighbor_routes = 2
    # ori_total_cost = compute_route_cost(cur_routes, distance_matrix)
    cur_routes = {}
    for i, route in enumerate(cur_solution): cur_routes[f"PATH{i}"] = route
    routes_before_insert = {}
    nodes_to_routes_dict = {}
    for route_name, route in cur_routes.items():
        for c in customers: nodes_to_routes_dict[c] = route_name
        new_route = [c for c in route if c not in customers]
        if len(new_route) > 0: routes_before_insert[route_name] = new_route
    customer_to_route_dict = {}
    for c in customers:
        route_cost_list = sorted([(route_name, route_insertion_cost(route, c, instance))
                                  for route_name, route in routes_before_insert.items()], key=lambda x: x[1][0])
        customer_to_route_dict[c] = [x for x in route_cost_list[:min(num_neighbor_routes, len(route_cost_list))] if (x[1][1] is not None)]
    
    min_total_cost_increase = float("inf")
    new_routes_after_insertion = None
    for i in range(num_neighbor_routes**(len(customers))):
        idx_list = [(i//(num_neighbor_routes**j))%num_neighbor_routes for j in range(len(customers))]
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
                min_cost, min_pos = route_insertion_cost(new_route, c, instance)
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
    final_solution = cur_solution
    if not math.isinf(min_total_cost_increase):
        new_routes = {}
        for route_name, route in routes_before_insert.items():
            if route_name in new_routes_after_insertion: new_routes[route_name] = new_routes_after_insertion[route_name]
            else: new_routes[route_name] = route
        final_solution = [route for route in new_routes.values()]
    return final_solution

if torch.cuda.is_available(): device = "cuda:0"
else: device = "cpu"

class Customer_Model(torch.nn.Module):
    def __init__(self):
        super(Customer_Model, self).__init__()
        self.output_dim = 32
        self.mlp = NN(input_dim=feature_dim, 
                     layers_info=[64, self.output_dim],
                     output_activation="tanh",
                     hidden_activations="tanh", initialiser="Xavier")
    
    def forward(self, x):
        return self.mlp(x)
    

class Route_Model(torch.nn.Module):
    def __init__(self):
        super(Route_Model, self).__init__()
        self.customer_model = Customer_Model()
        self.num_hidden = 2
        self.rnn = torch.nn.GRU(self.customer_model.output_dim, route_output_dim, self.num_hidden)
    
    def forward(self, x):
        x = x.reshape(-1, max_num_nodes_per_route, feature_dim).permute(1, 0, 2)
        x_rnn = []
        for i in range(max_num_nodes_per_route):
            x_rnn.append(self.customer_model(x[i, :, :]))
        x = torch.stack(x_rnn, axis=0)
        h = torch.zeros(self.num_hidden, x.size(1), route_output_dim).to(device)
        return self.rnn(x, h)


class Route_MLP_Model(torch.nn.Module):
    def __init__(self):
        super(Route_MLP_Model, self).__init__()
        self.mlp = NN(input_dim=max_num_nodes_per_route*feature_dim, 
                     layers_info=[256, 256, route_output_dim],
                     output_activation="tanh",
                     hidden_activations="tanh", initialiser="Xavier")
    
    def forward(self, x):
        return self.mlp(x)

    
   
class MLP_RL_Model(torch.nn.Module):
    def __init__(self, hyperparameters):
        super(MLP_RL_Model, self).__init__()
        self.key_to_use = hyperparameters['key_to_use']
        self.mlp_route = hyperparameters["linear_route"]
        if self.mlp_route: self.route_model = Route_MLP_Model()
        else: self.route_model = Route_Model()
        if self.key_to_use == 'Actor': self.final_layer = torch.nn.Softmax(dim=1)
        else: self.final_layer = torch.nn.Softmax(dim=1)
        self.exploration_rate = 1.0
        self.rate_delta = 0.01
        input_dim = max_num_nodes_per_route+route_output_dim*3
        self.mlp = NN(input_dim=input_dim, 
                      layers_info=hyperparameters["linear_hidden_units"] + [hyperparameters["output_dim"]],
                      output_activation=None,
                      batch_norm=hyperparameters["batch_norm"], dropout=hyperparameters["dropout"],
                      hidden_activations=hyperparameters["hidden_activations"], initialiser=hyperparameters["initialiser"],
                      columns_of_data_to_be_embedded=hyperparameters["columns_of_data_to_be_embedded"],
                      embedding_dimensions=hyperparameters["embedding_dimensions"], y_range=hyperparameters["y_range"],
                      random_seed=hyperparameters["seed"])

    def forward(self, state):
        num_samples = state.size(0)
        route_len = state[:, 0].cpu().numpy().astype(int)
        route_cost_mask = state[:, 1:1+max_num_nodes_per_route]
        # route_len_mask = np.zeros((num_samples, max_num_nodes_per_route))
        # for i in range(num_samples):
        #     route_len_mask[i, :route_len[i]] = 1.0
        # route_len_mask = torch.from_numpy(route_len_mask).to(device)
        state = state[:, 1+max_num_nodes_per_route:].reshape(num_samples, max_num_route+1, max_num_nodes_per_route*feature_dim)
        route_rnn_output_list = []
        if self.mlp_route: x_cr = self.route_model(state[:, 0, :])
        else:
            x_cr, _ = self.route_model(state[:, 0, :])
            x_cr = x_cr[-1, :, :]
        customers = state[:, 1:, :]
        for i in range(max_num_route):
            x_r = customers[:, i, :]
            if self.mlp_route: x_r = self.route_model(x_r)
            else:
                x_r, _ = self.route_model(x_r)
                x_r = x_r[-1, :, :]
            route_rnn_output_list.append(x_r)
        x_r_mean = torch.stack(route_rnn_output_list, dim=1).mean(axis=1)
        x_r_max, _ = torch.stack(route_rnn_output_list, dim=1).max(axis=1)
        x = torch.cat((route_cost_mask, x_cr, x_r_mean, x_r_max), axis=1)
        x = self.mlp(x)
        x = self.final_layer(x)
        if self.key_to_use != 'Actor': x = torch.mul(x, 100.0)
        out = torch.mul(x, route_cost_mask>0.0)
        return out
    
def get_route_mask(route_nums):
    num_samples = route_nums.size(0)
    route_nums_np = route_nums.cpu().numpy()
    route_num_mask = np.zeros((num_samples, max_num_route))
    route_len_mask = np.zeros((num_samples, max_num_route, max_num_nodes_per_route))
    for i in range(num_samples):
        route_num = np.count_nonzero(route_nums_np[i, :])
        route_num_mask[i, :route_num] = 1.0
        for j, route_len in enumerate(route_nums_np[i, :]):
            if route_len > 0: route_len_mask[i, j, :route_len] = 1.0
    route_num_mask = torch.from_numpy(route_num_mask).to(device)
    route_len_mask = torch.from_numpy(route_len_mask).to(device)
    return route_num_mask, route_len_mask

def fragment_reconstruction(route, instance):
    tsp_route = get_tsp_solution(route, instance)
    assert len(tsp_route) == len(route), f"wrong route len: {tsp_route}, {route}"
    new_route = [tsp_route[0]]
    for c in tsp_route[1:]:
        _cost, _pos = route_insertion_cost(new_route, c, instance)
        if _pos is None: return route
        new_route = new_route[:_pos] + [c] + new_route[_pos:]
    original_cost = tools.compute_route_driving_time(route, instance["duration_matrix"])
    new_cost = tools.compute_route_driving_time(new_route, instance["duration_matrix"])
    return route if new_cost-original_cost >=0 else new_route

def reorder_nodes_according_solution(nodes, solution):
    new_nodes = []
    for route in solution:
        for c in route:
            if c in nodes: new_nodes.append(c)
    return new_nodes

def swap_improvement(solution, instance):
    k1 = 2 #np.random.randint(1, 3)
    k2 = 2 #np.random.randint(1, 3)
    route_num = len(solution)
    chosen_routes = np.random.choice(list(range(route_num)), size=2, replace=False)
    route_idx1, route_idx2 = chosen_routes[0], chosen_routes[1]
    route1, route2 = solution[route_idx1], solution[route_idx2]
    len1, len2 = len(route1), len(route2)
    node_idx1 = (np.random.randint(len1-k1) if len1 > k1 else 0)
    node_idx2 = (np.random.randint(len2-k2) if len2 > k2 else 0)
    _route1 = route1[:node_idx1] + route1[node_idx1+k1:]
    _route2 = route2[:node_idx2] + route2[node_idx2+k2:]
    c1, c2 = route1[node_idx1:node_idx1+k1], route2[node_idx2:node_idx2+k2]
    if instance["demands"][_route1+c2].sum() > instance["capacity"] or instance["demands"][_route2+c1].sum() > instance["capacity"]: return solution, 0.0
    ori_cost1, ori_cost2 = tools.compute_route_driving_time(route1, instance["duration_matrix"]), tools.compute_route_driving_time(route2, instance["duration_matrix"])
    new_route1, reconstruced1 = get_tsptw_solution(_route1+c2, instance)
    cost_reduction = 0.0
    if reconstruced1:
        new_route2, reconstruced2 = get_tsptw_solution(_route2+c1, instance)
        if reconstruced2:
            new_cost1, new_cost2 = tools.compute_route_driving_time(new_route1, instance["duration_matrix"]), tools.compute_route_driving_time(new_route2, instance["duration_matrix"])
            cost_reduction = (ori_cost1 + ori_cost2) - (new_cost1 + new_cost2)
            if cost_reduction > 0:
                solution[route_idx1] = new_route1
                solution[route_idx2] = new_route2
    return solution, cost_reduction


def ruin_and_recreation(node, solution, instance):
    nb_customers = len(instance["demands"]) - 1
    duration_matrix = instance["duration_matrix"]
    num_nodes_ruin = 6 # nb_customers // np.random.randint(4, 8)
    if np.random.random() <= 1.0:
        dist_to_node = sorted([(c, duration_matrix[node][c]+duration_matrix[c][node]) for c in range(1, nb_customers+1)], key=lambda x: x[1])
        ruin_nodes = [dist_to_node[i][0] for i in range(num_nodes_ruin)]
    else:
        ruin_nodes = np.random.choice(list(range(1,1+nb_customers)), size=num_nodes_ruin)
    return heuristic_improvement_with_candidates(solution, ruin_nodes, instance)
  

def get_tsptw_problem(selected_customers, instance):
    all_customers = [depot] + selected_customers
    duration_matrix_str = ""
    time_window_str = ""
    for i, c in enumerate(all_customers):
        duration_matrix_str += " ".join([str(instance["duration_matrix"][c][_c]+instance["service_times"][c]) for _c in all_customers])
        duration_matrix_str += "\n"
        time_window_str += f"{i+1} {instance['time_windows'][c, 0]} {instance['time_windows'][c, 1]}\n"
    problem_str = f"""
NAME : tsptw
TYPE : TSPTW
DIMENSION : {len(all_customers)}
EDGE_WEIGHT_TYPE : EXPLICIT
EDGE_WEIGHT_FORMAT : FULL_MATRIX
EDGE_WEIGHT_SECTION
{duration_matrix_str}
TIME_WINDOW_SECTION
{time_window_str}
DEPOT_SECTION
1 
-1
EOF
"""
    return problem_str
    

def get_tsptw_solution(selected_customers, instance):
    problem_str = get_tsptw_problem(selected_customers, instance)
    problem = lkh.LKHProblem.parse(problem_str)
    solver_path = '/home/lesong/euro-neurips-vrp/LKH-3.0.7/LKH'
    route = lkh.solve(solver_path, problem=problem, max_trials=100, runs=1)
    if len(route) <= 0: return selected_customers, False
    else:
        new_route = [selected_customers[c-2] for c in route[0][1:]]
        return new_route, True


def get_tsp_solution(selected_customers, instance):
    nb_customers = len(selected_customers)
    M = np.zeros((nb_customers+1, nb_customers+1))
    for i, c1 in enumerate([depot] + selected_customers):
        for j, c2 in enumerate([depot] + selected_customers):
            M[i, j] = instance["duration_matrix"][c1][c2]
    route = elkai.solve_float_matrix(M)[1:]
    return [selected_customers[c-1] for c in route]


def solve_single_route(selected_customers,
                       instance,
                       lp_file_name = None,
                       bigm=10000000,
                       mip_gap=0.001,
                       solver_time_limit_minutes=1,
                       enable_solution_messaging=1,
                       solver_type='PULP_CBC_CMD'):
    _depot = "Customer_0"
    selected_customers = [0] + selected_customers
    num_customers = len(selected_customers)
    local_customers_var = [f"Customer_{i}" for i in range(num_customers)]
    local_transit_cost = np.zeros((num_customers, num_customers))
    for i, c1 in enumerate(selected_customers):
        for j, c2 in enumerate(selected_customers):
            local_transit_cost[i, j] = instance["duration_matrix"][c1, c2]
            
    local_assignment_var_dict = {}
    local_transit_cost_dict = {}
    for i in range(num_customers):
        for j in range(num_customers):
            local_assignment_var_dict[f"Customer_{i}", f"Customer_{j}"] = 0
            local_transit_cost_dict[f"Customer_{i}", f"Customer_{j}"] = local_transit_cost[i, j]

    # local_service_time = {}
    # local_demands = {}
    # local_earliest_start = {}
    # local_latest_end = {}
    # earliest_start = instance["time_windows"][:, 0]
    # latest_end = instance["time_windows"][:, 1]
    # service_time = instance["service_times"]
    # truck_capacity = instance["capacity"]
    # for i in range(num_customers):
    #     key = local_customers_var[i]
    #     local_demands[key] = instance["demands"][selected_customers[i]]
    #     local_earliest_start[key] = earliest_start[selected_customers[i]]
    #     local_latest_end[key] = latest_end[selected_customers[i]]
    #     local_service_time[key] = service_time[selected_customers[i]]

    # sub problem
    sub_model = LpProblem("SU_CVRPTW", LpMinimize)
    # time_var = LpVariable.dicts("Time", local_customers_var, 0, None, LpContinuous)
    assignment_var = LpVariable.dicts("Assign", local_assignment_var_dict.keys(), 0, 1, LpBinary) #LpBinary

    print('objective function')
    max_transportation_cost = np.max(list(local_transit_cost_dict.values()))
    
    sub_model += lpSum(
        (local_transit_cost_dict[from_loc, to_loc]-max_transportation_cost) * assignment_var[from_loc, to_loc]
        for from_loc, to_loc in local_assignment_var_dict.keys())

    # Each vehicle should leave from a depot
    print('Each vehicle should leave from a depot')
    sub_model += lpSum([assignment_var[_depot, customer]
                                for customer in local_customers_var]) == 1, "entryDepotConnection"

    # Flow in Flow Out
    print('Flow in Flow out')
    for customer in local_customers_var:
        sub_model += (assignment_var[customer, customer] == 0.0, f"no loop for {customer}")
        sub_model += lpSum([assignment_var[customer, _customer]
                            for _customer in local_customers_var]) == 1, f"must leave {customer}"
        sub_model += lpSum([assignment_var[_customer, customer]
                            for _customer in local_customers_var]) == 1, f"must enter {customer}"
        # sub_model += lpSum(
        #     [assignment_var[from_loc, customer] for from_loc in local_customers_var]) - lpSum(
        #     [assignment_var[customer, to_loc] for to_loc in local_customers_var]) == 0, "forTrip" + str(
        #     customer)

    # Each vehicle should enter a depot
    # print('Each vehicle should enter a depot')
    # sub_model += lpSum([assignment_var[customer, _depot]
    #                         for customer in local_customers_var]) == 1, "exitDepotConnection"

    # # vehicle Capacity
    # print('vehicle Capacity')
    # sub_model += lpSum(
    #     [float(local_demands[from_loc]) * assignment_var[from_loc, to_loc]
    #         for from_loc, to_loc in local_assignment_var_dict.keys()]) <= float(truck_capacity), "Capacity"

    # # Time intervals
    # print('time intervals')
    # for from_loc, to_loc in local_assignment_var_dict.keys():
    #     if to_loc == _depot: continue
    #     stop_time = local_service_time[from_loc]
    #     sub_model += time_var[to_loc] - time_var[from_loc] >= \
    #                     stop_time + bigm * assignment_var[
    #                         from_loc, to_loc] - bigm, "timewindow" + str(
    #         from_loc) + 'p' + str(to_loc)

    # # Time Windows
    # print('time windows')
    # for vertex in local_customers_var:
    #     time_var[vertex].bounds(float(local_earliest_start[vertex]),
    #                             float(local_latest_end[vertex]))

    if lp_file_name is not None:
        sub_model.writeLP('{}.lp'.format(str(lp_file_name)))

    print("Using solver ", solver_type)
    if solver_type == 'PULP_CBC_CMD':
        sub_model.solve(PULP_CBC_CMD(
            msg=enable_solution_messaging,
            timeLimit=60 * solver_time_limit_minutes,
            fracGap=mip_gap)
        )
    elif solver_type == "GUROBI_CMD":
        solver = getSolver('GUROBI_CMD', msg=enable_solution_messaging,
            timeLimit=60 * solver_time_limit_minutes)
        sub_model.solve(solver)

    if LpStatus[sub_model.status] in ('Optimal', 'Undefined'):
        print('Sub Model Status = {}'.format(LpStatus[sub_model.status]))
        print("Sub model optimized objective function= ", value(sub_model.objective))
        solution_objective = value(sub_model.objective)
        # get assignment variable values
        #print('getting solution for assignment variables')
        route_dict = {}
        for from_loc, to_loc in local_assignment_var_dict.keys():
            if assignment_var[from_loc, to_loc].value() > 0:
                route_dict[int(from_loc.split("_")[1])] = int(to_loc.split("_")[1])
        print(route_dict)
        route = []
        cur_node = depot
        while True:
            if cur_node in route_dict: cur_node = route_dict[cur_node]
            else: break
            if cur_node == depot:
                break
            route.append(cur_node)
        return route
    else:
        print('Model Status = {}'.format(LpStatus[sub_model.status]))
        raise Exception('No Solution Exists for the Sub problem')


def extend_candidate_points(route, node_idx, distance_matrix, all_customers):
    if len(route) <= 2: 
        M = route[:]
        prev_node = next_node = depot
    else:
        M = route[node_idx:node_idx+2]
        prev_node = (depot if node_idx == 0 else route[node_idx-1])
        next_node = (depot if node_idx >= len(route)-2 else route[node_idx+2])
    dist = [(c, distance_matrix[prev_node][c]+distance_matrix[c][next_node]) for c in all_customers if c not in route]
    dist = sorted(dist, key=lambda x: x[1])
    M.extend([dist[i][0] for i in range(min(4, len(dist)))])
    return M


def select_candidate_points(routes, distance_matrix, all_customers, only_short_routes=False):
    if only_short_routes:
        route_list = sorted([(r, len(r)) for r in routes.keys()], key=lambda x: x[1])
        route_list = [x[0] for x in route_list]
        route_name = np.random.choice(route_list[:min(5, len(route_list))])
    else: route_name = np.random.choice(list(routes.keys()))
    route = routes[route_name]
    node_idx = np.random.randint(0, len(route)-1)
    M = extend_candidate_points(route, node_idx, distance_matrix, all_customers)
    return M


def read_elem(filename):
    with open(filename) as f:
        return [str(elem) for elem in f.read().split()]

# The input files follow the "Solomon" format.
def read_input_cvrptw(filename):
    file_it = iter(read_elem(filename))

    for i in range(4): next(file_it)

    nb_trucks = int(next(file_it))
    truck_capacity = int(next(file_it))

    for i in range(13): next(file_it)

    warehouse_x = int(next(file_it))
    warehouse_y = int(next(file_it))

    for i in range(2): next(file_it)

    max_horizon = int(next(file_it))

    next(file_it)

    customers_x = []
    customers_y = []
    demands = []
    earliest_start = []
    latest_end = []
    service_time = []

    while (1):
        val = next(file_it, None)
        if val is None: break
        i = int(val) - 1
        customers_x.append(int(next(file_it)))
        customers_y.append(int(next(file_it)))
        demands.append(int(next(file_it)))
        ready = int(next(file_it))
        due = int(next(file_it))
        stime = int(next(file_it))
        earliest_start.append(ready)
        latest_end.append(due + stime)  # in input files due date is meant as latest start time
        service_time.append(stime)

    nb_customers = i + 1

    # Compute distance matrix
    distance_matrix = compute_distance_matrix(customers_x, customers_y)
    distance_warehouses = compute_distance_warehouses(warehouse_x, warehouse_y, customers_x, customers_y)

    return (nb_customers, nb_trucks, truck_capacity, distance_matrix, distance_warehouses, demands, service_time,
            earliest_start, latest_end, max_horizon, warehouse_x, warehouse_y, customers_x, customers_y)


# Computes the distance matrix
def compute_distance_matrix(customers_x, customers_y):
    nb_customers = len(customers_x)
    distance_matrix = [[None for i in range(nb_customers)] for j in range(nb_customers)]
    for i in range(nb_customers):
        distance_matrix[i][i] = 0
        for j in range(nb_customers):
            dist = compute_dist(customers_x[i], customers_x[j], customers_y[i], customers_y[j])
            distance_matrix[i][j] = dist
            distance_matrix[j][i] = dist
    return distance_matrix


# Computes the distances to warehouse
def compute_distance_warehouses(depot_x, depot_y, customers_x, customers_y):
    nb_customers = len(customers_x)
    distance_warehouses = [None] * nb_customers
    for i in range(nb_customers):
        dist = compute_dist(depot_x, customers_x[i], depot_y, customers_y[i])
        distance_warehouses[i] = dist
    return distance_warehouses


def compute_dist(xi, xj, yi, yj):
    return int(math.sqrt(math.pow(xi - xj, 2) + math.pow(yi - yj, 2)))
    # return int(round(math.sqrt(math.pow(xi - xj, 2) + math.pow(yi - yj, 2)), 2)*100)

def compute_dist_float(xi, xj, yi, yj):
    return round(math.sqrt(math.pow(xi - xj, 2) + math.pow(yi - yj, 2)), 2)


def compute_cost_from_routes(cur_routes, coords):
    total_cost = 0.0
    for j in range(len(cur_routes)):
        _route = [0] + list(cur_routes[j]) + [0]
        for i in range(len(_route)-1):
            total_cost += compute_dist_float(coords[_route[i]][0], coords[_route[i+1]][0], coords[_route[i]][1], coords[_route[i+1]][1])
    return round(total_cost, 2)


# depots = dat.depots1
# LOCATION_NAME   LATITUDE   LONGITUDE  TIME_WINDOW_START  TIME_WINDOW_END  MAXIMUM_CAPACITY

# customers = dat.customers1
# LOCATION_NAME   LATITUDE   LONGITUDE  STOP_TIME  TIME_WINDOW_START  TIME_WINDOW_END  DEMAND

# transportation_matrix = dat.transportation_matrix1
# FROM_LOCATION_NAME TO_LOCATION_NAME  FROM_LATITUDE  FROM_LONGITUDE  TO_LATITUDE  TO_LONGITUDE  DRIVE_MINUTES  HAVERSINE_DISTANCE_MILES  TRANSPORTATION_COST

# vehicles = dat.vehicles1.head(15)
# VEHICLE_NAME  CAPACITY  VEHICLE_FIXED_COST

# capacity = vehicles.iloc[0, :]['CAPACITY']

def solomon2df(nb_customers, nb_trucks, truck_capacity, distance_matrix, distance_warehouses, demands, service_time,
                earliest_start, latest_end, max_horizon, warehouse_x, warehouse_y, customers_x, customers_y):
    depots = pd.DataFrame(data={"LOCATION_NAME": ["depot"],
                                "LATITUE": [warehouse_x], "LONGITUDE": [warehouse_y],
                                "TIME_WINDOW_START": [0], "TIME_WINDOW_END": [max_horizon],
                                "MAXIMUM_CAPACITY": [5*np.sum(demands)]})
    customers_data = []
    customer_names = []
    for i in range(nb_customers):
        customer_names.append(f"Customer_{i}")
        customers_data.append([f"Customer_{i}", customers_x[i], customers_y[i], service_time[i],
                                 earliest_start[i],  latest_end[i], demands[i]])
    customers = pd.DataFrame(data=customers_data, columns="LOCATION_NAME   LATITUDE   LONGITUDE  STOP_TIME  TIME_WINDOW_START  TIME_WINDOW_END  DEMAND".split())
    transportation_matrix_data = []
    for i in range(nb_customers):
        for j in range(nb_customers):
            transportation_matrix_data.append([customer_names[i], customer_names[j],
                                               customers_x[i], customers_y[i],
                                               customers_x[j], customers_y[j],
                                               0, 0, distance_matrix[i][j]])
        transportation_matrix_data.append([customer_names[i], "depot",
                                            customers_x[i], customers_y[i],
                                            warehouse_x, warehouse_y,
                                            0, 0, distance_warehouses[i]])
        transportation_matrix_data.append(["depot", customer_names[i],
                                            warehouse_x, warehouse_y,
                                            customers_x[i], customers_y[i],
                                            0, 0, distance_warehouses[i]])

    transportation_matrix = pd.DataFrame(data=transportation_matrix_data, columns="FROM_LOCATION_NAME TO_LOCATION_NAME  FROM_LATITUDE  FROM_LONGITUDE  TO_LATITUDE  TO_LONGITUDE  DRIVE_MINUTES  HAVERSINE_DISTANCE_MILES  TRANSPORTATION_COST".split())
    vehicles_data = []
    for i in range(nb_trucks):
        vehicles_data.append([f"vehicle_{i}", truck_capacity, 1])
    vehicles = pd.DataFrame(data=vehicles_data, columns="VEHICLE_NAME  CAPACITY  VEHICLE_FIXED_COST".split())
    return depots, customers, transportation_matrix, vehicles