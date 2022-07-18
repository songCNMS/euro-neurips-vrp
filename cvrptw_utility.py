import numpy as np
from torch.utils.data import Dataset, DataLoader
from cvrptw_heuristic import extend_candidate_points


depot = "Customer_0"
feature_dim = 9 # (dist_to_prev_node, dist_to_next_node, service_time, earlieast_time, latest_time, arrival_time, demand, remaining_capacity, dist_to_depot)
selected_nodes_num = 800


def extract_features_for_nodes(node, node_to_route_dict, cur_routes,
                               truck_capacity, 
                               demands, service_time,
                               earliest_start, latest_end,
                               distance_matrix, max_distance):
    node_feature = np.zeros(feature_dim)
    if node == depot: return node_feature
    
    route_name = node_to_route_dict[node]
    route = cur_routes[route_name]
    max_duration = latest_end[depot]
    max_service_time = np.max(list(service_time.values()))
    
    node_remaining_demand = truck_capacity
    node_arrival_time = 0
    next_node = pre_node = None
    for i, c in enumerate(route):
        node_remaining_demand -= demands[c]
        node_arrival_time = max(node_arrival_time, earliest_start[c]) + service_time[c]
        if c == node: 
            next_node = (route[i+1] if i < len(route)-1 else depot)
            pre_node = (route[i-1] if i > 0 else depot)
            break
    node_feature[0] = distance_matrix[pre_node][node] / max_distance
    node_feature[1] = distance_matrix[node][next_node] / max_distance
    node_feature[2] = service_time[node] / max_service_time
    node_feature[3] = earliest_start[node] / max_duration
    node_feature[4] = latest_end[node] / max_duration
    node_feature[5] = node_arrival_time / max_duration
    node_feature[6] = demands[node] / truck_capacity
    node_feature[7] = node_remaining_demand / truck_capacity
    node_feature[8] = distance_matrix[node][depot] / max_distance
    return node_feature

def get_candidate_feateures(candidates, node_to_route_dict,
                            cur_routes, truck_capacity,
                            demands, service_time,
                            earliest_start, latest_end,
                            distance_matrix, max_distance):
    candidates_feature = np.zeros((2, feature_dim))
    candidates_feature[0, :] = extract_features_for_nodes(candidates[0], node_to_route_dict, cur_routes,
                                                    truck_capacity, demands, service_time,
                                                    earliest_start, latest_end,
                                                    distance_matrix, max_distance)
    if len(candidates) > 1:
        candidates_feature[1, :] = extract_features_for_nodes(candidates[1], node_to_route_dict, cur_routes,
                                                        truck_capacity, demands, service_time,
                                                        earliest_start, latest_end,
                                                        distance_matrix, max_distance)
    return candidates_feature

def get_customers_features(node_to_route_dict,
                           cur_routes, truck_capacity,
                           demands, service_time,
                           earliest_start, latest_end,
                           distance_matrix, max_distance):
    customers_features = np.zeros((1, selected_nodes_num*feature_dim))
    i = 0
    for _, route in cur_routes.items():
        i += 1
        for c in route:
            customers_features[0, i*feature_dim:(i+1)*feature_dim]\
                                        = extract_features_for_nodes(c, node_to_route_dict, cur_routes,
                                                                    truck_capacity, demands, service_time,
                                                                    earliest_start, latest_end,
                                                                    distance_matrix, max_distance)
            i += 1
    return customers_features

def extract_features_from_candidates(candidates, node_to_route_dict,
                                     cur_routes, truck_capacity, 
                                     demands, service_time,
                                     earliest_start, latest_end,
                                     distance_matrix, max_distance):
    candidates_feature = get_candidate_feateures(candidates, node_to_route_dict,
                                                cur_routes, truck_capacity, 
                                                demands, service_time,
                                                earliest_start, latest_end,
                                                distance_matrix, max_distance)
    customers_features = get_customers_features(node_to_route_dict,
                                                cur_routes, truck_capacity, 
                                                demands, service_time,
                                                earliest_start, latest_end,
                                                distance_matrix, max_distance)
    return  candidates_feature, customers_features

def map_node_to_route(cur_routes):
    node_to_route_dict = {}
    for route_name, route in cur_routes.items():
        for c in route: node_to_route_dict[c] = route_name
    return node_to_route_dict

import torch as pt
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

route_output_dim = 128
max_num_route = 30
max_num_nodes_per_route = 2

if pt.cuda.is_available(): device = "cuda:0"
else: device = "cpu"

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
        x = pt.torch.cat((x_r, x_c), axis=1)
        dout = pt.tanh(self.dropout(self.fc1(x)))
        dout = pt.tanh(self.fc2(dout))
        return self.fc3(dout)
    
def acc_mrse_compute(pred, label):
    pred = pred.cpu().data.numpy()
    label = label.cpu().data.numpy()
    return math.sqrt(np.mean((pred-label)**2))

def transform_data(candidate_features, customer_features, costs):
    labels = costs.astype(np.float32)
    num_samples = len(costs)
    candidates = candidate_features.reshape(num_samples, -1).astype(np.float32)
    customers = np.zeros((num_samples, max_num_route, max_num_nodes_per_route*feature_dim))
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
                customers[i, j, feature_dim*k:feature_dim*(k+1)] = all_routes_features[l:feature_dim:(l+1)*feature_dim]
                k += 1
    customers = customers.astype(np.float32)
    return candidates, customers, labels

class VRPTWDataset(Dataset):
    def __init__(self, candidate_features, customer_features, costs):
        self.candidates, self.customers, self.labels = transform_data(candidate_features, customer_features, costs)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.candidates[idx, :], self.customers[idx, :, :], self.labels[idx]


def select_candidate_points_ML(model, routes, distance_matrix, 
                               truck_capacity, all_customers,
                               demands, service_time,
                               earliest_start, latest_end,
                               max_distance):
    total_num_candidates = -1
    all_candidates = []
    candidates_list = []
    customers_list = []
    node_to_route_dict = map_node_to_route(routes)
    customers_features = get_customers_features(node_to_route_dict, routes, truck_capacity,
                                                demands, service_time, earliest_start,
                                                latest_end, distance_matrix, max_distance)
    for route_name, route in routes.items():
        for node_idx in range(len(route)-1):
            if total_num_candidates > 0 and np.random.random() > total_num_candidates / len(all_customers): continue
            candidate_points = [route[node_idx]]
            if node_idx == len(route)-2: candidate_points.append(depot)
            else: candidate_points.append(route[node_idx+1])
            candidates_feature = \
                         get_candidate_feateures(candidate_points, node_to_route_dict,
                                                 routes, truck_capacity,
                                                 demands, service_time,
                                                 earliest_start, latest_end,
                                                 distance_matrix, max_distance)
            all_candidates.append((route_name, node_idx))
            candidates_list.append(candidates_feature)
            customers_list.append(customers_features)
    _candidates = np.concatenate(candidates_list, axis=0)
    _customers = np.concatenate(customers_list, axis=0)
    _costs = np.zeros(len(candidates_list))
    candidates, customers, labels = transform_data(_candidates, _customers, _costs)
    candidates_ts = pt.from_numpy(candidates).to(device)
    customers_ts = pt.from_numpy(customers).to(device)
    with pt.no_grad():
        predict_cost_reductions = model(candidates_ts, customers_ts).squeeze(axis=1).cpu().detach().numpy()
    candidates_with_cost = []
    for (route_name, node_idx), cost in zip(all_candidates, predict_cost_reductions):
        candidates_with_cost.append((cost, (route_name, node_idx)))
    sorted_candidates = sorted(candidates_with_cost, key=lambda x: x[0], reverse=True)
    route_name, node_idx = sorted_candidates[0][1]
    M = extend_candidate_points(route_name, routes, node_idx, distance_matrix, all_customers)
    return M