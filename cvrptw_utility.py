import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import math
from nn_builder.pytorch.NN import NN


route_output_dim = 128
max_num_route = 48
max_num_nodes_per_route = 24
node_embedding_dim = 32
depot = "Customer_0"
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



if torch.cuda.is_available(): device = "cuda:0"
else: device = "cpu"

class Customer_Model(torch.nn.Module):
    def __init__(self):
        super(Customer_Model, self).__init__()
        self.output_dim = 32
        self.mlp = NN(input_dim=feature_dim, 
                     layers_info=[128, self.output_dim],
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


class MLP_Model(torch.nn.Module):
    def __init__(self):
        super(MLP_Model, self).__init__()
        self.route_model = Route_Model()
        self.candidate_model = torch.nn.Linear(2*feature_dim, 64)
        self.fc1 = torch.nn.Linear(route_output_dim+64, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, 1)
        self.dropout = torch.nn.Dropout(p=0.2)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        
    def forward(self, candidates, customers):
        x_c = self.candidate_model(candidates)
        route_rnn_output_list = []
        for i in range(max_num_route):
            x_r = customers[:, i, :]
            x_r, _ = self.route_model(x_r)
            x_r = x_r[-1, :, :]
            route_rnn_output_list.append(x_r)
        x_r = torch.stack(route_rnn_output_list, dim=1).mean(axis=1)
        x_r = self.dropout(self.x_r)
        x = torch.cat((x_r, x_c), axis=1)
        dout = torch.tanh(self.fc1(x))
        dout = torch.tanh(self.fc2(dout))
        return self.fc3(dout)
    
   
class MLP_RL_Model(torch.nn.Module):
    def __init__(self, hyperparameters):
        super(MLP_RL_Model, self).__init__()
        self.key_to_use = hyperparameters['key_to_use']
        self.mlp_route = hyperparameters["linear_route"]
        if self.mlp_route: self.route_model = Route_MLP_Model()
        else: self.route_model = Route_Model()
        if self.key_to_use == 'Actor': self.final_layer = torch.nn.Softmax(dim=1)
        else: self.final_layer = torch.nn.Softplus(beta=1, threshold=40)
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
        route_len_mask = np.zeros((num_samples, max_num_nodes_per_route))
        for i in range(num_samples):
            route_len_mask[i, :route_len[i]] = 1.0
        route_len_mask = torch.from_numpy(route_len_mask).to(device)
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
        out = torch.mul(x, route_len_mask)
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
    

class MLP_Route_RL_Model(torch.nn.Module):
    def __init__(self, hyperparameters):
        super(MLP_Route_RL_Model, self).__init__()
        self.mlp_route = hyperparameters["linear_route"]
        self.route_model = Route_Model()
        self.final_layer = torch.nn.Softmax(dim=1)
        self.node_selection_mlp = NN(input_dim=route_output_dim*2, 
                                    layers_info=hyperparameters["linear_hidden_units"] + [max_num_nodes_per_route],
                                    output_activation=None,
                                    batch_norm=hyperparameters["batch_norm"], dropout=hyperparameters["dropout"],
                                    hidden_activations=hyperparameters["hidden_activations"], initialiser=hyperparameters["initialiser"],
                                    columns_of_data_to_be_embedded=hyperparameters["columns_of_data_to_be_embedded"],
                                    embedding_dimensions=hyperparameters["embedding_dimensions"], y_range=hyperparameters["y_range"],
                                    random_seed=hyperparameters["seed"])

    def forward(self, state):
        num_samples = state.size(0)
        route_nums = state[num_samples, :max_num_route]
        _, route_len_mask = get_route_mask(route_nums)
        customers = state[num_samples, max_num_route:]
        customers = customers.reshape(num_samples, max_num_route, max_num_nodes_per_route*feature_dim)
        route_rnn_output_list = []
        for i in range(max_num_route):
            x_r = customers[:, i, :]
            x_r, _ = self.route_model(x_r)
            x_r = x_r[-1, :, :]
            route_rnn_output_list.append(x_r)
        x_g = torch.stack(route_rnn_output_list, dim=1)
        x_g_mean = x_g.mean(axis=1)
        node_out = []
        for i in range(max_num_route):
            r = route_rnn_output_list[i]
            node_selection_in = torch.cat((r, x_g_mean), axis=1)
            node_selection_out = self.node_selection_mlp(node_selection_in)
            node_selection_out = self.final_layer(node_selection_out)
            node_selection_out = torch.mul(node_selection_out, route_len_mask)
            node_out.append(node_selection_out)
        node_out = torch.cat(node_out, axis=1)
        return node_selection_out


def acc_mrse_compute(pred, label):
    pred = pred.cpu().data.numpy()
    label = label.cpu().data.numpy()
    return math.sqrt(np.mean((pred-label)**2))


def transform_customers_data(num_samples, customer_features):
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
    return customers


def transform_data(candidate_features, customer_features, costs):
    labels = costs.astype(np.float32)
    num_samples = len(costs)
    candidates = candidate_features.reshape(num_samples, -1).astype(np.float32)
    customers = transform_customers_data(num_samples, customer_features)
    return candidates, customers, labels

class VRPTWDataset(Dataset):
    def __init__(self, candidate_features, customer_features, costs):
        self.candidates, self.customers, self.labels = transform_data(candidate_features, customer_features, costs)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.candidates[idx, :], self.customers[idx, :, :], self.labels[idx]

def extend_candidate_points(route, node_idx, distance_matrix, all_customers):
    if len(route) <= 2: 
        M = route[:]
        if len(M) <= 1: M.append(depot)
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

def random_choice(ll, probs):
    r = np.random.random()
    cp = 0
    for i, p in enumerate(probs):
        cp += p
        if cp >= r: return ll[i]
    return ll[-1]

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
    candidates_ts = torch.from_numpy(candidates).to(device)
    customers_ts = torch.from_numpy(customers).to(device)
    with torch.no_grad():
        predict_cost_reductions = model(candidates_ts, customers_ts).squeeze(axis=1).cpu().detach().numpy()
    # candidates_with_cost = []
    # for (route_name, node_idx), cost in zip(all_candidates, predict_cost_reductions):
    #     candidates_with_cost.append((cost, (route_name, node_idx)))
    # sorted_candidates = sorted(candidates_with_cost, key=lambda x: x[0], reverse=True)
    # idx = np.random.randint(0, min(50, len(sorted_candidates)))
    min_cost, max_cost = np.min(predict_cost_reductions), np.max(predict_cost_reductions)
    norm_cost = [round((x-min_cost)/max_cost, 2) for x in predict_cost_reductions]
    probs = [c/np.sum(norm_cost) for c in norm_cost]
    i = random_choice(list(range(len(all_candidates))), probs)
    route_name, node_idx = all_candidates[i]
    M = extend_candidate_points(route_name, routes, node_idx, distance_matrix, all_customers)
    return M