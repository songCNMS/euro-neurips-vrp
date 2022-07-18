import numpy as np

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

def extract_features_from_candidates(candidates, node_to_route_dict,
                                     cur_routes, truck_capacity, all_customers,
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
    return  candidates_feature, customers_features
