from platform import node
from select import select
import numpy as np
import pandas as pd
from pulp import *
import sys
import pickle
from cvrptw import read_input_cvrptw
from tsp import get_tsp_solution


def customer_order_by_distance(nb_customers, distance_matrix):
    customers_order_dict = {}
    for i in range(nb_customers):
        dist = sorted([(j, distance_matrix[i][j]) for j in range(nb_customers) if i != j], key=lambda x: x[1])
        customers_order_dict[i] = [x[0] for x in dist]
    return customers_order_dict
        
def select_candidate_points(routes, distance_matrix, nb_customers):
    route_idx = np.random.randint(0, len(routes))
    route = routes[route_idx]
    if len(route) < 2: return []
    node_idx = np.random.randint(0, len(route)-1)
    M = [route[node_idx], route[node_idx+1]]
    dist = [(i, distance_matrix[route[node_idx]][i]+distance_matrix[i][route[node_idx+1]]) for i in range(nb_customers) if i not in M]
    min_i1, min_i2 = None, None
    min_dist1, min_dist2 = np.float("inf"), np.float("inf")
    for i, d in dist:
        if i in route: continue
        if d < min_dist1:
            min_dist2 = min_dist1
            min_i2 = min_i1
            min_dist1 = d
            min_i1 = i
        elif d < min_dist2:
            min_dist2 = d
            min_i2 = i
    M.extend([min_i1, min_i2])
    return M

def is_valid_pos(route, pos, customer, service_time, earliest_start, latest_end, max_horizon):
    new_route = (route[:pos] if pos > 0 else [])
    new_route.append(customer)
    if pos < len(route): new_route.extend(route[pos:])
    cur_time = 0.0
    for r in new_route:
        if cur_time > latest_end[r]: return False
        cur_time = max(cur_time, earliest_start[r]) + service_time[r]
    return (cur_time <= max_horizon)

def route_insertion_cost(route, customer, service_time, 
                         earliest_start, latest_end, max_horizon,
                         distance_matrix, distance_warehouses):
    route_len = len(route)
    min_cost = np.float("inf")
    min_pos = None
    for i in range(route_len+1):
        if is_valid_pos(route, i, customer, service_time, earliest_start, latest_end, max_horizon):
            if i == 0:
                old_cost = distance_warehouses[route[0]]
                new_cost = distance_warehouses[customer] + distance_matrix[customer][route[0]]
            elif i == route_len:
                old_cost = distance_warehouses[route[-1]]
                new_cost = distance_warehouses[customer] + distance_matrix[route[-1]][customer]
            else:
                old_cost = distance_matrix[route[i-1]][route[i]]
                new_cost = distance_matrix[route[i-1]][customer] + distance_matrix[customer][route[i]]
            if new_cost - old_cost < min_cost: 
                min_cost = new_cost - old_cost
                min_pos = i 
    return min_cost, min_pos

def path_selection(num_customers, 
                   paths_dict,
                   paths_cost_dict,
                   paths_customers_dict,
                   number_of_paths=None,
                   lp_file_name=None,
                   binary_model=False,
                   mip_gap=0.001,
                   solver_time_limit_minutes=10,
                   enable_solution_messaging=1,
                   solver_type='PULP_CBC_CMD'
                   ):
    customers_var = [f"Customer_{i}" for i in range(1, num_customers+1)]
    master_model = LpProblem("MA_CVRPTW", LpMinimize)
    if binary_model:
        path_var = LpVariable.dicts("Path", paths_dict.keys(), 0, 1, LpBinary)
    else:
        path_var = LpVariable.dicts("Path", paths_dict.keys(), 0, 1, LpContinuous)
    print('Master model objective function')
    master_model += lpSum((paths_cost_dict[path]+1000000) * path_var[path] for path in paths_dict.keys())

    print('Each customer belongs to one path')
    for customer in customers_var:
        master_model += lpSum(
            [paths_customers_dict[path, customer] * path_var[path] for path in
                paths_dict.keys()]) == 1, "Customer" + str(customer)

    if number_of_paths is not None:
        master_model += lpSum(
            [path_var[path] for path in
                paths_dict.keys()]) <= number_of_paths, "No of Vehicles"

    if lp_file_name is not None:
        master_model.writeLP('{}.lp'.format(str(lp_file_name)))

    if solver_type == 'PULP_CBC_CMD':
        master_model.solve(PULP_CBC_CMD(
            msg=enable_solution_messaging,
            timeLimit=60*solver_time_limit_minutes,
            gapRel=mip_gap)
        )
    elif solver_type == "GUROBI_CMD":
        solver = getSolver('GUROBI_CMD', msg=enable_solution_messaging,
            timeLimit=60*solver_time_limit_minutes)
        master_model.solve(solver)

    print('Master Model Status = {}'.format(LpStatus[master_model.status]))
    if master_model.status == 1:
        solution_master_model_objective = value(master_model.objective)
        total_cost = 0.0
        print('Master model objective = {}'.format(str(solution_master_model_objective)))
        price = {}
        chosen_routes = {}
        for customer in customers_var:
            if solver_type == "GUROBI_CMD":
                price[customer] = float(master_model.constraints["Customer" + str(customer)].pi)
            else:
                price[customer] = float(master_model.constraints["Customer" + str(customer)].pi)
        solution_master_path = []
        for path in path_var.keys():
            if path_var[path].value() and path_var[path].value() > 0:
                solution_master_path.append({'PATH_NAME': path,
                                             'VALUE': path_var[path].value(),
                                             'PATH': paths_dict[path]
                                             })
                total_cost += paths_cost_dict[path]
                chosen_routes[path] = paths_dict[path]
                print(f"path_name: {path}, {paths_dict[path]}")
        sys.stdout.flush()
        solution_master_path = pd.DataFrame(solution_master_path)
        solution_master_path['OBJECTIVE'] = solution_master_model_objective
        return total_cost, price, chosen_routes
    else:
        raise Exception('No Solution Exists')

def cvrptw_one_vehicle(selected_customers, 
                       truck_capacity, distance_matrix, 
                       distance_warehouses, demands, service_time,
                       earliest_start, latest_end, 
                       max_horizon,
                       prices = None,
                       lp_file_name = None,
                       bigm=10000000,
                       mip_gap=0.001,
                       solver_time_limit_minutes=10,
                       enable_solution_messaging=1,
                       solver_type='PULP_CBC_CMD'):
    
    depot = "Customer_0"
    num_customers = len(selected_customers) + 1
    local_customers_var = [f"Customer_{i}" for i in range(num_customers)]
    local_transit_cost = np.zeros((num_customers, num_customers))
    for i in range(num_customers):
        for j in range(num_customers):
            if i == 0 and j == 0: continue
            elif i == 0: local_transit_cost[i, j] = distance_warehouses[selected_customers[j-1]]
            elif j == 0: local_transit_cost[i, j] = distance_warehouses[selected_customers[i-1]]
            else: local_transit_cost[i, j] = distance_matrix[selected_customers[i-1]][selected_customers[j-1]]

    local_assignment_var_dict = {}
    local_transit_cost_dict = {}
    for i in range(num_customers):
        for j in range(num_customers):
            local_assignment_var_dict[f"Customer_{i}", f"Customer_{j}"] = 0
            local_transit_cost_dict[f"Customer_{i}", f"Customer_{j}"] = local_transit_cost[i, j]

    local_service_time = {}
    local_demands = {}
    local_earliest_start = {}
    local_latest_end = {}
    for i in range(num_customers):
        key = local_customers_var[i]
        local_demands[key] = (0 if i == 0 else demands[selected_customers[i-1]])
        local_earliest_start[key] = (0 if i == 0 else earliest_start[selected_customers[i-1]])
        local_latest_end[key] = (max_horizon if i == 0 else latest_end[selected_customers[i-1]])
        local_service_time[key] = (0 if i == 0 else service_time[selected_customers[i-1]])

    # sub problem
    sub_model = LpProblem("SU_CVRPTW", LpMinimize)
    time_var = LpVariable.dicts("Time", local_customers_var, 0, None, LpContinuous)
    assignment_var = LpVariable.dicts("Assign", local_assignment_var_dict.keys(), 0, 1, LpContinuous) #LpBinary

    print('objective function')
    max_transportation_cost = np.max(list(local_transit_cost_dict.values()))
    
    if prices is None:
        sub_model += lpSum(
            (local_transit_cost_dict[from_loc, to_loc]-max_transportation_cost) * assignment_var[from_loc, to_loc]
            for from_loc, to_loc in local_assignment_var_dict.keys())
    else:
        prices[depot] = 0.0
        sub_model += lpSum(
            (local_transit_cost_dict[from_loc, to_loc]-max_transportation_cost-prices[from_loc]) * assignment_var[from_loc, to_loc]
            for from_loc, to_loc in local_assignment_var_dict.keys())
    # Each vehicle should leave from a depot
    print('Each vehicle should leave from a depot')
    sub_model += lpSum([assignment_var[depot, customer]
                                for customer in local_customers_var]) == 1, "entryDepotConnection"

    # Flow in Flow Out
    print('Flow in Flow out')
    for customer in local_customers_var:
        sub_model += (assignment_var[customer, customer] == 0.0, f"no loop for {customer}")
        sub_model += lpSum(
            [assignment_var[from_loc, customer] for from_loc in local_customers_var]) - lpSum(
            [assignment_var[customer, to_loc] for to_loc in local_customers_var]) == 0, "forTrip" + str(
            customer)

    # Each vehicle should enter a depot
    print('Each vehicle should enter a depot')
    sub_model += lpSum([assignment_var[customer, depot]
                            for customer in local_customers_var]) == 1, "exitDepotConnection"

    # vehicle Capacity
    print('vehicle Capacity')
    sub_model += lpSum(
        [float(local_demands[from_loc]) * assignment_var[from_loc, to_loc]
            for from_loc, to_loc in local_assignment_var_dict.keys()]) <= float(truck_capacity), "Capacity"

    # Time intervals
    print('time intervals')
    for from_loc, to_loc in local_assignment_var_dict.keys():
        if to_loc == depot: continue
        stop_time = local_service_time[from_loc]
        sub_model += time_var[to_loc] - time_var[from_loc] >= \
                        stop_time + bigm * assignment_var[
                            from_loc, to_loc] - bigm, "timewindow" + str(
            from_loc) + 'p' + str(to_loc)

    # Time Windows
    print('time windows')
    for vertex in local_customers_var:
        time_var[vertex].bounds(float(local_earliest_start[vertex]),
                                float(local_latest_end[vertex]))

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
                route_dict[from_loc] = to_loc
        route = []
        cur_node = depot
        # route_data = []
        while True:
            if cur_node != depot:
                node = selected_customers[int(cur_node.split('_')[1])-1]
                if node in route: break
                else: route.append(node)
            # route_data.append([cur_node, route_dict[cur_node], local_demands[cur_node], time_var[cur_node].value(), local_earliest_start[cur_node], local_latest_end[cur_node]])
            if cur_node in route_dict: cur_node = route_dict[cur_node]
            else: break
            i += 1
            if cur_node == depot:
                break
        # route_df = pd.DataFrame(data=route_data, columns=['previous_node', "next_node", "demand", "arrive_time", "ready_time", "due_time"])
        return solution_objective, route, None
    else:
        print('Model Status = {}'.format(LpStatus[sub_model.status]))
        raise Exception('No Solution Exists for the Sub problem')

def route_exist(route, paths_dict):
    for _, _route in paths_dict.items():
        if "_".join([str(r) for r in route]) == "_".join([str(r) for r in _route]): return True
    return False

def add_path(route, paths_dict, paths_cost_dict, paths_customers_dict, nb_customers, distance_warehouses, distance_matrix):
    if not route_exist(route, paths_dict):
        total_num_path = len(paths_dict.keys())
        path_name = f"PATH_{total_num_path}"
        paths_dict[path_name] = [f"Customer_{c+1}" for c in route]
        paths_cost_dict[path_name] = distance_warehouses[route[0]] + distance_warehouses[route[-1]]
        for j in range(len(route)-1):
            paths_cost_dict[path_name] += distance_matrix[route[j]][route[j+1]]
        for j in range(nb_customers):
            paths_customers_dict[path_name, f"Customer_{j+1}"] = 0
        for j in route:
            paths_customers_dict[path_name, f"Customer_{j+1}"] = 1
    return paths_dict, paths_cost_dict, paths_customers_dict

import os
if __name__ == '__main__':
    problem_file = "/home/lesong/cvrptw/cvrp_benchmarks/homberger_100_customer_instances/c104.txt"
    dir_name = os.path.dirname(problem_file)
    file_name = os.path.splitext(os.path.basename(problem_file))[0]
    (nb_customers, nb_trucks, truck_capacity, distance_matrix, distance_warehouses, demands, service_time,
                earliest_start, latest_end, max_horizon, warehouse_x, warehouse_y, customers_x, customers_y) = read_input_cvrptw(problem_file)
    num_episodes = 100
    min_path_frag_len, max_path_frag_len = 8, 20
    if False:
        print("solving a non-constraint tsp problem")
        tsp_solution = get_tsp_solution(nb_customers, distance_warehouses, distance_matrix)
        total_num_path = nb_customers
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
        ## initialize path from tsp
        num_selected_customers = 0
        for i in range(3):
            if i == 0: _tsp_solution = tsp_solution[:]
            else:
                idx = np.random.randint(1, nb_customers-1)
                _tsp_solution = tsp_solution[idx:] + tsp_solution[:idx]
            while len(_tsp_solution) > 0:
                print(f"selected customers {num_selected_customers}")
                path_frag_len = np.random.randint(min_path_frag_len, max_path_frag_len)
                selected_customers = _tsp_solution[:min(path_frag_len, len(_tsp_solution))]
                solution_objective, route, route_df = cvrptw_one_vehicle(selected_customers, truck_capacity, distance_matrix, distance_warehouses, demands, service_time,
                                        earliest_start, latest_end, max_horizon, solver_type="PULP_CBC_CMD")
                paths_dict, paths_cost_dict, paths_customers_dict = add_path(route, paths_dict, paths_cost_dict, paths_customers_dict, nb_customers, distance_warehouses, distance_matrix)
                num_selected_customers += len(route)
                for c in route:
                    _tsp_solution.remove(c)
        print(f"selected customers {num_selected_customers}")

        with open(f"{dir_name}/{file_name}_path.txt", "wb") as f:
            pickle.dump(paths_dict, f)
        with open(f"{dir_name}/{file_name}_path_cost.txt", "wb") as f:
            pickle.dump(paths_cost_dict, f)
        with open(f"{dir_name}/{file_name}_path_customer.txt", "wb") as f:
            pickle.dump(paths_customers_dict, f)
    else:
        with open(f"{dir_name}/{file_name}_path.txt", "rb") as f:
            paths_dict = pickle.load(f)
        with open(f"{dir_name}/{file_name}_path_cost.txt", "rb") as f:
            paths_cost_dict = pickle.load(f)
        with open(f"{dir_name}/{file_name}_path_customer.txt", "rb") as f:
            paths_customers_dict = pickle.load(f)
            
#         candidate_route = """Route  1 : 90 87 86 83 82 84 85 88 89 91
# Route  2 : 20 24 25 27 29 30 28 26 23 22 21
# Route  3 : 5 3 7 8 11 9 6 4 2 1 75
# Route  4 : 67 65 62 74 72 61 64 68 66 69
# Route  5 : 32 33 31 35 37 38 39 36 34
# Route  6 : 43 42 41 40 44 46 45 48 51 50 52 49 47
# Route  7 : 13 17 18 19 15 16 14 12 10
# Route  8 : 57 55 54 53 56 58 60 59
# Route  9 : 98 96 95 94 92 93 97 100 99
# Route  10 : 81 78 76 71 70 73 77 79 80 63""".split("\n")
#         routes = [r.split()[3:] for r in candidate_route]
#         routes = [[int(c)-1 for c in r] for r in routes]
#         for route in routes:
#             paths_dict, paths_cost_dict, paths_customers_dict = add_path(route, paths_dict, paths_cost_dict, paths_customers_dict, nb_customers, distance_warehouses, distance_matrix)

    # fine tuning using dual-variable
    for i in range(num_episodes):
        print(f"master model, episode: {i}")
        total_cost, prices_dict, chosen_routes = path_selection(nb_customers, 
                                                                paths_dict,
                                                                paths_cost_dict,
                                                                paths_customers_dict,
                                                                solver_type='PULP_CBC_CMD')
        print("total cost: ", total_cost)
        print(f"sub model, episode: {i}")
        # all_customers = list(range(nb_customers))
        # prices = list(prices_dict.values())
        # min_price, max_price = np.min(prices), np.max(prices)
        # norm_prices = [(x-min_price)/max_price for x in prices]
        # total_norm_price = np.sum(norm_prices)
        # prob_prices = [x/total_norm_price for x in norm_prices]
        # path_frag_len = np.random.randint(min_path_frag_len, max_path_frag_len)
        # selected_customers = np.random.choice(all_customers, size=path_frag_len, replace=False, p=prob_prices)
        
        num_chosen_routes = len(chosen_routes.keys())
        num_sampled_routes = 2
        sampled_routes_idx = np.random.choice(list(chosen_routes.keys()), size=min(num_sampled_routes, num_chosen_routes), replace=False)
        candidate_customers = []
        for idx in sampled_routes_idx:
            candidate_customers.extend(chosen_routes[idx])
        selected_customers = [int(c.split('_')[1])-1 for c in candidate_customers]
        first_round = True
        while len(selected_customers) > 0:
            if first_round and len(selected_customers) >= 4: _selected_customers = np.random.choice(selected_customers, size=len(selected_customers)-2, replace=False)
            else: _selected_customers = selected_customers
            first_round = False
            solution_objective, route, route_df = cvrptw_one_vehicle(_selected_customers, truck_capacity, distance_matrix, distance_warehouses, demands, service_time,
                                    earliest_start, latest_end, max_horizon, prices=prices_dict, solver_type="PULP_CBC_CMD")
            print("objective: ", solution_objective, 'route: ', route)
            print("capacity: ", truck_capacity, "total demand: ", route_df["demand"].sum())
            paths_dict, paths_cost_dict, paths_customers_dict = add_path(route, paths_dict, paths_cost_dict, paths_customers_dict, nb_customers, distance_warehouses, distance_matrix)
            for r in route: selected_customers.remove(r)