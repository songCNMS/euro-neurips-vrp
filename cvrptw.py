import sys
import math
import pandas as pd
import numpy as np

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
    return round(math.sqrt(math.pow(xi - xj, 2) + math.pow(yi - yj, 2)), 2)


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


if __name__ == '__main__':
    problem_file = "/data/songlei/cvrptw-optimization/cvrp_benchmarks/homberger_100_customer_instances/c102.txt"
    (nb_customers, nb_trucks, truck_capacity, distance_matrix, distance_warehouses, demands, service_time,
                earliest_start, latest_end, max_horizon, warehouse_x, warehouse_y, customers_x, customers_y) = read_input_cvrptw(problem_file)


    # nb_customers = 50
    depots, customers, transportation_matrix, vehicles = solomon2df(nb_customers, nb_trucks, truck_capacity, distance_matrix, distance_warehouses, demands, service_time,
                earliest_start, latest_end, max_horizon, warehouse_x, warehouse_y, customers_x, customers_y)
    capacity = vehicles.iloc[0, :]['CAPACITY']
    vehicles = vehicles.head(18)

    print("depots:", depots)
    print("customers: ", customers)
    print("transportation_matrix: ", transportation_matrix)
    print("vehicles: ", vehicles)
    print("capacity: ", capacity)

    from cvrptw_optimization.single_depot_column_generation_pulp import run_single_depot_column_generation
    from cvrptw_optimization.src import single_depot_general_model_pulp_inputs as inputs
    from cvrptw_optimization.src import single_depot_general_model_pulp_formulation as formulation


    if False:
        model_inputs = inputs.ModelInputs(transportation_matrix, customers, depots, vehicles)
        print('Model')
        model = formulation.ModelFormulation(model_inputs.time_variables_dict,
                                            model_inputs.assignment_variables_dict,
                                            model_inputs.vertices_dict,
                                            model_inputs.vehicles_dict,
                                            model_inputs.customers_dict,
                                            model_inputs.transit_dict,
                                            model_inputs.transit_starting_customers_dict,
                                            depots['LOCATION_NAME'].iloc[0]
                                            )
        print('Formulating the problem')
        model.formulate_problem()

        print('Solving the model')
        # GUROBI_CMD
        model.solve_model(solver_type="GUROBI_CMD",
                        solver_time_limit_minutes=20)

        print('Getting model results')
        model.get_model_solution()

        print(model.solution_path['LOCATION_NAME'])
        print(model.solution_objective)
    else:
        solution, solution_statistics = run_single_depot_column_generation(
                                            depots,
                                            customers,
                                            transportation_matrix,
                                            vehicles,
                                            capacity,
                                            mip_gap=0.001,
                                            solver_time_limit_minutes=2,
                                            enable_solution_messaging=0,
                                            solver_type='PULP_CBC_CMD',
                                            max_iteration=2000)
        print(solution)
        print(solution_statistics)
        for path_name, route in solution.groupby("PATH_NAME"):
            location_list = route.sort_values(by="STOP_NUMBER")["LOCATION_NAME"].tolist()
            print(path_name, location_list)
