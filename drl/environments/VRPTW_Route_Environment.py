import random
import os
from collections import namedtuple
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from cvrptw import read_input_cvrptw, compute_cost_from_routes
import tools
from cvrptw_heuristic import heuristic_improvement, get_problem_dict, generate_init_solution, route_validity_check, heuristic_improvement_with_candidates, compute_route_cost
from cvrptw_utility import depot, device, feature_dim, max_num_nodes_per_route, max_num_route, route_output_dim, selected_nodes_num, extract_features_for_nodes, extend_candidate_points
from cvrptw_hybrid_heuristic import construct_solution_from_ge_solver



class VRPTW_Route_Environment(gym.Env):
    environment_name = "VRPTW Route Selection Environment"

    def __init__(self, instance, data_dir, seed=1):
        self.instance = instance
        self.data_dir = data_dir
        self.rd_seed = seed
        self.mode = 'train'
        self.reset()
        self.cur_step = 0
        self._max_episode_steps = 400
        self.max_episode_steps = self._max_episode_steps
        self.early_stop_steps = 10
        self.steps_not_improved = 0
        self.trials = 10
        self.reward_threshold = float("inf")
        self.id = "VRPTW_Route_Selection"
        self.num_states = max_num_nodes_per_route*feature_dim
        self.observation_space = spaces.Box(
            low=0.00, high=1.00, shape=(self.num_states, ), dtype=float
        )
        self.action_space = spaces.Discrete(max_num_route*max_num_nodes_per_route)
        self.cur_route_name = "PATH_0"
        self.cur_cost = None
        
    def seed(self, s):
        self.rd_seed = s

    def load_problem(self, problem_file, routes, cur_route_name):
        dir_name = os.path.dirname(f"{self.data_dir}/cvrp_benchmarks/homberger_{self.instance}_customer_instances/")
        if problem_file is None:
            problem_list = sorted(os.listdir(dir_name))
            # problem_file = np.random.choice(problem_list[:20])
            problem_file = "ORTEC-VRPTW-ASYM-0bdff870-d1-n458-k35.txt"
        self.problem_name = str.lower(os.path.splitext(os.path.basename(problem_file))[0])
        self.problem_file = f"{dir_name}/{problem_file}"
        if self.instance != 'ortec': self.problem = tools.read_solomon(self.problem_file)
        else: self.problem = tools.read_vrplib(self.problem_file)
        nb_customers = len(self.problem['is_depot']) - 1
        self.truck_capacity = self.problem['capacity']
        time_windows = self.problem['time_windows']
        earliest_start = [time_windows[i][0] for i in range(1, nb_customers+1)]
        latest_end = [time_windows[i][1] for i in range(1, nb_customers+1)]
        max_horizon = time_windows[0][1]
        duration_matrix = self.problem['duration_matrix']
        distance_warehouses = duration_matrix[0, 1:]
        distance_matrix = duration_matrix[1:, 1:] 
        service_times = self.problem['service_times']
        demands = self.problem['demands']
        self.max_distance = np.max(duration_matrix)
        self.all_customers, self.demands_dict, self.service_time_dict,\
            self.earliest_start_dict, self.latest_end_dict, self.distance_matrix_dict\
                = get_problem_dict(nb_customers,
                                   demands, service_times, 
                                   earliest_start, latest_end, max_horizon, 
                                   distance_warehouses, distance_matrix)
        self.coordinations = {}
        for i, c in enumerate([depot] + self.all_customers): self.coordinations[c] = self.problem['coords'][i]
        if routes is None:
            solution_file_name = f"{self.data_dir}/cvrp_benchmarks/RL_train_data/{self.problem_name}.npy"
            tmp_file_name =  f'tmp/{self.problem_name}'
            if os.path.exists(solution_file_name): _routes = np.load(solution_file_name, allow_pickle=True)
            else:
                if os.path.exists(tmp_file_name):
                    os.rmdir(tmp_file_name)
                os.makedirs(tmp_file_name, exist_ok=True)
                _routes, _ = construct_solution_from_ge_solver(self.problem, seed=self.rd_seed, tmp_dir=tmp_file_name, time_limit=240)
                np.save(solution_file_name, np.array(_routes))
            routes = {}
            for i, route in enumerate(_routes):
                path_name = f"PATH{i}"
                routes[path_name] = [f"Customer_{c}" for c in route]
        self.cur_routes = routes
        self.init_total_cost = self.get_route_cost()
        self.route_name_list = sorted(list(self.cur_routes.keys()))
            
    def reset(self, problem_file=None, routes=None, cur_route=None):
        # self.load_problem("ORTEC-VRPTW-ASYM-50d1f78d-d1-n329-k19.txt", routes, cur_route)
        self.load_problem(problem_file, routes, cur_route)
        self.steps_not_improved = 0
        self.cur_step = 0
        self.state = self.get_state()
        return self.state

    def get_route_cost(self):
        return compute_route_cost(self.cur_routes, self.distance_matrix_dict)
    
    def get_route_state(self, route):
        route_state = np.zeros(max_num_nodes_per_route*feature_dim)
        for i, c in enumerate(route): 
            if i + 1 >= max_num_nodes_per_route: break
            route_state[i*feature_dim:(i+1)*feature_dim] = \
                extract_features_for_nodes(c, route, self.truck_capacity, 
                                           self.demands_dict, self.service_time_dict,
                                           self.earliest_start_dict, self.latest_end_dict,
                                           self.distance_matrix_dict, self.max_distance, self.coordinations)
        return route_state
            
    def get_state(self):
        cur_routes_encode_state = np.zeros((max_num_route, max_num_nodes_per_route*feature_dim))
        route_len = np.zeros(max_num_route)
        for i, route_name in enumerate(self.route_name_list):
            route = self.cur_routes[route_name]
            cur_routes_encode_state[i, :] = self.get_route_state(route)
            route_len[i] = len(route)
        state = np.concatenate((route_len, cur_routes_encode_state.reshape(-1)), axis=0)
        return state
        
    def step(self, action):
        route_idx, node_idx = (action // max_num_nodes_per_route), (action % max_num_nodes_per_route)
        cur_route_name = self.route_name_list[route_idx]
        route = self.cur_routes.get(cur_route_name, [])
        self.cur_step += 1
        self.steps_not_improved += 1
        if node_idx >= len(route)-1: self.reward = 0.0
        else:
            M = extend_candidate_points(route, node_idx, self.distance_matrix_dict, self.all_customers)
            new_routes, ori_total_cost, cost_reduction =\
                    heuristic_improvement_with_candidates(self.cur_routes, M, self.truck_capacity, 
                                                          self.demands_dict, self.service_time_dict, 
                                                          self.earliest_start_dict, self.latest_end_dict,
                                                          self.distance_matrix_dict)
            if cost_reduction > 0: self.steps_not_improved = 0
            self.reward = max(0.0, 100*cost_reduction / self.max_distance)
            self.cur_routes = new_routes
        # if self.steps_not_improved > self.early_stop_steps:
        if self.reward <= 0.0:
            self.cur_route_idx = (self.cur_route_idx + 1) % len(self.route_name_list)
            self.cur_route_name = self.route_name_list[self.cur_route_idx]
        self.route_name_list = sorted(list(self.cur_routes.keys()))
        self.state = self.get_state()
        self.done = ((self.steps_not_improved >= self.early_stop_steps) | (self.cur_step >= self._max_episode_steps))
        return self.state, self.reward, self.done, {}

    def switch_mode(self, mode):
        self.mode = mode



