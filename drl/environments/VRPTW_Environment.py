import random
import os
from collections import namedtuple
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from cvrptw import read_input_cvrptw
import tools
from cvrptw_heuristic import heuristic_improvement, get_problem_dict, generate_init_solution, route_validity_check, heuristic_improvement_with_candidates
from cvrptw_utility import device, feature_dim, max_num_nodes_per_route, max_num_route, route_output_dim, selected_nodes_num, extract_features_for_nodes, extend_candidate_points
from cvrptw_hybrid_heuristic import construct_solution_from_ge_solver



class VRPTW_Environment(gym.Env):
    environment_name = "VRPTW Environment"

    def __init__(self, instance, data_dir, seed=1):
        self.instance = instance
        self.data_dir = data_dir
        self.rd_seed = seed
        self.mode = 'train'
        self.reset()
        self.cur_step = 0
        self._max_episode_steps = 1000
        self.early_stop_steps = 10
        self.steps_not_improved = 0
        self.trials = 10
        self.reward_threshold = np.float("inf")
        self.id = "VRPTW"
        self.num_states = (max_num_nodes_per_route+selected_nodes_num)*feature_dim
        self.observation_space = spaces.Box(
            low=0.00, high=1.00, shape=(self.num_states,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(max_num_nodes_per_route)
        self.cur_route_name = "PATH_0"
        self.cur_cost = None
        
    def seed(self, s):
        self.rd_seed = s

    def load_problem(self, problem_file, routes, cur_route_name):
        dir_name = os.path.dirname(f"{self.data_dir}/cvrp_benchmarks/homberger_{self.instance}_customer_instances/")
        if problem_file is None:
            problem_list = os.listdir(dir_name)
            problem_file = np.random.choice(problem_list)
        self.problem_name = problem_file
        self.problem_file = f"{dir_name}/{self.problem_name}"
        if self.instance != 'ortec':
            self.problem = tools.read_solomon(self.problem_file)
        else:
            self.problem = tools.read_vrplib(self.problem_file)
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
        if routes is None:
            if os.path.exists(f"RL_train_data/{self.problem_name}.npy"): _routes = np.load(f"RL_train_data/{self.problem_name}.npy", allow_pickle=True)
            else:
                if os.path.exists(f'tmp/{self.problem_name}'):
                    os.rmdir(f"tmp/{self.problem_name}")
                os.makedirs(f"tmp/{self.problem_name}", exist_ok=True)
                _routes, _ = construct_solution_from_ge_solver(self.problem, seed=self.rd_seed, tmp_dir=f"tmp/{self.problem_name}", time_limit=240)
                np.save(f"RL_train_data/{self.problem_name}.npy", np.array(_routes))
            routes = {}
            for i, route in enumerate(_routes):
                path_name = f"PATH{i}"
                routes[path_name] = [f"Customer_{c}" for c in route]
        self.cur_routes = routes
        self.route_name_list = sorted(list(self.cur_routes.keys()))
        self.cur_route_idx = 0
        self.cur_route_name = self.route_name_list[self.cur_route_idx]
            
    def reset(self, problem_file=None, routes=None, cur_route=None):
        # self.load_problem("ORTEC-VRPTW-ASYM-8b1620d9-d1-n346-k25.txt", routes, cur_route)
        self.load_problem(problem_file, routes, cur_route)
        self.steps_not_improved = 0
        self.cur_step = 0
        self.state = self.get_state()
        return np.copy(self.state)
    
    def get_route_state(self, route):
        route_state = np.zeros(max_num_nodes_per_route*feature_dim)
        for i, c in enumerate(route): 
            if i + 1 >= max_num_nodes_per_route: break
            route_state[i*feature_dim:(i+1)*feature_dim] = \
                extract_features_for_nodes(c, route, self.truck_capacity, 
                                           self.demands_dict, self.service_time_dict,
                                           self.earliest_start_dict, self.latest_end_dict,
                                           self.distance_matrix_dict, self.max_distance)
        return route_state
            
    def get_state(self):
        route = self.cur_routes[self.cur_route_name]
        cur_route_state = self.get_route_state(route)
        
        cur_routes_encode_state = np.zeros((max_num_route+1, max_num_nodes_per_route*feature_dim))
        cur_routes_encode_state[0, :] = cur_route_state
        for i, route_name in enumerate(self.route_name_list):
            route = self.cur_routes[route_name]
            cur_routes_encode_state[i+1, :] = self.get_route_state(route)
        return cur_routes_encode_state
        
    def step(self, action):
        node_idx = action
        route = self.cur_routes[self.cur_route_name]
        self.cur_step += 1
        self.steps_not_improved += 1
        self.done = False
        if node_idx >= len(route)-1: self.reward = 0.0
        else:
            M = extend_candidate_points(route, node_idx, self.distance_matrix_dict, self.all_customers)
            new_routes, ori_total_cost, cost_reduction =\
                    heuristic_improvement_with_candidates(self.cur_routes, M, self.truck_capacity, 
                                                        self.demands_dict, self.service_time_dict, 
                                                        self.earliest_start_dict, self.latest_end_dict,
                                                        self.distance_matrix_dict)
            if cost_reduction > 0: self.steps_not_improved = 0
            self.reward = cost_reduction / self.max_distance
            self.cur_routes = new_routes
        if self.steps_not_improved > self.early_stop_steps:
            self.cur_route_idx += 1
            if self.cur_route_idx >= len(self.route_name_list): self.done = True
            else:
                self.steps_not_improved = 0
                self.cur_route_name = self.route_name_list[self.cur_route_idx]
        self.state = self.get_state()
        self.done = (self.done | (self.cur_step >= self._max_episode_steps))
        return np.copy(self.state), self.reward, self.done, {}

    def switch_mode(self, mode):
        self.mode = mode



