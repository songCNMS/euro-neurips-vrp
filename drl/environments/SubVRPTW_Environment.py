import sys
sys.path.append("./")
sys.path.append("./drl")
import multiprocessing as mp
import random
import os
import sys
import math
from collections import namedtuple
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from sknetwork.embedding import Spectral
import _pickle as cPickle

import tools
from cvrptw_utility import *
from rl_data_generate import construct_solution_from_ge_solver


class SubVRPTW_Environment(gym.Env):
    environment_name = "VRPTW Environment"
    num_nodes_to_sample = 16

    def __init__(self, instance, data_dir, save_data=False, seed=1):
        self.rng = np.random.default_rng(seed)
        self.instance = instance
        self.data_dir = data_dir
        self.rd_seed = seed
        self.save_data = save_data
        self.mode = 'train'
        self.reset()
        self.cur_step = 0
        self._max_episode_steps = max_num_nodes_per_route*max_num_route
        self.max_episode_steps = self._max_episode_steps
        self.switch_route_early_stop = 1
        self.early_stop_steps = max_num_route * self.switch_route_early_stop
        self.steps_not_improved = 0
        self.steps_not_improvoed_same_route = 0
        self.trials = 10
        self.reward_threshold = float("inf")
        self.id = "SubVRPTW"
        self.num_states = 1+max_num_nodes_per_route+(max_num_route+1)*max_num_nodes_per_route*feature_dim
        self.observation_space = spaces.Box(
            low=0.00, high=1.00, shape=(self.num_states, ), dtype=float
        )
        self.action_space = spaces.Discrete(max_num_nodes_per_route)
        self.cur_route_name = "PATH_0"
        self.cur_cost = None
        
    def seed(self, s):
        self.rd_seed = s
        self.rng = np.random.default_rng(s)

    def load_problem(self, problem_file, routes, cur_route_name):
        dir_name = os.path.dirname(f"{self.data_dir}/cvrp_benchmarks/homberger_{self.instance}_customer_instances/")
        if problem_file is None:
            problem_list = sorted(os.listdir(dir_name))
            # problem_list = [p for p in problem_list if (p.split('_')[0] in ["R1", "C1", "RC1"])]
            # problem_list = ['ORTEC-VRPTW-ASYM-e2f2ccf7-d1-n285-k25.txt', 'ORTEC-VRPTW-ASYM-ca1ed34e-d1-n226-k21.txt', 'ORTEC-VRPTW-ASYM-02182cf8-d1-n327-k20.txt', 'ORTEC-VRPTW-ASYM-d9af647d-d1-n237-k16.txt']
            problem_file = self.rng.choice(problem_list)
            # problem_file = "ORTEC-VRPTW-ASYM-0bdff870-d1-n458-k35.txt"
        self.problem_name = str.lower(os.path.splitext(os.path.basename(problem_file))[0])
        self.problem_file = f"{dir_name}/{problem_file}"
        print("loading problem: ", self.problem_name, self.rd_seed)
        if self.instance != 'ortec':
            (nb_customers, nb_trucks, truck_capacity, 
            distance_matrix, distance_warehouses, demands, service_time,
                earliest_start, latest_end, max_horizon, 
                    warehouse_x, warehouse_y, customers_x, customers_y) = read_input_cvrptw(self.problem_file)
            self.problem = tools.read_solomon(self.problem_file)
        else:
            problem = tools.read_vrplib(self.problem_file)
            self.problem = problem
            nb_customers = len(problem['is_depot']) - 1
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
        
        self.truck_capacity = self.problem['capacity']
        self.max_distance = max(np.max(distance_matrix), np.max(distance_warehouses))
        self.all_customers, self.demands_dict, self.service_time_dict,\
            self.earliest_start_dict, self.latest_end_dict, self.distance_matrix_dict\
                = get_problem_dict(nb_customers,
                                   demands, service_time, 
                                   earliest_start, latest_end, max_horizon, 
                                   distance_warehouses, distance_matrix)
        
        spectral = Spectral(n_components=node_embedding_dim)
        self.node_embeddings = {}
        node_embeddings_array = spectral.fit_transform(self.problem['duration_matrix'])
        for i, c in enumerate([depot] + self.all_customers): self.node_embeddings[c] = node_embeddings_array[i, :]
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
        self.cur_route_idx = 0
        self.cur_route_name = self.route_name_list[self.cur_route_idx]
        duration_matrix = self.problem["duration_matrix"]
        route_idx = self.rng.integers(len(self.route_name_list))
        node_idx = np.random.randint(len(self.cur_routes[route_idx]))
        node = self.cur_routes[route_idx][node_idx]
        dist_to_node = sorted([(c, duration_matrix[node][c]+duration_matrix[c][node]) for c in range(1, nb_customers+1)], key=lambda x: x[1])
        ruin_nodes = [dist_to_node[i][0] for i in range(SubVRPTW_Environment.num_nodes_to_sample)]
        self.sub_problem = get_sub_instance(ruin_nodes, self.cur_routes, self.problem)
        self.sub_routes = [[s, e] for s, e in zip(self.sub_problem["ori_starts"], self.sub_problem["ori_stops"])]
        self.order_to_dispatch = []
        for route in self.sub_problem["ori_routes"]: self.order_to_dispatch.extend(route)
            
    def reset(self, problem_file=None, routes=None, cur_route=None):
        # self.load_problem("ORTEC-VRPTW-ASYM-50d1f78d-d1-n329-k19.txt", routes, cur_route)
        self.load_problem(problem_file, routes, cur_route)
        self.steps_not_improved = 0
        self.steps_not_improvoed_same_route = 0
        self.cur_step = 0
        self.switch_route_early_stop = 1
        self.early_stop_steps = len(self.route_name_list) * self.switch_route_early_stop
        self.state = self.get_state()
        if self.save_data: self.local_experience_buffer = [np.copy(self.state)]
        return self.state

    def get_route_cost(self):
        return compute_route_cost(self.cur_routes, self.distance_matrix_dict)
    
    def get_route_state(self, route):
        route_state = np.zeros(max_num_nodes_per_route*feature_dim)
        for i, c in enumerate(route): 
            if i + 1 >= max_num_nodes_per_route: break
            route_state[i*feature_dim:(i+1)*feature_dim] = \
                extract_features_for_nodes(c, True, self.problem, self.sub_problem, self.node_embeddings)
        return route_state
            
    def get_state(self):
        improvement_vec = np.zeros(max_num_nodes_per_route)
        cur_route = self.cur_routes.get(self.cur_route_name, [])
        improvement_vec[0] = 1.0
        for node_idx in range(1, min(max_num_nodes_per_route, len(cur_route))):
            improvement_vec[node_idx], _ = self.get_improve(cur_route, node_idx)
        max_improvement, min_improvement = np.max(improvement_vec), np.min(improvement_vec)
        if max_improvement - min_improvement > 0.0: improvement_vec = (improvement_vec-min_improvement) / (max_improvement - min_improvement)
        cur_route_state = self.get_route_state(cur_route)
        cur_routes_encode_state = np.zeros((max_num_route+1, max_num_nodes_per_route*feature_dim))
        cur_routes_encode_state[0, :] = cur_route_state
        for i, route_name in enumerate(self.route_name_list):
            if route_name not in self.cur_routes: continue
            if i+1 > max_num_route: break
            route = self.cur_routes[route_name]
            cur_routes_encode_state[i+1, :] = self.get_route_state(route)
        state = cur_routes_encode_state.reshape(-1)
        state = np.concatenate(([len(cur_route)], improvement_vec, state), axis=0)
        return state
        
    def get_improve(self, route, node_idx):
        M = extend_candidate_points(route, node_idx, self.distance_matrix_dict, self.all_customers)
        cur_routes, _, cost_reduction =\
                heuristic_improvement_with_candidates(self.cur_routes, M, self.truck_capacity, 
                                                    self.demands_dict, self.service_time_dict, 
                                                    self.earliest_start_dict, self.latest_end_dict,
                                                    self.distance_matrix_dict)
        # cost_reduction - positive: shorter path, negative: longer path (not improved)
        return self.reward_shaping(cost_reduction), cur_routes
    
    def reward_shaping(self, cost_reduction):
        return max(0.0, 0.0 if (cost_reduction is None) else cost_reduction/100.0)

    def step(self, action):
        node_idx = action
        route = self.cur_routes[self.cur_route_name]
        assert node_idx < len(route), f"state: {self.state[0]}, node: {node_idx}, route: {len(route)}"
        self.cur_step += 1
        self.steps_not_improved += 1
        if node_idx > 0:
            cost_reduction, _routes = self.get_improve(route, node_idx)
            if cost_reduction > 0.0:
                self.steps_not_improved = 0
                self.cur_routes = _routes
            self.reward = cost_reduction
        else:
            self.cur_route_idx = (self.cur_route_idx + 1) % len(self.route_name_list)
            self.reward = 0.0
        while True:
            self.cur_route_name = self.route_name_list[self.cur_route_idx]
            if len(self.cur_routes.get(self.cur_route_name, [])) > 0: break
            self.cur_route_idx = (self.cur_route_idx + 1) % len(self.route_name_list)
        self.state = self.get_state()
        self.done = ((self.steps_not_improved >= self.early_stop_steps) | (self.cur_step >= self.max_episode_steps))
        if self.save_data: self.local_experience_buffer.extend([action, self.reward, np.copy(self.state)])
        return self.state, self.reward, self.done, {}

    def switch_mode(self, mode):
        self.mode = mode
        
    def save_experience(self, output_dir):
        if self.save_data:
            loc_dir = f"{output_dir}/{self.problem_name}/"
            os.makedirs(loc_dir, exist_ok=True)
            f = f"{loc_dir}/{self.rng.integers(0, 100000)}.dp"
            with open(f, "wb") as output_file:
                cPickle.dump(self.local_experience_buffer, output_file)


def generate_env_data(idx, num_envs, instance, data_dir):
    env = VRPTW_Environment(instance, data_dir, save_data=True, seed=idx)
    for i in range(num_envs):
        state = env.reset()
        print("epoch: ", i, "problem: ", env.problem_name)
        done = False
        while not done:
            action = env.rng.integers(min(int(state[0]), max_num_nodes_per_route))
            state, _, done, _ = env.step(action)
            print("epoch: ", i, "problem: ", env.problem_name, "step: ", env.cur_step, "reward: ", env.reward)
        env.save_experience(f"{data_dir}/vrptw_{instance}/")
    

import argparse
import os
parser = argparse.ArgumentParser(description='Input of VRPTW Trainer')
if __name__ == "__main__":
    parser.add_argument('--instance', type=str, default="ortec")
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--mp", type=int, default=1)
    args = parser.parse_args()
    if args.remote:
        data_dir = os.getenv("AMLT_DATA_DIR", "cvrp_benchmarks/")
    else:
        data_dir = "./"

    if args.mp == 1: generate_env_data(1, 100, args.instance, data_dir)
    else:
        procs = []
        for idx in range(args.mp):
            proc = mp.Process(target=generate_env_data, args=(idx, 100, args.instance, data_dir))
            procs.append(proc)
            proc.start()
        for proc in procs:
            proc.join()