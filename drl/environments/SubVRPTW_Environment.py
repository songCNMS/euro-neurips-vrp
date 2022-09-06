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


def check_route_validity(route, route_idx, problem, sub_problem):
    if problem["demands"][route].sum() > sub_problem["capacities"][route_idx]: return False
    latest_stop_time = sub_problem["stop_times"][route_idx]
    start_depot = sub_problem["ori_starts"][route_idx]
    stop_depot = sub_problem["ori_stops"][route_idx]
    cur_time = max(sub_problem["start_times"][route_idx], problem["time_windows"][start_depot, 0])
    prev_node = start_depot
    for c in route + [stop_depot]:
        cur_time += problem["service_times"][prev_node] + problem["duration_matrix"][prev_node, c]
        if cur_time > problem["time_windows"][c, 1]: return False
        cur_time = max(cur_time, problem["time_windows"][c, 0])
        prev_node = c
    return cur_time <= latest_stop_time

def greedy_insertion(route, route_idx, node, problem, sub_problem):
    min_pos, min_cost = None, float("inf")
    if problem["demands"][route+[node]].sum() > sub_problem["capacities"][route_idx]: return min_pos, min_cost
    start_depot = sub_problem["ori_starts"][route_idx]
    stop_depot = sub_problem["ori_stops"][route_idx]
    full_route = [start_depot] + route + [stop_depot]
    for i in range(len(route)+1):
        new_route = route[:i] + [node] + route[i:]
        if check_route_validity(new_route, route_idx, problem, sub_problem):
            dist = (-problem["duration_matrix"][full_route[i], full_route[i+1]]
                     + problem["duration_matrix"][full_route[i], node]
                     + problem["duration_matrix"][node, full_route[i+1]])
            if dist < min_cost:
                min_pos, min_cost = i, dist
    return min_pos, min_cost

def compute_route_cost(routes, problem, sub_problem):
    start_depots = sub_problem["ori_starts"]
    stop_depots = sub_problem["ori_stops"]
    duration_matrix = problem["duration_matrix"]
    total_cost = 0.0
    for route_idx, route in enumerate(routes):
        route = [start_depots[route_idx]] + \
            routes[route_idx] + [stop_depots[route_idx]]
        total_cost += duration_matrix[route[:-1], route[1:]].sum()
    return total_cost

class SubVRPTW_Environment(gym.Env):
    environment_name = "VRPTW Environment"
    num_nodes_to_sample = 12

    def __init__(self, instance, data_dir, save_data=False, seed=1):
        self.rng = np.random.default_rng(seed)
        self.instance = instance
        self.data_dir = data_dir
        self.rd_seed = seed
        self.save_data = save_data
        self.mode = 'train'
        self._max_episode_steps = max_num_nodes*2
        self.max_episode_steps = self._max_episode_steps
        self.cur_step = 0
        self.trials = 10
        self.reward_threshold = float("inf")
        self.id = "SubVRPTW"
        self.num_states = 1+max_num_route+max_num_route*max_num_nodes_per_route*feature_dim+max_num_nodes*feature_dim
        self.observation_space = spaces.Box(
            low=0.00, high=1.00, shape=(self.num_states, ), dtype=float
        )
        self.action_space = spaces.Discrete(1+max_num_route)
        self.reset()

    def seed(self, s):
        self.rd_seed = s
        self.rng = np.random.default_rng(s)

    def load_problem(self, problem_file, routes, ruin_nodes):
        dir_name = os.path.dirname(f"{self.data_dir}/cvrp_benchmarks/homberger_{self.instance}_customer_instances/")
        if problem_file is None:
            problem_list = sorted(os.listdir(dir_name))
            problem_file = self.rng.choice(problem_list)
        self.problem_name = str.lower(os.path.splitext(os.path.basename(problem_file))[0])
        self.problem_file = f"{dir_name}/{problem_file}"
        print("loading problem: ", self.problem_name, self.rd_seed)
        if self.instance != 'ortec':
            self.problem = tools.read_solomon(self.problem_file)
        else:
            self.problem = tools.read_vrplib(self.problem_file)
        self.all_customers = list(range(len(self.problem["demands"])))
        duration_matrix = self.problem["duration_matrix"]
        spectral = Spectral(n_components=node_embedding_dim)
        self.node_embeddings = {}
        node_embeddings_array = spectral.fit_transform(duration_matrix)
        for i, c in enumerate(self.all_customers): self.node_embeddings[c] = node_embeddings_array[i, :]
        if routes is None:
            solution_file_name = f"{self.data_dir}/cvrp_benchmarks/RL_train_data/{self.problem_name}.npy"
            tmp_file_name =  f'tmp/{self.problem_name}'
            if os.path.exists(solution_file_name): routes = np.load(solution_file_name, allow_pickle=True)
            else:
                if os.path.exists(tmp_file_name):
                    os.rmdir(tmp_file_name)
                os.makedirs(tmp_file_name, exist_ok=True)
                routes, _ = construct_solution_from_ge_solver(self.problem, seed=self.rd_seed, tmp_dir=tmp_file_name, time_limit=240)
                np.save(solution_file_name, np.array(routes))
        if ruin_nodes is None:
            while True:
                route_idx = self.rng.integers(len(routes))
                node_idx = self.rng.integers(len(routes[route_idx]))
                node = routes[route_idx][node_idx]
                nb_customers = len(self.all_customers) - 1
                dist_to_node = sorted([(c, duration_matrix[node][c]+duration_matrix[c][node]) for c in range(1, nb_customers+1)], key=lambda x: x[1])
                ruin_nodes = [dist_to_node[i][0] for i in range(SubVRPTW_Environment.num_nodes_to_sample)]
                self.sub_problem = get_sub_instance(ruin_nodes, routes, self.problem)
                if 0 < len(self.sub_problem["ori_routes"]) <= max_num_route: break
        else: self.sub_problem = get_sub_instance(ruin_nodes, routes, self.problem)
        self.sub_routes = [[] for _ in range(len(self.sub_problem["ori_routes"]))]
        self.order_to_dispatch = []
        for route in self.sub_problem["ori_routes"]: self.order_to_dispatch.extend(route)
        depots = self.sub_problem["ori_starts"] + self.sub_problem["ori_stops"]
        self.reward_norm = np.max([self.problem["duration_matrix"][c1, c2] for c1 in depots+self.order_to_dispatch for c2 in depots+self.order_to_dispatch])
        self.init_total_cost = compute_route_cost(self.sub_problem["ori_routes"], self.problem, self.sub_problem)
        self.ori_full_routes = routes

    def get_route_cost(self):
        if len(self.order_to_dispatch) > 0:
            self.sub_routes = self.sub_problem["ori_routes"]
        return compute_route_cost(self.sub_routes, self.problem, self.sub_problem)

    def reset(self, problem_file=None, routes=None, ruin_nodes=None):
        # self.load_problem("ORTEC-VRPTW-ASYM-50d1f78d-d1-n329-k19.txt", routes, cur_route)
        self.load_problem(problem_file, routes, ruin_nodes)
        self.cur_step = 0
        self.state = self.get_state()
        if self.save_data: self.local_experience_buffer = [np.copy(self.state)]
        return self.state
    
    def get_route_state(self, route_idx, route):
        route_state = np.zeros(max_num_nodes_per_route*feature_dim)
        start, end = self.sub_problem["ori_starts"][route_idx], self.sub_problem["ori_stops"][route_idx]
        for i, c in enumerate([start]+route+[end]): 
            if i + 1 >= max_num_nodes_per_route: break
            route_state[i*feature_dim:(i+1)*feature_dim] = \
                extract_features_for_nodes(c, True, self.problem, self.sub_problem, self.node_embeddings)
        return route_state
            
    def get_state(self):
        cost_vec = np.zeros(max_num_route+1)
        if len(self.order_to_dispatch) > 0:
            cur_order = self.order_to_dispatch[0]
            for route_idx, route in enumerate(self.sub_routes):
                min_pos, min_cost = greedy_insertion(route, route_idx, cur_order, self.problem, self.sub_problem)
                if min_pos is not None: cost_vec[route_idx] = 1.0+(self.reward_norm - min_cost) / self.reward_norm
        if self.cur_step < self._max_episode_steps - max_num_nodes: cost_vec[-1] = 1.0
        if np.sum(cost_vec) == 0.0:  cost_vec[-1] = 1.0
        cur_routes_encode_state = np.zeros((max_num_route, max_num_nodes_per_route*feature_dim))
        for i, route in enumerate(self.sub_routes):
            if i >= max_num_route: break
            cur_routes_encode_state[i, :] = self.get_route_state(i, route)
        route_state = cur_routes_encode_state.reshape(-1)
        state = np.concatenate((cost_vec, route_state), axis=0)
        orders_to_dispatch_state = np.zeros((max_num_nodes,feature_dim))
        for i, order in enumerate(self.order_to_dispatch):
            if i >= max_num_nodes: break
            orders_to_dispatch_state[i, :] = extract_features_for_nodes(order, False, self.problem, self.sub_problem, self.node_embeddings)
        orders_to_dispatch_state = orders_to_dispatch_state.reshape(-1)
        state = np.concatenate((state, orders_to_dispatch_state), axis=0)
        return state

    def step(self, action):
        route_idx = int(action)
        self.cur_step += 1
        if route_idx == max_num_route:
            if len(self.order_to_dispatch) > 0: self.order_to_dispatch = self.order_to_dispatch[1:] + [self.order_to_dispatch[0]]
            self.reward = 0.0
        else:
            cur_order = self.order_to_dispatch[0]
            self.order_to_dispatch = self.order_to_dispatch[1:]
            route = self.sub_routes[route_idx]
            min_pos, min_cost = greedy_insertion(route, route_idx, cur_order, self.problem, self.sub_problem)
            self.reward = (self.reward_norm-min_cost)/self.reward_norm
            self.sub_routes[route_idx] = route[:min_pos] + [cur_order] + route[min_pos:]
        self.state = self.get_state()
        self.done = ((self.cur_step >= self._max_episode_steps) | (len(self.order_to_dispatch) <= 0))
        # if self.done and len(self.order_to_dispatch) > 0: self.reward = -10*len(self.order_to_dispatch)
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
    

import argparse
import os
parser = argparse.ArgumentParser(description='Input of VRPTW Trainer')
if __name__ == "__main__":
    parser.add_argument('--instance', type=str, default="ortec")
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--mp", type=int, default=1)
    args = parser.parse_args()
    pass