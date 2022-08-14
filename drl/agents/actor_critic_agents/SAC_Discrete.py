from cvrptw_utility import MLP_RL_Model
from drl.environments.VRPTW_Environment import VRPTW_Environment
from drl.environments.VRPTW_Route_Environment import VRPTW_Route_Environment
import torch
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
from agents.Base_Agent import Base_Agent
from utilities.data_structures.Replay_Buffer import Replay_Buffer
from agents.actor_critic_agents.SAC import SAC
from utilities.Utility_Functions import create_actor_distribution
from cvrptw_utility import MLP_RL_Model, MLP_Route_RL_Model
import wandb
import pandas as pd
import os
import random
torch.autograd.set_detect_anomaly(False)


class SAC_Discrete(SAC):
    """The Soft Actor Critic for discrete actions. It inherits from SAC for continuous actions and only changes a few
    methods."""
    agent_name = "SAC"
    def __init__(self, config):
        Base_Agent.__init__(self, config)
        assert self.action_types == "DISCRETE", "Action types must be discrete. Use SAC instead for continuous actions"
        assert self.config.hyperparameters["Actor"]["final_layer_activation"] == "Softmax", "Final actor layer must be softmax"
        self.hyperparameters = config.hyperparameters
        self.critic_local = self.create_NN(input_dim=self.state_size, output_dim=self.action_size, key_to_use="Critic")
        self.critic_local_2 = self.create_NN(input_dim=self.state_size, output_dim=self.action_size,
                                           key_to_use="Critic", override_seed=self.config.seed + 1)
        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(),
                                                 lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
        self.critic_optimizer_2 = torch.optim.Adam(self.critic_local_2.parameters(),
                                                   lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
        self.critic_target = self.create_NN(input_dim=self.state_size, output_dim=self.action_size,
                                            key_to_use="Critic")
        self.critic_target_2 = self.create_NN(input_dim=self.state_size, output_dim=self.action_size,
                                            key_to_use="Critic")
        Base_Agent.copy_model_over(self.critic_local, self.critic_target)
        Base_Agent.copy_model_over(self.critic_local_2, self.critic_target_2)
        self.memory = Replay_Buffer(self.hyperparameters["Critic"]["buffer_size"], self.hyperparameters["batch_size"],
                                    self.config.seed, device=self.device)

        self.actor_local = self.create_NN(input_dim=self.state_size, output_dim=self.action_size, key_to_use="Actor")
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(),
                                          lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-4)
        self.automatic_entropy_tuning = self.hyperparameters["automatically_tune_entropy_hyperparameter"]
        if self.automatic_entropy_tuning:
            # we set the max possible entropy as the target entropy
            self.target_entropy = -np.log((1.0 / self.action_size)) * 0.98
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-4)
        else:
            self.alpha = self.hyperparameters["entropy_term_weight"]
        assert not self.hyperparameters["add_extra_noise"], "There is no add extra noise option for the discrete version of SAC at moment"
        self.add_extra_noise = False
        self.do_evaluation_iterations = self.hyperparameters["do_evaluation_iterations"]
        self.eval_reward_list = []
        self.greedy_exploration = self.hyperparameters["greedy_exploration"]
        self.start_exploration_rate = self.hyperparameters["start_exploration_rate"]
        self.end_exploration_rate = self.hyperparameters["end_exploration_rate"]
        self.cost_reduction = self.config.cost_reduction

    def produce_action_and_action_info(self, state):
        """Given the state, produces an action, the probability of the action, the log probability of the action, and
        the argmax action"""
        action_probabilities = self.actor_local(state)
        # greedy_epsilon = self.start_exploration_rate - (self.episode_number*(self.end_exploration_rate - self.start_exploration_rate)) / self.config.num_episodes_to_run
        max_probability_action = torch.argmax(action_probabilities, dim=-1)
        # if random.random() >= greedy_epsilon:
        action_distribution = create_actor_distribution(self.action_types, action_probabilities, self.action_size)
        action = action_distribution.sample().cpu()
        # else:
        #     # action_shape = (state.size(0), )
        #     if len(state.shape) > 1: num_routes = state[:, 0].cpu().numpy()
        #     else: num_routes = np.array([state[0]])
        #     action = torch.from_numpy(np.random.randint(num_routes)).to(self.device)
            # action = torch.randint(0, self.action_size, action_shape)
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probabilities + z)
        return action, (action_probabilities, log_action_probabilities), max_probability_action

    def calculate_critic_losses(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch):
        """Calculates the losses for the two critics. This is the ordinary Q-learning loss except the additional entropy
         term is taken into account"""
        with torch.no_grad():
            next_state_action, (action_probabilities, log_action_probabilities), _ = self.produce_action_and_action_info(next_state_batch)
            qf1_next_target = self.critic_target(next_state_batch)
            qf2_next_target = self.critic_target_2(next_state_batch)
            min_qf_next_target = action_probabilities * (torch.min(qf1_next_target, qf2_next_target) - self.alpha * log_action_probabilities)
            min_qf_next_target = min_qf_next_target.sum(dim=1).unsqueeze(-1)
            next_q_value = reward_batch + (1.0 - mask_batch) * self.hyperparameters["discount_rate"] * (min_qf_next_target)
        qf1 = self.critic_local(state_batch).gather(1, action_batch.long())
        qf2 = self.critic_local_2(state_batch).gather(1, action_batch.long())
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        q_vals = qf1.cpu().detach().numpy()
        return qf1_loss, qf2_loss, q_vals

    def calculate_actor_loss(self, state_batch):
        """Calculates the loss for the actor. This loss includes the additional entropy term"""
        action, (action_probabilities, log_action_probabilities), _ = self.produce_action_and_action_info(state_batch)
        qf1_pi = self.critic_local(state_batch)
        qf2_pi = self.critic_local_2(state_batch)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        if self.cost_reduction: inside_term = self.alpha * log_action_probabilities + min_qf_pi
        else: inside_term = self.alpha * log_action_probabilities - min_qf_pi
        policy_loss = (action_probabilities * inside_term).sum(dim=1).mean()
        log_action_probabilities = torch.sum(log_action_probabilities * action_probabilities, dim=1)
        return policy_loss, log_action_probabilities

    def create_NN(self, input_dim, output_dim, key_to_use=None, override_seed=None, hyperparameters=None):
        """Creates a neural network for the agents to use"""
        if hyperparameters is None: hyperparameters = self.hyperparameters
        if key_to_use:
            hyperparameters = hyperparameters[key_to_use]
            hyperparameters['key_to_use'] = key_to_use
        if override_seed: hyperparameters["seed"] = override_seed
        else: hyperparameters["seed"] = self.config.seed
        default_hyperparameter_choices = {"output_activation": None, "hidden_activations": "relu", "dropout": 0.0,
                                          "initialiser": "default", "batch_norm": False,
                                          "columns_of_data_to_be_embedded": [],
                                          "embedding_dimensions": [], "y_range": (),
                                          "seed": 1}

        for key in default_hyperparameter_choices:
            if key not in hyperparameters.keys():
                hyperparameters[key] = default_hyperparameter_choices[key]
        hyperparameters["input_dim"] = input_dim
        hyperparameters["output_dim"] = output_dim
        hyperparameters["linear_route"] = self.config.linear_route
        # if isinstance(self.environment, VRPTW_Environment): return MLP_RL_Model(hyperparameters).to(self.device)
        # else: return MLP_Route_RL_Model(hyperparameters).to(self.device)
        return MLP_RL_Model(hyperparameters).to(self.device)
    
    def eval(self):
        self.eval_environment.switch_mode("eval")
        total_reward = 0.0
        # hybrid_res_df = pd.read_csv(f"./amlt/vrptw_hybrid_ges_only_240/vrptw_{self.eval_environment.instance}/hybrid_res.csv")
        hybrid_res_df = pd.read_csv(f"hybrid_res/hybrid_{self.eval_environment.instance}.csv")
        hybrid_res = {row['problem']: row['total_cost'] for _, row in hybrid_res_df.iterrows()}
        dir_name = os.path.dirname(f"{self.eval_environment.data_dir}/cvrp_benchmarks/homberger_{self.eval_environment.instance}_customer_instances/")
        problem_list = sorted(os.listdir(dir_name))
        # problem_list = [p for p in problem_list if int(p.split('-')[-2][1:]) ]
        # problem_list = ["ORTEC-VRPTW-ASYM-0bdff870-d1-n458-k35.txt"]
        eval_rounds = min(5, len(problem_list))
        init_cost, final_cost, hybrid_cost = 0.0, 0.0, 0.0
        problem_reward_list = []
        for i in range(eval_rounds):
            problem = problem_list[i]
            hybrid_cost += hybrid_res[problem]
            state = self.eval_environment.reset(problem_file=problem)
            init_cost += self.eval_environment.init_total_cost
            done = False
            _total_reward = 0.0
            while not done:
                action = self.actor_pick_action(state=state, eval=True)
                state, reward, done, _ = self.eval_environment.step(action)
                print(f"problem: {self.eval_environment.problem_name}, cur_step: {self.eval_environment.cur_step}, early_stop_round: {self.eval_environment.steps_not_improved}, action: {action}, reward: {reward} \n ")
                _total_reward = reward
            problem_reward_list.append(_total_reward)
            total_reward += _total_reward
            final_cost += self.eval_environment.get_route_cost()
        self.eval_environment.switch_mode("train")
        print("problem reward: ", problem_reward_list)
        return total_reward/eval_rounds, init_cost/eval_rounds, final_cost/eval_rounds, hybrid_cost/eval_rounds
    
    def print_summary_of_latest_evaluation_episode(self):
        """Prints a summary of the latest episode"""
        print(" ")
        print("----------------------------")
        print("Episode score {} ".format(self.total_episode_score_so_far))
        total_reward, init_cost, final_cost, hybrid_cost = self.eval()
        wandb.log({"cost_reduction": total_reward,
                   "init_cost": init_cost,
                   "final_cost": final_cost,
                   "hybrid_cost": hybrid_cost})
        self.eval_reward_list.append(total_reward)
        print("Eval reward {}, best reward {} ".format(total_reward, np.max(self.eval_reward_list)))
        print("History rewards: ", self.eval_reward_list)
        print("----------------------------")
        self.locally_save_policy(self.episode_number)
        if total_reward == np.max(self.eval_reward_list):
            self.locally_save_policy("best")