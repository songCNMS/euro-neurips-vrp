from agents.Base_Agent import Base_Agent
from utilities.OU_Noise import OU_Noise
from utilities.data_structures.Replay_Buffer import Replay_Buffer
from torch.optim import Adam
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from cvrptw_utility import max_num_nodes_per_route
import numpy as np
import os

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
TRAINING_EPISODES_PER_EVAL_EPISODE = 10
EPSILON = 1e-6

class SAC(Base_Agent):
    """Soft Actor-Critic model based on the 2018 paper https://arxiv.org/abs/1812.05905 and on this github implementation
      https://github.com/pranz24/pytorch-soft-actor-critic. It is an actor-critic algorithm where the agent is also trained
      to maximise the entropy of their actions as well as their cumulative reward"""
    agent_name = "SAC"
    def __init__(self, config):
        Base_Agent.__init__(self, config)
        assert self.action_types == "CONTINUOUS", "Action types must be continuous. Use SAC Discrete instead for discrete actions"
        assert self.config.hyperparameters["Actor"]["final_layer_activation"] != "Softmax", "Final actor layer must not be softmax"
        self.hyperparameters = config.hyperparameters
        self.critic_local = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1, key_to_use="Critic")
        self.critic_local_2 = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                           key_to_use="Critic", override_seed=self.config.seed + 1)
        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(),
                                                 lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
        self.critic_optimizer_2 = torch.optim.Adam(self.critic_local_2.parameters(),
                                                   lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
        self.critic_target = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                           key_to_use="Critic")
        self.critic_target_2 = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                            key_to_use="Critic")
        Base_Agent.copy_model_over(self.critic_local, self.critic_target)
        Base_Agent.copy_model_over(self.critic_local_2, self.critic_target_2)
        self.memory = Replay_Buffer(self.hyperparameters["Critic"]["buffer_size"], self.hyperparameters["batch_size"],
                                    self.config.seed, device=config.device)
        self.actor_local = self.create_NN(input_dim=self.state_size, output_dim=self.action_size * 2, key_to_use="Actor")
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(),
                                          lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-4)
        self.automatic_entropy_tuning = self.hyperparameters["automatically_tune_entropy_hyperparameter"]
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(self.environment.action_space.shape).to(self.device)).item() # heuristic value from the paper
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-4)
        else:
            self.alpha = self.hyperparameters["entropy_term_weight"]

        self.add_extra_noise = self.hyperparameters["add_extra_noise"]
        if self.add_extra_noise:
            self.noise = OU_Noise(self.action_size, self.config.seed, self.hyperparameters["mu"],
                                  self.hyperparameters["theta"], self.hyperparameters["sigma"])

        self.do_evaluation_iterations = self.hyperparameters["do_evaluation_iterations"]

        self.eval_reward_list = []

    def save_result(self):
        """Saves the result of an episode of the game. Overriding the method in Base Agent that does this because we only
        want to keep track of the results during the evaluation episodes"""
        if self.episode_number == 1 or not self.do_evaluation_iterations:
            self.game_full_episode_scores.extend([self.total_episode_score_so_far])
            self.rolling_results.append(np.mean(self.game_full_episode_scores[-1 * self.rolling_score_window:]))
            self.save_max_result_seen()

        # elif self.episode_number % TRAINING_EPISODES_PER_EVAL_EPISODE == 0:
        else:
            # self.game_full_episode_scores.extend([self.total_episode_score_so_far for _ in range(TRAINING_EPISODES_PER_EVAL_EPISODE)])
            self.game_full_episode_scores.append(self.total_episode_score_so_far)
            self.rolling_results.extend([np.mean(self.game_full_episode_scores[-1 * self.rolling_score_window:]) for _ in range(TRAINING_EPISODES_PER_EVAL_EPISODE)])
            self.save_max_result_seen()

    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        Base_Agent.reset_game(self)
        if self.add_extra_noise: self.noise.reset()

    def step(self):
        """Runs an episode on the game, saving the experience and running a learning step if appropriate"""
        eval_ep = (self.episode_number+1) % TRAINING_EPISODES_PER_EVAL_EPISODE == 0 and self.do_evaluation_iterations
        self.episode_step_number_val = 0
        _done = False
        while not _done:
            self.episode_step_number_val += 1
            if self.is_vec_env: self.action = np.array([self.pick_action(eval_ep, state=state) for i, state in enumerate(self.state)])
            else: self.action = self.pick_action(eval_ep)
            self.conduct_action(self.action)
            # if eval_ep:
            #     print("state: ", self.environment.state)
            #     print(f"step: {self.environment.local_env.cur_step}, action: {self.action}, reward: {self.environment.reward}, rep: {self.environment.local_env.cur_replenish_amount}, instock: {self.environment.local_env.in_stocks}, sales: {self.environment.local_env.cur_sales}, demand: {self.environment.local_env.cur_demand}")
            if self.time_for_critic_and_actor_to_learn():
                for _ in range(self.hyperparameters["learning_updates_per_learning_session"]):
                    self.learn()
            if not self.is_vec_env:
                mask = False if self.episode_step_number_val >= self.environment._max_episode_steps else self.done
                if not eval_ep: self.save_experience(experience=(self.state, self.action, self.reward, self.next_state, mask))
            else:
                for i in range(self.environment.num_envs):
                    mask = False if self.episode_step_number_val >= self.environment.envs[i].max_episode_steps else self.done[i]
                    if not eval_ep: self.save_experience(experience=(self.state[i], self.action[i], self.reward[i], self.next_state[i], mask))
            self.state = self.next_state
            self.global_step_number += 1
            _done = (np.any(self.done) if self.is_vec_env else self.done)
        print(self.total_episode_score_so_far)
        if eval_ep: 
            self.print_summary_of_latest_evaluation_episode()
            # self.environment.switch_mode("train")
        self.episode_number += 1

    def pick_action(self, eval_ep, state=None):
        """Picks an action using one of three methods: 1) Randomly if we haven't passed a certain number of steps,
         2) Using the actor in evaluation mode if eval_ep is True  3) Using the actor in training mode if eval_ep is False.
         The difference between evaluation and training mode is that training mode does more exploration"""
        if state is None: state = self.state
        if eval_ep: action = self.actor_pick_action(state=state, eval=True)
        elif self.global_step_number < self.hyperparameters["min_steps_before_learning"]:
            num_routes = int(state[0])
            cost_improve_vec = np.array(state[1:1+max_num_nodes_per_route])[:num_routes]
            cost_sum = np.sum(cost_improve_vec)
            if cost_sum > 0.0: node_prob = [c/cost_sum for c in cost_improve_vec]
            else: node_prob = None
            action = np.random.choice(range(num_routes), p=node_prob)
            # action = self.environment.action_space.sample()
            print("Picking random action ", action)
        else: action = self.actor_pick_action(state=state)
        if self.add_extra_noise:
            action += self.noise.sample()
        return action

    def actor_pick_action(self, state=None, eval=False):
        """Uses actor to pick an action in one of two ways: 1) If eval = False and we aren't in eval mode then it picks
        an action that has partly been randomly sampled 2) If eval = True then we pick the action that comes directly
        from the network and so did not involve any random sampling"""
        if state is None: state = self.state
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        if len(state.shape) == 1: state = state.unsqueeze(0)
        if eval == False: action, _, _ = self.produce_action_and_action_info(state)
        else:
            with torch.no_grad():
                _, z, action = self.produce_action_and_action_info(state)
        action = action.detach().cpu().numpy()
        return action[0]

    def produce_action_and_action_info(self, state):
        """Given the state, produces an action, the log probability of the action, and the tanh of the mean action"""
        actor_output = self.actor_local(state)
        mean, log_std = actor_output[:, :self.action_size], actor_output[:, self.action_size:]
        mean = torch.clamp(mean, -2.0, 2.0)
        log_std = torch.clamp(log_std, -5.0, 2.0)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  #rsample means it is sampled using reparameterisation trick
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + EPSILON)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, torch.tanh(mean)

    def time_for_critic_and_actor_to_learn(self):
        """Returns boolean indicating whether there are enough experiences to learn from and it is time to learn for the
        actor and critic"""
        return self.global_step_number > self.hyperparameters["min_steps_before_learning"] and \
               self.enough_experiences_to_learn_from() and self.global_step_number % self.hyperparameters["update_every_n_steps"] == 0

    def learn(self):
        """Runs a learning iteration for the actor, both critics and (if specified) the temperature parameter"""
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.sample_experiences()
        qf1_loss, qf2_loss = self.calculate_critic_losses(state_batch, action_batch, reward_batch, next_state_batch, mask_batch)
        self.update_critic_parameters(qf1_loss, qf2_loss)

        policy_loss, log_pi = self.calculate_actor_loss(state_batch)
        if self.automatic_entropy_tuning: alpha_loss = self.calculate_entropy_tuning_loss(log_pi)
        else: alpha_loss = None
        self.update_actor_parameters(policy_loss, alpha_loss)

    def sample_experiences(self):
        return  self.memory.sample()

    def calculate_critic_values(self, state_batch, action_batch):
        qf1 = self.critic_local(torch.cat((state_batch, action_batch), 1))
        qf2 = self.critic_local_2(torch.cat((state_batch, action_batch), 1))
        min_qf_pi = torch.min(qf1, qf2)
        return min_qf_pi

    def calculate_critic_losses(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch):
        """Calculates the losses for the two critics. This is the ordinary Q-learning loss except the additional entropy
         term is taken into account"""
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.produce_action_and_action_info(next_state_batch)
            qf1_next_target = self.critic_target(torch.cat((next_state_batch, next_state_action), 1))
            qf2_next_target = self.critic_target_2(torch.cat((next_state_batch, next_state_action), 1))
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + (1.0 - mask_batch) * self.hyperparameters["discount_rate"] * (min_qf_next_target)
        qf1 = self.critic_local(torch.cat((state_batch, action_batch), 1))
        qf2 = self.critic_local_2(torch.cat((state_batch, action_batch), 1))
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        return qf1_loss, qf2_loss

    def calculate_actor_loss(self, state_batch):
        """Calculates the loss for the actor. This loss includes the additional entropy term"""
        action, log_pi, _ = self.produce_action_and_action_info(state_batch)
        qf1_pi = self.critic_local(torch.cat((state_batch, action), 1))
        qf2_pi = self.critic_local_2(torch.cat((state_batch, action), 1))
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        return policy_loss, log_pi

    def calculate_entropy_tuning_loss(self, log_pi):
        """Calculates the loss for the entropy temperature parameter. This is only relevant if self.automatic_entropy_tuning
        is True."""
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        return alpha_loss

    def update_critic_parameters(self, critic_loss_1, critic_loss_2):
        """Updates the parameters for both critics"""
        self.take_optimisation_step(self.critic_optimizer, self.critic_local, critic_loss_1,
                                    self.hyperparameters["Critic"]["gradient_clipping_norm"])
        self.take_optimisation_step(self.critic_optimizer_2, self.critic_local_2, critic_loss_2,
                                    self.hyperparameters["Critic"]["gradient_clipping_norm"])
        self.soft_update_of_target_network(self.critic_local, self.critic_target,
                                           self.hyperparameters["Critic"]["tau"])
        self.soft_update_of_target_network(self.critic_local_2, self.critic_target_2,
                                           self.hyperparameters["Critic"]["tau"])

    def update_actor_parameters(self, actor_loss, alpha_loss):
        """Updates the parameters for the actor and (if specified) the temperature parameter"""
        self.take_optimisation_step(self.actor_optimizer, self.actor_local, actor_loss,
                                    self.hyperparameters["Actor"]["gradient_clipping_norm"])
        if alpha_loss is not None:
            self.take_optimisation_step(self.alpha_optim, None, alpha_loss, None)
            self.alpha = self.log_alpha.exp()

    def eval(self):
        n_agents = self.eval_environment.n_agents
        self.eval_environment.switch_mode("eval")
        eval_rounds = 5
        total_reward = 0.0
        for _ in range(eval_rounds):
            for i in range(1, n_agents+1):
                state = self.eval_environment.reset(local_agent_idx=i, relative_start_day=self.eval_environment.episode_duration_training)
                done = False
                while not done:
                    action = self.actor_pick_action(state=state, eval=True)
                    state, reward, done, _ = self.eval_environment.step(action)
                    total_reward += reward
                self.eval_environment.local_env.tracker.render_sku(self.config.log_path, [self.eval_environment.sku_name])
        self.eval_environment.switch_mode("train")        
        return total_reward / eval_rounds

    def joint_eval(self):
        self.eval_environment.switch_mode("eval")
        eval_rounds = 1
        total_reward = 0.0
        for _ in range(eval_rounds):
            states = self.eval_environment.joint_reset(relative_start_day=self.eval_environment.episode_duration_training)
            done = False
            while not done:
                actions = np.array([self.actor_pick_action(state=state, eval=True)[0] for state in states])
                states, rewards, dones, _ = self.eval_environment.joint_step(actions)
                total_reward += np.sum(rewards)
                done = dones[0]
            self.eval_environment.joint_env.tracker.render_sku(self.config.log_path, self.eval_environment.joint_env.sku_names)
        self.eval_environment.switch_mode("train")        
        return total_reward / eval_rounds

    def generate_C_trajectory(self):
        states = self.environment.joint_reset(relative_start_day=0, sampler_seq_len=self.environment.episode_duration_training)
        done = False
        self.environment.stock_trajectory[:, 0, 0] = self.environment.joint_env.in_stocks_end_day[:]
        self.environment.stock_trajectory[:, 1, 0] = self.environment.joint_env.inventory_to_load[:]
        self.environment.stock_trajectory[:, 2, 0] = self.environment.joint_env.in_stocks[:]
        i = 0
        while not done:
            i += 1
            actions = np.array([self.actor_pick_action(state=state, eval=True)[0] for state in states])
            states, _, dones, _ = self.environment.joint_step(actions)
            done = dones[0]
            self.environment.stock_trajectory[:, 0, i] = self.environment.joint_env.in_stocks_end_day[:]
            self.environment.stock_trajectory[:, 1, i] = self.environment.joint_env.inventory_to_load[:]
            self.environment.stock_trajectory[:, 2, i] = self.environment.joint_env.in_stocks[:]

    def print_summary_of_latest_evaluation_episode(self):
        """Prints a summary of the latest episode"""
        print(" ")
        print("----------------------------")
        print("Episode score {} ".format(self.total_episode_score_so_far))
        # self.environment.local_env.tracker.render_sku(self.config.log_path, self.environment.sku_name)
        if self.config.joint_training: total_reward = self.joint_eval()
        else: total_reward = self.eval()
        self.eval_reward_list.append(total_reward)
        print("Eval reward {}, best reward {} ".format(total_reward, np.max(self.eval_reward_list)))
        print("History rewards: ", self.eval_reward_list)
        print("----------------------------")
        self.locally_save_policy(self.episode_number)
        if total_reward == np.max(self.eval_reward_list):
            self.locally_save_policy("best")
        if self.episode_number > self.config.generate_trajectory_warmup_rounds:
            print("generating stock trajectory")
            self.generate_C_trajectory()
        

    def locally_save_policy(self, ep):
        loc_path = f"{self.config.log_path}/SAC_checkpoints_{ep}/"
        os.makedirs(loc_path, exist_ok=True)
        torch.save(self.critic_local.state_dict(), f"{loc_path}/critic.pt")
        torch.save(self.critic_local_2.state_dict(), f"{loc_path}/critic2.pt")
        torch.save(self.actor_local.state_dict(), f"{loc_path}/actor.pt")
        torch.save(self.critic_optimizer.state_dict(), f"{loc_path}/critic_otorch.pt")
        torch.save(self.critic_optimizer_2.state_dict(), f"{loc_path}/critic2_otorch.pt")
        torch.save(self.actor_optimizer.state_dict(), f"{loc_path}/actor_otorch.pt")

    def load_policy(self, ep):
        loc_path = f"{self.config.log_path}/SAC_checkpoints_{ep}/"
        self.critic_local.load_state_dict(torch.load(f"{loc_path}/critic.pt", map_location=self.device))
        self.critic_local_2.load_state_dict(torch.load(f"{loc_path}/critic2.pt", map_location=self.device))
        self.actor_local.load_state_dict(torch.load(f"{loc_path}/actor.pt", map_location=self.device))
        self.critic_optimizer.load_state_dict(torch.load(f"{loc_path}/critic_otorch.pt", map_location=self.device))
        self.critic_optimizer_2.load_state_dict(torch.load(f"{loc_path}/critic2_otorch.pt", map_location=self.device))
        self.actor_optimizer.load_state_dict(torch.load(f"{loc_path}/actor_otorch.pt", map_location=self.device))
        Base_Agent.copy_model_over(self.critic_local, self.critic_target)
        Base_Agent.copy_model_over(self.critic_local_2, self.critic_target_2)
