import sys
sys.path.append("/data/songlei/replenishment-env/replenishment")
sys.path.append("/data/songlei/replenishment-env/replenishment/drl")
from agents.policy_gradient_agents.PPO import PPO
from agents.actor_critic_agents.DDPG import DDPG
from agents.actor_critic_agents.SAC import SAC
from agents.actor_critic_agents.TD3 import TD3
from agents.Trainer import Trainer
from agents.hierarchical_agents.DIAYN import DIAYN
from utilities.data_structures.Config import Config

from environments.Inventory_Environment import Inventory_Environment
import os
import random
import argparse
parser = argparse.ArgumentParser(description='Input of Inventory Trainer')
parser.add_argument('--capacity', metavar='N', type=int, help='storage capacity')
parser.add_argument('--cuda', metavar='C', type=int, help='CUDA Device ID')
parser.add_argument('--joint', action='store_true')
parser.add_argument('--warmup', type=int, help='warm up rounds before training')

args = parser.parse_args()
config = Config()
config.seed = random.randint(0, 100)
config.joint_training = args.joint
config.environment = Inventory_Environment(50, args.capacity, 100, joint_training=config.joint_training) # n_agents, max_capacity, sampler_seq_len
jt = ("jt" if config.joint_training else "lt")
config.log_path = f'/data/songlei/replenishment-env/replenishment/drl/results/data_and_graphs/inventory_{jt}_{args.capacity}/'
config.file_to_save_data_results = f"{config.log_path}/Inventory.pkl"
config.file_to_save_results_graph = f"{config.log_path}/Inventory.png"
config.use_GPU = False
if config.use_GPU: config.device = f"cuda:{args.cuda}"
else: config.device = "cpu"
config.num_episodes_to_run = 10000
config.show_solution_score = False
config.visualise_individual_results = False
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
config.runs_per_agent = 3
config.overwrite_existing_results_file = False
config.randomise_random_seed = True
config.save_model = False
config.generate_trajectory_warmup_rounds = args.warmup


actor_critic_agent_hyperparameters = {
        "Actor": {
            "learning_rate": 0.0005,
            "linear_hidden_units": [512, 256, 128],
            "hidden_activations": "tanh",
            "final_layer_activation": None,
            "batch_norm": False,
            "tau": 0.01,
            "gradient_clipping_norm": 1.0,
            "initialiser": "xavier_uniform"
        },

        "Critic": {
            "learning_rate": 0.001,
            "linear_hidden_units": [512, 256, 128],
            "hidden_activations": "tanh",
            "final_layer_activation": None,
            "batch_norm": False,
            "buffer_size": 1000000,
            "tau": 0.01,
            "gradient_clipping_norm": 1.0,
            "initialiser": "xavier_uniform"
        },

        "min_steps_before_learning": 10000,
        "batch_size": 512,
        "discount_rate": 0.99,
        "mu": 0.0, #for O-H noise
        "theta": 0.15, #for O-H noise
        "sigma": 0.25, #for O-H noise
        "action_noise_std": 0.2,  # for TD3
        "action_noise_clipping_range": 0.5,  # for TD3
        "update_every_n_steps": 4,
        "learning_updates_per_learning_session": 128,
        "automatically_tune_entropy_hyperparameter": True,
        "entropy_term_weight": None,
        "add_extra_noise": False,
        "do_evaluation_iterations": True,
        "clip_rewards": False
    }

dqn_agent_hyperparameters =   {
        "learning_rate": 0.001,
        "batch_size": 256,
        "buffer_size": 40000,
        "epsilon": 1.0,
        "epsilon_decay_rate_denominator": 3,
        "discount_rate": 0.99,
        "tau": 0.01,
        "alpha_prioritised_replay": 0.6,
        "beta_prioritised_replay": 0.1,
        "incremental_td_error": 1e-8,
        "update_every_n_steps": 3,
        "linear_hidden_units": [256, 256, 256],
        "final_layer_activation": "None",
        "batch_norm": False,
        "gradient_clipping_norm": 5,
        "clip_rewards": False
    }


manager_hyperparameters = dqn_agent_hyperparameters
manager_hyperparameters.update({"timesteps_to_give_up_control_for": 5})


config.hyperparameters = {
    "Policy_Gradient_Agents": {
            "learning_rate": 0.001,
            "linear_hidden_units": [256, 256, 256],
            "final_layer_activation": "TANH",
            "learning_iterations_per_round": 10,
            "discount_rate": 0.99,
            "batch_norm": False,
            "clip_epsilon": 0.2,
            "episodes_per_learning_round": 10,
            "normalise_rewards": True,
            "gradient_clipping_norm": 5,
            "mu": 0.0,
            "theta": 0.15,
            "sigma": 0.2,
            "epsilon_decay_rate_denominator": 1,
            "clip_rewards": False
        },

    "Actor_Critic_Agents": actor_critic_agent_hyperparameters,
    "DIAYN": {
        "DISCRIMINATOR": {
            "learning_rate": 0.001,
            "linear_hidden_units": [32, 32],
            "final_layer_activation": None,
            "gradient_clipping_norm": 5

        },
        "AGENT": actor_critic_agent_hyperparameters,
        "MANAGER": manager_hyperparameters,
        "num_skills": 10,
        "num_unsupservised_episodes": 500
    }
}

if __name__ == "__main__":
    os.makedirs(config.log_path, exist_ok=True)

    AGENTS = [SAC] #SAC] #, DDPG, PPO, TD3] ] #,
    trainer = Trainer(config, AGENTS)
    trainer.run_games_for_agents()






