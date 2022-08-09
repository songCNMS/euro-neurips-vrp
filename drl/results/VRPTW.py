import sys
sys.path.append("./")
sys.path.append("./drl")
import multiprocessing as mp
from yaml import parse
from agents.actor_critic_agents.A2C import A2C
from agents.DQN_agents.Dueling_DDQN import Dueling_DDQN
from agents.actor_critic_agents.SAC_Discrete import SAC_Discrete
from agents.actor_critic_agents.A3C import A3C
from agents.policy_gradient_agents.PPO import PPO
from agents.Trainer import Trainer
from agents.DQN_agents.DDQN import DDQN
from agents.DQN_agents.DDQN_With_Prioritised_Experience_Replay import DDQN_With_Prioritised_Experience_Replay
from agents.DQN_agents.DQN import DQN
from agents.DQN_agents.DQN_With_Fixed_Q_Targets import DQN_With_Fixed_Q_Targets
from utilities.data_structures.Config import Config
from environments.VRPTW_Environment import VRPTW_Environment
from cvrptw_utility import device
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import random
import argparse
import wandb
import os

from datetime import datetime
import platform


if __name__ == "__main__":
    mp.set_start_method("spawn")
    os.environ["WANDB_API_KEY"] = "116a4f287fd4fbaa6f790a50d2dd7f97ceae4a03"
    wandb.login()
    parser = argparse.ArgumentParser(description='Input of VRPTW Trainer')
    parser.add_argument('--cuda', metavar='C', type=int, help='CUDA Device ID')
    parser.add_argument('--warmup', type=int, help='warm up rounds before training')
    parser.add_argument('--instance', type=str, default="ortec")
    parser.add_argument('--exp_name', type=str, default="exp")
    parser.add_argument("--remote", action="store_true")
    args = parser.parse_args()

    config = Config()
    config.seed = random.randint(0, 100)
    if args.remote: 
        config.data_dir = os.getenv("AMLT_DATA_DIR", "cvrp_benchmarks/")
        config.output_dir = os.environ['AMLT_OUTPUT_DIR']
    else:
        config.data_dir = "./"
        config.output_dir = "./logs/"
    
    # wrappable_env = VRPTW_Environment(args.instance, config.data_dir, seed=random.randint(0, 100))
    N_ENVS = 8
    vec_env = make_vec_env(
        lambda: VRPTW_Environment(args.instance, config.data_dir, seed=random.randint(0, 100)),
        n_envs=N_ENVS,
        vec_env_cls=DummyVecEnv
    )
    config.environment = vec_env
    # config.environment = VRPTW_Environment(args.instance, config.data_dir, seed=config.seed)
    config.eval_environment = VRPTW_Environment(args.instance, config.data_dir, seed=config.seed)
    config.log_path = config.output_dir
    config.file_to_save_data_results = f"{config.log_path}/VRPTW.pkl"
    config.file_to_save_results_graph = f"{config.log_path}/VRPTW.png"
    config.use_GPU = True
    if config.use_GPU: config.device = device
    else: config.device = "cpu"
    config.num_episodes_to_run = 10000
    config.show_solution_score = False
    config.visualise_individual_results = False
    config.visualise_overall_agent_results = True
    config.standard_deviation_results = 1.0
    config.runs_per_agent = 1
    config.overwrite_existing_results_file = False
    config.randomise_random_seed = True
    config.save_model = False
    config.generate_trajectory_warmup_rounds = args.warmup
    config.debug_mode = False
    config.linear_route = False
    config.is_vec_env = True


    config.hyperparameters = {
        "DQN_Agents": {
            "learning_rate": 0.01,
            "batch_size": 256,
            "buffer_size": 40000,
            "epsilon": 1.0,
            "epsilon_decay_rate_denominator": 1,
            "discount_rate": 0.99,
            "tau": 0.01,
            "alpha_prioritised_replay": 0.6,
            "beta_prioritised_replay": 0.1,
            "incremental_td_error": 1e-8,
            "update_every_n_steps": 1,
            "linear_hidden_units": [30, 15],
            "final_layer_activation": "None",
            "batch_norm": False,
            "gradient_clipping_norm": 0.7,
            "learning_iterations": 1,
            "clip_rewards": False
        },
        "Stochastic_Policy_Search_Agents": {
            "policy_network_type": "Linear",
            "noise_scale_start": 1e-2,
            "noise_scale_min": 1e-3,
            "noise_scale_max": 2.0,
            "noise_scale_growth_factor": 2.0,
            "stochastic_action_decision": False,
            "num_policies": 10,
            "episodes_per_policy": 1,
            "num_policies_to_keep": 5,
            "clip_rewards": False
        },
        "Policy_Gradient_Agents": {
            "learning_rate": 0.05,
            "linear_hidden_units": [20, 20],
            "final_layer_activation": "SOFTMAX",
            "learning_iterations_per_round": 5,
            "discount_rate": 0.99,
            "batch_norm": False,
            "clip_epsilon": 0.1,
            "episodes_per_learning_round": 4,
            "normalise_rewards": True,
            "gradient_clipping_norm": 7.0,
            "mu": 0.0, #only required for continuous action games
            "theta": 0.0, #only required for continuous action games
            "sigma": 0.0, #only required for continuous action games
            "epsilon_decay_rate_denominator": 1.0,
            "clip_rewards": False,
            "output_activation": None, 
            "hidden_activations": "relu", 
            "dropout": 0.0,
            "initialiser": "Xavier", 
        },

        "Actor_Critic_Agents":  {
            "learning_rate": 0.0003,
            "linear_hidden_units": [128, 128],
            "gradient_clipping_norm": 1.0,
            "discount_rate": 1.0,
            "epsilon_decay_rate_denominator": 1.0,
            "normalise_rewards": True,
            "exploration_worker_difference": 2.0,
            "clip_rewards": False,

            "Actor": {
                "learning_rate": 0.0001,
                "linear_hidden_units": [128, 128],
                "hidden_activations": "tanh",
                "final_layer_activation": "Softmax",
                "batch_norm": False,
                "tau": 0.01,
                "gradient_clipping_norm": 5.0,
                "initialiser": "Xavier"
            },

            "Critic": {
                "learning_rate": 0.0003,
                "linear_hidden_units": [128, 128],
                "hidden_activations": "tanh",
                "final_layer_activation": None,
                "batch_norm": False,
                "buffer_size": 51200,
                "tau": 0.01,
                "gradient_clipping_norm": 5.0,
                "initialiser": "Xavier"
            },

            "min_steps_before_learning": 64,
            "batch_size": 512,
            "discount_rate": 0.99,
            "mu": 0.0, #for O-H noise
            "theta": 0.15, #for O-H noise
            "sigma": 0.25, #for O-H noise
            "action_noise_std": 0.2,  # for TD3
            "action_noise_clipping_range": 0.5,  # for TD3
            "update_every_n_steps": 8, # how frequency learn is run
            "learning_updates_per_learning_session": 1, # how many iterations per learn
            "automatically_tune_entropy_hyperparameter": False,
            "entropy_term_weight": 1.0,
            "add_extra_noise": False,
            "do_evaluation_iterations": True,
            "greedy_exploration": True,
            "start_exploration_rate": 1.0,
            "end_exploration_rate": 0.1
        }
    }

    # AGENTS = [SAC_Discrete, DDQN, Dueling_DDQN, DQN, DQN_With_Fixed_Q_Targets,
    #           DDQN_With_Prioritised_Experience_Replay, A2C, PPO, A3C ]
    AGENTS = [SAC_Discrete]
    exp_name = datetime.now().strftime("%m%d-%H%M")
    if not config.linear_route: exp_name += f'_GRU_{AGENTS[0].agent_name}_{args.exp_name}'
    wandb.init(dir=f"{config.output_dir}/", project="VRPTW_SAC", config=vars(config), name=exp_name, group=f"{platform.node()}")
    trainer = Trainer(config, AGENTS)
    trainer.run_games_for_agents()
    vec_env.close()





