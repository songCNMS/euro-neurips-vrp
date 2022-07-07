# Controller has an environment and tests it against a dynamic solver program
import subprocess
import argparse
import tools
import json
import sys
import numpy as np
import threading
from environment import VRPEnvironment
import os
from datetime import datetime
import pandas as pd
from cvrptw import compute_cost_from_routes



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", help="Instance to solve")
    parser.add_argument("--instance_seed", type=int, default=1, help="Seed to use for the dynamic instance")
    parser.add_argument("--static", action='store_true', help="Add this flag to solve the static variant of the problem (by default dynamic)")
    parser.add_argument("--epoch_tlim", type=int, default=600, help="Time limit per epoch")
    parser.add_argument("--timeout", type=int, default=3600, help="Global timeout (seconds) to use")
    parser.add_argument("--remote", action="store_true")
    
    try:
        split_idx = sys.argv.index("--")
    except ValueError:
        print("Usage: python controller.py {options} -- {solver}")
        sys.exit()

    args = parser.parse_args(sys.argv[1:split_idx])
    if args.remote: 
        data_dir = os.getenv("AMLT_DATA_DIR", "cvrp_benchmarks/")
        output_dir = os.environ['AMLT_OUTPUT_DIR']
    else:
        data_dir = "./"
        output_dir = "./"
    
    is_solo = (args.instance != 'ortec')
    solver_cmd = sys.argv[split_idx+1:]
    
    dir_name = os.path.dirname(f"{data_dir}/cvrp_benchmarks/homberger_{args.instance}_customer_instances/")
    problem_list = sorted(os.listdir(dir_name))
    res_file_name = f"{output_dir}/ges_res_{args.instance}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    sota_res = pd.read_csv("sota_res.csv")
    sota_res_dict = {row["problem"]: (row["distance"], row["vehicle"]) for _, row in sota_res.iterrows()}
    result_list = []
    for problem in problem_list:
        problem_file = os.path.join(dir_name, problem)
        if is_solo: static_instance = tools.read_solomon(problem_file)
        else: static_instance = tools.read_vrplib(problem_file)

        # Create environment
        env = VRPEnvironment(args.instance_seed, static_instance, args.epoch_tlim, args.static)

        done = False

        # Start subprocess and interact with it
        with subprocess.Popen(solver_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) as p:

            # Set global timeout, bit ugly but this will also interrupt waiting for input
            timeout_timer = threading.Timer(args.timeout, lambda: p.kill())
            timeout_timer.daemon = True
            timeout_timer.start()

            for line in p.stdout:
                line = line.strip()
                request = json.loads(line)
                if request['action'] == 'step':
                    solution = [np.array(route) for route in request['data']]
                    observation, reward, done, info = env.step(solution)

                    response = dict(
                        observation=observation,
                        reward=reward,
                        done=done,
                        info=info
                    )
                elif request['action'] == 'reset':
                    assert env.reset_counter == 0, "Can only reset environment once!"
                    observation, info = env.reset()
                    response = dict(
                        observation=observation,
                        info=info
                    )
                else:
                    raise Exception("Invalid request")
                
                response_str = tools.json_dumps_np(response)
                p.stdin.write(response_str)
                p.stdin.write('\n')
                p.stdin.flush()

            # Cancel timer (does nothing if timer already triggered) and wait for it to finish
            timeout_timer.cancel()
            timeout_timer.join()

            # Catch remaining output and wait at most 10 secs for solver thread to finish gracefully
            return_code = p.wait(10)
            # assert return_code == 0, "Solver did not exit succesfully"

        # assert done, "Environment is not finished"
        # Write results
        if done and (return_code == 0):
            print(f"------ Controller ------")
            print(f"Cost of solution: {sum(env.final_costs.values())}")
            print("Solution:")
            print(tools.json_dumps_np(env.final_solutions))
            if is_solo:
                route_num = len(env.final_solutions[env.end_epoch])
                total_cost = compute_cost_from_routes(env.final_solutions[env.end_epoch], static_instance['coords'])
            else: route_num, total_cost = len(env.final_solutions[env.end_epoch]), sum(env.final_costs.values())
        else: route_num, total_cost = -1, -1
        problem_name =  str.lower(os.path.splitext(os.path.basename(problem_file))[0])
        sota = sota_res_dict.get(problem_name, (1, 1))
        result_list.append([problem, route_num, total_cost, sota[1], sota[0]])
        res_df = pd.DataFrame(data=result_list, columns=['problem', 'vehicles', 'total_cost', 'sota_vehicles', 'sota_cost'])
        res_df.loc[:, "gap"] = (res_df["total_cost"] - res_df["sota_cost"])/res_df["sota_cost"]
        res_df.to_csv(res_file_name, index=False)
    print(res_df.head())
