# Step 1: Compile GES solver
```shell
virtualenv -p python3.8 env
source env/bin/activate
pip install -r requirements.txt
cd baselines/hgs_vrptw
make all
cd ../..
```

# Step 2: Run rl_data_generate.py to generate data required to train RL models
```python rl_data_generate.py --instance ortec --mp --data --remote```
* --instance, which benchmark to use, options are "ortec", "200", "400", "600", "800", "1000"
* --mp, accelerate using multiprocesses
* --data, generate data, default
* --remote, only turn on if run on amulet

# Step 3: Run RL algorithm
```python drl/results/VRPTW.py --instance ortec --exp_name test --eval```
* --instance, same as the above
* --exp_name, a customized name for the run
* --eval, turn to eval mode
