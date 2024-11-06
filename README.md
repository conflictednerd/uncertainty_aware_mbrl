# Installation
+ Use python 3.10.
+ Create a virutal env and install dependencies with `pip install -r requirements.txt`.
You might need to install the development version (`sudo apt-get install python3.10-dev`) if box2d installation fails.
+ You might also need to install `swig` and/or `wheel`.
+ **NOTE:** With the new update to `metaworld` installation of dependencies through `requirements.txt` may fail. If this happened:
    + Install metaworld using `pip install git+https://github.com/Farama-Foundation/Metaworld.git@master#egg=metaworld`.
    + Remove metaworld from `requirements.txt`. (last line)
    + Install the requirements.
    + Make sure that you are using `mujoco==2.3.0` and `gymnasium==0.29.1`, otherwise SAC training will not work!
# Running
+ Run `python -m src.main exp_name="Name of your choosing"`.
+ Config can be modified in config/ directory or by overriding the default config through command line ([see hydra's docs for more details about setting up config](https://hydra.cc/docs/intro/)).
+ `exp_name` must be specified through command line. `env_id` specifies the environment (in metaworld). It is specified in the `cfg.policy`. You can add a new policy config file in `config/policy` and use it through command line by running `python -m src.main exp_name="NAME" policy="YOUR_POLICY_CONFIG_NAME"

# Code Structure
+ `main.py` is the starting point of the program. Configs are passed to it through hydra.
+ All neural networks are defined in `nets.py`.
+ `sac.py` contains code for training a policy with soft actor critic.
+ `utils.py` contains various miscellaneous utility functions.
+ `wm.py` contains the code related to the world model.