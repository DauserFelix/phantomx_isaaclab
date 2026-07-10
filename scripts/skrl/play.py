# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent from skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from unittest import runner

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO", "SAC"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import random
import time

import gymnasium as gym
import skrl
import torch
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict

from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import phantomx_thesis.tasks  # noqa: F401

# Custom SAC components (Fixes 1-4 from RSL-RL-SAC paper)
from phantomx_thesis.tasks.direct.phantomx_thesis.agents.skrl_sac_models import (
    PhysicsBoundedActor,
    SACCritic,
    compute_action_scaling_direct,
    get_sac_hidden_dims,
)
from phantomx_thesis.tasks.direct.phantomx_thesis.agents.skrl_sac_agent import CustomSAC
from skrl.resources.preprocessors.torch import RunningStandardScaler

_SKRL_PREPROCESSORS = {
    "RunningStandardScaler": RunningStandardScaler,
}

# Map custom agent class names back to their base algorithm so the config entry point resolves.
_CUSTOM_CLASS_TO_ALGORITHM = {"CUSTOMSAC": "SAC"}

# If an explicit checkpoint is given but no algorithm was specified, auto-detect from the
# checkpoint's saved params/agent.yaml so the correct model architecture is loaded.
if args_cli.checkpoint and args_cli.algorithm == "PPO" and args_cli.agent is None:
    import yaml as _yaml
    _params_yaml = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(args_cli.checkpoint))),
        "params", "agent.yaml",
    )
    if os.path.exists(_params_yaml):
        with open(_params_yaml) as _f:
            _saved_class = _yaml.safe_load(_f).get("agent", {}).get("class", "PPO").upper()
        # Normalize custom subclass names to the base algorithm name
        _saved_class = _CUSTOM_CLASS_TO_ALGORITHM.get(_saved_class, _saved_class)
        if _saved_class != "PPO":
            print(f"[INFO] Auto-detected algorithm '{_saved_class}' from checkpoint params. Overriding --algorithm.")
            args_cli.algorithm = _saved_class

# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    """Play with skrl agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

        # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    experiment_cfg["seed"] = args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    env_cfg.seed = experiment_cfg["seed"]

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    if algorithm == "sac":
        # Instantiate custom SAC models so checkpoint weights load correctly
        device = env.device
        obs_space = env.observation_space
        act_space = env.action_space
        num_envs = env.num_envs

        # Read hidden_dims from the YAML instead of hardcoding them — must match whatever
        # architecture the loaded checkpoint was actually trained with (see train.py).
        policy_hidden_dims, critic_hidden_dims = get_sac_hidden_dims(experiment_cfg)
        print(f"[INFO] SAC policy hidden_dims: {policy_hidden_dims}")
        print(f"[INFO] SAC critic hidden_dims: {critic_hidden_dims}")

        action_range, action_bias = compute_action_scaling_direct(env, device)
        policy = PhysicsBoundedActor(
            observation_space=obs_space,
            action_space=act_space,
            device=device,
            action_range=action_range,
            action_bias=action_bias,
            hidden_dims=policy_hidden_dims,
            initial_log_std=experiment_cfg["models"]["policy"].get("initial_log_std", -1.9),
        )
        critic_kwargs = dict(observation_space=obs_space, action_space=act_space,
                             device=device, hidden_dims=critic_hidden_dims)
        play_agent = CustomSAC(
            models={
                "policy": policy,
                "critic_1": SACCritic(**critic_kwargs),
                "critic_2": SACCritic(**critic_kwargs),
                "target_critic_1": SACCritic(**critic_kwargs),
                "target_critic_2": SACCritic(**critic_kwargs),
            },
            memory=None,
            observation_space=obs_space,
            action_space=act_space,
            device=device,
            cfg={
                k: (
                    _SKRL_PREPROCESSORS[v] if k.endswith("_preprocessor") and isinstance(v, str) and v in _SKRL_PREPROCESSORS
                    else {"size": obs_space, "device": device} if k == "observation_preprocessor_kwargs"
                    else {} if k.endswith("_kwargs") and v is None
                    else v
                )
                for k, v in experiment_cfg["agent"].items()
                if k not in ("class", "rewards_shaper_scale")
            },
        )
    else:
        runner = Runner(env, experiment_cfg)
        play_agent = runner.agent

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    play_agent.load(resume_path)
    play_agent.training = False

    # reset environment
    obs, _ = env.reset()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        # run everything in inference mode
        with torch.inference_mode():

            # agent stepping
            outputs = play_agent.act(obs, None, timestep=timestep, timesteps=timestep)
            # - multi-agent (deterministic) actions
            if hasattr(env, "possible_agents"):
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            # - single-agent (deterministic) actions
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
            # env stepping
            obs, _, _, _, _ = env.step(actions)
        timestep += 1
        if args_cli.video and timestep == args_cli.video_length:
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
