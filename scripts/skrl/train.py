# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
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
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument(
    "--resume_step",
    type=int,
    default=None,
    help=(
        "Env step to resume the velocity curriculum at (see PhantomxThesisEnv._apply_velocity_curriculum). "
        "agent.load() only restores network/optimizer weights, not env.common_step_counter, so without this "
        "a resumed run's curriculum silently restarts at step 0. If omitted and --checkpoint matches the "
        "'agent_<N>.pt' periodic-checkpoint naming pattern, N is used as a best-effort default (this does not "
        "apply to 'best_agent.pt', which has no step number in its name)."
    ),
)
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
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
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
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

import logging
import os
import random
import re
import time
from datetime import datetime

import gymnasium as gym
import skrl
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
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)

import phantomx_thesis.tasks  # noqa: F401

# Custom SAC components (all 4 fixes from RSL-RL-SAC paper)
from phantomx_thesis.tasks.direct.phantomx_thesis.agents.skrl_sac_models import (
    PhysicsBoundedActor,
    SACCritic,
    compute_action_scaling_direct,
    get_sac_hidden_dims,
)
from phantomx_thesis.tasks.direct.phantomx_thesis.agents.skrl_sac_memory import NStepMemory
from phantomx_thesis.tasks.direct.phantomx_thesis.agents.skrl_sac_agent import CustomSAC
from phantomx_thesis.tasks.direct.phantomx_thesis.agents.timeout_wrapper import TimeoutAwareWrapper
from phantomx_thesis.tasks.direct.phantomx_thesis.phantomx_thesis_env_cfg import PhantomxThesisSACEnvCfg

# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    # The Ray Tune workflow extracts experiment name using the logging line below, hence,
    # do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f"_{agent_cfg['agent']['experiment']['experiment_name']}"
    # set directory into agent config
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # get checkpoint path (to resume training)
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    # resolve the velocity-curriculum resume step (Problem 2 fix): agent.load() below only
    # restores network/optimizer weights, never env.common_step_counter, so without this the
    # curriculum would silently restart at step 0 on every resume. --resume_step always wins;
    # as a best-effort default, fall back to parsing "agent_<N>.pt" (skrl's periodic-checkpoint
    # naming). "best_agent.pt" has no step number and is intentionally left unresolved.
    resume_step = args_cli.resume_step
    if resume_step is None and resume_path:
        match = re.search(r"agent_(\d+)\.pt$", os.path.basename(resume_path))
        if match:
            resume_step = int(match.group(1))
            print(f"[INFO] --resume_step not given; inferred {resume_step} from checkpoint filename.")
    if resume_step is not None and not resume_path:
        logger.warning(
            f"--resume_step={resume_step} was given without --checkpoint. The curriculum will start at "
            f"step {resume_step} even though no weights are being resumed."
        )
    resume_step_msg = str(resume_step) if resume_step is not None else "0 (default — no --resume_step given/inferred)"

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # guard against running the wrong algorithm against the wrong task variant: the SAC-specific
    # reward/curriculum tuning (PhantomxThesisSACEnvCfg) must only ever be paired with --algorithm SAC,
    # and vice versa, since the two env_cfg variants deliberately have different reward economics
    is_sac_env_cfg = isinstance(env_cfg, PhantomxThesisSACEnvCfg)
    if algorithm == "sac" and not is_sac_env_cfg:
        raise ValueError(
            f"--algorithm SAC requires the SAC task variant (env_cfg must be PhantomxThesisSACEnvCfg, "
            f"got {type(env_cfg).__name__}). Use --task Template-Phantomx-Thesis-SAC-Direct-v0."
        )
    if algorithm != "sac" and is_sac_env_cfg:
        raise ValueError(
            f"--algorithm {args_cli.algorithm} was requested against the SAC task variant "
            f"({type(env_cfg).__name__}), which uses SAC-specific reward/curriculum tuning. "
            f"Use --task Template-Phantomx-Thesis-Direct-v0 for {args_cli.algorithm}."
        )

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Resume the velocity curriculum at the right stage (Problem 2 fix). Placed before the
    # PPO/SAC branch below and before any wrapping so it applies identically to both algorithm
    # paths. No-op (env.common_step_counter stays at its default 0) unless --resume_step was
    # given or inferred above.
    if resume_step is not None:
        env.unwrapped.set_curriculum_step(resume_step)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    print("=" * 80)
    print("Action space:", env.action_space)
    try:
        print("low :", env.action_space.low)
        print("high:", env.action_space.high)
    except Exception as e:
        print(e)
    print("=" * 80)
    print(agent_cfg["models"]["policy"])
    print("=" * 80)

    if algorithm == "sac":
        # ------------------------------------------------------------------ #
        #  Custom SAC with all 4 fixes from the RSL-RL-SAC paper              #
        # ------------------------------------------------------------------ #
        from skrl.trainers.torch import SequentialTrainer

        # Fix 4: wrap env to cache pre-reset observations for timeout episodes
        env = TimeoutAwareWrapper(env)

        device = env.device
        obs_space = env.observation_space
        act_space = env.action_space
        num_envs = env.num_envs

        # Read hidden_dims from the YAML instead of hardcoding them, so the network actually
        # trained always matches what params/agent.yaml (and the PPO path) declare.
        policy_hidden_dims, critic_hidden_dims = get_sac_hidden_dims(agent_cfg)
        print(f"[INFO] SAC policy hidden_dims: {policy_hidden_dims}")
        print(f"[INFO] SAC critic hidden_dims: {critic_hidden_dims}")

        # Fix 1+2: physics-bounded actor with improved weight init
        action_range, action_bias = compute_action_scaling_direct(env, device)
        actor_kwargs = dict(
            observation_space=obs_space,
            action_space=act_space,
            device=device,
            action_range=action_range,
            action_bias=action_bias,
            hidden_dims=policy_hidden_dims,
            initial_log_std=agent_cfg["models"]["policy"].get("initial_log_std", -1.9),
        )
        policy = PhysicsBoundedActor(**actor_kwargs)

        critic_kwargs = dict(
            observation_space=obs_space,
            action_space=act_space,
            device=device,
            hidden_dims=critic_hidden_dims,
        )
        critic_1 = SACCritic(**critic_kwargs)
        critic_2 = SACCritic(**critic_kwargs)
        target_critic_1 = SACCritic(**critic_kwargs)
        target_critic_2 = SACCritic(**critic_kwargs)

        # Fix 3: N-step replay buffer
        mem_cfg = agent_cfg.get("memory", {})
        print(f"[INFO] SAC memory: memory_size={mem_cfg.get('memory_size', 500_000)}, "
              f"num_envs={num_envs}, replacement={mem_cfg.get('replacement', False)}")
        memory = NStepMemory(
            memory_size=mem_cfg.get("memory_size", 500_000),
            num_envs=num_envs,
            device=device,
            n_steps=mem_cfg.get("n_steps", 3),
            discount_factor=agent_cfg["agent"].get("discount_factor", 0.97),
            replacement=mem_cfg.get("replacement", False),
        )

        # SAC agent config — pass everything from YAML except class-specific keys
        agent_dict = {k: v for k, v in agent_cfg["agent"].items()
                      if k not in ("class", "rewards_shaper_scale")}

        from skrl.resources.preprocessors.torch import RunningStandardScaler
        agent_dict["observation_preprocessor"] = RunningStandardScaler
        agent_dict["observation_preprocessor_kwargs"] = {"size": obs_space}
        agent_dict["state_preprocessor"] = None
        agent_dict["state_preprocessor_kwargs"] = {}

        agent = CustomSAC(
            models={
                "policy": policy,
                "critic_1": critic_1,
                "critic_2": critic_2,
                "target_critic_1": target_critic_1,
                "target_critic_2": target_critic_2,
            },
            memory=memory,
            observation_space=obs_space,
            action_space=act_space,
            device=device,
            cfg=agent_dict,
        )

        trainer_cfg = agent_cfg.get("trainer", {})
        trainer_cfg.pop("class", None)

        trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)

        if resume_path:
            print(f"[INFO] Loading model checkpoint from: {resume_path}")
            print(f"[INFO] Velocity curriculum resumed at step: {resume_step_msg}")
            agent.load(resume_path)

        trainer.train()

    else:
        # ------------------------------------------------------------------ #
        #  All other algorithms (PPO, AMP, IPPO, MAPPO) — standard Runner     #
        # ------------------------------------------------------------------ #
        runner = Runner(env, agent_cfg)

        if resume_path:
            print(f"[INFO] Loading model checkpoint from: {resume_path}")
            print(f"[INFO] Velocity curriculum resumed at step: {resume_step_msg}")
            runner.agent.load(resume_path)

        runner.run()

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
