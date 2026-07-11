# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Template-Phantomx-Thesis-Direct-v0",
    entry_point=f"{__name__}.phantomx_thesis_env:PhantomxThesisEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.phantomx_thesis_env_cfg:PhantomxThesisEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Eigene Task-Variante fuer SAC: gleiche Env-Klasse (Physik/Reward/Observation-Code bleibt
# gemeinsam), aber eigene Cfg mit SAC-spezifischer Reward-/Curriculum-Feinabstimmung
# (PhantomxThesisSACEnvCfg) — siehe phantomx_thesis_env_cfg.py. Verhindert, dass die
# PPO-Task ungewollt SAC-Werte erbt oder umgekehrt (zwei gym-IDs statt einer, nach dem
# Muster von isaaclab_tasks/direct/anymal_c/__init__.py Flat/Rough-Varianten).
gym.register(
    id="Template-Phantomx-Thesis-SAC-Direct-v0",
    entry_point=f"{__name__}.phantomx_thesis_env:PhantomxThesisEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.phantomx_thesis_env_cfg:PhantomxThesisSACEnvCfg",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)