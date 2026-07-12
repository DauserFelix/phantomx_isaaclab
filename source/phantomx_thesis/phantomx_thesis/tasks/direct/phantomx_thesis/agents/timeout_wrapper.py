from __future__ import annotations

import torch


class TimeoutAwareWrapper:
    """Wraps a SKRL-wrapped VecEnv to cache pre-reset observations for truncated episodes.

    Isaac Lab resets truncated envs before returning the next observation, so the
    returned obs[i] after truncation is the post-reset state, not the final state
    of the episode. This wrapper stores the previous step's obs and injects it into
    infos["time_outs_obs"] so CustomSAC can use the correct next_obs for bootstrapping.

    Usage (in train.py, after SkrlVecEnvWrapper):
        env = SkrlVecEnvWrapper(raw_env)
        env = TimeoutAwareWrapper(env)
    """

    def __init__(self, env):
        self.env = env
        self._prev_obs: torch.Tensor | None = None

    def step(self, actions):
        obs, reward, terminated, truncated, infos = self.env.step(actions)

        if self._prev_obs is not None and truncated.any():
            infos["time_outs_obs"] = self._prev_obs.clone()

        self._prev_obs = obs.clone()
        return obs, reward, terminated, truncated, infos

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        obs = result[0] if isinstance(result, tuple) else result
        self._prev_obs = obs.clone()
        return result

    def __getattr__(self, name):
        return getattr(self.env, name)
