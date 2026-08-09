from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

import gymnasium
from gymnasium import spaces

from skrl.models.torch import Model
from skrl.models.torch.gaussian import GaussianMixin
from skrl.models.torch.deterministic import DeterministicMixin


Normal.set_default_validate_args(False)


def compute_action_scaling_direct(env, device: str | torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute physics-based action bounds for a DirectRLEnv (Fix 1).

    Der nutzbare Actor-Bereich hängt davon ab, ob die Env die Actions vorher klemmt:

    strict_action_pipeline=True (Default):
        _pre_physics_step klemmt auf [-1, 1], BEVOR mit action_scale skaliert wird.
        Alles, was der Actor jenseits von ±1 ausgibt, ist physikalisch identisch zu ±1 —
        ein totes Band ohne Q-Gradient, in dem tanh sättigt, ohne je eine andere Wirkung
        zu haben. Der Actor-Bereich muss deshalb exakt dem Clamp entsprechen:
            action_range = 1.0

    strict_action_pipeline=False:
        Ohne Clamp darf der Actor die volle Gelenkmarge ausnutzen:
            action_range = joint_pos_limit / action_scale

    action_bias = 0.0 (symmetrisch um die Default-Pose)

    Returns:
        action_range: (num_actions,) tensor — half-range per action dimension
        action_bias:  (num_actions,) tensor — centre offset per action dimension
    """
    cfg = env.unwrapped.cfg
    num_actions = env.action_space.shape[0]

    if getattr(cfg, "strict_action_pipeline", False):
        range_value = 1.0
    else:
        range_value = float(cfg.joint_pos_limit) / float(cfg.action_scale)

    action_range = torch.full((num_actions,), range_value, device=device)
    action_bias = torch.zeros(num_actions, device=device)
    return action_range, action_bias


def _extract_layers(agent_cfg: dict, role: str) -> list[int]:
    """Read the hidden-layer sizes for one model role from the YAML ``models`` config.

    :raises ValueError: If ``models.<role>.network`` is missing, is not a single-container
        list, or its ``layers`` entry is not a non-empty list of ints.
    """
    try:
        network = agent_cfg["models"][role]["network"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"agent_cfg['models']['{role}']['network'] is missing or malformed — cannot "
            "determine hidden_dims for the custom SAC models."
        ) from exc

    if not isinstance(network, list) or len(network) != 1:
        raise ValueError(
            f"agent_cfg['models']['{role}']['network'] must be a single-container list "
            f"(got {network!r}). PhysicsBoundedActor/SACCritic only support one sequential "
            "trunk per role."
        )

    layers = network[0].get("layers")
    if not isinstance(layers, list) or not layers or not all(isinstance(x, int) for x in layers):
        raise ValueError(
            f"agent_cfg['models']['{role}']['network'][0]['layers'] must be a non-empty list "
            f"of ints (got {layers!r})."
        )
    return list(layers)


def get_sac_hidden_dims(agent_cfg: dict) -> tuple[list[int], list[int]]:
    """Resolve policy/critic hidden_dims from the YAML config instead of hardcoding them.

    Also validates that critic_2/target_critic_1/target_critic_2 declare the same
    architecture as critic_1: Polyak averaging requires the online and target critics
    to share identical shapes, so a divergent YAML edit must fail loudly instead of
    silently only ever using critic_1's value.

    :raises ValueError: If any required config path is missing/malformed, or if the
        critic roles disagree with each other.
    """
    policy_hidden_dims = _extract_layers(agent_cfg, "policy")
    critic_hidden_dims = _extract_layers(agent_cfg, "critic_1")

    for role in ("critic_2", "target_critic_1", "target_critic_2"):
        role_dims = _extract_layers(agent_cfg, role)
        if role_dims != critic_hidden_dims:
            raise ValueError(
                f"agent_cfg['models']['{role}']['network'][0]['layers'] ({role_dims}) does not "
                f"match critic_1's layers ({critic_hidden_dims}). All four critic roles must "
                "share the same architecture for Polyak/target-network updates to work."
            )

    return policy_hidden_dims, critic_hidden_dims


# ---------------------------------------------------------------------------
# Actor — Fix 1 (physics-bounded tanh squashing) + Fix 2 (weight init)
# ---------------------------------------------------------------------------

class PhysicsBoundedActor(GaussianMixin, Model):
    """SAC actor with physics-based action bounds and improved weight initialisation.

    Fix 1: a = action_range * tanh(raw) + action_bias
           log π corrected for the tanh change-of-variables.
    Fix 2: mean head initialised N(0, 1e-3); log_std bias = log(0.15) ≈ -1.9
           so the initial policy is near-deterministic with small, safe actions.
    """

    def __init__(
        self,
        observation_space: gymnasium.Space,
        action_space: gymnasium.Space,
        device: str | torch.device,
        *,
        action_range: torch.Tensor,
        action_bias: torch.Tensor,
        hidden_dims: list[int] = [128, 128],
        clip_log_std: bool = True,
        min_log_std: float = -5.0,
        max_log_std: float = 2.0,
        initial_log_std: float = -1.9,
    ) -> None:
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        GaussianMixin.__init__(
            self,
            clip_actions=False,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
        )

        num_obs = self.num_observations
        num_act = self.num_actions

        # Shared trunk
        layers: list[nn.Module] = []
        in_dim = num_obs
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ELU()]
            in_dim = h
        self.net = nn.Sequential(*layers)

        # Mean head — near-zero init (Fix 2)
        self.mean_head = nn.Linear(in_dim, num_act)
        nn.init.normal_(self.mean_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.mean_head.bias)

        # Log-std head — bias = log(0.15) ≈ -1.9 (Fix 2)
        self.log_std_head = nn.Linear(in_dim, num_act)
        nn.init.normal_(self.log_std_head.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.log_std_head.bias, initial_log_std)

        # Physics bounds (Fix 1) — registered as buffers so they move with .to(device)
        self.register_buffer("action_range", action_range.to(device))
        self.register_buffer("action_bias", action_bias.to(device))

    def compute(self, inputs: dict, role: str = ""):
        obs = inputs["observations"]
        x = self.net(obs)
        mean_raw = self.mean_head(x)
        log_std = self.log_std_head(x)
        return mean_raw, {"log_std": log_std}

    def act(self, inputs: dict, *, role: str = ""):
        """Sample action with tanh squashing and corrected log-probability (Fix 1)."""
        mean_raw, outputs = self.compute(inputs, role)
        log_std = outputs["log_std"]

        # Clip log_std
        if self._g_clip_log_std:
            log_std = torch.clamp(log_std, self._g_min_log_std, self._g_max_log_std)

        dist = Normal(mean_raw, log_std.exp())
        self._g_distribution = dist

        # Sample raw action and squash (Fix 1)
        raw = dist.rsample()
        tanh_raw = torch.tanh(raw)
        action = self.action_range * tanh_raw + self.action_bias

        # log π(a|s) = log N(raw) − Σ log(range * (1 − tanh²(raw)) + ε)
        log_prob = dist.log_prob(raw) - torch.log(
            self.action_range * (1.0 - tanh_raw.pow(2)) + 1e-6
        )
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        outputs["log_std"] = log_std
        outputs["log_prob"] = log_prob
        outputs["mean_actions"] = self.action_range * torch.tanh(mean_raw) + self.action_bias
        return action, outputs


# ---------------------------------------------------------------------------
# Critic — standard Q-network [128, 128]
# ---------------------------------------------------------------------------

class SACCritic(DeterministicMixin, Model):
    """SAC Q-network: Q(s, a) → scalar."""

    def __init__(
        self,
        observation_space: gymnasium.Space,
        action_space: gymnasium.Space,
        device: str | torch.device,
        *,
        hidden_dims: list[int] = [128, 128],
    ) -> None:
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self, clip_actions=False)

        num_obs = self.num_observations
        num_act = self.num_actions

        layers: list[nn.Module] = []
        in_dim = num_obs + num_act
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ELU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def compute(self, inputs: dict, role: str = ""):
        obs = inputs["observations"]
        act = inputs["taken_actions"]
        x = torch.cat([obs, act], dim=-1)
        return self.net(x), {}
