from __future__ import annotations

from typing import Any

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config
from skrl.agents.torch import Agent
from skrl.agents.torch.sac.sac import SAC


class CustomSAC(SAC):
    """SAC with N-step returns and timeout-aware bootstrapping (Fixes 3 + 4).

    Changes vs. SKRL's base SAC:
      - record_transition(): stores "truncated" in NStepMemory; for timeout episodes
        replaces next_observations with the pre-reset obs from TimeoutAwareWrapper.
      - init(): registers "truncated" and "bootstrap" extra tensors in NStepMemory.
      - update(): uses γ^n_steps and bootstrap_mask from NStepMemory.
    """

    def init(self, *, trainer_cfg: dict[str, Any] | None = None) -> None:
        super().init(trainer_cfg=trainer_cfg)

        # Extend tensor list for CustomSAC sampling
        if self.memory is not None:
            self.memory.create_tensor(name="truncated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="bootstrap", size=1, dtype=torch.float32)

            self._tensors_names = [
                "observations",
                "states",
                "actions",
                "rewards",
                "next_observations",
                "next_states",
                "terminated",
                "bootstrap",
            ]

    def record_transition(
        self,
        *,
        observations: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        # Base Agent bookkeeping (episode stats, etc.) — skip SAC's memory write
        Agent.record_transition(
            self,
            observations=observations,
            states=states,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            next_states=next_states,
            terminated=terminated,
            truncated=truncated,
            infos=infos,
            timestep=timestep,
            timesteps=timesteps,
        )

        if not self.training:
            return

        # Fix 4a: for truncated (timeout) episodes, replace next_obs with the
        # pre-reset observation cached by TimeoutAwareWrapper.
        if truncated.any() and isinstance(infos, dict) and "time_outs_obs" in infos:
            mask = truncated.squeeze(-1)          # (num_envs,) bool
            pre_reset_obs = infos["time_outs_obs"]  # (num_envs, obs_dim)
            next_observations = next_observations.clone()
            next_observations[mask] = pre_reset_obs[mask]

        # Fix 4b: bootstrap flag — 1.0 for timeout truncations (legit to bootstrap)
        bootstrap = truncated.float()

        if self.cfg.rewards_shaper is not None:
            rewards = self.cfg.rewards_shaper(rewards, timestep, timesteps)

        self.memory.add_samples(
            observations=observations,
            states=states,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            next_states=next_states,
            terminated=terminated,
            truncated=truncated,
            bootstrap=bootstrap,
        )

    def update(self, *, timestep: int, timesteps: int) -> None:
        """SAC update using N-step returns and bootstrap-aware targets (Fixes 3+4)."""

        for gradient_step in range(self.cfg.gradient_steps):

            # NStepMemory.sample() returns N-step accumulated rewards and bootstrap_mask
            (
                sampled_observations,
                sampled_states,
                sampled_actions,
                sampled_rewards,           # N-step returns R_n
                sampled_next_observations, # s_{t+n} (last valid next state)
                sampled_next_states,
                sampled_terminated,        # base step's terminated flag
                sampled_bootstrap,         # 1 where γ^n * V should be added
            ) = self.memory.sample(names=self._tensors_names, batch_size=self.cfg.batch_size)[0]

            # γ^n (set by NStepMemory during sample())
            gamma_n = getattr(self.memory, "_last_gamma_n", self.cfg.discount_factor)

            with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                inputs = {
                    "observations": self._observation_preprocessor(sampled_observations, train=True),
                    "states": self._state_preprocessor(sampled_states, train=True),
                }
                next_inputs = {
                    "observations": self._observation_preprocessor(sampled_next_observations, train=True),
                    "states": self._state_preprocessor(sampled_next_states, train=True),
                }

                with torch.no_grad():
                    next_actions, next_outputs = self.policy.act(next_inputs, role="policy")
                    next_log_prob = next_outputs["log_prob"]

                    target_q1, _ = self.target_critic_1.act(
                        {**next_inputs, "taken_actions": next_actions}, role="target_critic_1"
                    )
                    target_q2, _ = self.target_critic_2.act(
                        {**next_inputs, "taken_actions": next_actions}, role="target_critic_2"
                    )
                    target_q = torch.min(target_q1, target_q2) - self._entropy_coefficient * next_log_prob

                    # Fix 4c: use bootstrap_mask instead of (1 - terminated)
                    # bootstrap_mask = 1 for continuing steps AND for timeout-truncated steps,
                    # = 0 only for real episode terminations.
                    target_values = sampled_rewards + gamma_n * sampled_bootstrap * target_q

                # Critic loss
                q1, _ = self.critic_1.act({**inputs, "taken_actions": sampled_actions}, role="critic_1")
                q2, _ = self.critic_2.act({**inputs, "taken_actions": sampled_actions}, role="critic_2")
                critic_loss = (F.mse_loss(q1, target_values) + F.mse_loss(q2, target_values)) / 2

            self.critic_optimizer.zero_grad()
            self.scaler.scale(critic_loss).backward()
            if config.torch.is_distributed:
                self.critic_1.reduce_parameters()
                self.critic_2.reduce_parameters()
            if self.cfg.grad_norm_clip > 0:
                self.scaler.unscale_(self.critic_optimizer)
                nn.utils.clip_grad_norm_(
                    itertools.chain(self.critic_1.parameters(), self.critic_2.parameters()),
                    self.cfg.grad_norm_clip,
                )
            self.scaler.step(self.critic_optimizer)

            with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                # Actor loss
                actions, outputs = self.policy.act(inputs, role="policy")
                log_prob = outputs["log_prob"]
                q1_pi, _ = self.critic_1.act({**inputs, "taken_actions": actions}, role="critic_1")
                q2_pi, _ = self.critic_2.act({**inputs, "taken_actions": actions}, role="critic_2")
                policy_loss = (
                    self._entropy_coefficient * log_prob - torch.min(q1_pi, q2_pi)
                ).mean()

            self.policy_optimizer.zero_grad()
            self.scaler.scale(policy_loss).backward()
            if config.torch.is_distributed:
                self.policy.reduce_parameters()
            if self.cfg.grad_norm_clip > 0:
                self.scaler.unscale_(self.policy_optimizer)
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_norm_clip)
            self.scaler.step(self.policy_optimizer)

            # Entropy tuning
            if self.cfg.learn_entropy:
                with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                    entropy_loss = -(
                        self.log_entropy_coefficient * (log_prob + self._target_entropy).detach()
                    ).mean()
                self.entropy_optimizer.zero_grad()
                self.scaler.scale(entropy_loss).backward()
                self.scaler.step(self.entropy_optimizer)
                self._entropy_coefficient = torch.exp(self.log_entropy_coefficient.detach())

            self.scaler.update()

            self.target_critic_1.update_parameters(self.critic_1, polyak=self.cfg.polyak)
            self.target_critic_2.update_parameters(self.critic_2, polyak=self.cfg.polyak)

            if self.policy_scheduler:
                self.policy_scheduler.step()
            if self.critic_scheduler:
                self.critic_scheduler.step()

            if self.write_interval > 0:
                self.track_data("Loss / Policy loss", policy_loss.item())
                self.track_data("Loss / Critic loss", critic_loss.item())
                self.track_data("Q-network / Q1 (max)", torch.max(q1).item())
                self.track_data("Q-network / Q1 (min)", torch.min(q1).item())
                self.track_data("Q-network / Q1 (mean)", torch.mean(q1).item())
                self.track_data("Q-network / Q2 (max)", torch.max(q2).item())
                self.track_data("Q-network / Q2 (min)", torch.min(q2).item())
                self.track_data("Q-network / Q2 (mean)", torch.mean(q2).item())
                self.track_data("Target / Target (max)", torch.max(target_values).item())
                self.track_data("Target / Target (min)", torch.min(target_values).item())
                self.track_data("Target / Target (mean)", torch.mean(target_values).item())
                self.track_data("N-step / gamma_n", gamma_n)
                if self.cfg.learn_entropy:
                    self.track_data("Loss / Entropy loss", entropy_loss.item())
                    self.track_data("Coefficient / Entropy coefficient", self._entropy_coefficient.item())
                if self.policy_scheduler:
                    self.track_data("Learning / Policy LR", self.policy_scheduler.get_last_lr()[0])
                if self.critic_scheduler:
                    self.track_data("Learning / Critic LR", self.critic_scheduler.get_last_lr()[0])
