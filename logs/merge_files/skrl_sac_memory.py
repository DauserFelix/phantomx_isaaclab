from __future__ import annotations

from typing import Literal

import numpy as np
import torch

from skrl.memories.torch.random import RandomMemory


class NStepMemory(RandomMemory):
    """Replay buffer with N-step return computation (Fix 3).

    Extends RandomMemory with two extra tensors ("truncated", "bootstrap") and
    overrides sample() to return N-step discounted returns instead of single-step.

    N-step target for SAC update:
        R_n = Σ_{k=0}^{n-1} γ^k * continuing_k * r_{t+k}
        target = R_n + γ^n * bootstrap_mask * min-Q(s_{t+n}, π)

    Args:
        n_steps: Number of steps to aggregate (default 3, per RSL-RL-SAC paper).
        discount_factor: γ used during N-step accumulation (must match agent cfg).
        All other args forwarded to RandomMemory.
    """

    def __init__(
        self,
        *,
        memory_size: int,
        num_envs: int = 1,
        device: str | torch.device | None = None,
        export: bool = False,
        export_format: Literal["pt", "npz", "csv"] = "pt",
        export_directory: str = "",
        replacement: bool = False,
        n_steps: int = 3,
        discount_factor: float = 0.97,
    ) -> None:
        super().__init__(
            memory_size=memory_size,
            num_envs=num_envs,
            device=device,
            export=export,
            export_format=export_format,
            export_directory=export_directory,
            replacement=replacement,
        )
        self.n_steps = n_steps
        self.discount_factor = discount_factor

    # CustomSAC.init() calls create_tensor for the standard SAC tensors, then we add ours.
    # We override to add "truncated" alongside the standard "terminated".
    def _ensure_extra_tensors(self) -> None:
        """Create extra tensors if they don't exist yet. Called lazily from sample()."""
        if "truncated" not in self.tensors:
            self.create_tensor(name="truncated", size=1, dtype=torch.bool)
        if "bootstrap" not in self.tensors:
            self.create_tensor(name="bootstrap", size=1, dtype=torch.float32)

    def sample(
        self,
        names: list[str],
        *,
        batch_size: int,
        mini_batches: int = 1,
        sequence_length: int = 1,
    ) -> list[list[torch.Tensor]]:
        """Sample a batch and compute N-step returns.

        Returns the same list-of-lists structure as RandomMemory, but:
          - "rewards" is replaced with the N-step accumulated return R_n
          - "next_observations" is replaced with s_{t+n} (or last valid s_{t+k+1})
          - "terminated" is unchanged (original termination flag for the base step)
          - "bootstrap" is a NEW tensor: 1 where V(s_{t+n}) should be added, 0 otherwise
        """
        self._ensure_extra_tensors()

        total_size = self.memory_size * self.num_envs
        # Exclude the last (n_steps-1)*num_envs flat indices to guarantee n future steps exist
        valid_size = max(1, len(self) - (self.n_steps - 1) * self.num_envs)

        if self._replacement:
            base_indexes = torch.randint(0, valid_size, (batch_size,))
        else:
            base_indexes = torch.randperm(valid_size, dtype=torch.long)[:batch_size]

        # ---- N-step return computation ----------------------------------------
        rewards_view = self.tensors_view["rewards"]        # (total, 1)
        next_obs_view = self.tensors_view["next_observations"]
        term_view = self.tensors_view["terminated"]        # (total, 1) bool
        trunc_view = self.tensors_view["truncated"]        # (total, 1) bool

        B = base_indexes.shape[0]

        # Initialise from step 0 (the base transition)
        idx0 = base_indexes
        r0 = rewards_view[idx0]                            # (B, 1)
        term0 = term_view[idx0].float()
        trunc0 = trunc_view[idx0].float()
        done0 = (term0 + trunc0).clamp(max=1.0)

        n_step_returns = r0.clone()                        # (B, 1)
        n_next_obs = next_obs_view[idx0].clone()           # (B, obs_dim)
        # bootstrap_mask: 1 if we should add γ^n * V to the target
        # = 1 - term  (truncation is still valid to bootstrap from)
        n_bootstrap = 1.0 - term0                         # (B, 1)

        # still_going: 1 while the episode has not ended yet at any prior step
        still_going = 1.0 - done0                         # (B, 1)

        gamma_k = self.discount_factor                     # γ^1 for step k=1

        for k in range(1, self.n_steps):
            future_idx = (base_indexes + k * self.num_envs) % total_size

            r_k = rewards_view[future_idx]
            term_k = term_view[future_idx].float()
            trunc_k = trunc_view[future_idx].float()
            done_k = (term_k + trunc_k).clamp(max=1.0)

            # Accumulate reward only where episode was still active
            n_step_returns = n_step_returns + gamma_k * still_going * r_k

            # Update n_next_obs and bootstrap to this step's values where still active
            n_next_obs = (
                n_next_obs * (1.0 - still_going)
                + next_obs_view[future_idx] * still_going
            )
            n_bootstrap = (
                n_bootstrap * (1.0 - still_going)
                + (1.0 - term_k) * still_going
            )

            still_going = still_going * (1.0 - done_k)
            gamma_k *= self.discount_factor

        # gamma_k is now γ^n_steps — stored so CustomSAC.update() can use it
        self._last_gamma_n = gamma_k

        # ---- Assemble output matching the caller's expected order ---------------
        # names expected by CustomSAC._tensors_names:
        # ["observations", "states", "actions", "rewards",
        #  "next_observations", "next_states", "terminated", "bootstrap"]
        result_tensors: dict[str, torch.Tensor] = {}

        for name in names:
            if name == "rewards":
                result_tensors[name] = n_step_returns
            elif name == "next_observations":
                result_tensors[name] = n_next_obs
            elif name == "next_states":
                # states not used in our setup (same as observations)
                result_tensors[name] = (
                    n_next_obs if "next_states" not in self.tensors
                    else self.tensors_view["next_states"][base_indexes]
                )
            elif name == "bootstrap":
                result_tensors[name] = n_bootstrap
            elif name in self.tensors:
                result_tensors[name] = self.tensors_view[name][base_indexes]
            else:
                result_tensors[name] = None  # type: ignore[assignment]

        ordered = [result_tensors[n] for n in names]

        self.sampling_indexes = base_indexes

        if mini_batches > 1:
            batches = np.array_split(np.arange(B), mini_batches)
            return [[t[b] if t is not None else None for t in ordered] for b in batches]

        return [ordered]
