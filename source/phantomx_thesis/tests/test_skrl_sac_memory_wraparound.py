"""Regression tests for the NStepMemory ring-buffer wrap-around (Problem 1, SAC code review).

Bug: once the buffer has wrapped (`filled=True`), `NStepMemory.sample()` still excludes
base_indexes using a *static* window (the physically-last `(n_steps-1)*num_envs` flat indices).
That window only coincides with the real "risk zone" (the `(n_steps-1)` rows immediately behind
the *current*, moving `memory_index`) at the single instant the buffer first fills. From then on
the real risk zone drifts through the buffer while the exclusion window stays fixed, so sampled
N-step returns can silently mix rewards from an unrelated, already-overwritten episode into a
newer episode's return. `terminated`/`truncated` flags do not catch this: they mark the end of
the OLD (overwritten) episode's own stream, not the ring-buffer seam, so `still_going` has no way
to know a completely different episode's data is being read.

Deliberately placed outside the `phantomx_thesis` package (source/phantomx_thesis/tests/, not
.../phantomx_thesis/phantomx_thesis/.../agents/tests/): the real package's __init__.py chain
pulls in isaaclab -> pxr (USD), which is only available inside a running Isaac Sim/Kit process,
not a plain `pytest` invocation. The module under test (skrl_sac_memory.py) has no such
dependency, so it's imported directly off an explicit sys.path entry instead.

Run: /isaac-sim/python.sh -m pytest <this file> -v
(needs the Isaac Sim-bundled torch/skrl; plain system `pytest` has neither).
"""
from __future__ import annotations

import sys
from pathlib import Path

_AGENTS_DIR = (
    Path(__file__).resolve().parents[1]
    / "phantomx_thesis" / "tasks" / "direct" / "phantomx_thesis" / "agents"
)
sys.path.insert(0, str(_AGENTS_DIR))

import torch
import pytest

from skrl_sac_memory import NStepMemory


def _make_memory(memory_size: int, n_steps: int = 3, discount_factor: float = 0.97) -> NStepMemory:
    mem = NStepMemory(
        memory_size=memory_size, num_envs=1, device="cpu",
        n_steps=n_steps, discount_factor=discount_factor, replacement=False,
    )
    # Mirrors CustomSAC.init(): observations carry a scalar "episode_id" marker so we can
    # unambiguously detect cross-episode contamination in next_observations/rewards.
    for name in ["observations", "next_observations", "states", "next_states", "actions"]:
        mem.create_tensor(name=name, size=1, dtype=torch.float32)
    mem.create_tensor(name="rewards", size=1, dtype=torch.float32)
    mem.create_tensor(name="terminated", size=1, dtype=torch.bool)
    mem.create_tensor(name="truncated", size=1, dtype=torch.bool)
    mem.create_tensor(name="bootstrap", size=1, dtype=torch.float32)
    return mem


def _add(mem: NStepMemory, episode_id: int, reward: float, terminated: bool, truncated: bool = False) -> None:
    """One env-step. observations/next_observations are stamped with episode_id so a sampled
    transition's "future" data can be checked against the base transition's own episode."""
    marker = torch.tensor([[float(episode_id)]])
    mem.add_samples(
        observations=marker, next_observations=marker, states=marker, next_states=marker,
        actions=torch.zeros(1, 1),
        rewards=torch.tensor([[float(reward)]]),
        terminated=torch.tensor([[bool(terminated)]]),
        truncated=torch.tensor([[bool(truncated)]]),
        bootstrap=torch.tensor([[1.0]]),
    )


NAMES = ["observations", "states", "actions", "rewards", "next_observations", "next_states", "terminated", "bootstrap"]


def test_prefill_phase_has_no_corruption():
    """Before the buffer ever wraps, valid_size == len(self) tracks the write frontier exactly,
    so no future-index lookup can cross into unwritten or unrelated data. This should always
    pass — it documents that Problem 1 is specifically a post-wrap issue, not a general one."""
    mem = _make_memory(memory_size=10, n_steps=3)
    episode_id = 1
    for i in range(6):  # buffer not yet full (memory_size=10)
        _add(mem, episode_id=episode_id, reward=100 + i, terminated=False)

    assert not mem.filled
    batch = mem.sample(names=NAMES, batch_size=64)[0]
    next_obs = batch[NAMES.index("next_observations")].squeeze(-1)
    # every returned "future" observation must belong to the same (only) episode written so far
    assert torch.all(next_obs == episode_id), next_obs.tolist()


def test_wraparound_does_not_mix_episodes():
    """Reproduces the exact scenario from the diagnosis: episode 1 fills the buffer completely
    and terminates; episode 2 then overwrites the first few rows and is still open (not enough
    of its own future data exists yet). A correct implementation must never hand out an N-step
    sample for episode 2 whose "future" steps are actually episode 1's (overwritten) data.

    This test is EXPECTED TO FAIL against the current sample() implementation — it is the
    regression guard for the Problem 1 fix, not a pre-existing passing test.
    """
    mem = _make_memory(memory_size=5, n_steps=3)

    # Episode 1: exactly fills the 5-row buffer, terminates on the last row.
    for i in range(5):
        _add(mem, episode_id=1, reward=1000 + i, terminated=(i == 4))
    assert mem.filled
    assert mem.memory_index == 0

    # Episode 2: overwrites rows 0,1,2. Still open (no termination), and critically has NOT
    # yet written enough of its own future data for a full 3-step return from its own step 2.
    for i in range(3):
        _add(mem, episode_id=2, reward=2000 + i, terminated=False)
    assert mem.memory_index == 3

    # Deterministically inspect every currently-sampleable base row (valid_size=3 with
    # replacement=False -> randperm over exactly those 3 rows, batch_size>=3 returns all of them).
    batch = mem.sample(names=NAMES, batch_size=64)[0]
    base_rows = mem.sampling_indexes.tolist()
    base_obs = batch[NAMES.index("observations")].squeeze(-1).tolist()
    next_obs = batch[NAMES.index("next_observations")].squeeze(-1).tolist()
    rewards = batch[NAMES.index("rewards")].squeeze(-1).tolist()

    contaminated = []
    for row, base_ep, next_ep, ret in zip(base_rows, base_obs, next_obs, rewards):
        # A transition's N-step window must stay within its own episode's genuinely-written
        # data. Since episode 2 only has 3 rows of its own, and its own reward stream is
        # {2000, 2001, 2002}, any accumulated return that is NOT explainable purely from
        # episode 2's own (partial, un-terminated) rewards proves contamination from episode 1.
        if base_ep == 2 and next_ep != 2:
            contaminated.append((row, base_ep, next_ep, ret))

    assert not contaminated, (
        f"N-step sample(s) mixed data from a different (overwritten) episode after ring-buffer "
        f"wrap-around: {contaminated}. Buffer contents were rewards={mem.tensors_view['rewards'].squeeze(-1).tolist()}, "
        f"episode_ids={mem.tensors_view['observations'].squeeze(-1).tolist()}, "
        f"terminated={mem.tensors_view['terminated'].squeeze(-1).tolist()}"
    )


def test_wraparound_stress_multi_env_replacement_true():
    """Broader confidence check beyond the minimal 1-env reproduction: multiple envs, several
    full wrap-arounds, short randomly-terminating episodes (maximises seam crossings), and
    replacement=True (the actual production setting in skrl_sac_cfg.yaml). Every env gets its
    own running episode_id counter; asserts zero cross-episode contamination over many samples.
    """
    torch.manual_seed(0)
    num_envs = 4
    memory_size = 11  # small + short episodes -> lots of seam crossings per env
    mem = NStepMemory(memory_size=memory_size, num_envs=num_envs, device="cpu", n_steps=3, discount_factor=0.97, replacement=True)
    for name in ["observations", "next_observations", "states", "next_states", "actions"]:
        mem.create_tensor(name=name, size=1, dtype=torch.float32)
    mem.create_tensor(name="rewards", size=1, dtype=torch.float32)
    mem.create_tensor(name="terminated", size=1, dtype=torch.bool)
    mem.create_tensor(name="truncated", size=1, dtype=torch.bool)
    mem.create_tensor(name="bootstrap", size=1, dtype=torch.float32)

    episode_id = torch.zeros(num_envs)
    step_in_episode = torch.zeros(num_envs)
    for step in range(400):  # >> memory_size * several laps
        term = torch.rand(num_envs) < 0.3  # short, irregular episodes
        obs = episode_id.clone().unsqueeze(-1)
        reward = (episode_id * 1000 + step_in_episode).unsqueeze(-1)
        mem.add_samples(
            observations=obs, next_observations=obs, states=obs, next_states=obs,
            actions=torch.zeros(num_envs, 1), rewards=reward,
            terminated=term.unsqueeze(-1), truncated=torch.zeros(num_envs, 1, dtype=torch.bool),
            bootstrap=torch.ones(num_envs, 1),
        )
        episode_id += term.float()
        step_in_episode = torch.where(term, torch.zeros(num_envs), step_in_episode + 1)

    assert mem.filled

    names = ["observations", "states", "actions", "rewards", "next_observations", "next_states", "terminated", "bootstrap"]
    for _ in range(20):  # repeat sampling many times (randint is stochastic)
        batch = mem.sample(names=names, batch_size=512)[0]
        base_obs = batch[names.index("observations")].squeeze(-1)
        next_obs = batch[names.index("next_observations")].squeeze(-1)
        bootstrap = batch[names.index("bootstrap")].squeeze(-1)
        # Wherever bootstrap==1 (episode was still open n_steps later), next_observations must
        # be from the SAME episode as the base observation — any mismatch is contamination
        # from an unrelated, overwritten episode crossing the ring-buffer seam.
        mismatched = (bootstrap == 1.0) & (next_obs != base_obs)
        assert not mismatched.any(), (
            f"contamination found: base_ep={base_obs[mismatched].tolist()}, "
            f"next_ep={next_obs[mismatched].tolist()}"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
