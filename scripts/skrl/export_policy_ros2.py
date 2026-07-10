"""Export a trained SKRL PPO or SAC policy checkpoint to a self-contained
TorchScript bundle for use in ROS2 / Gazebo.

The exported file contains:
  - RunningStandardScaler weights (observation normalization)
  - Policy network weights (actor only, no critic)
  - forward(obs_raw) -> mean_actions   (deterministic, no sampling)

For smooth deployment, use the SmoothPolicyRunner wrapper in your ROS2 node
(see bottom of file / --demo flag).

Usage:
  # Export PPO checkpoint:
  python export_policy_ros2.py \\
      --checkpoint logs/.../best_agent.pt \\
      --algorithm  ppo

  # Export SAC checkpoint:
  python export_policy_ros2.py \\
      --checkpoint logs/.../best_agent.pt \\
      --algorithm  sac

  # Quick demo (no Isaac Sim needed):
  python export_policy_ros2.py \\
      --checkpoint logs/.../best_agent.pt \\
      --algorithm  ppo \\
      --demo
"""

import argparse
import os
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Joint configuration (alphabetical Isaac Lab order, 18 DOF)
# ---------------------------------------------------------------------------
JOINT_NAMES = [
    "j_c1_lf",    "j_c1_lm",    "j_c1_lr",
    "j_c1_rf",    "j_c1_rm",    "j_c1_rr",
    "j_thigh_lf", "j_thigh_lm", "j_thigh_lr",
    "j_thigh_rf", "j_thigh_rm", "j_thigh_rr",
    "j_tibia_lf", "j_tibia_lm", "j_tibia_lr",
    "j_tibia_rf", "j_tibia_rm", "j_tibia_rr",
]

# Default joint positions (matching phantomx.py init_state)
DEFAULT_JOINT_POS = torch.tensor([
    0.0, 0.0, 0.0,   # c1 (coxa):   all 0
    0.0, 0.0, 0.0,
    0.5, 0.5, 0.5,   # thigh (femur): all 0.5 rad
    0.5, 0.5, 0.5,
    0.5, 0.5, 0.5,   # tibia (knee):  all 0.5 rad
    0.5, 0.5, 0.5,
])

ACTION_SCALE     = 0.5      # from env_cfg
JOINT_POS_LIMIT  = 0.5235   # ±30° from default


# ---------------------------------------------------------------------------
# Utility: RunningStandardScaler
# ---------------------------------------------------------------------------
class RunningStandardScalerModule(nn.Module):
    def __init__(self, size: int, clip_threshold: float = 5.0, epsilon: float = 1e-8):
        super().__init__()
        self.clip_threshold = clip_threshold
        self.epsilon = epsilon
        self.register_buffer("running_mean", torch.zeros(size))
        self.register_buffer("running_variance", torch.ones(size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        std = torch.sqrt(self.running_variance + self.epsilon)
        return torch.clamp((x - self.running_mean) / std,
                           -self.clip_threshold, self.clip_threshold)


# ---------------------------------------------------------------------------
# PPO policy network  [66 → 128 → 128 → 18]  (no tanh on output)
# ---------------------------------------------------------------------------
class PPOPolicyModule(nn.Module):
    def __init__(self, obs_dim: int = 66, act_dim: int = 18, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden),  nn.ELU(),
        )
        self.policy_layer = nn.Linear(hidden, act_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.policy_layer(self.net(obs))


# ---------------------------------------------------------------------------
# SAC policy network  [66 → 128 → 128 → tanh → ×1.047]
# ---------------------------------------------------------------------------
class SACPolicyModule(nn.Module):
    def __init__(self, obs_dim: int = 66, act_dim: int = 18, hidden: int = 128,
                 action_range: float = 1.047):
        super().__init__()
        self.action_range = action_range
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden),  nn.ELU(),
        )
        self.mean_head = nn.Linear(hidden, act_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.action_range * torch.tanh(self.mean_head(self.net(obs)))


# ---------------------------------------------------------------------------
# TorchScript bundle: Scaler + Policy  (stateless, safe for JIT)
# forward(obs_raw) -> raw mean_actions  (NOT yet clipped/scaled)
# ---------------------------------------------------------------------------
class PolicyBundle(nn.Module):
    def __init__(self, scaler: RunningStandardScalerModule, policy: nn.Module):
        super().__init__()
        self.scaler = scaler
        self.policy = policy

    def forward(self, obs_raw: torch.Tensor) -> torch.Tensor:
        return self.policy(self.scaler(obs_raw))


# ---------------------------------------------------------------------------
# SmoothPolicyRunner  — Python wrapper for ROS2 nodes
#
# Wraps the loaded TorchScript bundle and adds:
#   1. EMA smoothing      (prevents jittery joint commands)
#   2. Max action rate    (limits joint velocity ≤ velocity_limit)
#   3. Env action pipeline (clamp → scale → joint-limit clamp)
#   4. Episode reset      (returns to default pose gracefully)
#
# Parameters:
#   ema_alpha         Blend factor: 0=frozen, 1=raw policy (default 0.2)
#   max_action_delta  Max change per step in action space [-1,1] (default 0.04)
#                     Relation to joint velocity:
#                       v_joint [rad/s] = max_action_delta × action_scale / step_dt
#                       0.04 × 0.5 / 0.02 = 1.0 rad/s  (just above hw limit 0.8)
#   action_scale      Must match env_cfg.action_scale (default 0.5)
#   joint_pos_limit   Must match env_cfg.joint_pos_limit (default 0.5235 rad)
# ---------------------------------------------------------------------------
class SmoothPolicyRunner:
    def __init__(
        self,
        policy_bundle: torch.nn.Module,
        *,
        ema_alpha: float = 0.20,
        max_action_delta: float = 0.04,
        action_scale: float = ACTION_SCALE,
        joint_pos_limit: float = JOINT_POS_LIMIT,
        default_joint_pos: torch.Tensor | None = None,
        device: str = "cpu",
    ):
        self.policy   = policy_bundle.to(device).eval()
        self.device   = device
        self.alpha    = ema_alpha
        self.max_delta = max_action_delta
        self.action_scale    = action_scale
        self.joint_pos_limit = joint_pos_limit

        self.default_pos = (
            DEFAULT_JOINT_POS.clone() if default_joint_pos is None
            else default_joint_pos.clone()
        ).to(device)

        self._smooth_action = torch.zeros(18, device=device)
        self._prev_action   = torch.zeros(18, device=device)

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Call at episode start / after robot falls. Resets smooth state."""
        self._smooth_action = torch.zeros(18, device=self.device)
        self._prev_action   = torch.zeros(18, device=self.device)

    # ------------------------------------------------------------------
    def step(self, obs_raw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs_raw: (66,) or (1, 66) raw observation tensor

        Returns:
            joint_targets: (18,) absolute joint position targets [rad]
        """
        if obs_raw.dim() == 1:
            obs_raw = obs_raw.unsqueeze(0)          # (1, 66)

        with torch.inference_mode():
            raw = self.policy(obs_raw.to(self.device)).squeeze(0)  # (18,)

        # 1. Clamp raw policy output to action space [-1, 1]
        raw_clipped = raw.clamp(-1.0, 1.0)

        # 2. EMA smoothing: blend new policy output with previous smooth action
        #    smooth = α × raw + (1-α) × smooth_prev
        self._smooth_action = (
            self.alpha * raw_clipped
            + (1.0 - self.alpha) * self._smooth_action
        )

        # 3. Action rate limiting: clamp change relative to last sent action
        delta = self._smooth_action - self._prev_action
        delta = delta.clamp(-self.max_delta, self.max_delta)
        action = self._prev_action + delta
        self._prev_action = action.clone()

        # 4. Convert to joint position targets
        joint_targets = self.default_pos + self.action_scale * action
        joint_targets = joint_targets.clamp(
            self.default_pos - self.joint_pos_limit,
            self.default_pos + self.joint_pos_limit,
        )

        return joint_targets   # (18,) in rad, alphabetical joint order

    # ------------------------------------------------------------------
    @property
    def joint_names(self) -> list[str]:
        return JOINT_NAMES

    @property
    def default_joint_pos(self) -> torch.Tensor:
        return self.default_pos


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------
def load_ppo(ckpt: dict, obs_dim: int, act_dim: int) -> PolicyBundle:
    scaler = RunningStandardScalerModule(obs_dim)
    sp = ckpt.get("observation_preprocessor", ckpt.get("state_preprocessor", {}))
    if sp:
        scaler.running_mean.copy_(sp["running_mean"])
        scaler.running_variance.copy_(sp["running_variance"])
    else:
        print("[WARN] No observation_preprocessor in checkpoint — using identity scaler.")

    policy = PPOPolicyModule(obs_dim, act_dim)
    pw = ckpt["policy"]
    policy.net[0].weight.data.copy_(pw["net_container.0.weight"])
    policy.net[0].bias.data.copy_(pw["net_container.0.bias"])
    policy.net[2].weight.data.copy_(pw["net_container.2.weight"])
    policy.net[2].bias.data.copy_(pw["net_container.2.bias"])
    policy.policy_layer.weight.data.copy_(pw["policy_layer.weight"])
    policy.policy_layer.bias.data.copy_(pw["policy_layer.bias"])

    return PolicyBundle(scaler, policy)


def load_sac(ckpt: dict, obs_dim: int, act_dim: int, action_range: float) -> PolicyBundle:
    scaler = RunningStandardScalerModule(obs_dim)
    sp = ckpt.get("observation_preprocessor", {})
    if sp:
        scaler.running_mean.copy_(sp["running_mean"])
        scaler.running_variance.copy_(sp["running_variance"])
    else:
        print("[WARN] No observation_preprocessor found — using identity scaler.")

    policy = SACPolicyModule(obs_dim, act_dim, action_range=action_range)
    pw = ckpt["policy"]
    policy.net[0].weight.data.copy_(pw["net.0.weight"])
    policy.net[0].bias.data.copy_(pw["net.0.bias"])
    policy.net[2].weight.data.copy_(pw["net.2.weight"])
    policy.net[2].bias.data.copy_(pw["net.2.bias"])
    policy.mean_head.weight.data.copy_(pw["mean_head.weight"])
    policy.mean_head.bias.data.copy_(pw["mean_head.bias"])

    return PolicyBundle(scaler, policy)


# ---------------------------------------------------------------------------
# Demo: simulate 200 steps and show action statistics
# ---------------------------------------------------------------------------
def run_demo(runner: SmoothPolicyRunner, steps: int = 200) -> None:
    import math

    obs = torch.zeros(66)
    # Fill in plausible gravity vector (robot upright)
    obs[6]  = 0.0   # gx
    obs[7]  = 0.0   # gy
    obs[8]  = -1.0  # gz (downward)

    max_joint_vel = 0.0
    max_delta_rad = 0.0
    prev_target   = runner.default_joint_pos.clone()

    print(f"\n{'Step':>5} | {'target[0]':>10} | {'target[6]':>10} | "
          f"{'Δjoint_max':>12} | {'jvel_max [r/s]':>14}")
    print("-" * 65)

    for i in range(steps):
        # Simulate slowly changing velocity command
        obs[9]  = 0.3 * math.sin(i / 30)   # vx_cmd
        obs[10] = 0.0                        # vy_cmd
        obs[11] = 0.0                        # yaw_cmd

        target = runner.step(obs)
        delta  = (target - prev_target).abs().max().item()
        jvel   = delta / 0.02                # step_dt = 0.02 s

        max_joint_vel = max(max_joint_vel, jvel)
        max_delta_rad = max(max_delta_rad, delta)
        prev_target   = target.clone()

        if i % 20 == 0:
            print(f"{i:>5} | {target[0].item():>10.4f} | {target[6].item():>10.4f} | "
                  f"{delta:>12.5f} | {jvel:>14.3f}")

    print("-" * 65)
    print(f"\nMax joint velocity seen: {max_joint_vel:.3f} rad/s  "
          f"(hw limit: 0.8 rad/s)")
    print(f"Max joint step size:     {max_delta_rad:.5f} rad\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output",     default=None,
                        help="Output .pt path (default: next to checkpoint)")
    parser.add_argument("--algorithm",  default="ppo", choices=["ppo", "sac"])
    parser.add_argument("--obs_dim",    type=int,   default=66)
    parser.add_argument("--act_dim",    type=int,   default=18)
    parser.add_argument("--action_range", type=float, default=1.047,
                        help="SAC only: joint_pos_limit / action_scale")
    # Smoothing parameters
    parser.add_argument("--ema_alpha",       type=float, default=0.20,
                        help="EMA blend: 0=frozen, 1=raw (default 0.20)")
    parser.add_argument("--max_action_delta", type=float, default=0.04,
                        help="Max action change per step in [-1,1] space. "
                             "0.04 ≈ 1.0 rad/s joint velocity. "
                             "0.016 = at hw limit (0.8 rad/s). (default 0.04)")
    parser.add_argument("--demo", action="store_true",
                        help="Run 200-step simulation to check smoothness")
    args = parser.parse_args()

    if args.output is None:
        base   = os.path.dirname(args.checkpoint)
        args.output = os.path.join(base, f"policy_ros2_{args.algorithm}.pt")

    print(f"[INFO] Loading checkpoint:  {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    if args.algorithm == "ppo":
        bundle = load_ppo(ckpt, args.obs_dim, args.act_dim)
    else:
        bundle = load_sac(ckpt, args.obs_dim, args.act_dim, args.action_range)

    bundle.eval()

    # Raw forward-pass sanity check
    dummy = torch.zeros(1, args.obs_dim)
    with torch.no_grad():
        raw_out = bundle(dummy)
    print(f"[INFO] Raw policy output (zero obs): "
          f"range=[{raw_out.min():.2f}, {raw_out.max():.2f}]")

    # Export TorchScript
    scripted = torch.jit.script(bundle)
    scripted.save(args.output)
    print(f"[INFO] TorchScript bundle saved → {args.output}")

    # Print smoothing configuration
    step_dt = 0.02
    max_jvel = args.max_action_delta * ACTION_SCALE / step_dt
    print(f"\n[INFO] Smoothing configuration:")
    print(f"  ema_alpha        = {args.ema_alpha}  "
          f"(τ ≈ {step_dt / max(args.ema_alpha, 1e-6):.2f} s settling time)")
    print(f"  max_action_delta = {args.max_action_delta}  "
          f"→ max joint velocity = {max_jvel:.2f} rad/s "
          f"({'OK' if max_jvel <= 0.8 else 'ABOVE hw limit 0.8 rad/s'})")

    # Optional demo
    if args.demo:
        runner = SmoothPolicyRunner(
            bundle,
            ema_alpha=args.ema_alpha,
            max_action_delta=args.max_action_delta,
        )
        run_demo(runner)

    # Print ROS2 usage snippet
    print(f"""
═══════════════════════════════════════════════════════════════
  ROS2 Node Usage
═══════════════════════════════════════════════════════════════
import torch
from export_policy_ros2 import SmoothPolicyRunner, JOINT_NAMES

policy_bundle = torch.jit.load("{args.output}")
runner = SmoothPolicyRunner(
    policy_bundle,
    ema_alpha={args.ema_alpha},          # smoother ↔ more responsive
    max_action_delta={args.max_action_delta},   # max change/step in [-1,1] space
)

# On episode start / robot fall:
runner.reset()

# In your 50 Hz timer callback:
obs = build_obs_vector()               # torch.Tensor (66,), raw sensor values
joint_targets = runner.step(obs)       # torch.Tensor (18,) in rad

# Publish to Gazebo:
for name, pos in zip(runner.joint_names, joint_targets.tolist()):
    ...  # send to /joint_group_position_controller/commands
═══════════════════════════════════════════════════════════════
""")


if __name__ == "__main__":
    main()
