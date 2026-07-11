from __future__ import annotations

import torch
import gymnasium as gym

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor

from .phantomx_thesis_env_cfg import PhantomxThesisEnvCfg


class PhantomxThesisEnv(DirectRLEnv):
    cfg: PhantomxThesisEnvCfg

    def __init__(self, cfg: PhantomxThesisEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(
            self.num_envs,
            gym.spaces.flatdim(self.single_action_space),
            device=self.device
        )
        self._previous_actions = torch.zeros(
            self.num_envs,
            gym.spaces.flatdim(self.single_action_space),
            device=self.device
        )

        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # Get specific body indices for termination (all 6 tibias/feet)
        self._die_body_ids, _ = self._contact_sensor.find_bodies([
            "tibia_lf", "tibia_lm", "tibia_lr",  # Left feet
            "tibia_rf", "tibia_rm", "tibia_rr"   # Right feet
        ])

        # MP_BODY index for height measurement (physical body, 10cm above base_link)
        self._mp_body_idx, _ = self._robot.find_bodies(["MP_BODY"])

        # Tripod gait indices into the 6-element foot array (order: lf=0, lm=1, lr=2, rf=3, rm=4, rr=5)
        self._TRIPOD_A = [0, 4, 2]  # lf, rm, lr
        self._TRIPOD_B = [3, 1, 5]  # rf, lm, rr

        self._has_stood_up = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "flat_orientation_l2",
                "alive",
                "height_tracking",
                "movement_penalty",
                "foot_contact",
                "tripod_gait",
                "lazy_legs",
            ]
        }

    # --------------------- SETUP ---------------------
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # --------------------- ACTION ---------------------
    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()

        q_def = self._robot.data.default_joint_pos
        self._processed_actions = q_def + self.cfg.action_scale * self._actions

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    # --------------------- OBSERVATIONS ---------------------
    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()

        # Sensor-Messungen mit Rauschen (Sim-to-Real: modelliert Encoder/IMU-Noise)
        lin_vel  = self._robot.data.root_lin_vel_b  + torch.randn_like(self._robot.data.root_lin_vel_b)  * 0.01
        ang_vel  = self._robot.data.root_ang_vel_b  + torch.randn_like(self._robot.data.root_ang_vel_b)  * 0.01
        gravity  = self._robot.data.projected_gravity_b + torch.randn_like(self._robot.data.projected_gravity_b) * 0.01
        jpos_rel = (self._robot.data.joint_pos - self._robot.data.default_joint_pos) + torch.randn(self.num_envs, self._robot.num_joints, device=self.device) * 0.01
        jvel     = self._robot.data.joint_vel   + torch.randn(self.num_envs, self._robot.num_joints, device=self.device) * 0.05

        obs = torch.cat(
            [
                lin_vel,           # 3
                ang_vel,           # 3
                gravity,           # 3
                self._commands,    # 3
                jpos_rel,          # 18
                jvel,              # 18
                self._actions,     # 18
            ],
            dim=-1,
        )  # total: 66

        return {"policy": obs}

    # --------------------- REWARDS ---------------------
    def _get_rewards(self) -> torch.Tensor:

        # linear velocity tracking
        lin_vel_error = torch.sum(
            torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]),
            dim=1
        )
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)

        # yaw rate tracking
        yaw_rate_error = torch.square(
            self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2]
        )
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)

        # z velocity penalty (body should not bounce)
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])

        # angular velocity x/y penalty (no roll/pitch)
        ang_vel_error = torch.sum(
            torch.square(self._robot.data.root_ang_vel_b[:, :2]),
            dim=1
        )

        # joint torques penalty
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)

        # joint acceleration penalty
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)

        # action rate penalty
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)

        # flat orientation penalty
        flat_orientation = torch.sum(
            torch.square(self._robot.data.projected_gravity_b[:, :2]),
            dim=1
        )

        # MP_BODY height tracking — relative to local terrain surface
        terrain_z = self._terrain.env_origins[:, 2]
        base_height = self._robot.data.body_pos_w[:, self._mp_body_idx[0], 2] - terrain_z
        height_error = torch.square(base_height - self.cfg.target_base_height)
        height_reward = torch.exp(-height_error / 0.02)

        # Alive reward
        alive_reward = torch.ones_like(lin_vel_error)

        # Movement penalty: penalizes not moving forward (unconditional — wie Working-Model 21.04.)
        forward_speed = self._robot.data.root_lin_vel_b[:, 0]
        is_moving_forward = forward_speed > self.cfg.movement_speed_x
        movement_penalty = (~is_moving_forward).float()

        # Foot contact reward — bonus for stable tripod support base (≥3 feet on ground)
        foot_forces = self._contact_sensor.data.net_forces_w[:, self._die_body_ids, :]
        foot_contact_bool = torch.norm(foot_forces, dim=-1) > 1.0
        num_feet_in_contact = foot_contact_bool.float().sum(dim=-1)
        foot_contact_reward = torch.clamp(num_feet_in_contact / 3.0, max=1.0)

        # Tripod gait reward — belohnt alternierenden 3-3 Kontakt (lf+rm+lr vs rf+lm+rr)
        tripod_a = foot_contact_bool[:, self._TRIPOD_A].float().sum(dim=-1)
        tripod_b = foot_contact_bool[:, self._TRIPOD_B].float().sum(dim=-1)
        tripod_score = torch.abs(tripod_a - tripod_b) / 3.0

        # Lazy leg penalty — Beine die dauerhaft (>1s) in der Luft hängen
        current_air_times = self._contact_sensor.data.current_air_time[:, self._die_body_ids]
        lazy_legs = (current_air_times > 1.0).float().sum(dim=-1)

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped  * self.cfg.lin_vel_reward_scale       * self.step_dt,
            "track_ang_vel_z_exp":  yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale       * self.step_dt,
            "lin_vel_z_l2":         z_vel_error            * self.cfg.z_vel_reward_scale          * self.step_dt,
            "ang_vel_xy_l2":        ang_vel_error          * self.cfg.ang_vel_reward_scale        * self.step_dt,
            "dof_torques_l2":       joint_torques          * self.cfg.joint_torque_reward_scale   * self.step_dt,
            "dof_acc_l2":           joint_accel            * self.cfg.joint_accel_reward_scale    * self.step_dt,
            "action_rate_l2":       action_rate            * self.cfg.action_rate_reward_scale    * self.step_dt,
            "flat_orientation_l2":  flat_orientation       * self.cfg.flat_orientation_reward_scale * self.step_dt,
            "alive":                alive_reward           * self.cfg.alive_reward_scale          * self.step_dt,
            "height_tracking":      height_reward          * self.cfg.height_reward_scale,
            "movement_penalty":     -movement_penalty      * self.cfg.movement_penalty_scale      * self.step_dt,
            "foot_contact":         foot_contact_reward    * self.cfg.foot_contact_reward_scale   * self.step_dt,
            "tripod_gait":          tripod_score           * self.cfg.tripod_gait_reward_scale    * self.step_dt,
            "lazy_legs":           -lazy_legs              * self.cfg.lazy_leg_penalty_scale      * self.step_dt,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        for key, value in rewards.items():
            self._episode_sums[key] += value

        return reward

    # --------------------- TERMINATION ---------------------
    def _get_dones(self):
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # MP_BODY height relative to local terrain surface
        mp_body_height = self._robot.data.body_pos_w[:, self._mp_body_idx[0], 2] - self._terrain.env_origins[:, 2]
        gravity = self._robot.data.projected_gravity_b
        tilt = torch.sum(torch.square(gravity[:, :2]), dim=1)

        died = (
            (mp_body_height < self.cfg.termination_height) |
            (mp_body_height > 0.30) |
            (tilt > self.cfg.termination_tilt)
        )

        return died, time_out

    # --------------------- RESET ---------------------
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if self.common_step_counter < 2:
            print(f"env_origins[:5]:\n{self._terrain.env_origins[:5]}")
            print(f"unique origins: {self._terrain.env_origins.unique(dim=0).shape[0]} / {self.num_envs}")

        if len(env_ids) == self.num_envs and self.common_step_counter > 0:
            self.episode_length_buf[:] = torch.randint_like(
                self.episode_length_buf,
                high=int(self.max_episode_length)
            )

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._has_stood_up[env_ids] = False

        # Velocity curriculum
        steps = self.common_step_counter
        if steps < 100_000:
            self._commands[env_ids] = 0.0
        elif steps < 350_000:
            self._commands[env_ids] = 0.0
            self._commands[env_ids, 0] = torch.rand(len(env_ids), device=self.device) * 0.5
        else:
            curriculum_factor = min(1.0, (steps - 350_000) / 700_000)
            max_vel = 0.3 + curriculum_factor * 0.7   # 0.3 → 1.0 m/s
            self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(
                -max_vel, max_vel
            )

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_pos += torch.randn_like(joint_pos) * 0.10  # ±~5.7° initial randomization

        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        default_root_state[:, 2] += torch.randn(len(env_ids), device=self.device) * 0.01

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Logging episode statistics
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0

        self.extras["log"] = dict()
        self.extras["log"].update(extras)

        extras = dict()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)

    def get_IO_descriptors(self) -> dict:
        return {
            "observations": {
                "policy": [
                    {"name": "root_lin_vel_b",         "size": 3,  "description": "Root linear velocity in body frame (x, y, z)"},
                    {"name": "root_ang_vel_b",          "size": 3,  "description": "Root angular velocity in body frame (roll, pitch, yaw)"},
                    {"name": "projected_gravity_b",     "size": 3,  "description": "Projected gravity vector in body frame"},
                    {"name": "commands",                "size": 3,  "description": "Velocity commands (vx, vy, yaw_rate)"},
                    {"name": "joint_pos_rel",           "size": 18, "description": "Joint positions relative to default pose"},
                    {"name": "joint_vel",               "size": 18, "description": "Joint velocities"},
                    {"name": "actions",                 "size": 18, "description": "Previous actions"},
                ]
            },
            "actions": [
                {
                    "name": "joint_position_targets",
                    "size": 18,
                    "description": "Joint position targets (scaled deviation from default_joint_pos)",
                    "scale": self.cfg.action_scale,
                }
            ],
            "articulations": {
                "robot": {
                    "num_joints": self._robot.num_joints,
                    "joint_names": self._robot.joint_names,
                }
            },
            "scene": {
                "num_envs": self.num_envs,
                "terrain": str(type(self._terrain).__name__),
                "target_base_height": self.cfg.target_base_height,
            },
        }
