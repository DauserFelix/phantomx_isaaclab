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

        # Femur (j_thigh_*) joint indices — default pose has all 6 at +0.5 rad (see
        # isaaclab_assets/robots/phantomx.py). Used by the femur-flip penalty below.
        self._femur_joint_ids, _ = self._robot.find_joints(["j_thigh_.*"])

        # Tripod gait indices into the 6-element foot array (order: lf=0, lm=1, lr=2, rf=3, rm=4, rr=5)
        self._TRIPOD_A = [0, 4, 2]  # lf, rm, lr
        self._TRIPOD_B = [3, 1, 5]  # rf, lm, rr

        self._has_stood_up = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Accumulator for the true (non-reward) average forward walking speed per episode —
        # separate from _episode_sums, since it's a raw m/s measurement, not a reward term, and
        # must be averaged over the actual number of steps taken, not the nominal episode length.
        self._episode_speed_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

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
                "femur_flip_l2",
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
        # previous_actions vor dem Ueberschreiben von self._actions einfangen — Cross-Step-Lag
        # bleibt identisch zur alten Zuweisung in _get_observations() (siehe Verifikations-Script),
        # aber die Zuweisung sitzt jetzt direkt neben der Stelle, die sie betrifft.
        self._previous_actions = self._actions.clone()

        if self.cfg.strict_action_pipeline:
            self._actions = torch.clamp(actions, -1.0, 1.0)
        else:
            self._actions = actions.clone()

        q_def = self._robot.data.default_joint_pos
        processed_actions = q_def + self.cfg.action_scale * self._actions
        if self.cfg.strict_action_pipeline:
            processed_actions = torch.clamp(
                processed_actions,
                q_def - self.cfg.joint_pos_limit,
                q_def + self.cfg.joint_pos_limit,
            )
        self._processed_actions = processed_actions

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    # --------------------- OBSERVATIONS ---------------------
    def _get_observations(self) -> dict:
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
        lin_vel_error_mapped = torch.exp(-lin_vel_error / self.cfg.lin_vel_kernel_width)

        # yaw rate tracking
        yaw_rate_error = torch.square(
            self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2]
        )
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / self.cfg.ang_vel_kernel_width)

        # z velocity penalty (body should not bounce)
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])

        # angular velocity x/y penalty (no roll/pitch)
        ang_vel_error = torch.sum(
            torch.square(self._robot.data.root_ang_vel_b[:, :2]),
            dim=1
        )

        # joint torques penalty
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)

        # joint acceleration penalty — bestraft ruckartige/hektische Bewegung. Vor der Skalierung
        # auf joint_accel_clamp gedeckelt: normale Bewegungsdynamik bleibt unclamped (voller
        # Gradient), aber ein einzelner Physik-Instabilitäts-Spike kann den Batch nicht mehr
        # vergiften wie vor dem Q-Divergenz-Fix (siehe clamp_reward — gleiches Prinzip, hier
        # auf den Einzelterm statt auf die Summe angewendet).
        joint_accel = torch.clamp(
            torch.sum(torch.square(self._robot.data.joint_acc), dim=1),
            max=self.cfg.joint_accel_clamp,
        )

        # action rate penalty
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)

        # flat orientation penalty
        flat_orientation = torch.sum(
            torch.square(self._robot.data.projected_gravity_b[:, :2]),
            dim=1
        )

        # MP_BODY height tracking — relativ zum lokalen Terrain-Ursprung (No-op solange
        # terrain_type="plane", relevant sobald generiertes Terrain mit env-abhängigem
        # Z-Offset genutzt wird)
        terrain_z = self._terrain.env_origins[:, 2]
        base_height = self._robot.data.body_pos_w[:, self._mp_body_idx[0], 2] - terrain_z
        height_error = torch.square(base_height - self.cfg.target_base_height)
        height_reward = torch.exp(-height_error / 0.02)

        # Alive reward
        alive_reward = torch.ones_like(lin_vel_error)

        # Movement penalty: penalizes not moving forward (unconditional — wie Working-Model 21.04.)
        forward_speed = self._robot.data.root_lin_vel_b[:, 0]
        movement_penalty = self._compute_movement_penalty(forward_speed, self._commands[:, 0])

        # Raw speed accumulation for the true average-forward-speed metric (nicht Teil des
        # Rewards) — nur die x-Komponente (Vorwärtsrichtung), konsistent mit movement_penalty/
        # commands[:, 0]. Kein seitlicher Anteil.
        self._episode_speed_sum += forward_speed

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

        # Femur flip penalty — j_thigh_* steht per Default bei +0.5 rad. Rotiert ein Femur unter
        # 0, kippt das Bein in eine anatomisch verkehrte Konfiguration ("Femur zeigt nach unten
        # statt nach oben"), bei der die Tibia flach auf dem Boden liegt und nur noch geschleift
        # statt normal aufgesetzt wird (beobachtet: 4/6 Beine korrekt, 2/6 gekippt). Bestraft nur
        # die negative Auslenkung (relu), damit normale positive Federung/Schrittbewegung des
        # Femurs beim Gehen nicht eingeschränkt wird.
        femur_pos = self._robot.data.joint_pos[:, self._femur_joint_ids]
        femur_flip_penalty = torch.sum(torch.clamp(-femur_pos, min=0.0), dim=-1)

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
            "height_tracking":      height_reward          * self.cfg.height_reward_scale         * self.step_dt,
            "movement_penalty":     -movement_penalty      * self.cfg.movement_penalty_scale      * self.step_dt,
            "foot_contact":         foot_contact_reward    * self.cfg.foot_contact_reward_scale   * self.step_dt,
            "tripod_gait":          tripod_score           * self.cfg.tripod_gait_reward_scale    * self.step_dt,
            "lazy_legs":           -lazy_legs              * self.cfg.lazy_leg_penalty_scale      * self.step_dt,
            "femur_flip_l2":        femur_flip_penalty     * self.cfg.femur_flip_penalty_scale    * self.step_dt,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

        # Clamp the total reward against rare physics-instability outliers (e.g. a single-step
        # solver blow-up driving joint_acc/applied_torque to extreme, likely non-physical
        # values — dof_acc_l2/dof_torques_l2 are unclamped squared terms, so one such step can
        # dominate an entire off-policy replay batch). Deliberately clamps only the *summed*
        # scalar, not the individual `rewards` dict entries below, so Episode_Reward/* logging
        # stays unclamped and can still pinpoint which term actually spiked. SAC-only (see
        # PhantomxThesisSACEnvCfg.clamp_reward) — PPO's reward path is unchanged by default.
        if self.cfg.clamp_reward:
            reward = torch.clamp(reward, -self.cfg.reward_clamp_value, self.cfg.reward_clamp_value)

        if self.common_step_counter % 100 == 0:
            for key, value in rewards.items():
                print(f"{key:22s} min={value.min().item():8.3f}  mean={value.mean().item():8.3f}")

        for key, value in rewards.items():
            self._episode_sums[key] += value

        return reward

    def _compute_movement_penalty(self, forward_speed: torch.Tensor, command_speed: torch.Tensor) -> torch.Tensor:
        if not self.cfg.continuous_movement_penalty:
            # Binary: wer nicht schneller als das aktuell aktive Curriculum-Kommando läuft,
            # bekommt volle Strafe (vorher fix gegen cfg.movement_speed_x geprüft — das ist
            # der Max-Wert der letzten Curriculum-Stufe, in früheren Stufen mit kleinerem
            # Kommando also unerreichbar und damit fast permanent aktiv).
            is_moving_forward = forward_speed > command_speed
            return (~is_moving_forward).float()

        # Continuous: proportionales Defizit zwischen Kommando und Ist-Geschwindigkeit, nur wenn
        # überhaupt ein nennenswertes Vorwärtskommando anliegt (Schwelle relativ zu movement_speed_x,
        # sonst wäre der Gate bei kleinem movement_speed_x tot).
        has_forward_command = command_speed.abs() > 0.1 * self.cfg.movement_speed_x
        velocity_deficit = torch.clamp(command_speed - forward_speed, min=0.0)
        return velocity_deficit * has_forward_command.float()

    # --------------------- TERMINATION ---------------------
    def _get_dones(self):
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # MP_BODY height relative zum lokalen Terrain-Ursprung (No-op solange terrain_type="plane")
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

        # Capture the PRE-reset episode length now — super()._reset_idx() below and the manual
        # randomization further down both overwrite episode_length_buf, so this is the only
        # point where it still reflects how long the just-ended episode actually ran (needed to
        # turn _episode_speed_sum into a true per-step average, not a length-normalized reward).
        episode_steps = self.episode_length_buf[env_ids].clamp(min=1).float()

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

        self._apply_velocity_curriculum(env_ids)

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

        # True average forward walking speed (m/s) over the episode that just ended — divided
        # by the actual number of steps taken (episode_steps, captured before any reset
        # touched episode_length_buf), not by the nominal max episode length like the reward
        # terms above. A robot that dies after 1s at 5cm/s and one that walks 40s at 5cm/s both
        # show ~0.05 here, unlike Episode_Reward/* which would differ a lot between the two.
        avg_forward_speed = self._episode_speed_sum[env_ids] / episode_steps
        extras["Metrics/avg_forward_speed_mps"] = torch.mean(avg_forward_speed)
        self._episode_speed_sum[env_ids] = 0.0

        self.extras["log"] = dict()
        self.extras["log"].update(extras)

        extras = dict()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)

    def _apply_velocity_curriculum(self, env_ids: torch.Tensor):
        """Setzt self._commands[env_ids] gemäß Trainings-Curriculum.

        movement_speed_x/yaw_rotation_speed_x aus der Cfg sind die einzige Quelle der
        Wahrheit für die Ziel-Geschwindigkeiten. Zwei Rampenformen je nach
        cfg.use_sac_curriculum — Default (False) reproduziert exakt das aktuelle
        PPO-Verhalten (100k/350k-Stufen).
        """
        steps = self.common_step_counter
        max_speed = self.cfg.movement_speed_x
        max_yaw = self.cfg.yaw_rotation_speed_x

        if not self.cfg.use_sac_curriculum:
            if steps < 100_000:
                self._commands[env_ids] = 0.0
            elif steps < 350_000:
                self._commands[env_ids] = 0.0
                self._commands[env_ids, 0] = max_speed / 2
            else:
                self._commands[env_ids] = 0.0
                self._commands[env_ids, 0] = max_speed
                self._commands[env_ids, 2] = torch.zeros_like(
                    self._commands[env_ids, 2]
                ).uniform_(-max_yaw, max_yaw)
            return

        # SAC-Variante: Zwei-Phasen-Curriculum "hoch → niedrig", invertiert ggü. PPO. Grund:
        # bei sehr niedrigen Zielgeschwindigkeiten (movement_speed_x=0.05 für SAC) ist der
        # exp(-error/kernel_width)-Tracking-Reward für Stillstand fast so hoch wie für echtes
        # Laufen (z.B. exp(-0.03²/0.1) ≈ 0.991 bei Kommando 0.03 m/s) — kaum Lernanreiz,
        # überhaupt loszulaufen. BOOTSTRAP_SPEED liegt deutlich über dem eigentlichen Ziel,
        # damit der Reward-Unterschied zwischen Stehen und Laufen groß genug ist, um ein
        # echtes Gangmuster zu erzwingen; Phase 2 transferiert dieses Gangmuster dann auf die
        # tatsächliche Zielgeschwindigkeit herunter, statt sie von Null unter schwachem
        # Gradienten zu lernen.
        BOOTSTRAP_SPEED = 0.15   # m/s — unverifiziert, ob physisch komfortabel erreichbar
                                # (Aktuator velocity_limit=0.8 rad/s in phantomx.py); in
                                # TensorBoard (track_lin_vel_xy_exp, Episodenlänge) beobachten
        if steps < 300_000:
            self._commands[env_ids] = 0.0
            self._commands[env_ids, 0] = BOOTSTRAP_SPEED
        else:
            self._commands[env_ids] = 0.0
            self._commands[env_ids, 0] = max_speed
            if max_yaw > 0:
                self._commands[env_ids, 2] = torch.zeros_like(
                    self._commands[env_ids, 2]
                ).uniform_(-max_yaw, max_yaw)

    def set_curriculum_step(self, step: int) -> None:
        """Override common_step_counter, z.B. um das Velocity-Curriculum nach einem Checkpoint-
        Resume an der richtigen Stelle fortzusetzen (siehe train.py --resume_step). Wird nur
        aufgerufen, wenn --resume_step explizit gesetzt ist — der normale Trainingspfad (Step 0,
        kein Resume) ruft diese Methode nie auf und bleibt unverändert."""
        self.common_step_counter = step

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
