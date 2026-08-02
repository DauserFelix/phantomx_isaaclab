from __future__ import annotations

import os

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

        # All bodies EXCEPT the last link (tibia/foot) — used to terminate the episode if any of
        # these touch the ground (body, coxa, or femur dragging/collapsing is never a valid stance,
        # unlike a tibia foot touching down).
        self._non_foot_body_ids, _ = self._contact_sensor.find_bodies([
            "MP_BODY", "base_link", "c1_.*", "c2_.*", "thigh_.*"
        ])

        # MP_BODY index for height measurement (physical body, 10cm above base_link)
        self._mp_body_idx, _ = self._robot.find_bodies(["MP_BODY"])

        # Tibia body indices in the ROBOT ARTICULATION namespace (self._robot.data.body_pos_w) —
        # NOT to be confused with self._die_body_ids above, which indexes the same body NAMES but
        # in the ContactSensor namespace (self._contact_sensor.data.*). body_pos_w cannot be
        # indexed with self._die_body_ids — a separate call is required.
        # preserve_order=True is REQUIRED here (and below): resolve_matching_names() with the
        # default preserve_order=False returns names in the order they appear in the target's OWN
        # internal body list, not in query order (verified against isaaclab/utils/string.py — the
        # docstring is easy to misread the other way around). self._die_body_ids above does NOT
        # set preserve_order=True, so its element order relative to this array is not guaranteed —
        # the URDF itself declares tibia links as rf,rm,rr,lf,lm,lr, not lf,lm,lr,rf,rm,rr, and
        # whether the ContactSensor's internal body list preserves that order (or e.g. sorts
        # alphabetically) could not be verified in this sandbox (no CUDA/pxr available to inspect
        # the live PhysX body view or the binary USD file). Rather than assume self._die_body_ids
        # happens to be lf,lm,lr,rf,rm,rr-ordered, a SEPARATE, order-safe ContactSensor query is
        # used below (self._swing_contact_body_ids) specifically for this reward.
        self._tibia_robot_body_ids, _ = self._robot.find_bodies([
            "tibia_lf", "tibia_lm", "tibia_lr",  # Left feet
            "tibia_rf", "tibia_rm", "tibia_rr"   # Right feet
        ], preserve_order=True)

        # Order-safe ContactSensor-namespace counterpart to self._tibia_robot_body_ids above (same
        # query, same preserve_order=True) — used only for the swing_foot_height reward's contact
        # mask, kept deliberately separate from self._die_body_ids (whose ordering guarantee
        # relative to the robot-namespace array above could not be verified, see comment above).
        self._swing_contact_body_ids, _ = self._contact_sensor.find_bodies([
            "tibia_lf", "tibia_lm", "tibia_lr",
            "tibia_rf", "tibia_rm", "tibia_rr"
        ], preserve_order=True)

        # Tripod gait indices into the 6-element foot array (order: lf=0, lm=1, lr=2, rf=3, rm=4, rr=5)
        self._TRIPOD_A = [0, 4, 2]  # lf, rm, lr
        self._TRIPOD_B = [3, 1, 5]  # rf, lm, rr

        self._has_stood_up = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Action-latency ring buffer (see cfg.enable_action_latency): holds the last
        # (max_latency+1) processed PD targets per env, so _apply_action() can apply an
        # env-specific delayed target instead of the just-computed one. Sized once for the max
        # possible delay in the configured range — per-env delay itself is drawn in _reset_idx().
        # No-op-shaped even when disabled (buffer still allocated, just never advanced/read) to
        # avoid a second code path; the actual behavior gate is _apply_action()'s cfg check.
        self._action_latency_max_steps = max(self.cfg.action_latency_steps_range)
        self._action_latency_buffer = torch.zeros(
            self._action_latency_max_steps + 1, self.num_envs, self._robot.num_joints,
            device=self.device,
        )
        self._action_latency_buffer_idx = 0
        self._action_latency_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        # State for the "imu_integration" lin_vel_b estimation mode (see cfg.lin_vel_estimation_mode):
        # a leaky integrator of noisy, biased acceleration, mirroring the real hardware pipeline
        # (phantomx_policy_real_data_interface.py), which has no odometry sensor. _lin_vel_imu_bias
        # is a slow random walk (unlike _lin_vel_imu_estimate, it is NEVER reset by contact/attitude
        # info — it purely accumulates), _lin_vel_imu_estimate is the leaky-integrated velocity
        # actually fed into the observation in "imu_integration" mode.
        self._lin_vel_imu_estimate = torch.zeros(self.num_envs, 3, device=self.device)
        self._lin_vel_imu_bias = torch.zeros(self.num_envs, 3, device=self.device)
        self._previous_root_lin_vel_b = torch.zeros(self.num_envs, 3, device=self.device)

        # Accumulator for the true (non-reward) average forward walking speed per episode —
        # separate from _episode_sums, since it's a raw m/s measurement, not a reward term, and
        # must be averaged over the actual number of steps taken, not the nominal episode length.
        self._episode_speed_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # Same pattern as _episode_speed_sum, for the true (non-reward) average MP_BODY height —
        # height_reward only shows how close the height is to the target via an exp-kernel, not
        # the raw height itself, so this gives a directly readable value in TensorBoard.
        self._episode_height_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "action_rate_l2",
                "flat_orientation_l2",
                "alive",
                "height_tracking",
                "movement_penalty",
                "foot_contact",
                "tripod_gait",
                "lazy_legs",
                "swing_foot_height",
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
        self._init_terrain_grid()

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _init_terrain_grid(self):
        """Verteilt die Envs beim Start GLEICHMÄSSIG über die leichtesten Terrain-Rows UND alle
        Cols, statt IsaacLabs Default (torch.randint pro Env, siehe TerrainImporter.
        _compute_env_origins_curriculum) zu übernehmen.

        Hintergrund: mit dem vorherigen max_init_terrain_level=0 landete JEDE Env exakt auf Row 0
        -> alle Envs auf identischer Y-Koordinate in einer einzigen Reihe, der Rest des Feldes
        blieb leer (live in der Sim beobachtet). torch.randint(0, N) für N>0 hätte dasselbe
        Klumpen-Risiko in schwächerer Form (kein Garant für Abdeckung aller Rows/Cols, vor allem
        bei kleinem N relativ zu num_envs). Diese Methode ersetzt das durch ein deterministisches
        Round-Robin-Raster: Env i bekommt Row i % init_terrain_rows und Col (i // init_terrain_rows)
        % num_cols. Row wird zuerst durchgetaktet (statt Col wie in einer naheliegenderen
        Variante), damit auch bei num_envs < init_terrain_rows*num_cols (SAC: 128 Envs vs.
        8*32=256 Zellen) noch ALLE init_terrain_rows Zeilen erreicht werden, nur mit weniger
        Spalten-Abdeckung — Zeilen-Diversität (= Difficulty-Streuung) ist das eigentliche Ziel,
        Spalten-Diversität (= Sub-Terrain-Variante) nur sekundär.

        No-op auf terrain_type="plane" (terrain_origins ist dann None, siehe TerrainImporter.
        configure_env_origins) — Flat-Task-IDs bleiben unverändert.
        """
        if self._terrain.terrain_origins is None:
            return

        num_rows, num_cols = self._terrain.terrain_origins.shape[:2]
        init_rows = max(1, min(self.cfg.init_terrain_rows, num_rows))

        env_idx = torch.arange(self.cfg.terrain.num_envs, device=self.device)
        rows = env_idx % init_rows
        cols = torch.div(env_idx, init_rows, rounding_mode="floor") % num_cols

        self._terrain.terrain_levels = rows.to(torch.long)
        self._terrain.terrain_types = cols.to(torch.long)
        self._terrain.max_terrain_level = num_rows
        self._terrain.env_origins[:] = self._terrain.terrain_origins[
            self._terrain.terrain_levels, self._terrain.terrain_types
        ]

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

        # Action-latency ring buffer (cfg.enable_action_latency): always maintained (cheap write),
        # so toggling the flag doesn't require re-seeding a freshly-allocated buffer with stale
        # zeros. Write the just-computed target at the current write head; _apply_action() below
        # reads back per-env with its own delay offset.
        self._action_latency_buffer[self._action_latency_buffer_idx] = self._processed_actions

    def _apply_action(self):
        if self.cfg.enable_action_latency:
            # Per-env delayed read: env i gets the target written _action_latency_steps[i] writes
            # ago. Buffer has max_steps+1 slots so every configured delay (including the max) has
            # a valid, already-written slot to read from — this only holds as long as
            # action_latency_steps_range's upper bound matches _action_latency_max_steps (set once
            # from the same range at env construction), so the buffer is never read before it was
            # first written for that offset.
            read_idx = (
                self._action_latency_buffer_idx - self._action_latency_steps
            ) % (self._action_latency_max_steps + 1)
            env_idx = torch.arange(self.num_envs, device=self.device)
            target = self._action_latency_buffer[read_idx, env_idx]
        else:
            target = self._processed_actions
        self._robot.set_joint_position_target(target)
        self._action_latency_buffer_idx = (
            self._action_latency_buffer_idx + 1
        ) % (self._action_latency_max_steps + 1)

    def _compute_lin_vel_observation(self) -> torch.Tensor:
        """root_lin_vel_b observation, in one of two modes (cfg.lin_vel_estimation_mode).

        "ground_truth" (default, unchanged prior behavior): the simulator's exact body-frame
        linear velocity plus small IID Gaussian sensor noise — matches what the Gazebo deployment
        gets from a real odometry plugin, but NOT what the real-hardware deployment can obtain
        (no odometry sensor on the physical robot).

        "imu_integration": mirrors phantomx_policy_real_data_interface.py's actual estimation
        path — lin_vel_b is never read directly, only reconstructed by leaky-integrating a noisy,
        BIASED acceleration signal. The bias is a slow random walk (accumulates without bound,
        unlike the noise term, which averages out) — this is the qualitatively different failure
        mode a simple IID-noise-on-ground-truth model cannot represent, since real accelerometer
        bias drifts over the course of an episode rather than resetting every step.
        """
        root_lin_vel_b = self._robot.data.root_lin_vel_b
        if self.cfg.lin_vel_estimation_mode == "ground_truth":
            return root_lin_vel_b + torch.randn_like(root_lin_vel_b) * 0.01

        # "imu_integration": approximate the body-frame acceleration a real IMU would have
        # measured this step via finite difference of the (otherwise inaccessible in this
        # simplified model) ground-truth velocity — a real accelerometer measures acceleration
        # directly, but this finite-difference stands in for it without adding a full rigid-body
        # force model. self._previous_root_lin_vel_b is updated at the end of this branch.
        accel = (root_lin_vel_b - self._previous_root_lin_vel_b) / self.step_dt
        self._previous_root_lin_vel_b = root_lin_vel_b.clone()

        self._lin_vel_imu_bias += torch.randn_like(self._lin_vel_imu_bias) * self.cfg.lin_vel_imu_bias_std
        noisy_accel = accel + self._lin_vel_imu_bias + torch.randn_like(accel) * self.cfg.lin_vel_imu_noise_std

        self._lin_vel_imu_estimate = (
            self.cfg.lin_vel_imu_leak * self._lin_vel_imu_estimate + noisy_accel * self.step_dt
        )
        return self._lin_vel_imu_estimate

    # --------------------- OBSERVATIONS ---------------------
    def _get_observations(self) -> dict:
        # Sensor-Messungen mit Rauschen (Sim-to-Real: modelliert Encoder/IMU-Noise)
        lin_vel = self._compute_lin_vel_observation()
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

        # Periodischer Joint-State-Dump von env 0 fuers Terminal (zum manuellen Abgreifen und
        # z.B. per Hand in ROS spielen, um die aktuelle Pose zu ueberpruefen). Rohe, unverrauschte
        # joint_pos (nicht das obs-jpos_rel), da das die tatsaechliche physikalische Pose ist.
        # Zusaetzlich in eine joint_states.txt im Run-Log-Ordner geschrieben (self.cfg.log_dir,
        # von train.py gesetzt), damit der Verlauf nach Trainingsende nicht nur im Terminal-
        # Scrollback, sondern dauerhaft bei den restlichen Run-Artefakten dokumentiert ist.
        if self.common_step_counter % 50_000 == 0:
            joint_pos_env0 = self._robot.data.joint_pos[0].tolist()
            lin_vel_env0 = self._robot.data.root_lin_vel_b[0].tolist()
            ang_vel_env0 = self._robot.data.root_ang_vel_b[0].tolist()
            mean_base_height = torch.mean(
                self._robot.data.body_pos_w[:, self._mp_body_idx[0], 2] - self._terrain.env_origins[:, 2]
            ).item()
            line = (f"[JOINT_STATE step={self.common_step_counter}] "
                    f"names={self._robot.joint_names} pos={joint_pos_env0} "
                    f"lin_vel_b={lin_vel_env0} ang_vel_b={ang_vel_env0} "
                    f"mean_base_height={mean_base_height:.4f}")
            print(line)
            log_dir = getattr(self.cfg, "log_dir", None)
            if log_dir:
                with open(os.path.join(log_dir, "joint_states.txt"), "a") as f:
                    f.write(line + "\n")

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

        # Raw height accumulation for the true average-body-height metric (nicht Teil des
        # Rewards) — height_reward zeigt nur die Kernel-Nähe zum Ziel, nicht die Rohhöhe selbst.
        self._episode_height_sum += base_height

        # Alive reward
        alive_reward = torch.ones_like(lin_vel_error)

        # Movement penalty: penalizes not moving forward (unconditional — wie Working-Model 21.04.)
        # sowie (PPO, ab Phase 3) proportional zu schnelles Laufen — siehe _compute_movement_penalty.
        # Phasenabhängig (Grenzen identisch zu _apply_velocity_curriculum):
        #   Phase 1 (<75k, Stillstand):        komplett inaktiv (0.0) — kein Zielkommando vorhanden,
        #                                       gegen das getrackt werden könnte.
        #   Phase 2 (75k-300k, halbe Speed):    NUR nach unten (Deficit-Clamp, kein Overshoot-Penalty)
        #                                       — der Roboter darf hier schneller als das
        #                                       Zwischenziel laufen, ohne bestraft zu werden; nur
        #                                       Zu-langsam-Sein kostet Reward.
        #   Phase 3 (>=300k, volle Speed):      volle, symmetrische _compute_movement_penalty (wie
        #                                       bisher) — jetzt muss auch Überschreiten vermieden
        #                                       werden.
        forward_speed = self._robot.data.root_lin_vel_b[:, 0]
        if self.common_step_counter < 75_000:
            movement_penalty = torch.zeros_like(forward_speed)
        elif self.common_step_counter < 300_000:
            movement_penalty = torch.clamp(self._commands[:, 0] - forward_speed, min=0.0)
        else:
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

        # Lazy leg penalty — Beine die dauerhaft (>2.5s) in der Luft hängen
        current_air_times = self._contact_sensor.data.current_air_time[:, self._die_body_ids]
        lazy_legs = (current_air_times > 3.0).float().sum(dim=-1)

        # Swing foot height reward — belohnt Fußhöhe über Terrain NUR während der Schwungphase,
        # geclippt auf swing_foot_height_target. Deckelung ist zwingend (kein "je höher desto
        # besser"), sonst Anreiz zu übertriebenem Anheben; min=0.0 schützt zusätzlich gegen
        # negative Werte durch Terrain-Rauschen.
        # Nutzt bewusst self._swing_contact_body_ids (eigener, preserve_order=True-Query) statt
        # dem oben berechneten foot_contact_bool/self._die_body_ids — deren Elementreihenfolge
        # relativ zu self._tibia_robot_body_ids (Robot-Namespace) konnte in dieser Sandbox nicht
        # verifiziert werden (siehe Kommentar im __init__). Kostet einen zusätzlichen, aber
        # günstigen net_forces_w-Tensor-Read pro Step.
        swing_contact_forces = self._contact_sensor.data.net_forces_w[:, self._swing_contact_body_ids, :]
        swing_contact_bool = torch.norm(swing_contact_forces, dim=-1) > 1.0
        foot_z = self._robot.data.body_pos_w[:, self._tibia_robot_body_ids, 2] - terrain_z.unsqueeze(-1)
        swing_mask = ~swing_contact_bool
        foot_height_clipped = torch.clamp(foot_z, min=0.0, max=self.cfg.swing_foot_height_target)
        swing_foot_height = torch.sum(foot_height_clipped * swing_mask.float(), dim=-1)

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped  * self.cfg.lin_vel_reward_scale       * self.step_dt,
            "track_ang_vel_z_exp":  yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale       * self.step_dt,
            "lin_vel_z_l2":         z_vel_error            * self.cfg.z_vel_reward_scale          * self.step_dt,
            "ang_vel_xy_l2":        ang_vel_error          * self.cfg.ang_vel_reward_scale        * self.step_dt,
            "action_rate_l2":       action_rate            * self.cfg.action_rate_reward_scale    * self.step_dt,
            "flat_orientation_l2":  flat_orientation       * self.cfg.flat_orientation_reward_scale * self.step_dt,
            "alive":                alive_reward           * self.cfg.alive_reward_scale          * self.step_dt,
            "height_tracking":      height_reward          * self.cfg.height_reward_scale         * self.step_dt,
            "movement_penalty":     -movement_penalty      * self.cfg.movement_penalty_scale      * self.step_dt,
            "foot_contact":         foot_contact_reward    * self.cfg.foot_contact_reward_scale   * self.step_dt,
            "tripod_gait":          tripod_score           * self.cfg.tripod_gait_reward_scale    * self.step_dt,
            "lazy_legs":           -lazy_legs              * self.cfg.lazy_leg_penalty_scale      * self.step_dt,
            "swing_foot_height":    swing_foot_height       * self.cfg.swing_foot_height_reward_scale * self.step_dt,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

        for key, value in rewards.items():
            self._episode_sums[key] += value

        return reward

    def _compute_movement_penalty(self, forward_speed: torch.Tensor, command_speed: torch.Tensor) -> torch.Tensor:
        if not self.cfg.continuous_movement_penalty:
            # Symmetrisch-proportionales |v_actual - cmd| statt binärem Deficit (hart, 0/1) +
            # linearem Overshoot (schwach) — der alte Split war für ein FESTES cfg.movement_speed_x
            # kalibriert (deficit_penalty=1.0 bestrafte jedes Unterschreiten des Curriculum-Maximums
            # gleich stark, unabhängig vom Abstand; overshoot_penalty wuchs dagegen nur linear und
            # war bei kleinen cmd-Werten kaum spürbar, siehe Kernel-Analyse). Sobald command_speed
            # nicht mehr konstant ist (Ranged-Command-Sampling), verstärkt diese Asymmetrie sich zu
            # einem reinen "immer maximal schnell laufen"-Anreiz statt echtem Tracking — analog zum
            # bereits in Log 07-14 §3 für dieselbe Formel gefundenen und behobenen Overshoot-Bias.
            # torch.abs() macht command_speed selbst zum eindeutigen, symmetrischen Minimum.
            return torch.abs(forward_speed - command_speed)

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

        # Terminiert, sobald irgendein Body außer der Tibia (letztes Glied) Bodenkontakt hat —
        # Körper/Coxa/Femur schleifen ist nie eine gültige Stützpose, anders als ein aufsetzender Fuß.
        non_foot_forces = self._contact_sensor.data.net_forces_w[:, self._non_foot_body_ids, :]
        non_foot_contact = torch.any(torch.norm(non_foot_forces, dim=-1) > 1.0, dim=-1)

        died = (
            (mp_body_height < self.cfg.termination_height) |
            (mp_body_height > 0.40) |
            (tilt > self.cfg.termination_tilt) |
            non_foot_contact
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

        # Re-draw the per-env action-delay for the upcoming episode (cfg.enable_action_latency).
        # Drawn even when disabled — cheap, and keeps _apply_action()'s indexing logic branch-free
        # regardless of when the flag is toggled mid-experimentation.
        low, high = self.cfg.action_latency_steps_range
        self._action_latency_steps[env_ids] = torch.randint(
            low, high + 1, (len(env_ids),), device=self.device
        )
        # Seed the buffer's reset slots with the post-reset default pose target, not stale zeros —
        # otherwise a freshly reset env would apply a bogus all-zero PD target for up to
        # action_latency_max_steps until the buffer "catches up" with real actions.
        self._action_latency_buffer[:, env_ids] = self._robot.data.default_joint_pos[env_ids]

        # Reset the IMU-integration lin_vel_b estimator state (cfg.lin_vel_estimation_mode) —
        # a fresh episode must not carry over accumulated bias/velocity from a previous rollout.
        self._lin_vel_imu_estimate[env_ids] = 0.0
        self._lin_vel_imu_bias[env_ids] = 0.0
        self._previous_root_lin_vel_b[env_ids] = 0.0

        # Terrain-Difficulty-Umstufung MUSS vor _apply_velocity_curriculum laufen — sie braucht
        # noch die alten (während der beendeten Episode aktiven) self._commands, die die
        # Velocity-Curriculum-Methode direkt danach überschreibt.
        self._apply_terrain_curriculum(env_ids)
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

        # True average MP_BODY height (m) over the episode that just ended — same per-step-average
        # pattern as avg_forward_speed_mps above, directly comparable to target_base_height.
        avg_body_height = self._episode_height_sum[env_ids] / episode_steps
        extras["Metrics/avg_body_height_m"] = torch.mean(avg_body_height)
        self._episode_height_sum[env_ids] = 0.0

        self.extras["log"] = dict()
        self.extras["log"].update(extras)

        extras = dict()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)

    def _apply_velocity_curriculum(self, env_ids: torch.Tensor):
        """Setzt self._commands[env_ids] gemäß Trainings-Curriculum.

        movement_speed_x/yaw_rotation_speed_x aus der Cfg sind die einzige Quelle der
        Wahrheit für die Ziel-Geschwindigkeiten. Einheitliches 3-Stufen-Curriculum für PPO
        UND SAC (kein algorithmus-spezifischer Pfad mehr): bis 75k Steps steht der Roboter
        (Command=0), von 75k bis 300k halbe Zielgeschwindigkeit (sanfterer Übergang aus dem
        Stillstand als ein direkter Sprung auf volle Speed), ab 300k volle Zielgeschwindigkeit
        inkl. randomisiertem Yaw.
        """
        steps = self.common_step_counter
        max_speed = self.cfg.movement_speed_x
        max_yaw = self.cfg.yaw_rotation_speed_x

        if steps < 75_000:
            self._commands[env_ids] = 0.0
        elif steps < 300_000:
            self._commands[env_ids] = 0.0
            self._commands[env_ids, 0] = max_speed / 2.0
        else:
            self._commands[env_ids] = 0.0
            self._commands[env_ids, 0] = max_speed
            self._commands[env_ids, 2] = torch.zeros_like(
                self._commands[env_ids, 2]
            ).uniform_(-max_yaw, max_yaw)

    def _apply_terrain_curriculum(self, env_ids: torch.Tensor):
        """Stuft Envs anhand der in der gerade beendeten Episode zurückgelegten Distanz auf
        leichtere/schwerere Terrain-Rows um (IsaacLab-Standardmuster, siehe
        isaaclab_tasks/manager_based/locomotion/velocity/mdp/curriculums.py:terrain_levels_vel).

        Muss VOR _apply_velocity_curriculum() aufgerufen werden, da self._commands an dieser
        Stelle noch die während der beendeten Episode aktiven Kommandos enthält (move_down
        braucht genau diese, nicht die neu zu setzenden für die kommende Episode).

        No-op auf terrain_type="plane" (terrain_generator ist dann None) — Flat-Task-IDs
        bleiben unverändert.

        No-op beim allerersten Reset (common_step_counter == 0, ausgelöst durch DirectRLEnv.
        __init__ -> self.reset(), bevor der Roboter je einen Physik-Schritt gemacht hat):
        root_pos_w ist zu diesem Zeitpunkt noch nicht auf eine sinnvolle Pose gesetzt (Default-
        USD-Prim-Pose, meist nahe dem Welt-Ursprung), während env_origins bei generator-Terrain
        bereits reale, teils weit vom Ursprung entfernte Koordinaten sind (Terrain-Grid bis zu
        ~250m). Ohne diesen Guard wird distance dadurch riesig -> move_up feuert sofort für
        JEDE Env, noch bevor max_init_terrain_level=0 überhaupt einen Trainingsschritt bewirken
        konnte — die Envs werden im Init-Reset direkt wieder auf zufällige, teils maximale
        Terrain-Level hochgestuft (das beobachtete "Roboter fällt sofort beim Spawn durch den
        Boden").
        """
        if self._terrain.cfg.terrain_generator is None:
            return
        if self.common_step_counter == 0:
            return

        distance = torch.norm(
            self._robot.data.root_pos_w[env_ids, :2] - self._terrain.env_origins[env_ids, :2],
            dim=1,
        )
        move_up = distance > self._terrain.cfg.terrain_generator.size[0] / 2
        move_down = distance < torch.norm(self._commands[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5
        move_down &= ~move_up
        self._terrain.update_env_origins(env_ids, move_up, move_down)

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
