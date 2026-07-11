# phantomx_thesis_env_cfg.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains import TerrainGeneratorCfg
import isaaclab.terrains.trimesh.mesh_terrains_cfg as mesh_gen
import isaaclab.terrains.height_field.hf_terrains_cfg as hf_gen
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg
from isaaclab_assets.robots.phantomx import PHANTOMX_CFG  # isort: skip

# =====================================================
# TERRAIN CONFIGURATION
# =====================================================
# Unstrukturiertes Terrain aus mehreren Sub-Terrain-Typen.
# Jeder Typ hat eine 'proportion' (Anteil am Gesamtterrain).
# 'curriculum=True' bedeutet: leichtere Terrains zuerst,
# der Roboter wird nach Performance auf schwierigere versetzt.

# ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
#     seed=42,
#     size=(8.0, 8.0),            # Größe jedes Sub-Terrains in Metern
#     border_width=0.5,          # Breiter Rand damit Roboter nicht rausfällt
#     num_rows=30,                # Anzahl Terrain-Reihen (Difficulty-Levels)
#     num_cols=32,                # Anzahl Terrain-Spalten (Variationen pro Level)
#     horizontal_scale=0.1,       # Auflösung des Height-Fields (m/pixel)
#     vertical_scale=0.005,       # Vertikale Skalierung (m/unit)
#     slope_threshold=0.75,       # Max Steigung bevor Terrain als Wand gilt
#     difficulty_range=(0.0, 1.0),
#     use_cache=False,
#     curriculum=False,            # Curriculum: einfach → schwer
#     sub_terrains={
#         # Flaches Terrain als Einstieg (20%)
#         "flat": mesh_gen.MeshPlaneTerrainCfg(
#             proportion=0.2,
#         ),
#         # Zufälliges Rauschen - leicht uneben (20%)
#         # Gut für Hexapod: simuliert Gras/Kies/unebenen Boden
#         "random_rough": hf_gen.HfRandomUniformTerrainCfg(
#             proportion=0.2,
#             noise_range=(0.02, 0.08),   # Höhe der Unebenheiten in Metern
#             noise_step=0.02,
#             border_width=0.25,
#         ),
#         # Diskrete Hindernisse - Klötze/Steine (20%)
#         # Herausfordernd für Hexapod: Beine müssen hochheben
#         "discrete_obstacles": hf_gen.HfDiscreteObstaclesTerrainCfg(
#             proportion=0.2,
#             obstacle_height_mode="fixed",
#             obstacle_width_range=(0.05, 0.2),   # Breite der Hindernisse
#             obstacle_height_range=(0.02, 0.06), # Höhe: konservativ für Hexapod
#             num_obstacles=60,
#             platform_width=2.0,
#         ),
#         # Geneigte Pyramide (20%)
#         # Trainiert Gleichgewicht auf Schrägen
#         "pyramid_slope": hf_gen.HfPyramidSlopedTerrainCfg(
#             proportion=0.2,
#             slope_range=(0.0, 0.3),     # Neigungswinkel in rad (0.3 ≈ 17°)
#             platform_width=2.0,
#             border_width=0.25,
#         ),
#         # Treppenstufen (20%)
#         # Schwierigste Variante - Beine müssen klar heben
#         "pyramid_stairs": mesh_gen.MeshPyramidStairsTerrainCfg(
#             proportion=0.2,
#             step_height_range=(0.02, 0.08), # Stufenhöhe: klein für Hexapod
#             step_width=0.3,
#             platform_width=3.0,
#             border_width=1.0,
#             holes=False,
#         ),
#     },
# )




GRAVEL_TERRAIN_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(8.0, 8.0),
    border_width=0.5,
    num_rows=16,
    num_cols=32,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    curriculum=False,
    sub_terrains={
        "flat": mesh_gen.MeshPlaneTerrainCfg(proportion=0.2),
        "gravel": hf_gen.HfRandomUniformTerrainCfg(
            proportion=0.8,
            noise_range=(0.01, 0.04),
            noise_step=0.01,
            border_width=0.25,
        ),
    },
)


@configclass
class EventCfg:
    """Configuration for randomization."""
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.7, 1.0),
            "dynamic_friction_range": (0.5, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="MP_BODY"),
            "mass_distribution_params": (-0.5, 0.5),
            "operation": "add",
        },
    )


@configclass
class PhantomxThesisEnvCfg(DirectRLEnvCfg):
    # =====================================================
    # ENVIRONMENT SETUP
    # =====================================================
    episode_length_s = 40.0
    decimation = 4
    action_scale = 0.5        # Policy gibt direkte ±0.5 rad Abweichung von Default-Pose
    joint_pos_limit: float = 0.5235  # ±30° um Default-Stellung (π/6) — verhindert mechanisch gefährliche Posen
    action_space = 18  # PhantomX: 6 legs × 3 joints = 18 DOF

    # Observation space:
    #   root_lin_vel_b (3) + root_ang_vel_b (3) + projected_gravity_b (3)
    #   + commands (3) + joint_pos_offset (18) + joint_vel (18) + actions (18)
    #   Total = 66
    observation_space = 66
    state_space = 0

    obs_groups = {
        "actor": "policy",
        "critic": "policy",
    }

    # =====================================================
    # SIMULATION
    # =====================================================
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # =====================================================
    # TERRAIN - Unstrukturiertes Rough Terrain
    # =====================================================
    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="generator",           # "generator" statt "plane"
    #     terrain_generator=ROUGH_TERRAINS_CFG,
    #     collision_group=-1,
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="multiply",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #         restitution=0.0,
    #     ),
    #     visual_material=sim_utils.MdlFileCfg(
    #         mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
    #         project_uvw=True,
    #     ),
    #     debug_vis=False,
    # )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=5,
        update_period=0.005,
        track_air_time=True,
    )

    # =====================================================
    # SCENE
    # =====================================================
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=512,
        env_spacing=3.0,
        replicate_physics=True,
    )

    # =====================================================
    # EVENTS (RANDOMIZATION)
    # =====================================================
    events: EventCfg = EventCfg()

    # =====================================================
    # ROBOT Movement Params
    # =====================================================
    robot: ArticulationCfg = PHANTOMX_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    target_base_height = 0.20    # MP_BODY at normal standing height (~20cm above ground)
    movement_speed_x = 0.10      # 10 cm/s — Wert aus funktionierendem Modell (21.04.)
    yaw_rotation_speed_x = 0.0   # 0 rad/s

    # =====================================================
    # REWARD SCALES - TUNED FOR HEXAPOD LOCOMOTION
    # =====================================================
    #🎯 TRACKING REWARDS (positive)
    lin_vel_reward_scale = 10.0
    yaw_rate_reward_scale = 4.0

    height_reward_scale = 0.1    # kein step_dt in env → 0.1/step max (wie 21.04. Working-Model)

    # 🚫 PENALTIES (negative)
    z_vel_reward_scale = -2.0
    ang_vel_reward_scale = -5
    joint_torque_reward_scale = -2e-5
    joint_accel_reward_scale = -2.5e-7
    action_rate_reward_scale = -0.02
    flat_orientation_reward_scale = -3.0

    movement_penalty_scale = 10.0

    alive_reward_scale = 0.3

    # 🦿 FOOT CONTACT REWARD — Bonus für stabile Stützbasis (≥3 Beine am Boden)
    foot_contact_reward_scale = 1.0

    # 🔄 TRIPOD GAIT REWARD — belohnt wechselndes 3-3 Kontaktmuster (alle Beine aktiv)
    tripod_gait_reward_scale = 2.0

    # 😴 LAZY LEG PENALTY — Strafe für Beine die >1s dauerhaft in der Luft hängen
    lazy_leg_penalty_scale = 0.5

    # # =====================================================
    # # Optuna Hyperparameter Tuning - Werte werden per Environment Variable gesetzt bzw angepasst (10.06.2025)
    # # =====================================================
    # # =====================================================
    # # PRIMÄRE REWARDS — hardcoded, Optuna darf nicht ändern
    # # =====================================================
    # lin_vel_reward_scale: float = 15.0      # hoch und fest — Roboter MUSS laufen
    # alive_reward_scale: float = 0.5         # moderat und fest
    # height_reward_scale: float = 2.0        # fest — wichtig für Stabilität
    # yaw_rate_reward_scale: float = 4.0      # fest

    # # =====================================================
    # # PENALTIES — hardcoded auf sichere Werte
    # # =====================================================
    # movement_penalty_scale: float = 0.0     # deaktiviert — lin_vel reicht
    # z_vel_reward_scale: float = -2.0        # fest
    # joint_accel_reward_scale: float = -2.5e-7  # fest

    # # =====================================================
    # # SEKUNDÄRE REWARDS — Optuna optimiert WIE der Roboter läuft
    # # =====================================================
    # flat_orientation_reward_scale: float = float(
    #     os.environ.get("OPTUNA_ORIENTATION_SCALE", "-3.0")
    # )
    # joint_torque_reward_scale: float = float(
    #     os.environ.get("OPTUNA_TORQUE_SCALE", "-2e-5")
    # )
    # action_rate_reward_scale: float = float(
    #     os.environ.get("OPTUNA_ACTION_RATE_SCALE", "-0.02")
    # )
    # ang_vel_reward_scale: float = float(
    #     os.environ.get("OPTUNA_ANG_VEL_SCALE", "-5.0")
    # )

    # =====================================================
    # TERMINATION THRESHOLDS - RELAXED FOR LEARNING
    # =====================================================
    termination_height = 0.1    # MP_BODY < 15cm → kollabiert (≙ base_link < 5cm + 10cm Offset)
    termination_tilt = 0.4     # gx²+gy² > 0.40 → ~39° Neigung — lockerer für Early Training