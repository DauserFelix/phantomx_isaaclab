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
from isaaclab_assets.robots.phantomx import PHANTOMX_CFG  # isort: skipF

# =====================================================
# TERRAIN CONFIGURATION
# =====================================================
# Unstrukturiertes Terrain aus mehreren Sub-Terrain-Typen.
# Jeder Typ hat eine 'proportion' (Anteil am Gesamtterrain).
# 'curriculum=True' bedeutet: leichtere Terrains zuerst,
# der Roboter wird nach Performance auf schwierigere versetzt.

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=1,
    size=(8.0, 8.0),            # Größe jedes Sub-Terrains in Metern
    border_width=0.5,          # Breiter Rand damit Roboter nicht rausfällt
    num_rows=30,                # Anzahl Terrain-Reihen (Difficulty-Levels)
    num_cols=32,                # Anzahl Terrain-Spalten (Variationen pro Level)
    horizontal_scale=0.1,       # Auflösung des Height-Fields (m/pixel)
    vertical_scale=0.005,       # Vertikale Skalierung (m/unit)
    slope_threshold=0.75,       # Max Steigung bevor Terrain als Wand gilt
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    curriculum=True,            # Curriculum: einfach → schwer
    sub_terrains={
        # Flaches Terrain als Einstieg (20%)
        "flat": mesh_gen.MeshPlaneTerrainCfg(
            proportion=0.2,
        ),
        # Zufälliges Rauschen - leicht uneben (20%)
        # Gut für Hexapod: simuliert Gras/Kies/unebenen Boden
        # 0.03-0.12 -> 0.02-0.07 (~60% der vorherigen Werte, auf User-Wunsch entschärft): das
        # 0.12-Maximum lag zu nah an target_base_height=0.11/termination_height=0.10 — Rauschen
        # allein konnte fast die gesamte Standhöhe auffressen. 0.07 lässt spürbaren Sicherheits-
        # abstand, bleibt aber klar uneben (>0 wie ursprünglich, kein Rückbau auf reine Ebene).
        "random_rough": hf_gen.HfRandomUniformTerrainCfg(
            proportion=0.2,
            noise_range=(0.02, 0.07),   # Höhe der Unebenheiten in Metern
            noise_step=0.02,
            border_width=0.25,
        ),
        # Diskrete Hindernisse - Klötze/Steine (20%)
        # Herausfordernd für Hexapod: Beine müssen hochheben
        # 0.03-0.09 -> 0.02-0.055 (~60% der vorherigen Werte, gleiche Entschärfung wie bei den
        # Gravel-Blöcken unten, auf User-Wunsch): bleibt deutlich unter termination_height=0.10.
        "discrete_obstacles": hf_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.2,
            obstacle_height_mode="fixed",
            obstacle_width_range=(0.05, 0.2),   # Breite der Hindernisse
            obstacle_height_range=(0.02, 0.055), # Höhe: konservativ für Hexapod
            num_obstacles=60,
            platform_width=2.0,
        ),
        # Feiner Gravel/Kies (20%) — ersetzt pyramid_slope (Pyramiden auf User-Wunsch entfernt,
        # wenig Trainingsnutzen für den Hexapod). Etwas gröber als der obere "random_rough"-Block,
        # eigener Name nur wegen num_cols-Aufteilung (proportion) — HfRandomUniformTerrainCfg
        # ignoriert den difficulty-Parameter (siehe IsaacLab-Doku), Differenzierung fein/grob läuft
        # daher rein über fixe noise_range, nicht über die Curriculum-Row.
        # 0.02-0.10 -> 0.015-0.06 (~60%, gleiche Entschärfung wie random_rough oben).
        "gravel_fine": hf_gen.HfRandomUniformTerrainCfg(
            proportion=0.2,
            noise_range=(0.015, 0.06),   # etwas feiner als gravel_coarse unten
            noise_step=0.01,
            border_width=0.25,
        ),
        # Grober Gravel/Schotter (20%) — ersetzt pyramid_stairs (gleicher Grund wie oben).
        # 0.05-0.12 -> 0.03-0.07 (~60%, auf User-Wunsch entschärft): bleibt spürbar gröber als
        # gravel_fine, aber mit deutlicherem Sicherheitsabstand zu termination_height=0.10 als
        # zuvor (0.12 lag zu nah an der Standhöhe).
        "gravel_coarse": hf_gen.HfRandomUniformTerrainCfg(
            proportion=0.2,
            noise_range=(0.03, 0.07),   # grober/unregelmäßiger als gravel_fine
            noise_step=0.02,
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
            # War (0.25,0.45)/(0.2,0.35) für PLA-Füße auf Holzboden. Hardware jetzt umgebaut:
            # Gummistopper auf den Tibia-Spitzen, Untergrund glattes Parkett — griffiger als
            # blankes PLA auf Holz, aber "glatt" begrenzt den Wert bewusst unter dem alten,
            # noch früheren Ausgangswert (0.7,1.0)/(0.5,0.8). Kombiniert mit Boden-mu=1.0
            # (multiply) ergibt das effektiv denselben Bereich wie unten angegeben.
            "static_friction_range": (0.4, 0.6),
            "dynamic_friction_range": (0.35, 0.5),
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
    episode_length_s = 80.0  # 4000 Policy-Steps bei 50Hz (decimation=4, sim.dt=1/200).
    decimation = 4
    action_scale = 0.5       # Policy gibt direkte ±0.75 rad Abweichung von Default-Pose
    joint_pos_limit: float = 1.0  # ≈57.3° um Default-Stellung — proportional mit action_scale mitgezogen
                                   # (0.75/1.0, dieselbe relative Marge wie vorher 0.5/0.7854), damit das
                                   # Clamp in _pre_physics_step weiterhin nur als Sicherheitsnetz wirkt und
                                   # nicht bei Policy-Sättigung nahe ±1 ständig scharf eingreift
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
    # Anzahl der leichtesten Terrain-Rows, über die die Envs beim Start GLEICHMÄSSIG (nicht per
    # torch.randint) verteilt werden — siehe _setup_scene()/_init_terrain_grid() in env.py.
    # max_init_terrain_level=0 (Vorgänger-Fix) zwang JEDE Env auf exakt Row 0 -> alle Envs auf
    # derselben Y-Koordinate, 29/30 Terrain-Reihen blieben leer (live beobachtet). Mit >1 Rows
    # verteilt _init_terrain_grid() deterministisch über Rows UND Cols, bleibt aber auf niedriger
    # Difficulty (Rows 0..init_terrain_rows-1 von 30).
    # 3 -> 8: Row 8 entspricht difficulty=(8+η)/30≈27-30% — auf User-Wunsch spürbar unebener ab
    # Start, bewusst noch klar unter den höchsten Rows (25-29, >80% Difficulty), damit nicht ein
    # Teil der Envs von Anfang an auf der schwierigsten Stufe (Grund des ursprünglichen
    # max_init_terrain_level-Fixes) landet.
    init_terrain_rows: int = 8

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",           # "generator" statt "plane"
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=0,            # von _init_terrain_grid() in env.py überschrieben —
                                              # bleibt hier nur als IsaacLab-interner Fallback/Default
                                              # falls _init_terrain_grid() aus irgendeinem Grund
                                              # (z.B. terrain_type="plane") übersprungen wird.
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )

    # terrain = TerrainImporterCfg(
    # prim_path="/World/ground",
    # terrain_type="plane",
    # collision_group=-1,
    # physics_material=sim_utils.RigidBodyMaterialCfg(
    #     friction_combine_mode="multiply",
    #     restitution_combine_mode="multiply",
    #     static_friction=1.0,
    #     dynamic_friction=1.0,
    #     restitution=0.0,
    # ),
    #
    # debug_vis=False,
    # )

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
        env_spacing=3.0,    # Erhöht von 2.5 auf 8.0 wegen Terrain-Größe (8x8m)
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
    target_base_height = 0.10   # MP_BODY at normal standing height (~20cm above ground)   -> 0.11-0.14 ist glaub sehr gut, da kann der roboter dann auch noch seine beine nach oben klappen!
    movement_speed_x = 0.10     # max. Vorwärtsgeschwindigkeit (Curriculum: Stillstand bis 70k, danach volle Speed)
    yaw_rotation_speed_x = 0.0   # max. Yaw-Rate, ab 70k Schritten zusammen mit voller Speed aktiv

    # =====================================================
    # ALGORITHM-VARIANT SWITCHES — Defaults reproduzieren exakt das heutige PPO-Verhalten.
    # Nur PhantomxThesisSACEnvCfg (siehe unten) überschreibt diese Werte.
    # =====================================================
    lin_vel_kernel_width: float = 0.01    # Breite des exp(-error/width) Tracking-Kernels (lin_vel).
                                           # War 0.25 (praktisch blind: bei kompl. Stillstand noch
                                           # 97.8% Reward). Zwischenzeitlich mehrfach verengt (0.05→
                                           # 0.005→0.01), weil vermutet wurde, zu breiter Kernel
                                           # belohne Stillstand zu stark — half aber nicht (PPO blieb
                                           # bei 342k Steps bei ~0.01 m/s hängen). Tiefenanalyse gegen
                                           # den nachweislich erfolgreichen Referenzlauf
                                           # 2026-07-31_18-19-46_ppo_torch (erreichte 0.0955 m/s bei
                                           # 600k Steps) zeigte: dieser Lauf nutzte bereits 0.05, nicht
                                           # enger — die Verengung ging also in die falsche Richtung.
                                           # Zurückgesetzt auf den nachweislich funktionierenden Wert
                                           # 0.05. Wahrscheinlichere Ursache des aktuellen Problems:
                                           # Curriculum-Timing (75k/300k statt 100k/350k im Referenzlauf).
    ang_vel_kernel_width: float = 0.01    # dito für yaw_rate
    # True: Actions vor Nutzung auf [-1,1] und q_def±joint_pos_limit klemmen. Für PPO UND SAC aktiv
    # (identisch zu SAC, das bereits so lief) — schließt eine dokumentierte Sim-to-Real-Lücke: das
    # ROS2-Deployment (phantomx_policy_simulation_interface.py) klemmt Actions/Joint-Targets schon
    # länger, PPO wurde bisher aber komplett ungeklemmt trainiert.
    strict_action_pipeline: bool = True
    continuous_movement_penalty: bool = False  # True: kontinuierlicher velocity-deficit statt binärer Penalty

    # =====================================================
    # SIM-TO-REAL GAP MODELING — enable_action_latency standardmäßig AKTIV,
    # lin_vel_estimation_mode wieder auf "ground_truth" zurückgesetzt (siehe Kommentar dort:
    # Lauf 2026-07-27_15-11-56_ppo_torch hatte BEIDE Maßnahmen gleichzeitig mit einer bereits
    # unabhängig vorhandenen PPO-σ-Explosion überlagert, wodurch der Effekt der einzelnen
    # Maßnahmen nicht isolierbar war — Latenz bleibt aktiv, IMU-Integration wird für einen
    # saubereren nächsten Vergleichslauf zunächst wieder deaktiviert). Adressiert eine der in der
    # Sim-to-Real-Gap-Analyse identifizierten Lücken: fehlende Latenz zwischen Policy-Aktion und
    # tatsächlicher Anwendung (Servobus-/Inferenzlatenz existierte bisher gar nicht in der
    # Simulation). Auf False zurücksetzen, um exakt das Vor-Sim-to-Real-Gap-Verhalten zu
    # reproduzieren (z.B. für einen A/B-Vergleich).
    # =====================================================
    enable_action_latency: bool = True    # True: verzögert die tatsächlich angewendete PD-Ziel-
                                           # position um action_latency_steps_range Steps (siehe
                                           # _apply_action() in env.py) — bildet Servobus-/ROS2-
                                           # Messaging-/Inferenzlatenz nach, die im Training bisher
                                           # nicht existierte (Policy-Aktion wirkte bisher instantan
                                           # im selben Schritt).
    action_latency_steps_range: tuple[int, int] = (0, 2)  # pro Env bei jedem Reset gleichverteilt
                                                            # gezogene Verzögerung in Policy-Steps
                                                            # (nicht Physik-Substeps) — 0 bewusst im
                                                            # Bereich enthalten, damit ein Teil der
                                                            # Envs weiterhin die unverzögerte
                                                            # Dynamik sieht (kein Alles-oder-Nichts).

    lin_vel_estimation_mode: str = "ground_truth"  # ZURÜCKGESETZT (war "imu_integration", siehe
                                                    # Kommentar oben) — bisheriges Verhalten
                                                    # (root_lin_vel_b + kleines Gaussian-Rauschen).
                                                    # "imu_integration": simuliert die reale
                                                    # Real-Data-Deployment-Pipeline (keine
                                                    # Odometrie vorhanden) — lin_vel_b wird NICHT
                                                    # direkt aus der Simulation gelesen, sondern
                                                    # aus einer rauschbehafteten, driftenden
                                                    # Beschleunigungs-Integration rekonstruiert
                                                    # (siehe _get_observations() in env.py).
    lin_vel_imu_bias_std: float = 0.02     # m/s² Standardabweichung des Random-Walk-Bias auf die
                                            # integrierte Beschleunigung — Kernunterschied zu
                                            # simplem Rauschen: akkumuliert unbegrenzt über die Zeit
                                            # statt sich über viele Steps wegzumitteln.
    lin_vel_imu_noise_std: float = 0.05    # m/s² Standardabweichung des hochfrequenten
                                            # (nicht-akkumulierenden) Beschleunigungsmess-Rauschens,
                                            # zusätzlich zum Bias oben.
    lin_vel_imu_leak: float = 0.99         # Leaky-Integrator-Faktor (siehe reale ROS2-Pipeline,
                                            # phantomx_policy_real_data_interface.py:
                                            # "self.lin_vel_b = 0.99 * self.lin_vel_b + a_body*dt")
                                            # — verhindert unbegrenzten Drift, führt aber selbst zu
                                            # einem systematischen Abklingen der geschätzten
                                            # Geschwindigkeit ohne neue Beschleunigung.

    # =======================================   ==============
    # REWARD SCALES - TUNED FOR HEXAPOD LOCOMOTION
    # =====================================================
    #🎯 TRACKING REWARDS (positive)
    lin_vel_reward_scale = 12.0      #12 hat gut funktioniert
    yaw_rate_reward_scale = 5.0

    height_reward_scale = 2.5    # war 5.0 — im Stillstand fast maximal (height_error≈0), lieferte
                                  # zusammen mit foot_contact_reward mehr Netto-Reward fürs Stehen
                                  # als movement_penalty fürs Stillstehen kostete (Roboter blieb bei
                                  # ~0.004 m/s trotz movement_speed_x=0.1 hängen, Session 2026-08-01).
                                  # Halbiert, um diesen Netto-Vorteil des Stillstands zu verkleinern,
                                  # ohne die Kernel-Form (0.02) oder andere Terme anzufassen.

    # 🚫 PENALTIES (negative)
    z_vel_reward_scale = -5.0
    ang_vel_reward_scale = -7.0
    action_rate_reward_scale = -0.02        #default -0.02
    flat_orientation_reward_scale = -5.0

    # movement_penalty_scale deckt jetzt beide Richtungen ab: Unterschreiten (binär, PPO) plus
    # proportionales Überschreiten (siehe _compute_movement_penalty) — bewusst ein gemeinsamer
    # Scale-Wert statt eines eigenen, damit nicht zwei Stellschrauben für dasselbe Tracking-Ziel
    # gepflegt werden müssen.
    movement_penalty_scale = 35       #10 hat gut funktioniert #bei SAC:25 hat perfect funktioniert

    alive_reward_scale = 0.3

    # 🦿 FOOT CONTACT REWARD — Bonus für stabile Stützbasis (≥3 Beine am Boden)
    foot_contact_reward_scale = 1.0

    # 🔄 TRIPOD GAIT REWARD — belohnt wechselndes 3-3 Kontaktmuster (alle Beine aktiv)
    tripod_gait_reward_scale = 3.0

    # 😴 LAZY LEG PENALTY — Strafe für Beine die >3s dauerhaft in der Luft hängen
    lazy_leg_penalty_scale = 0.5

    # 🦵 SWING FOOT HEIGHT REWARD — Bonus für Fußhöhe über Terrain WÄHREND der Schwungphase
    # (kein Bonus im Stand). Adressiert das beobachtete "Zuckeln mit minimalen Tibia-Bewegungen"
    # statt sauberem Anheben/Schwingen — kein bestehender Term (foot_contact/tripod_gait sind
    # reine Kontakt-Booleans, lazy_legs bestraft nur extrem lange Luftzeit >3s) erfasst bisher
    # die tatsächliche Fußhöhe während der Schwungphase.
    swing_foot_height_target: float = 0.04  # m — Ziel-Clearance über Terrain-Referenz. Startwert
                                              # konservativ: ~1/3 der Femur-Segmentlänge (≈6.6cm),
                                              # deutlich unter target_base_height=0.11m, aber klar
                                              # über dem Terrain-Rauschen der leichteren
                                              # Sub-Terrains (random_rough/gravel_fine: 0.015-0.06m).
    swing_foot_height_reward_scale: float = 3.0  # AKTIVIERT (erster Testwert, siehe
                                                  # height_reward_scale=2.5 als Größenvergleich) —
                                                  # Episode_Reward/swing_foot_height weiter
                                                  # beobachten, bei Bedarf nachjustieren.

    # =====================================================
    # TERMINATION THRESHOLDS - RELAXED FOR LEARNING
    # =====================================================
    termination_height = 0.00    # MP_BODY < 15cm → kollabiert (≙ base_link < 5cm + 10cm Offset)
    termination_tilt = 0.03     # gx²+gy² > 0.10 → ~18° Neigung — exakter Wert aus funktionierendem Modell


@configclass
class PhantomxThesisFlatEnvCfg(PhantomxThesisEnvCfg):
    """Flat-Terrain-Variante der PPO-Basis-Cfg: identischer Reward-/Observation-/Curriculum-Code,
    nur die Bodenebene ist wieder eine reine Plane statt des generierten Rough-Terrains — für ein
    "normales" Modell auf ebenem Boden, ohne die Rough-Terrain-Cfg der Basisklasse anzufassen."""

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


@configclass
class PhantomxThesisSACEnvCfg(PhantomxThesisEnvCfg):
    """SAC-Variante: eigene Reward-/Curriculum-Feinabstimmung (Ziel: langsames, sauberes Gehen
    bis 7.5 cm/s), ohne die PPO-Cfg oben zu berühren. Physik-/Observation-Code bleibt in
    PhantomxThesisEnv gemeinsam — nur diese Werte weichen vom PPO-Default ab."""

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=128,   # auf skrl_sac_cfg.yaml memory_size=125000 abgestimmt (~16M Transitions gesamt)
        env_spacing=3.0,
        replicate_physics=True,
    )

    target_base_height = 0.10
    movement_speed_x = 0.10
    yaw_rotation_speed_x = 0.0

    termination_height = 0.000

    # lin_vel_kernel_width nochmal deutlich verengt (0.1 → 0.005): bei movement_speed_x=0.05 gab
    # 0.1 komplettem Stillstand (error=command=0.05) noch exp(-0.05²/0.1)=exp(-0.025)≈0,975 —
    # 97,5% des maximalen Tracking-Rewards fürs Nichtstun, praktisch kein Anreiz zu laufen (live
    # in der Simulation bestätigt: Roboter steht, bewegt sich kaum). Bei 0.005 sinkt derselbe
    # Stillstand-Fall auf exp(-0.05²/0.005)=exp(-0.5)≈0,61 — deutlich sichtbare Lücke zu echtem
    # Tracking (~0,96-0,98 bei 80% Zielgeschwindigkeit) — Startwert, mit avg_forward_speed_mps
    # in TensorBoard beobachten.
    lin_vel_kernel_width = 0.005
    ang_vel_kernel_width = 0.1   # unverändert — für SAC aktuell wirkungslos (yaw_rotation_speed_x=0,
                                  # commands[:,2] immer 0), nicht Teil des Geh-Problems

    # strict_action_pipeline: kein eigener SAC-Override mehr — jetzt auch für PPO True in der
    # Basisklasse, beide erben denselben Wert (vermeidet das in Log 07-12/Punkt 16 dokumentierte
    # Bugmuster stiller, ungenutzter Overrides).
    continuous_movement_penalty = True   # kontinuierlicher velocity-deficit statt binaerer Penalty —
                                          # die neue Overshoot-Strafe lebt nur im binären (PPO-)Zweig
                                          # von _compute_movement_penalty, SAC bleibt also unverändert


@configclass
class PhantomxThesisSACFlatEnvCfg(PhantomxThesisSACEnvCfg):
    """Flat-Terrain-Variante der SAC-Cfg: identischer Reward-/Curriculum-Code wie
    PhantomxThesisSACEnvCfg, nur die Bodenebene ist wieder eine reine Plane statt des von der
    PPO-Basisklasse geerbten generierten Rough-Terrains."""

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