"""Configuration for the PhantomX Hexapod robot.

Robot has 6 legs (rf, rm, rr, lf, lm, lr) with 3 revolute joints each:
  - j_c1_*   : coxa  (hip yaw)
  - j_thigh_*: femur (hip pitch)
  - j_tibia_*: tibia (knee)

Total: 18 revolute joints

Hardware limits (from URDF / Dynamixel AX-12A):
  Joint range:      ±2.6179939 rad  (~±150°)
  Effort (max):     2.8 Nm
  Velocity (max):   5.6548668 rad/s  (~54 RPM)

Simulation / Training (ImplicitActuatorCfg — overrides hardware limits):
  Effort limit:     1.5 Nm   (conservative to prevent sim instability)
  Velocity limit:   0.8 rad/s (conservative — matches ROS2 max_action_delta)
  Stiffness (Kp):  10.0
  Damping   (Kd):   2.0
"""

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils

PHANTOMX_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        #für train.py verwenden
        usd_path="/workspace/projects/phantomx_thesis/source/phantomx_thesis/usd_files_phantomx/full_phantom_isaacsim/full_phantom_isaacsim.usd",  # <-- adjust this path

        # #für play.py verwenden (mit controller)
        # usd_path="/workspace/projects/phantomx_thesis/source/phantomx_thesis/usd_files_phantomx/isaacsim_ros2_interface/phantomx_isaacsim_ros2_setup.usd",  # <-- adjust this path

        #für play.py verwenden (ohne controller, auch zum trainieren)
        # usd_path="/workspace/projects/phantomx_thesis/source/phantomx_thesis/usd_files_phantomx/isaacsim_ros2_interface/phantomx_isaacsim_ros2_wc.usd",  # <-- adjust this path
        
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=10.0,       # war 1000 — begrenzt entlaufene Roboter
            max_angular_velocity=100.0,     # war 1000
            max_depenetration_velocity=1.0, # low value prevents PhysX from ejecting robot on spawn
            enable_gyroscopic_forces=True,
            disable_gravity=False,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4, # war 1 — mehr Iterationen = stabilere Kontakte
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.18),   # höher gespawnt damit Füße nicht in Terrain-Bumps clippen
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            # Right Front
            "j_c1_rf":    0.0,      #default=0.0
            "j_thigh_rf": -1.2,      #default=0.5
            "j_tibia_rf": -1.0,      #default=0.5
            # Right Middle
            "j_c1_rm":    0.0,
            "j_thigh_rm": -1.2,
            "j_tibia_rm": -1.0,
            # Right Rear
            "j_c1_rr":    0.0,
            "j_thigh_rr": -1.2,
            "j_tibia_rr": -1.0,
            # Left Front
            "j_c1_lf":    0.0,
            "j_thigh_lf": -1.2,
            "j_tibia_lf": -1.0,
            # Left Middle
            "j_c1_lm":    0.0,
            "j_thigh_lm": -1.2,
            "j_tibia_lm": -1.0,
            # Left Rear
            "j_c1_lr":    0.0,
            "j_thigh_lr": -1.2,
            "j_tibia_lr": -1.0,
        },
        joint_vel={
            "j_c1_rf": 0.0, "j_thigh_rf": 0.0, "j_tibia_rf": 0.0,
            "j_c1_rm": 0.0, "j_thigh_rm": 0.0, "j_tibia_rm": 0.0,
            "j_c1_rr": 0.0, "j_thigh_rr": 0.0, "j_tibia_rr": 0.0,
            "j_c1_lf": 0.0, "j_thigh_lf": 0.0, "j_tibia_lf": 0.0,
            "j_c1_lm": 0.0, "j_thigh_lm": 0.0, "j_tibia_lm": 0.0,
            "j_c1_lr": 0.0, "j_thigh_lr": 0.0, "j_tibia_lr": 0.0,
        },
    ),
    actuators={
        "coxa_joints": ImplicitActuatorCfg(
            joint_names_expr=["j_c1_.*"],
            effort_limit=1.5,
            velocity_limit=0.8,
            stiffness=10.0,
            damping=2.0,
        ),
        "femur_joints": ImplicitActuatorCfg(
            joint_names_expr=["j_thigh_.*"],
            effort_limit=1.5,
            velocity_limit=0.8,
            stiffness=10.0,
            damping=2.0,
        ),
        "tibia_joints": ImplicitActuatorCfg(
            joint_names_expr=["j_tibia_.*"],
            effort_limit=1.5,
            velocity_limit=0.8,
            stiffness=10.0,
            damping=2.0,
        ),
    },
    soft_joint_pos_limit_factor=0.9,  # stay within 90% of ±150° hardware limits
)
"""Configuration of PhantomX Hexapod using implicit actuators."""
