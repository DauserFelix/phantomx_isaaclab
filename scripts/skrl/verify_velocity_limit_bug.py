"""Diagnose-Skript: verifiziert/falsifiziert den Verdacht, dass velocity_limit=0.8 (OHNE
_sim-Suffix) in isaaclab_assets/robots/phantomx.py NIE wirksam ist, und dass PhysX
stattdessen den URDF-Fallback (~5.6549 rad/s, aus full_phantom_isaacsim.urdf) tatsaechlich
an den Solver schreibt (siehe Log 2026-07-31 Punkt 3 fuer die vollstaendig gelesene
Quellcode-Kette: actuator_pd.py -> actuator_base.py -> articulation.py).

Muss auf einer GPU-Maschine mit funktionierendem CUDA/PhysX-Stack laufen (in der
Analyse-Sandbox nicht moeglich).

Aufruf:
    /workspace/isaaclab/isaaclab.sh -p verify_velocity_limit_bug.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify velocity_limit vs. velocity_limit_sim bug.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym

import phantomx_thesis.tasks  # noqa: F401 registers the task IDs
from isaaclab_tasks.utils import load_cfg_from_registry

CFG_VALUE = 0.8            # velocity_limit, wie in phantomx.py:97/104/111 gesetzt
URDF_FALLBACK = 5.6548668   # <limit ... velocity="5.6548668"/> in full_phantom_isaacsim.urdf


def main():
    task_id = "Template-Phantomx-Thesis-PPO-Direct-v0"
    env_cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
    env_cfg.scene.num_envs = 1

    env = gym.make(task_id, cfg=env_cfg).unwrapped
    robot = env._robot

    print("=" * 80)
    print("VERIFY: velocity_limit (0.8, nie wirksam?) vs. velocity_limit_sim (URDF-Fallback?)")
    print("=" * 80)

    # (1) Primaerquelle: der Tensor, der laut articulation.py:1773
    #     (write_joint_velocity_limit_to_sim) tatsaechlich an den PhysX-Solver geschrieben wird.
    joint_vel_limits = robot.data.joint_vel_limits[0]  # [num_joints], env 0
    print(f"robot.data.joint_vel_limits (an PhysX geschrieben): {joint_vel_limits.tolist()}")

    close_to_cfg = torch.allclose(
        joint_vel_limits, torch.full_like(joint_vel_limits, CFG_VALUE), atol=1e-3
    )
    close_to_urdf = torch.allclose(
        joint_vel_limits, torch.full_like(joint_vel_limits, URDF_FALLBACK), atol=1e-2
    )

    print(f"  allclose(0.8)              -> {close_to_cfg}")
    print(f"  allclose(5.6548668, URDF)  -> {close_to_urdf}")
    print("-" * 80)

    # (2) Zweite, unabhaengige Quelle: direkt von den Actuator-Objekten lesen
    #     (velocity_limit_sim ist das, was tatsaechlich an PhysX geht; velocity_limit sollte
    #     laut actuator_pd.py:91 explizit auf None verworfen worden sein).
    print("Actuator-Objekte (zweite, unabhaengige Quelle):")
    for name, actuator in robot.actuators.items():
        vlim = getattr(actuator, "velocity_limit", None)
        vlim_sim = getattr(actuator, "velocity_limit_sim", None)
        print(f"  actuator '{name}': velocity_limit={vlim!r}  velocity_limit_sim={vlim_sim!r}")
    print("-" * 80)

    # (3) Dritte, unabhaengige Bestaetigung: die von Isaac Lab selbst geloggte Warnung
    #     (actuator_pd.py:81-91) faengt man am zuverlaessigsten ueber den Python-Logger ab,
    #     nicht per stdout-Capture. Hier stattdessen direkt die im Actuator-Cfg gespeicherten
    #     Rohwerte gegen die Erwartung aus der Quellcode-Analyse pruefen:
    #     cfg.velocity_limit sollte nach ImplicitActuator.__init__ auf None verworfen sein,
    #     wenn velocity_limit_sim ursprünglich None war und velocity_limit gesetzt war.
    print("Actuator-Cfg-Rohwerte (sollten velocity_limit=None zeigen, wenn Bug real ist):")
    for name, actuator in robot.actuators.items():
        cfg_vlim = getattr(actuator.cfg, "velocity_limit", "N/A")
        cfg_vlim_sim = getattr(actuator.cfg, "velocity_limit_sim", "N/A")
        print(f"  actuator '{name}'.cfg: velocity_limit={cfg_vlim!r}  velocity_limit_sim={cfg_vlim_sim!r}")
    print("-" * 80)

    if close_to_urdf and not close_to_cfg:
        verdict = "BESTAETIGT"
        detail = (
            f"joint_vel_limits liegt bei URDF-Fallback ({URDF_FALLBACK} rad/s), NICHT bei "
            f"dem in phantomx.py konfigurierten Wert (0.8 rad/s). velocity_limit=0.8 war nie "
            f"wirksam. Handlungsempfehlung: Entscheidung noetig (siehe Log 2026-07-31 Punkt 13.2) "
            f"— entweder velocity_limit_sim=0.8 explizit setzen (echte Trainingsauswirkung, "
            f"Neutraining noetig) oder ~5.65 als akzeptierte, bereits trainierte Baseline "
            f"uebernehmen (dann nur der ROS2-Rate-Limiter-Referenzwert anzupassen)."
        )
    elif close_to_cfg and not close_to_urdf:
        verdict = "FALSIFIZIERT"
        detail = (
            "joint_vel_limits liegt bei 0.8 rad/s, dem konfigurierten Cfg-Wert. Der Verdacht "
            "aus Log 2026-07-31 Punkt 3 ist damit widerlegt — velocity_limit war wirksam "
            "(z.B. weil eine neuere Isaac-Lab-Version die _sim-Suffix-Regel anders handhabt, "
            "oder die Quellcode-Kette wurde in dieser Isaac-Lab-Version geaendert). Quellcode "
            "erneut pruefen (actuator_pd.py:81-96), falls dieses Ergebnis ueberrascht."
        )
    else:
        verdict = "UNKLAR"
        detail = (
            f"joint_vel_limits={joint_vel_limits.tolist()} passt weder eindeutig zu 0.8 noch "
            f"zu {URDF_FALLBACK}. Manuelle Nachpruefung noetig — evtl. sind nicht alle 18 "
            f"Gelenke identisch limitiert, oder ein anderer Wert wurde zwischenzeitlich in der "
            f"Cfg gesetzt (Log-Historie zeigt wiederholt undokumentierte Zwischenaenderungen)."
        )

    print(f"FAZIT: {verdict}")
    print(detail)
    print("=" * 80)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
