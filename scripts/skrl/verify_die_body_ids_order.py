"""Diagnose-Skript: verifiziert/falsifiziert den Verdacht, dass self._die_body_ids
(phantomx_thesis_env.py, ContactSensor-Query OHNE preserve_order=True) in einer anderen
Reihenfolge vorliegt als die im Code angenommene (lf, lm, lr, rf, rm, rr), wodurch
_TRIPOD_A=[0,4,2] / _TRIPOD_B=[3,1,5] die Beine ggf. falsch links/rechts zuordnen.

Muss auf einer GPU-Maschine mit funktionierendem CUDA/PhysX-Stack laufen (in der
Analyse-Sandbox nicht möglich). num_envs=1, keine Trainingswirkung, kein Checkpoint noetig.

Aufruf:
    /workspace/isaaclab/isaaclab.sh -p verify_die_body_ids_order.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify self._die_body_ids ordering assumption.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

import phantomx_thesis.tasks  # noqa: F401 registers the task IDs
from isaaclab_tasks.utils import load_cfg_from_registry

EXPECTED_ORDER = ["tibia_lf", "tibia_lm", "tibia_lr", "tibia_rf", "tibia_rm", "tibia_rr"]

# The 4 usage sites of self._die_body_ids in phantomx_thesis_env.py and whether they need
# real positional/left-right semantics (only tripod_gait does; the other 3 only sum/count).
USAGE_SITES = {
    "foot_contact (net_forces_w sum/threshold)": False,
    "tripod_gait (_TRIPOD_A/_TRIPOD_B grouping)": True,
    "lazy_legs (current_air_time sum/threshold)": False,
    "termination (non_foot_body_ids, separate query, not _die_body_ids)": False,
}


def main():
    task_id = "Template-Phantomx-Thesis-PPO-Direct-v0"
    env_cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
    env_cfg.scene.num_envs = 1

    env = gym.make(task_id, cfg=env_cfg).unwrapped

    contact_sensor = env._contact_sensor
    robot = env._robot

    query_names = [
        "tibia_lf", "tibia_lm", "tibia_lr",
        "tibia_rf", "tibia_rm", "tibia_rr",
    ]

    # (1) Reproduce the exact call used in __init__ (no preserve_order) — this is the
    #     currently-live self._die_body_ids.
    die_body_ids, die_body_names = contact_sensor.find_bodies(query_names)

    # (2) Same query, WITH preserve_order=True — guaranteed to match query_names exactly.
    ordered_body_ids, ordered_body_names = contact_sensor.find_bodies(query_names, preserve_order=True)

    # (3) Third, independent cross-check: robot-articulation namespace, preserve_order=True
    #     (this is what self._tibia_robot_body_ids / self._swing_contact_body_ids already use).
    robot_body_ids, robot_body_names = robot.find_bodies(query_names, preserve_order=True)

    print("=" * 80)
    print("VERIFY: self._die_body_ids ordering (preserve_order=True vs. default False)")
    print("=" * 80)
    print(f"Query names (assumed order):      {query_names}")
    print(f"ContactSensor default (no order):  ids={die_body_ids} names={die_body_names}")
    print(f"ContactSensor preserve_order=True: ids={ordered_body_ids} names={ordered_body_names}")
    print(f"Robot     preserve_order=True:     ids={robot_body_ids} names={robot_body_names}")
    print("-" * 80)

    matches_expected = die_body_names == EXPECTED_ORDER
    matches_ordered = die_body_names == ordered_body_names

    if matches_expected and matches_ordered:
        verdict = "FALSIFIZIERT"
        detail = (
            "self._die_body_ids liefert bereits die erwartete Reihenfolge "
            f"{EXPECTED_ORDER}. Kein Bug — _TRIPOD_A/_TRIPOD_B sind korrekt zugeordnet."
        )
    else:
        verdict = "BESTAETIGT"
        detail = (
            "self._die_body_ids weicht von der im Code angenommenen Reihenfolge "
            f"{EXPECTED_ORDER} ab. Tatsaechliche Reihenfolge: {die_body_names}."
        )

    print(f"FAZIT: {verdict}")
    print(detail)
    print("-" * 80)

    print("Betroffene Nutzungsstellen (self._die_body_ids in phantomx_thesis_env.py):")
    for site, needs_positional_semantics in USAGE_SITES.items():
        risk = "BETROFFEN falls Bug real" if needs_positional_semantics else "unbetroffen (nur Summe/Zaehlung)"
        print(f"  - {site}: {risk}")
    print("-" * 80)

    if verdict == "BESTAETIGT":
        # Compute the correct _TRIPOD_A/_TRIPOD_B indices for the ACTUAL order returned by
        # self._die_body_ids, so tripod_gait can be fixed directly against real body names.
        name_to_correct_index = {name: EXPECTED_ORDER.index(name) for name in die_body_names}
        # _TRIPOD_A/_TRIPOD_B are defined in terms of EXPECTED_ORDER's semantic slots
        # (lf, rm, lr) / (rf, lm, rr). We need: for the ACTUAL order die_body_names, which
        # index holds "lf", which holds "rm", etc.
        tripod_a_names = ["tibia_lf", "tibia_rm", "tibia_lr"]
        tripod_b_names = ["tibia_rf", "tibia_lm", "tibia_rr"]
        actual_index_of = {name: i for i, name in enumerate(die_body_names)}

        try:
            corrected_tripod_a = [actual_index_of[n] for n in tripod_a_names]
            corrected_tripod_b = [actual_index_of[n] for n in tripod_b_names]
            print("Automatisch berechnete KORREKTE Indizes fuer die tatsaechliche Reihenfolge:")
            print(f"  self._TRIPOD_A = {corrected_tripod_a}  # {tripod_a_names}")
            print(f"  self._TRIPOD_B = {corrected_tripod_b}  # {tripod_b_names}")
        except KeyError as e:
            print(f"  Konnte Indizes nicht automatisch berechnen (unerwarteter Name: {e}).")
            print(f"  Bitte manuell gegen die oben ausgegebene Reihenfolge {die_body_names} zuordnen.")

    print("=" * 80)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
