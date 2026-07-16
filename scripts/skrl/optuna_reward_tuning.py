# scripts/skrl/optuna_reward_tuning.py

import optuna
import subprocess
import json
import os
import re
import torch

def run_training(trial_params: dict, trial_number: int) -> float:
    """
    Startet einen kurzen PPO-Trainingslauf mit den gegebenen Reward-Parametern.
    Gibt den mittleren episodischen Reward zurück.
    """
    log_dir = f"/workspace/projects/phantomx_thesis/logs/optuna/trial_{trial_number}"
    os.makedirs(log_dir, exist_ok=True)

    # Parameter als Umgebungsvariablen übergeben
    env = os.environ.copy()
    env["OPTUNA_LIN_VEL_SCALE"]       = str(trial_params["lin_vel_reward_scale"])
    env["OPTUNA_ALIVE_SCALE"]          = str(trial_params["alive_reward_scale"])
    env["OPTUNA_ORIENTATION_SCALE"]    = str(trial_params["flat_orientation_reward_scale"])
    env["OPTUNA_TORQUE_SCALE"]         = str(trial_params["joint_torque_reward_scale"])
    env["OPTUNA_ACTION_RATE_SCALE"]    = str(trial_params["action_rate_reward_scale"])
    env["OPTUNA_LOG_DIR"]              = log_dir

    # Kurzen Trainingslauf starten (300K steps)
    result = subprocess.run(
        [
            "/workspace/isaaclab/isaaclab.sh", "-p",
            "scripts/skrl/train_optuna.py",
            "--task", "Template-Phantomx-Thesis-PPO-Direct_uneven-v0",
            "--headless",
            "--num_envs", "64",       # weniger Envs für schnellere Trials
        ],
        env=env,
        capture_output=True,
        text=True,
        cwd="/workspace/projects/phantomx_thesis"
    )

    # Mean Reward aus Log-Datei lesen
    reward = parse_mean_reward(log_dir)
    return reward


def parse_mean_reward(log_dir: str) -> float:
    """Liest den finalen mean reward aus dem SKRL-Log."""
    # SKRL schreibt Logs als TensorBoard — alternativ eine reward.txt anlegen
    reward_file = os.path.join(log_dir, "final_reward.txt")
    if os.path.exists(reward_file):
        with open(reward_file) as f:
            return float(f.read().strip())
    return -999.0  # Fallback wenn Training fehlschlug


def objective(trial: optuna.Trial) -> float:
    params = {
        "flat_orientation_reward_scale": trial.suggest_float(
            "orientation", -6.0, -0.5
        ),
        "joint_torque_reward_scale": trial.suggest_float(
            "torque", -1e-4, -1e-6, log=True
        ),
        "action_rate_reward_scale": trial.suggest_float(
            "action_rate", -0.1, -0.001, log=True
        ),
        "ang_vel_reward_scale": trial.suggest_float(
            "ang_vel", -8.0, -0.5
        ),
    }
    return run_short_training(params)


if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",
        study_name="hexapod_reward_tuning",
        storage="sqlite:///optuna_rewards.db",  # Ergebnisse persistent speichern
        load_if_exists=True,                     # Fortsetzen falls unterbrochen
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=0,
        ),
    )

    study.optimize(objective, n_trials=20, n_jobs=1)

    print("\n=== Beste Parameter ===")
    print(json.dumps(study.best_params, indent=2))
    print(f"Bester Reward: {study.best_value:.4f}")

    # Ergebnis speichern
    with open("best_reward_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)