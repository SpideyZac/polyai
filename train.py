"""
Distributed PPO trainer for PolyTrackEnv.

Usage:
    python train.py --export-string <your_export_string> [--checkpoint <path>]
"""

import argparse
import json
import logging
import os
import time

import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from src_py.polytrack import PolyTrackEnv

CHECKPOINT_DIR = "./checkpoints"
LOG_FILE = "./training_log.jsonl"
CHECKPOINT_INTERVAL = 10
REBUILD_INTERVAL = 60
WORKER_RESTART_DELAY = 5
RAY_ADDRESS = "auto"

CPUS_PER_WORKER = 1

LEARN_TIME_THRESHOLD = 0.4
MAX_LEARNERS = 4
CPUS_PER_LEARNER = 2
GPUS_PER_LEARNER = 1

MAX_WORKER_RESTARTS = -1

TRAIN_BATCH_SIZE = 50_000
MINIBATCH_SIZE = 1024
NUM_SGD_EPOCHS = 10
ENTROPY_COEFF = 0.01
CLIP_PARAM = 0.2
VF_CLIP_PARAM = 10.0
GAMMA = 0.995
LAMBDA_GAE = 0.95

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)


def log_event(data: dict) -> None:
    """Append a JSON-encoded event to the log file."""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")


def cluster_resources() -> tuple[int, int]:
    """Return (total_cpus, total_gpus) from the current Ray cluster."""
    r = ray.cluster_resources()
    return int(r.get("CPU", 1)), int(r.get("GPU", 0))


def get_num_rollout_workers(num_learners: int, total_cpus: int) -> int:
    """Return the number of rollout workers to use, given the learner count and total CPUs."""
    reserved = num_learners * CPUS_PER_LEARNER
    return max(1, (total_cpus - reserved) // CPUS_PER_WORKER)


def desired_num_learners(metrics: dict, current: int, total_gpus: int) -> int:
    """Return updated learner count, scaling up by one if SGD is the bottleneck
    and a free GPU exists."""
    s, l = metrics["sample_time_ms"], metrics["learn_time_ms"]
    if s != s or l != l or s <= 0:  # pylint: disable=comparison-with-itself
        return current
    if l / (s + l) > LEARN_TIME_THRESHOLD and total_gpus > current:
        new = min(current + 1, MAX_LEARNERS, total_gpus)
        if new != current:
            log.warning(
                "learn_time is %.0f%% of iteration time - scaling learners %d -> %d",
                l / (s + l) * 100,
                current,
                new,
            )
            return new
    return current


def build_trainer(num_learners: int, num_workers: int, export_string: str) -> Algorithm:
    """Construct a new PPO trainer with the given number of learners and workers."""
    log.info("Building trainer - workers: %d  learners: %d", num_workers, num_learners)
    config = (
        PPOConfig()
        .environment(
            env="PolyTrackEnv",
            env_config={"export_string": export_string},
        )
        .framework("torch")
        .env_runners(
            num_env_runners=num_workers,
            num_cpus_per_env_runner=CPUS_PER_WORKER,
            num_envs_per_env_runner=1,
            create_env_on_local_worker=False,
            observation_filter="MeanStdFilter",
        )
        .fault_tolerance(
            max_num_env_runner_restarts=MAX_WORKER_RESTARTS,
        )
        .resources(
            num_gpus=0,
            num_cpus_for_local_worker=0,
        )
        .learners(
            num_learners=num_learners,
            num_cpus_per_learner=CPUS_PER_LEARNER,
            num_gpus_per_learner=GPUS_PER_LEARNER,
        )
        .training(
            train_batch_size=TRAIN_BATCH_SIZE,
            minibatch_size=MINIBATCH_SIZE,
            num_epochs=NUM_SGD_EPOCHS,
            clip_param=CLIP_PARAM,
            vf_clip_param=VF_CLIP_PARAM,
            entropy_coeff=ENTROPY_COEFF,
            gamma=GAMMA,
            lambda_=LAMBDA_GAE,
            use_critic=True,
            use_gae=True,
        )
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .reporting(
            min_sample_timesteps_per_iteration=TRAIN_BATCH_SIZE,
            metrics_num_episodes_for_smoothing=50,
        )
    )
    return config.build()


def save_checkpoint(trainer: Algorithm) -> str:
    """Save a checkpoint and return its path."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint = trainer.save(CHECKPOINT_DIR)
    if hasattr(checkpoint, "checkpoint") and hasattr(checkpoint.checkpoint, "path"):
        return checkpoint.checkpoint.path  # type: ignore[union-attr]
    return str(checkpoint)


def restore_weights(trainer: Algorithm, checkpoint_path: str) -> None:
    """Restore weights only, avoiding Algorithm.from_checkpoint() which would
    overwrite the current config with the checkpoint's saved one."""
    source = Algorithm.from_checkpoint(checkpoint_path)
    trainer.set_weights(source.get_weights())
    source.stop()
    log.info("Weights restored from %s", checkpoint_path)


def extract_metrics(result: dict) -> dict:
    """Extract relevant metrics from the trainer result dict, with defaults."""
    runners = result.get("env_runners", {})
    timers = result.get("timers", {})
    return {
        "reward_mean": runners.get("episode_reward_mean", float("nan")),
        "reward_max": runners.get("episode_reward_max", float("nan")),
        "reward_min": runners.get("episode_reward_min", float("nan")),
        "ep_len_mean": runners.get("episode_len_mean", float("nan")),
        "episodes_total": result.get("episodes_total", 0),
        "timesteps_total": result.get("num_env_steps_sampled_lifetime", 0),
        "checkpoints_hit": runners.get("checkpoints_hit_mean", float("nan")),
        "finish_rate": runners.get("finished_mean", float("nan")),
        "sample_time_ms": timers.get("sample_time_ms", float("nan")),
        "learn_time_ms": timers.get("learn_time_ms", float("nan")),
    }


def log_metrics(
    iteration: int, num_workers: int, num_learners: int, metrics: dict
) -> None:
    """Log key metrics to console."""
    m = metrics
    log.info(
        "[iter %04d] reward=%.2f (min=%.2f max=%.2f)  ep_len=%.0f  "
        "episodes=%d  steps=%d  checkpoints=%.1f  finish_rate=%.2f  "
        "sample_ms=%.0f  learn_ms=%.0f  workers=%d  learners=%d",
        iteration,
        m["reward_mean"],
        m["reward_min"],
        m["reward_max"],
        m["ep_len_mean"],
        m["episodes_total"],
        m["timesteps_total"],
        m["checkpoints_hit"],
        m["finish_rate"],
        m["sample_time_ms"],
        m["learn_time_ms"],
        num_workers,
        num_learners,
    )


def rebuild(
    num_learners: int,
    num_workers: int,
    export_string: str,
    checkpoint_path: str | None,
) -> Algorithm:
    """Stop the current trainer, build a fresh one, and restore weights."""
    trainer = build_trainer(num_learners, num_workers, export_string)
    if checkpoint_path:
        restore_weights(trainer, checkpoint_path)
    return trainer


def main() -> None:
    """Main training loop."""
    parser = argparse.ArgumentParser(description="Train PPO on PolyTrackEnv")
    parser.add_argument("--export-string", required=True)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    ray.init(address=RAY_ADDRESS, ignore_reinit_error=True)
    log.info("Ray cluster resources: %s", ray.cluster_resources())

    register_env("PolyTrackEnv", PolyTrackEnv)

    total_cpus, _ = cluster_resources()
    num_learners = 1
    num_workers = get_num_rollout_workers(num_learners, total_cpus)
    trainer = build_trainer(num_learners, num_workers, args.export_string)
    checkpoint_path = args.checkpoint

    if checkpoint_path:
        log.info("Resuming from: %s", checkpoint_path)
        restore_weights(trainer, checkpoint_path)

    last_rebuild_time = time.time()
    iteration = 0

    try:
        while True:
            iteration += 1

            try:
                result = trainer.train()
            except Exception as exc:  # pylint: disable=broad-exception-caught
                log.error("Iteration %d failed: %s", iteration, exc, exc_info=True)
                time.sleep(WORKER_RESTART_DELAY)
                continue

            metrics = extract_metrics(result)
            log_metrics(iteration, num_workers, num_learners, metrics)
            log_event(
                {
                    "iteration": iteration,
                    "workers": num_workers,
                    "learners": num_learners,
                    "time": time.time(),
                    **metrics,
                }
            )

            if iteration % CHECKPOINT_INTERVAL == 0:
                checkpoint_path = save_checkpoint(trainer)
                log.info("[iter %04d] checkpoint -> %s", iteration, checkpoint_path)

            now = time.time()
            if now - last_rebuild_time >= REBUILD_INTERVAL:
                new_total_cpus, new_total_gpus = cluster_resources()
                new_learners = desired_num_learners(
                    metrics, num_learners, new_total_gpus
                )
                new_workers = get_num_rollout_workers(new_learners, new_total_cpus)

                if new_workers != num_workers or new_learners != num_learners:
                    log.info(
                        "Rebuilding - workers: %d -> %d  learners: %d -> %d",
                        num_workers,
                        new_workers,
                        num_learners,
                        new_learners,
                    )
                    log_event(
                        {
                            "event": "rebuild",
                            "old_workers": num_workers,
                            "new_workers": new_workers,
                            "old_learners": num_learners,
                            "new_learners": new_learners,
                            "time": now,
                        }
                    )

                    checkpoint_path = save_checkpoint(trainer)
                    trainer.stop()

                    num_learners = new_learners
                    num_workers = new_workers
                    trainer = rebuild(
                        num_learners, num_workers, args.export_string, checkpoint_path
                    )

                last_rebuild_time = now

    except KeyboardInterrupt:
        log.info("Interrupted - saving final checkpoint")
        checkpoint_path = save_checkpoint(trainer)
        log.info("Final checkpoint: %s", checkpoint_path)

    finally:
        trainer.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
