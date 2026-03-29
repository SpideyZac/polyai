"""
Distributed PPO trainer for PolyTrackEnv.

Usage:
    python train.py --export-string <your_export_string> [--checkpoint <path>]
"""

import argparse
import json
import logging
import math
import os
import time

import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.connectors.env_to_module import MeanStdFilter
from ray.tune.registry import register_env

from src_py.polytrack import PolyTrackEnv

# Where checkpoints are written to disk.
CHECKPOINT_DIR = "./checkpoints"
# Append-only JSONL file; one dict per iteration for offline analysis.
LOG_FILE = "./training_log.jsonl"
# Write a checkpoint every N iterations. Checkpointing serialises model
# weights + MeanStdFilter state so it is not free - every iteration is wasteful.
CHECKPOINT_INTERVAL = 10
# How often (seconds) to check whether the cluster has changed size or
# whether learner scaling is warranted.
REBUILD_INTERVAL = 60
# Seconds to sleep after a caught training exception before retrying,
# preventing a tight crash loop on a persistent failure.
WORKER_RESTART_DELAY = 5
# Ray cluster address. "auto" works when running on a node started with
# `ray start --head`. Set explicitly if auto-detection is unreliable.
RAY_ADDRESS = "auto"

# CPUs reserved per rollout worker. Each worker is one Python process running
# one env. Raycasting (BVH + glam, 65 rays, max_distance=10) is too cheap
# per-ray for Rayon parallelism to help - thread overhead exceeds compute
# savings. One worker per CPU outperforms fewer workers with multiple threads.
CPUS_PER_WORKER = 1

# Fraction of total iteration time spent in the learner that triggers scaling up.
# 0.4 means "if SGD takes >40% of wall time, add a learner".
LEARN_TIME_SCALE_UP_THRESHOLD = 0.4
# Fraction below which a learner is removed. Should be well below the scale-up
# threshold to create a dead-band and prevent oscillation.
LEARN_TIME_SCALE_DOWN_THRESHOLD = 0.15
# Minimum seconds between any consecutive scale events (up or down).
SCALE_COOLDOWN = 300
# Iterations to wait before allowing any scaling decision. The EMA needs
# time to stabilise - early measurements are noisy and can trigger premature
# scale events before the training signal is meaningful.
SCALE_WARMUP_ITERATIONS = 20
# Smoothing factor for the exponential moving average of timing metrics.
# Higher = faster to react, lower = more stable. 0.2 is a good default.
EMA_ALPHA = 0.2
# Hard cap on learner workers regardless of GPU availability.
MAX_LEARNERS = 4
# CPUs per learner worker. Used for data loading off the GPU; more than 2
# never helps for a flat MLP.
CPUS_PER_LEARNER = 2
# GPUs per learner worker. Always 1 - data-parallel multi-GPU adds all-reduce
# overhead that exceeds compute savings at flat MLP scale.
GPUS_PER_LEARNER = 1

# -1 means Ray restarts dead rollout workers indefinitely rather than
# crashing the run. Set to a positive integer to give up after N restarts.
MAX_WORKER_RESTARTS = -1

# Steps per worker per batch. Per-learner train batch = num_workers *
# STEPS_PER_WORKER / num_learners, so total data collected per iteration
# scales automatically when workers are added or removed. This value should
# be at least avg_episode_length * 2 so the value function sees multiple
# complete episodes per batch.
STEPS_PER_WORKER = 10_000
# Train batch is split into minibatches for each SGD epoch. Larger values
# improve GPU utilisation; 1024 is appropriate for a 122-dim flat MLP.
MINIBATCH_SIZE = 1024
# Full passes over the train batch per iteration. PPO's clip objective bounds
# how stale the data can be. Reduce if the policy collapses after updates.
NUM_SGD_EPOCHS = 10
# Bonus proportional to policy entropy, encouraging exploration. Anneal
# toward 0 manually once the agent reliably reaches the first few checkpoints.
ENTROPY_COEFF = 0.01
# Clips the probability ratio between old and new policy to
# [1 - CLIP_PARAM, 1 + CLIP_PARAM]. Limits update size. 0.2 is standard.
CLIP_PARAM = 0.2
# Clips the value function loss. Set to ~max expected undiscounted return.
VF_CLIP_PARAM = 10.0
# Discount factor. Higher values are needed for long episodes - at 5000
# steps, rewards from early in the episode are near-zero at 0.99.
GAMMA = 0.995
# GAE lambda. Controls bias-variance tradeoff: 1.0 = Monte Carlo,
# 0.0 = TD(0). 0.95 is the PPO default and rarely needs changing.
LAMBDA_GAE = 0.95

# Run deterministic evaluation every N training iterations.
EVAL_INTERVAL = 10
# Number of episodes to run per evaluation. More episodes = less noisy
# eval signal but more wall-clock time spent not training.
EVAL_EPISODES = 10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)


def _sanitise(v: object) -> object:
    """Replace float NaN with None so log lines are valid JSON."""
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def log_event(data: dict) -> None:
    """Append a JSON-serialisable dict as one line to the JSONL log file."""
    sanitised = {k: _sanitise(v) for k, v in data.items()}
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(sanitised) + "\n")


def cluster_resources() -> tuple[int, int]:
    """Return (total_cpus, total_gpus) from the live Ray cluster."""
    r = ray.cluster_resources()
    return int(r.get("CPU", 1)), int(r.get("GPU", 0))


def get_num_rollout_workers(num_learners: int, total_cpus: int) -> int:
    """Return how many rollout workers fit after reserving CPUs for learners."""
    reserved = num_learners * CPUS_PER_LEARNER
    return max(1, (total_cpus - reserved) // CPUS_PER_WORKER)


def get_train_batch_size_per_learner(num_workers: int, num_learners: int) -> int:
    """Return per-learner train batch size scaled to current worker/learner counts.

    Total data collected per iteration = num_learners * return value, which
    keeps per-worker contribution constant as the cluster grows or shrinks.
    """
    return max(STEPS_PER_WORKER * num_workers // num_learners, MINIBATCH_SIZE)


class LearnerScaler:
    """Tracks EMA of sample/learn times and decides when to scale learners up or down."""

    def __init__(self) -> None:
        self._ema_sample: float | None = None
        self._ema_learn: float | None = None
        self._last_scale_time: float = 0.0

    def reset(self) -> None:
        """Clear EMA state after a rebuild so stale ratios don't drive scaling."""
        self._ema_sample = None
        self._ema_learn = None

    def update(self, sample_s: float, learn_s: float) -> None:
        """Ingest one iteration's timing values into the running EMA."""
        if math.isnan(sample_s) or math.isnan(learn_s):
            return
        if self._ema_sample is None:
            self._ema_sample = sample_s
            self._ema_learn = learn_s
        else:
            self._ema_sample = EMA_ALPHA * sample_s + (1 - EMA_ALPHA) * self._ema_sample
            # pylint: disable=line-too-long
            self._ema_learn = EMA_ALPHA * learn_s + (1 - EMA_ALPHA) * self._ema_learn  # type: ignore[operator]

    def desired_learners(self, current: int, total_gpus: int, iteration: int) -> int:
        """Return the desired learner count based on smoothed timing ratios.

        Scales up by one when learn fraction exceeds LEARN_TIME_SCALE_UP_THRESHOLD
        and a free GPU exists. Scales down by one when learn fraction drops below
        LEARN_TIME_SCALE_DOWN_THRESHOLD and more than one learner is running.
        Both directions share SCALE_COOLDOWN and SCALE_WARMUP_ITERATIONS.
        """
        if self._ema_sample is None or self._ema_sample <= 0:
            return current
        if iteration < SCALE_WARMUP_ITERATIONS:
            return current
        if time.time() - self._last_scale_time < SCALE_COOLDOWN:
            return current

        ratio = self._ema_learn / (self._ema_sample + self._ema_learn)  # type: ignore[operator]

        if ratio > LEARN_TIME_SCALE_UP_THRESHOLD and total_gpus > current:
            new = min(current + 1, MAX_LEARNERS, total_gpus)
            if new != current:
                self._last_scale_time = time.time()
                log.warning(
                    "EMA learn_time %.0f%% of iteration - scaling learners %d -> %d",
                    ratio * 100,
                    current,
                    new,
                )
                return new

        if ratio < LEARN_TIME_SCALE_DOWN_THRESHOLD and current > 1:
            new = current - 1
            self._last_scale_time = time.time()
            log.info(
                "EMA learn_time %.0f%% of iteration - scaling learners down %d -> %d",
                ratio * 100,
                current,
                new,
            )
            return new

        return current


def build_trainer(num_learners: int, num_workers: int, export_string: str) -> Algorithm:
    """Construct and return a PPO Algorithm with the given worker and learner counts."""
    batch_per_learner = get_train_batch_size_per_learner(num_workers, num_learners)
    log.info(
        "Building trainer - workers: %d  learners: %d  batch_per_learner: %d",
        num_workers,
        num_learners,
        batch_per_learner,
    )
    config = (
        PPOConfig()
        .environment(
            env="PolyTrackEnv",
            env_config={"export_string": export_string},
        )
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .env_runners(
            num_env_runners=num_workers - 1,  # One evaluation runner
            num_cpus_per_env_runner=CPUS_PER_WORKER,
            num_envs_per_env_runner=1,  # TBD
            # pylint: disable=line-too-long
            env_to_module_connector=lambda env, spaces, device: MeanStdFilter(multi_agent=False),  # type: ignore
        )
        .fault_tolerance(
            max_num_env_runner_restarts=MAX_WORKER_RESTARTS,
        )
        .resources(
            num_cpus_for_main_process=0,
        )
        .learners(
            num_learners=num_learners,
            num_cpus_per_learner=CPUS_PER_LEARNER,
            num_gpus_per_learner=GPUS_PER_LEARNER,
        )
        .training(
            train_batch_size_per_learner=batch_per_learner,
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
        .evaluation(
            evaluation_interval=EVAL_INTERVAL,
            evaluation_duration=EVAL_EPISODES,
            evaluation_duration_unit="episodes",
            evaluation_num_env_runners=1,
            evaluation_config=PPOConfig.overrides(explore=False),
        )
        .reporting(
            min_sample_timesteps_per_iteration=batch_per_learner * num_learners,
            metrics_num_episodes_for_smoothing=50,
        )
    )
    return config.build_algo()


def save_checkpoint(trainer: Algorithm) -> str:
    """Save a checkpoint and return its path as a string."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    return str(trainer.save(CHECKPOINT_DIR))


def restore_weights(trainer: Algorithm, checkpoint_path: str) -> None:
    """Restore model weights and optimizer state from a checkpoint path."""
    trainer.restore(checkpoint_path)
    log.info("Full state restored from %s", checkpoint_path)


def extract_metrics(result: dict) -> dict:
    """Pull all relevant metrics out of the RLlib result dict.

    Timer values come from result["time_taken_this_iter_s"] (total wall time)
    and the per-phase breakdown under result["env_runners"] and
    result["learners"] on the new API stack. The old "timers" key with
    "sample_time_ms" / "learn_time_ms" no longer exists.
    """
    runners = result.get("env_runners", {})
    eval_runner = result.get("evaluation", {}).get("env_runners", {})
    learner = result.get("learners", {}).get("default_policy", {})

    # New API stack exposes per-phase timing under these keys (seconds).
    # EnvRunnerGroup sampling time is recorded on the runners sub-dict;
    # learner update time is recorded on the learners sub-dict.
    sample_s: float = runners.get("sample_time_s", float("nan"))
    learn_s: float = result.get("learners", {}).get("update_time_s", float("nan"))

    return {
        # Mean episode return across all rollout workers this iteration.
        "reward_mean": runners.get("episode_reward_mean", float("nan")),
        # Best and worst episode returns this iteration.
        "reward_max": runners.get("episode_reward_max", float("nan")),
        "reward_min": runners.get("episode_reward_min", float("nan")),
        # Average number of steps per episode. Approaches MAX_FRAMES if the
        # agent is not finishing; approaches 0 if it crashes immediately.
        "ep_len_mean": runners.get("episode_len_mean", float("nan")),
        # Cumulative episode and step counts since training started.
        "episodes_total": result.get("episodes_total", 0),
        "timesteps_total": result.get("num_env_steps_sampled_lifetime", 0),
        # Mean number of track checkpoints hit per episode. Populated when
        # PolyTrackEnv.step() returns {"checkpoints_hit": n} in info.
        "checkpoints_hit": runners.get("checkpoints_hit_mean", float("nan")),
        # Fraction of episodes that ended with terminated=True (reached the
        # finish line) vs truncated (hit MAX_FRAMES). Populated when
        # PolyTrackEnv.step() returns {"finished": 0 or 1} in info.
        "finish_rate": runners.get("finished_mean", float("nan")),
        # Deterministic evaluation reward - cleaner signal than training
        # reward because exploration noise is disabled.
        "eval_reward_mean": eval_runner.get("episode_reward_mean", float("nan")),
        # Deterministic evaluation episode length.
        "eval_ep_len_mean": eval_runner.get("episode_len_mean", float("nan")),
        # PPO surrogate policy loss. Should decrease over time; a sudden
        # spike suggests the policy update was too large.
        "policy_loss": learner.get("policy_loss", float("nan")),
        # Value function loss. High values mean the critic is struggling to
        # predict returns - consider raising VF_CLIP_PARAM if it stays high.
        "vf_loss": learner.get("vf_loss", float("nan")),
        # KL divergence between old and new policy. Values consistently
        # above ~0.02 suggest CLIP_PARAM or NUM_SGD_EPOCHS is too large.
        "kl": learner.get("mean_kl_loss", float("nan")),
        # Policy entropy. Should start high and decay as the policy
        # specialises. If it collapses to near zero early, raise ENTROPY_COEFF.
        "entropy": learner.get("entropy", float("nan")),
        # Wall time spent collecting rollouts (seconds). Should dominate
        # learn_time for this architecture.
        "sample_time_s": sample_s,
        # Wall time spent on SGD updates (seconds).
        "learn_time_s": learn_s,
    }


def log_metrics(
    iteration: int, num_workers: int, num_learners: int, metrics: dict
) -> None:
    """Write a one-line iteration summary to the console."""
    m = metrics
    log.info(
        "[iter %04d] reward=%.2f (min=%.2f max=%.2f)  eval=%.2f  "
        "ep_len=%.0f  episodes=%d  steps=%d  "
        "checkpoints=%.1f  finish_rate=%.2f  "
        "kl=%.4f  entropy=%.3f  vf_loss=%.2f  "
        "sample_s=%.1f  learn_s=%.1f  workers=%d  learners=%d",
        iteration,
        m["reward_mean"],
        m["reward_min"],
        m["reward_max"],
        m["eval_reward_mean"],
        m["ep_len_mean"],
        m["episodes_total"],
        m["timesteps_total"],
        m["checkpoints_hit"],
        m["finish_rate"],
        m["kl"],
        m["entropy"],
        m["vf_loss"],
        m["sample_time_s"],
        m["learn_time_s"],
        num_workers,
        num_learners,
    )


def do_rebuild(
    num_learners: int,
    num_workers: int,
    export_string: str,
    checkpoint_path: str | None,
) -> Algorithm:
    """Build a fresh trainer, restoring state from checkpoint if provided."""
    trainer = build_trainer(num_learners, num_workers, export_string)
    if checkpoint_path:
        trainer.restore(checkpoint_path)
    return trainer


def main() -> None:
    """Parse args, initialise Ray, then run the training loop."""
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
    scaler = LearnerScaler()

    if checkpoint_path:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
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
            scaler.update(metrics["sample_time_s"], metrics["learn_time_s"])
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
                new_learners = scaler.desired_learners(
                    num_learners, new_total_gpus, iteration
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
                    trainer = do_rebuild(
                        num_learners, num_workers, args.export_string, checkpoint_path
                    )
                    scaler.reset()

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
