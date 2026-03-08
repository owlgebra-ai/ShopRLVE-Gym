"""Episode replay and environment probing for ShopRLVE-GYM debugging.

Provides:
    - replay_episode: Re-run an episode from its trace to verify determinism.
    - probe_env:      Quick diagnostic -- run N episodes with DummyModelFn and
                      return statistics.

Usage:
    from shop_rlve.debug.replay import replay_episode, probe_env

    stats = probe_env("PD", difficulty=3, n_episodes=20)
    print(stats["mean_reward"], stats["success_rate"])
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any

import numpy as np

from shop_rlve.envs.base import EpisodeResult
from shop_rlve.server.openenv import ShopRLVEEnv
from shop_rlve.training.rollout import DummyModelFn, RolloutResult, run_rollout

logger = logging.getLogger(__name__)


def replay_episode(
    env: ShopRLVEEnv,
    trace: dict[str, Any],
    step_by_step: bool = False,
) -> EpisodeResult:
    """Re-run an episode from its trace data to verify determinism.

    Replays the episode using the same env_id, difficulty, and seed from
    the trace.  The original assistant messages are fed back as actions so
    the environment follows the exact same path.  At the end, the reward
    is compared with the original to check determinism.

    Args:
        env:          ShopRLVEEnv instance (should have the same catalog).
        trace:        Episode trace dict (from ``ShopRLVEEnv.get_episode_trace()``).
        step_by_step: If True, print each step's state to the console.

    Returns:
        EpisodeResult with r_task and is_correct from the replayed episode.
    """
    env_id: str = trace.get("env_id", "PD")
    difficulty: int = trace.get("difficulty", 0)
    seed: int = trace.get("seed", 42)
    original_reward: float | None = trace.get("reward")

    if step_by_step:
        print(f"[Replay] Starting: env={env_id}, d={difficulty}, seed={seed}")

    # Reset the env with the same params
    obs = env.reset(env_id=env_id, difficulty=difficulty, seed=seed)

    # Extract assistant messages from the original conversation
    conversation: list[dict[str, str]] = trace.get("conversation", [])
    tool_history: list[dict[str, Any]] = trace.get("tool_results_history", [])

    # Build action sequence from the original conversation
    # The conversation alternates user / assistant.  We need to
    # reconstruct the action JSON the model originally produced.
    actions: list[str] = []
    tool_idx = 0
    for msg in conversation:
        if msg.get("role") == "assistant":
            # Reconstruct the action JSON
            # Collect tool calls that belong to this assistant turn
            turn_tools: list[dict[str, Any]] = []
            while tool_idx < len(tool_history):
                tr = tool_history[tool_idx]
                turn_tools.append({
                    "name": tr.get("name", ""),
                    "args": tr.get("args", {}),
                })
                tool_idx += 1
                # Heuristic: one batch of tool calls per assistant turn
                break

            action_dict: dict[str, Any] = {
                "assistant_message": msg.get("content", ""),
                "tool_calls": turn_tools,
            }
            actions.append(json.dumps(action_dict))

    # If the original episode ended with a done answer, the last action
    # should include that.  Since we are replaying from the trace and the
    # trace conversation may not capture the full answer payload, we add
    # a fallback done answer on the final action.
    if actions:
        last_action = json.loads(actions[-1])
        if "answer" not in last_action or last_action.get("answer") is None:
            last_action["answer"] = {
                "env": env_id,
                "done": True,
                "recommended_product_ids": trace.get("seen_product_ids", [])[:5],
            }
            actions[-1] = json.dumps(last_action)

    # Replay the episode
    done = False
    reward = 0.0
    info: dict[str, Any] = {}
    turn = 0

    for action_json in actions:
        if done:
            break
        obs, reward, done, info = env.step(action_json)
        turn += 1

        if step_by_step:
            print(
                f"  [Replay] Turn {turn}: reward={reward:.4f}, done={done}"
            )

    # If we ran out of actions but the episode is not done, force termination
    if not done:
        fallback_action = json.dumps({
            "assistant_message": "I apologize, ending the conversation.",
            "tool_calls": [],
            "answer": {"env": env_id, "done": True},
        })
        obs, reward, done, info = env.step(fallback_action)

    # Check determinism
    is_correct: bool = info.get("is_correct", False)
    if original_reward is not None and abs(reward - original_reward) > 1e-6:
        logger.warning(
            "Replay MISMATCH: original_reward=%.6f, replay_reward=%.6f "
            "(delta=%.6f).  Determinism check FAILED.",
            original_reward,
            reward,
            abs(reward - original_reward),
        )
    elif original_reward is not None:
        logger.info(
            "Replay determinism check PASSED (reward=%.6f)", reward
        )

    return EpisodeResult(
        r_task=info.get("reward_breakdown", {}).get("r_task", reward),
        is_correct=is_correct,
        details={
            "replay_reward": reward,
            "original_reward": original_reward,
            "turns": turn,
            "deterministic": (
                original_reward is not None and abs(reward - original_reward) < 1e-6
            ),
        },
    )


def probe_env(
    env_id: str,
    difficulty: int,
    n_episodes: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    """Run N episodes with DummyModelFn and return diagnostic statistics.

    This is a key debug lever for checking whether an environment is
    functioning correctly at a given difficulty level.

    Args:
        env_id:     Environment identifier (PD, SUB, CART, etc.).
        difficulty: Fixed difficulty level to probe.
        n_episodes: Number of episodes to run.
        seed:       Master random seed.

    Returns:
        Dict with keys:
            - env_id:           str
            - difficulty:       int
            - n_episodes:       int
            - mean_reward:      float
            - std_reward:       float
            - mean_turns:       float
            - success_rate:     float (fraction of is_correct episodes)
            - reward_histogram: list[int] (10 bins from -1 to 1)
            - rewards:          list[float]
            - min_reward:       float
            - max_reward:       float
    """
    # Determine the collection that contains this env_id
    from shop_rlve.training.collections import COLLECTIONS

    collection = "C8"  # C8 contains all envs
    for cname, cenvs in COLLECTIONS.items():
        if env_id in cenvs:
            collection = cname
            break

    env = ShopRLVEEnv(collection=collection, seed=seed)
    env.dump_dir = ""  # Disable trace dumping during probe

    # Get some product IDs from the catalog for DummyModelFn
    product_ids = [p.id for p in env._products[:20]]

    rewards: list[float] = []
    turns_list: list[int] = []
    correct_count = 0

    for i in range(n_episodes):
        ep_seed = seed + i
        dummy = DummyModelFn(
            env_id=env_id,
            product_ids=product_ids,
            seed=ep_seed,
        )

        result: RolloutResult = run_rollout(
            env=env,
            model_fn=dummy,
            env_id=env_id,
            difficulty=difficulty,
            seed=ep_seed,
        )

        rewards.append(result.reward)
        turns_list.append(result.turns)
        if result.is_correct:
            correct_count += 1

    env.close()

    rewards_arr = np.array(rewards, dtype=np.float64)

    # Build histogram (10 bins from -1 to 1)
    hist_counts, _ = np.histogram(rewards_arr, bins=10, range=(-1.0, 1.0))

    return {
        "env_id": env_id,
        "difficulty": difficulty,
        "n_episodes": n_episodes,
        "mean_reward": float(np.mean(rewards_arr)),
        "std_reward": float(np.std(rewards_arr)),
        "mean_turns": float(np.mean(turns_list)),
        "success_rate": correct_count / max(n_episodes, 1),
        "reward_histogram": hist_counts.tolist(),
        "rewards": rewards,
        "min_reward": float(np.min(rewards_arr)),
        "max_reward": float(np.max(rewards_arr)),
    }
