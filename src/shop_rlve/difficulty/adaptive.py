"""RLVE adaptive difficulty state machine (Spec Section 7).

Implements per-environment adaptive difficulty tracking. Each environment
maintains independent state (l_i, h_i, a_i, b_i) that controls the
difficulty sampling range and promotion logic.

Algorithm:
    Per-env state: (l_i, h_i, a_i, b_i) initialized to (0, 0, 0, 0)
    Sampling: d ~ UniformInt([l_i, h_i])
    Update: when d == h_i, track a_i (correct), b_i (total)
    Promotion: when b_i >= tau_num:
        if a_i/b_i >= tau_acc then h_i += 1, clamp window d_delta
        Reset: a_i=0, b_i=0

Constants:
    tau_acc  = 0.9  (accuracy threshold for promotion)
    tau_num  = 32   (minimum rollouts before difficulty check)
    d_delta  = 4    (maximum difficulty window width: h_i - l_i <= d_delta)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-environment adaptive state
# ---------------------------------------------------------------------------


@dataclass
class AdaptiveState:
    """Per-environment adaptive difficulty state.

    Spec Section 7:
        (l_i, h_i, a_i, b_i) initialized to (0, 0, 0, 0)

    Attributes:
        low:           Lower bound of difficulty range (l_i).
        high:          Upper bound of difficulty range (h_i).
        correct_count: Number of correct completions at h_i (a_i).
        total_count:   Total completions at h_i (b_i).
    """

    low: int = 0
    high: int = 0
    correct_count: int = 0
    total_count: int = 0

    def accuracy(self) -> float:
        """Compute current accuracy at the frontier (a_i / b_i).

        Returns:
            Accuracy in [0, 1], or 0.0 if no trials yet.
        """
        if self.total_count == 0:
            return 0.0
        return self.correct_count / self.total_count

    def as_dict(self) -> dict[str, int]:
        """Serialize state to a dictionary."""
        return {
            "low": self.low,
            "high": self.high,
            "correct_count": self.correct_count,
            "total_count": self.total_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, int]) -> AdaptiveState:
        """Deserialize state from a dictionary.

        Args:
            data: Dict with keys: low, high, correct_count, total_count.

        Returns:
            AdaptiveState instance.
        """
        return cls(
            low=data.get("low", 0),
            high=data.get("high", 0),
            correct_count=data.get("correct_count", 0),
            total_count=data.get("total_count", 0),
        )


# ---------------------------------------------------------------------------
# Adaptive Difficulty Engine
# ---------------------------------------------------------------------------


class AdaptiveDifficultyEngine:
    """RLVE-style adaptive difficulty engine.

    Maintains per-environment difficulty state and implements the promotion
    logic from Spec Section 7. Each environment independently tracks
    performance at its difficulty frontier and promotes when accuracy
    exceeds the threshold.

    Attributes:
        env_ids:   List of environment IDs being tracked.
        tau_acc:   Accuracy threshold for promotion (default: 0.9).
        tau_num:   Minimum rollouts before promotion check (default: 32).
        d_delta:   Maximum window width h_i - l_i (default: 4).

    Example:
        >>> engine = AdaptiveDifficultyEngine(["PD", "SUB"], tau_num=8)
        >>> d = engine.sample_difficulty("PD")
        >>> d
        0
        >>> engine.update("PD", difficulty=0, is_correct=True)
        >>> engine.get_state("PD").correct_count
        1
    """

    def __init__(
        self,
        env_ids: list[str],
        tau_acc: float = 0.9,
        tau_num: int = 32,
        d_delta: int = 4,
    ) -> None:
        """Initialize the adaptive difficulty engine.

        Args:
            env_ids:  List of environment identifiers to track.
            tau_acc:  Accuracy threshold for promotion (default: 0.9).
            tau_num:  Minimum trials at frontier before promotion check (default: 32).
            d_delta:  Maximum difficulty window width (default: 4).
        """
        self.env_ids = list(env_ids)
        self.tau_acc = tau_acc
        self.tau_num = tau_num
        self.d_delta = d_delta

        # Initialize per-env state
        self._states: dict[str, AdaptiveState] = {
            env_id: AdaptiveState() for env_id in env_ids
        }

    def sample_difficulty(self, env_id: str, rng: np.random.Generator | None = None) -> int:
        """Sample a difficulty level for the given environment.

        Spec Section 7:
            d ~ UniformInt([l_i, h_i])

        Args:
            env_id: Environment identifier.
            rng:    Optional numpy random generator for reproducibility.
                    If None, uses numpy default.

        Returns:
            Sampled difficulty level (integer in [l_i, h_i]).

        Raises:
            KeyError: If env_id is not tracked.
        """
        state = self._get_state(env_id)
        if rng is None:
            rng = np.random.default_rng()

        # UniformInt on [low, high] (inclusive)
        if state.low == state.high:
            return state.low
        return int(rng.integers(state.low, state.high + 1))

    def update(
        self,
        env_id: str,
        difficulty: int,
        is_correct: bool,
    ) -> dict[str, Any] | None:
        """Update adaptive state after an episode completion.

        Spec Section 7:
            - Only track stats when d == h_i (frontier difficulty)
            - Increment b_i (total) and a_i (if correct)
            - When b_i >= tau_num: check accuracy
            - If a_i/b_i >= tau_acc: promote h_i += 1, clamp window
            - Reset a_i = 0, b_i = 0 after check

        Args:
            env_id:     Environment identifier.
            difficulty: The difficulty level that was used for the episode.
            is_correct: Whether the agent achieved IsCorrect on this episode.

        Returns:
            Promotion info dict if a promotion occurred, None otherwise.
            Promotion dict contains: env_id, old_low, old_high, new_low,
            new_high, accuracy, trials.

        Raises:
            KeyError: If env_id is not tracked.
        """
        state = self._get_state(env_id)

        # Only track when at the frontier (d == h_i)
        if difficulty != state.high:
            return None

        # Update counters
        state.total_count += 1
        if is_correct:
            state.correct_count += 1

        # Check for promotion
        if state.total_count >= self.tau_num:
            accuracy = state.accuracy()
            old_low = state.low
            old_high = state.high

            if accuracy >= self.tau_acc:
                # Promote: h_i += 1
                state.high += 1

                # Clamp window: ensure h_i - l_i <= d_delta
                if state.high - state.low > self.d_delta:
                    state.low = state.high - self.d_delta

                promotion_info = {
                    "env_id": env_id,
                    "old_low": old_low,
                    "old_high": old_high,
                    "new_low": state.low,
                    "new_high": state.high,
                    "accuracy": accuracy,
                    "trials": state.total_count,
                }

                logger.info(
                    "AdaptiveDifficulty PROMOTION [%s]: "
                    "[%d, %d] -> [%d, %d] (accuracy=%.3f, trials=%d)",
                    env_id,
                    old_low,
                    old_high,
                    state.low,
                    state.high,
                    accuracy,
                    state.total_count,
                )

                # Reset counters
                state.correct_count = 0
                state.total_count = 0

                return promotion_info
            else:
                # Did not meet threshold; reset counters and keep current bounds
                logger.debug(
                    "AdaptiveDifficulty NO_PROMOTE [%s]: "
                    "accuracy=%.3f < tau_acc=%.3f at [%d, %d] (trials=%d)",
                    env_id,
                    accuracy,
                    self.tau_acc,
                    old_low,
                    old_high,
                    state.total_count,
                )

                state.correct_count = 0
                state.total_count = 0

        return None

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    def get_state(self, env_id: str) -> AdaptiveState:
        """Get the adaptive state for an environment.

        Args:
            env_id: Environment identifier.

        Returns:
            Copy of the AdaptiveState for the environment.

        Raises:
            KeyError: If env_id is not tracked.
        """
        state = self._get_state(env_id)
        return AdaptiveState(
            low=state.low,
            high=state.high,
            correct_count=state.correct_count,
            total_count=state.total_count,
        )

    def get_all_states(self) -> dict[str, AdaptiveState]:
        """Get adaptive states for all tracked environments.

        Returns:
            Dict mapping env_id -> AdaptiveState (copies).
        """
        return {env_id: self.get_state(env_id) for env_id in self._states}

    def _get_state(self, env_id: str) -> AdaptiveState:
        """Get the internal mutable state for an environment.

        Args:
            env_id: Environment identifier.

        Returns:
            The internal AdaptiveState reference.

        Raises:
            KeyError: If env_id is not tracked.
        """
        if env_id not in self._states:
            raise KeyError(
                f"Environment '{env_id}' is not tracked. "
                f"Tracked environments: {sorted(self._states.keys())}"
            )
        return self._states[env_id]

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, env_id: str | None = None) -> None:
        """Reset adaptive state for one or all environments.

        Args:
            env_id: Environment to reset. If None, resets all environments.

        Raises:
            KeyError: If env_id is provided and not tracked.
        """
        if env_id is not None:
            if env_id not in self._states:
                raise KeyError(
                    f"Environment '{env_id}' is not tracked. "
                    f"Tracked environments: {sorted(self._states.keys())}"
                )
            self._states[env_id] = AdaptiveState()
            logger.debug("AdaptiveDifficulty RESET [%s]", env_id)
        else:
            for eid in self._states:
                self._states[eid] = AdaptiveState()
            logger.debug("AdaptiveDifficulty RESET ALL (%d envs)", len(self._states))

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the entire engine state to a dictionary.

        Returns:
            Dict with keys: env_ids, tau_acc, tau_num, d_delta, states.
        """
        return {
            "env_ids": list(self.env_ids),
            "tau_acc": self.tau_acc,
            "tau_num": self.tau_num,
            "d_delta": self.d_delta,
            "states": {
                env_id: state.as_dict()
                for env_id, state in self._states.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AdaptiveDifficultyEngine:
        """Deserialize an engine from a dictionary.

        Args:
            data: Dict produced by to_dict().

        Returns:
            AdaptiveDifficultyEngine with restored state.
        """
        engine = cls(
            env_ids=data["env_ids"],
            tau_acc=data.get("tau_acc", 0.9),
            tau_num=data.get("tau_num", 32),
            d_delta=data.get("d_delta", 4),
        )
        for env_id, state_data in data.get("states", {}).items():
            if env_id in engine._states:
                engine._states[env_id] = AdaptiveState.from_dict(state_data)
        return engine

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of all environment states.

        Returns:
            Multi-line string with per-env difficulty ranges and stats.
        """
        lines = ["AdaptiveDifficultyEngine Summary:"]
        lines.append(f"  tau_acc={self.tau_acc}, tau_num={self.tau_num}, d_delta={self.d_delta}")
        for env_id in sorted(self._states):
            state = self._states[env_id]
            acc_str = f"{state.accuracy():.3f}" if state.total_count > 0 else "N/A"
            lines.append(
                f"  [{env_id}] range=[{state.low}, {state.high}] "
                f"frontier_trials={state.total_count} "
                f"frontier_accuracy={acc_str}"
            )
        return "\n".join(lines)
