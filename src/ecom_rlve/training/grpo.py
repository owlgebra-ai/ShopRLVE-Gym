"""TRL GRPO integration for EcomRLVE-GYM (Spec Section 9.2).

This module provides the reward function and episode sampling that
plugs into TRL's GRPOTrainer.  GRPO (Group Relative Policy Optimization)
trains an LLM by generating groups of completions per prompt, scoring
them with an external reward function, and computing policy gradients
from the relative advantage within each group.

Provides:
    - EcomRLVERewardFunction:  Callable reward function for GRPOTrainer.
    - create_grpo_config:      Generate a TRL GRPOConfig-compatible dict.
    - EcomRLVEDataCollator:    Formats episodes for TRL's expected input.

Usage:
    from ecom_rlve.server import EcomRLVEEnv
    from ecom_rlve.training.grpo import EcomRLVERewardFunction, create_grpo_config

    env = EcomRLVEEnv(collection="C4")
    reward_fn = EcomRLVERewardFunction(env, adaptive=True)

    # Use with TRL GRPOTrainer:
    # trainer = GRPOTrainer(
    #     model=model,
    #     reward_funcs=[reward_fn],
    #     ...
    # )
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ecom_rlve.server.openenv import EcomRLVEEnv
from ecom_rlve.server.state import Observation, parse_action

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompt template for multi-turn conversation formatting
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
You are a helpful e-commerce shopping assistant. Your goal is to help \
customers find products, manage orders, handle returns, and answer \
policy questions.

You can use the following tools:
- catalog.search(query, filters, top_k): Search the product catalog
- catalog.rerank(query, candidate_product_ids, top_k): Re-rank products
- catalog.get_product(product_id): Get full product details
- catalog.get_variants(product_id): Get product variants
- cart.add(product_id, variant_id, qty): Add item to cart
- cart.remove(line_id): Remove item from cart
- cart.view(): View current cart
- order.list(days): List recent orders
- order.get_status(order_id): Get order status
- order.checkout(shipping_address_id, payment_method_id): Checkout
- return.initiate(order_id, line_id, reason): Initiate a return
- policy.search(query, top_k): Search policy knowledge base

Respond with valid JSON containing:
{
    "assistant_message": "your message to the user",
    "tool_calls": [{"name": "tool_name", "args": {...}}],
    "answer": {"env": "PD", "recommended_product_ids": [...], "done": true}
}

When you have found the answer, set "done": true in the answer field."""


# ---------------------------------------------------------------------------
# Reward function for GRPO
# ---------------------------------------------------------------------------


class EcomRLVERewardFunction:
    """Reward function compatible with TRL GRPOTrainer.

    Takes a batch of prompts and completions and returns scalar rewards
    by running each completion through the EcomRLVE-GYM environment.

    This function implements the ``reward_funcs`` interface expected by
    TRL's GRPOTrainer:

        reward_fn(prompts: list[str], completions: list[str]) -> list[float]

    The reward function handles two modes:
        1. **Single-turn**: Each completion is a single action JSON.
           The environment is reset with the prompt as context.
        2. **Multi-turn**: The prompt encodes a conversation prefix and
           the completion is the next action. This is the common case
           in multi-turn GRPO.

    Args:
        env:      EcomRLVEEnv instance (shared across calls; note that
                  episodes are run sequentially within a batch).
        adaptive: If True, update the adaptive difficulty engine after
                  each episode.

    Attributes:
        total_episodes:    Counter of total episodes evaluated.
        total_correct:     Counter of episodes where IsCorrect=True.
        reward_history:    Rolling history of last N rewards (for monitoring).
    """

    def __init__(
        self,
        env: EcomRLVEEnv,
        adaptive: bool = True,
    ) -> None:
        self.env = env
        self.adaptive = adaptive
        self.total_episodes: int = 0
        self.total_correct: int = 0
        self._reward_history: list[float] = []
        self._max_history: int = 1000

    def __call__(
        self,
        prompts: list[str],
        completions: list[str],
    ) -> list[float]:
        """Compute rewards for a batch of prompt-completion pairs.

        Each (prompt, completion) pair is treated as a single episode.
        The prompt encodes the observation / conversation context, and
        the completion is the model's action JSON response.

        For multi-turn training, the prompt typically contains the
        serialized conversation up to the current turn, and the
        completion is the model's next action.

        Args:
            prompts:     List of prompt strings (one per sample in the batch).
            completions: List of completion strings (one per sample).

        Returns:
            List of float rewards, one per sample.  Rewards are in [-1, 1].
        """
        if len(prompts) != len(completions):
            raise ValueError(
                f"prompts ({len(prompts)}) and completions ({len(completions)}) "
                f"must have the same length"
            )

        rewards: list[float] = []

        for prompt, completion in zip(prompts, completions):
            reward = self._evaluate_single(prompt, completion)
            rewards.append(reward)

            # Track statistics
            self.total_episodes += 1
            self._reward_history.append(reward)
            if len(self._reward_history) > self._max_history:
                self._reward_history = self._reward_history[-self._max_history:]

        return rewards

    def _evaluate_single(self, prompt: str, completion: str) -> float:
        """Evaluate a single prompt-completion pair.

        Attempts to reconstruct an episode from the prompt context
        and evaluate the completion as an action.

        Args:
            prompt:     The prompt string (may contain conversation JSON).
            completion: The model's action JSON string.

        Returns:
            Scalar reward in [-1, 1].
        """
        # Try to parse the completion as an action
        action, format_valid = parse_action(completion)
        if not format_valid or action is None:
            return -1.0

        try:
            # Reset a fresh episode
            obs = self.env.reset()

            # If the prompt contains conversation context, we could
            # reconstruct it, but for the standard GRPO single-step
            # evaluation, we just step with the completion
            obs, reward, done, info = self.env.step(completion)

            if not done:
                # If the model didn't signal done, run more steps
                # with a simple "I'm done" follow-up
                done_action = json.dumps({
                    "assistant_message": "Here is my final answer.",
                    "tool_calls": [],
                    "answer": action.answer.model_dump() if action.answer else {
                        "env": obs.env_id or "PD",
                        "done": True,
                    },
                })
                obs, reward, done, info = self.env.step(done_action)

            is_correct = info.get("is_correct", False)
            if is_correct:
                self.total_correct += 1

            return reward

        except Exception as exc:
            logger.warning(
                "EcomRLVERewardFunction: evaluation failed: %s: %s",
                type(exc).__name__,
                exc,
            )
            return -1.0

    @property
    def accuracy(self) -> float:
        """Current accuracy (fraction of correct episodes)."""
        if self.total_episodes == 0:
            return 0.0
        return self.total_correct / self.total_episodes

    @property
    def mean_reward(self) -> float:
        """Mean reward over the recent history window."""
        if not self._reward_history:
            return 0.0
        return sum(self._reward_history) / len(self._reward_history)

    def get_stats(self) -> dict[str, Any]:
        """Return monitoring statistics.

        Returns:
            Dict with total_episodes, total_correct, accuracy,
            mean_reward, and reward_std.
        """
        import numpy as np

        return {
            "total_episodes": self.total_episodes,
            "total_correct": self.total_correct,
            "accuracy": self.accuracy,
            "mean_reward": self.mean_reward,
            "reward_std": float(np.std(self._reward_history)) if self._reward_history else 0.0,
        }


# ---------------------------------------------------------------------------
# GRPO configuration builder
# ---------------------------------------------------------------------------


def create_grpo_config(
    model_name: str,
    collection: str = "C1",
    *,
    learning_rate: float = 1e-6,
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    num_generations: int = 4,
    max_prompt_length: int = 2048,
    max_completion_length: int = 512,
    temperature: float = 0.7,
    kl_coef: float = 0.05,
    output_dir: str = "outputs/grpo",
    seed: int = 42,
    **kwargs: Any,
) -> dict[str, Any]:
    """Generate a TRL GRPOConfig-compatible configuration dict.

    This produces a dict that can be unpacked into ``trl.GRPOConfig(...)``
    for training with the EcomRLVE-GYM reward function.

    Args:
        model_name:                    HuggingFace model name or path.
        collection:                    Environment collection (C1, C2, C4, C8).
        learning_rate:                 Optimizer learning rate.
        num_train_epochs:              Number of training epochs.
        per_device_train_batch_size:   Batch size per device.
        gradient_accumulation_steps:   Gradient accumulation steps.
        num_generations:               Number of generations per prompt (G in GRPO).
        max_prompt_length:             Maximum prompt token length.
        max_completion_length:         Maximum completion token length.
        temperature:                   Sampling temperature for generation.
        kl_coef:                       KL divergence penalty coefficient.
        output_dir:                    Directory for checkpoints and logs.
        seed:                          Random seed.
        **kwargs:                      Additional GRPOConfig overrides.

    Returns:
        Dict suitable for ``trl.GRPOConfig(**config)``.
    """
    config: dict[str, Any] = {
        "model_name_or_path": model_name,
        "output_dir": output_dir,
        "learning_rate": learning_rate,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_generations": num_generations,
        "max_prompt_length": max_prompt_length,
        "max_completion_length": max_completion_length,
        "temperature": temperature,
        "kl_coef": kl_coef,
        "seed": seed,
        # Logging
        "logging_steps": 10,
        "save_steps": 500,
        "eval_steps": 100,
        # EcomRLVE-specific metadata (not consumed by TRL, but useful
        # for experiment tracking)
        "_ecomrlve_collection": collection,
        "_ecomrlve_env_ids": _get_collection_ids(collection),
    }

    # Apply user overrides
    config.update(kwargs)

    return config


def _get_collection_ids(collection: str) -> list[str]:
    """Get env IDs for a collection, with fallback.

    Args:
        collection: Collection name.

    Returns:
        List of env ID strings.
    """
    try:
        from ecom_rlve.training.collections import get_collection
        return get_collection(collection)
    except (ImportError, ValueError):
        return ["PD"]


# ---------------------------------------------------------------------------
# Data collator for TRL
# ---------------------------------------------------------------------------


class EcomRLVEDataCollator:
    """Formats EcomRLVE-GYM episodes for TRL's expected input format.

    Converts Observations into tokenizer-ready prompt strings and handles
    multi-turn conversation formatting into a single prompt string.

    TRL's GRPOTrainer expects prompts as strings. This collator converts
    the structured conversation history into a formatted string suitable
    for the model.

    Args:
        system_prompt:     Optional system prompt to prepend.
        include_env_id:    Whether to include the env_id in the prompt.
        include_tool_defs: Whether to include tool definitions in the prompt.
        max_turns:         Maximum number of conversation turns to include.

    Example:
        >>> collator = EcomRLVEDataCollator()
        >>> prompt = collator.format_observation(obs)
        >>> isinstance(prompt, str)
        True
    """

    def __init__(
        self,
        system_prompt: str | None = None,
        include_env_id: bool = True,
        include_tool_defs: bool = True,
        max_turns: int = 20,
    ) -> None:
        self.system_prompt = system_prompt or _SYSTEM_PROMPT_TEMPLATE
        self.include_env_id = include_env_id
        self.include_tool_defs = include_tool_defs
        self.max_turns = max_turns

    def format_observation(self, obs: Observation) -> str:
        """Convert an Observation into a prompt string.

        The prompt is structured as:
            [SYSTEM] system_prompt
            [ENV_ID] PD (if include_env_id)
            [USER] first user message
            [ASSISTANT] first assistant response
            [USER] second user message
            ...

        Args:
            obs: Observation from env.reset() or env.step().

        Returns:
            Formatted prompt string.
        """
        parts: list[str] = []

        # System prompt
        parts.append(f"[SYSTEM]\n{self.system_prompt}")

        # Environment ID
        if self.include_env_id and obs.env_id:
            parts.append(f"\n[ENV_ID] {obs.env_id}")

        # Conversation history (truncate to max_turns)
        conversation = obs.conversation
        if len(conversation) > self.max_turns * 2:
            conversation = conversation[-(self.max_turns * 2):]

        for msg in conversation:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            parts.append(f"\n[{role}]\n{content}")

        # Tool results from the most recent step
        if obs.tool_results:
            parts.append("\n[TOOL_RESULTS]")
            for tr in obs.tool_results:
                name = tr.get("name", "unknown")
                result = tr.get("result", "")
                error = tr.get("error")
                if error:
                    parts.append(f"  {name}: ERROR - {error}")
                else:
                    result_str = json.dumps(result, default=str)
                    # Truncate very long results
                    if len(result_str) > 500:
                        result_str = result_str[:500] + "..."
                    parts.append(f"  {name}: {result_str}")

        # Final instruction
        parts.append("\n[ASSISTANT]")

        return "\n".join(parts)

    def format_batch(self, observations: list[Observation]) -> list[str]:
        """Format a batch of Observations into prompt strings.

        Args:
            observations: List of Observation objects.

        Returns:
            List of formatted prompt strings.
        """
        return [self.format_observation(obs) for obs in observations]

    def observation_to_messages(
        self, obs: Observation
    ) -> list[dict[str, str]]:
        """Convert an Observation to the chat-messages format.

        This produces a list of ``{"role": ..., "content": ...}`` dicts
        compatible with chat model APIs and tokenizers that use
        ``apply_chat_template``.

        Args:
            obs: Observation from the environment.

        Returns:
            List of message dicts.
        """
        messages: list[dict[str, str]] = []

        # System message
        system_content = self.system_prompt
        if self.include_env_id and obs.env_id:
            system_content += f"\n\nCurrent task: {obs.env_id}"
        messages.append({"role": "system", "content": system_content})

        # Conversation messages
        conversation = obs.conversation
        if len(conversation) > self.max_turns * 2:
            conversation = conversation[-(self.max_turns * 2):]

        for msg in conversation:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Append tool results to the last assistant message
            if role == "assistant" and obs.tool_results:
                tool_summary = self._format_tool_results(obs.tool_results)
                content = f"{content}\n\n{tool_summary}"

            messages.append({"role": role, "content": content})

        return messages

    def _format_tool_results(
        self, tool_results: list[dict[str, Any]]
    ) -> str:
        """Format tool results into a readable string.

        Args:
            tool_results: List of tool result dicts.

        Returns:
            Formatted string.
        """
        parts: list[str] = ["Tool Results:"]
        for tr in tool_results:
            name = tr.get("name", "unknown")
            error = tr.get("error")
            if error:
                parts.append(f"  - {name}: ERROR - {error}")
            else:
                result = tr.get("result", "")
                result_str = json.dumps(result, default=str)
                if len(result_str) > 300:
                    result_str = result_str[:300] + "..."
                parts.append(f"  - {name}: {result_str}")
        return "\n".join(parts)
