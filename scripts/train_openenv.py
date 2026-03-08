#!/usr/bin/env python3
"""ShopRLVE-GYM OpenEnv RL Training Script.

Trains a language model to act as an e-commerce shopping assistant using
Reinforcement Learning (GRPO) with ShopRLVE-GYM environments as the
reward signal.  Follows the Unsloth + TRL GRPOTrainer pattern from the
OpenEnv 2048 notebook, adapted for multi-turn e-commerce conversation.

Usage:
    # Basic run with defaults (Qwen3-1.7B, C1, 300 steps)
    python scripts/train_openenv.py

    # Full options
    python scripts/train_openenv.py \
        --model Qwen/Qwen3-1.7B \
        --collection C1 \
        --max_steps 300 \
        --lora_rank 16 \
        --num_generations 4 \
        --load_in_4bit \
        --output_dir outputs/shoprlve_grpo

Requires: pip install unsloth trl transformers datasets torch
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Ensure shop_rlve is importable (add src/ to path if needed)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from shop_rlve.server.openenv import ShopRLVEEnv
from shop_rlve.server.state import parse_action
from shop_rlve.training.collections import COLLECTIONS, get_collection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("shoprlve.train_openenv")


# ===================================================================
# System prompt (what the model is told about its role & tools)
# ===================================================================

SYSTEM_PROMPT = """\
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

When you have found the answer, set "done": true in the answer field.\
"""


# ===================================================================
# Environment wrapper — one persistent ShopRLVEEnv instance
# ===================================================================

class ShopRLVEOpenEnv:
    """Thin wrapper around ShopRLVEEnv that stores state for reward
    computation across the GRPO generate → reward pipeline.

    GRPOTrainer generates completions first, then calls reward functions.
    We must be able to reconstruct an episode for each (prompt, completion)
    pair.  Since each prompt is generated from a fresh env.reset(), we
    cache the Observation so the reward function can evaluate it.
    """

    def __init__(self, collection: str = "C1", seed: int = 42) -> None:
        self.env = ShopRLVEEnv(collection=collection, seed=seed)
        self.env.dump_dir = ""       # Disable disk trace during training
        self.env.trace_episodes = False
        self.env.validate_rewards = True
        self.collection = collection
        self.env_ids = get_collection(collection)
        self._episode_counter = 0

    def sample_prompt(self, tokenizer: Any) -> tuple[list[dict[str, str]], str, int]:
        """Reset the env and produce a chat-messages prompt.

        Returns:
            Tuple of (messages, env_id, episode_seed) where messages is
            a list of {"role": ..., "content": ...} dicts suitable for
            tokenizer.apply_chat_template(), and env_id + episode_seed
            can be used to deterministically re-create the exact same
            episode for reward evaluation.
        """
        # Cycle through envs uniformly
        env_id = self.env_ids[self._episode_counter % len(self.env_ids)]
        self._episode_counter += 1

        # Use a deterministic seed so the reward function can
        # reconstruct the exact same episode (same target products,
        # constraints, user message, etc.)
        episode_seed = self._episode_counter * 1000 + 42

        obs = self.env.reset(env_id=env_id, seed=episode_seed)

        # Build messages: [system, user]
        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        for msg in obs.conversation:
            messages.append({"role": msg["role"], "content": msg["content"]})

        return messages, env_id, episode_seed

    def evaluate_completion(
        self,
        completion: str,
        env_id: str,
        episode_seed: int,
    ) -> dict[str, Any]:
        """Run one episode: reset with the SAME env_id and seed that
        generated the prompt, then step with the completion.

        Because env.reset() is deterministic for a given (env_id, seed),
        this re-creates the exact same problem instance (same hidden
        goal, constraints, target products, user message, persona) that
        the model saw during generation.

        Args:
            completion:    The model's raw JSON action string.
            env_id:        Environment ID used when the prompt was created.
            episode_seed:  Seed used when the prompt was created.

        Returns:
            Dict with reward, is_correct, turn, termination_reason,
            and reward_breakdown.
        """
        # Re-create the exact same episode the model was prompted with
        obs = self.env.reset(env_id=env_id, seed=episode_seed)

        # Step with the model completion
        obs, reward, done, info = self.env.step(completion)

        # If the model didn't signal done, force a terminal answer
        if not done:
            action, valid = parse_action(completion)
            fallback_env = obs.env_id or env_id
            fallback_answer: dict[str, Any] = {"env": fallback_env, "done": True}
            if action and action.answer:
                fallback_answer = action.answer.model_dump()
                fallback_answer["done"] = True
            else:
                fallback_answer["recommended_product_ids"] = []

            done_action = json.dumps({
                "assistant_message": "Here is my final answer.",
                "tool_calls": [],
                "answer": fallback_answer,
            })
            obs, reward, done, info = self.env.step(done_action)

        return {
            "reward": reward,
            "is_correct": info.get("is_correct", False),
            "turn": info.get("turn", 0),
            "termination_reason": info.get("termination_reason", "unknown"),
            "reward_breakdown": info.get("reward_breakdown", {}),
        }


# ===================================================================
# Reward functions for GRPOTrainer
# ===================================================================

# Global env wrapper -- initialized in main()
_OPENENV: ShopRLVEOpenEnv | None = None
_PRINT_COUNTER: int = 0


def _extract_json_from_completion(text: str) -> str | None:
    """Extract the first JSON object from a completion string.

    Handles cases where the model wraps JSON in markdown code blocks
    or emits thinking tokens before the JSON.
    """
    # Try direct JSON parse first
    text = text.strip()

    # Strip markdown code fences if present
    if "```" in text:
        first = text.find("```") + 3
        second = text.find("```", first)
        if second > first:
            candidate = text[first:second].strip()
            candidate = candidate.removeprefix("json\n").removeprefix("json").strip()
            text = candidate

    # Find the first '{' and last '}'
    start = text.find("{")
    if start == -1:
        return None
    end = text.rfind("}")
    if end == -1 or end < start:
        return None

    candidate = text[start : end + 1]
    try:
        json.loads(candidate)
        return candidate
    except json.JSONDecodeError:
        return None


def format_reward(completions: list[list[dict[str, str]]], **kwargs: Any) -> list[float]:
    """Reward: does the completion parse as valid ShopRLVE action JSON?

    Checks that the output contains a valid JSON object with the
    required 'assistant_message' field.

    +1.0  valid JSON with assistant_message
    -0.5  valid JSON but missing required fields
    -2.0  invalid JSON / no JSON found
    """
    scores: list[float] = []
    for completion in completions:
        response = completion[0]["content"]
        extracted = _extract_json_from_completion(response)

        if extracted is None:
            scores.append(-2.0)
            continue

        action, valid = parse_action(extracted)
        if valid and action is not None:
            scores.append(1.0)
        else:
            scores.append(-0.5)

    return scores


def tool_usage_reward(completions: list[list[dict[str, str]]], **kwargs: Any) -> list[float]:
    """Reward: does the completion use tools appropriately?

    +1.0  has well-formed tool_calls with valid tool names
    +0.5  has answer with done=true but no tool calls (acceptable for
          simple tasks)
    -0.5  has tool_calls but with invalid tool names
    -1.0  no JSON / unparseable
    """
    VALID_TOOL_PREFIXES = {"catalog.", "cart.", "order.", "return.", "policy."}

    scores: list[float] = []
    for completion in completions:
        response = completion[0]["content"]
        extracted = _extract_json_from_completion(response)

        if extracted is None:
            scores.append(-1.0)
            continue

        action, valid = parse_action(extracted)
        if not valid or action is None:
            scores.append(-1.0)
            continue

        if action.tool_calls:
            all_valid = all(
                any(tc.name.startswith(prefix) for prefix in VALID_TOOL_PREFIXES)
                for tc in action.tool_calls
            )
            scores.append(1.0 if all_valid else -0.5)
        elif action.answer and action.answer.done:
            scores.append(0.5)
        else:
            # No tools and no done answer — unhelpful
            scores.append(-0.25)

    return scores


def env_reward(completions: list[list[dict[str, str]]], **kwargs: Any) -> list[float]:
    """Reward: run the completion through the ShopRLVE-GYM environment
    and return the environment's scalar reward.

    This is the core reward that evaluates actual e-commerce task
    performance: product recommendation quality, cart correctness,
    return handling, etc.

    TRL GRPOTrainer passes extra dataset columns through **kwargs,
    so we receive `env_id` and `episode_seed` lists that let us
    reconstruct the exact same episode the prompt came from.

    Reward range: [-1.0, 1.0] from the environment, scaled by 5.0
    to make it the dominant signal.
    """
    global _OPENENV, _PRINT_COUNTER
    assert _OPENENV is not None, "ShopRLVEOpenEnv not initialized"

    # Extract episode identifiers from kwargs (set in the dataset)
    env_ids = kwargs.get("env_id", [])
    episode_seeds = kwargs.get("episode_seed", [])

    scores: list[float] = []
    for i, completion in enumerate(completions):
        response = completion[0]["content"]
        extracted = _extract_json_from_completion(response)

        should_print = (_PRINT_COUNTER % 10 == 0)
        _PRINT_COUNTER += 1

        if extracted is None:
            if should_print:
                logger.info("[env_reward] No JSON found in completion")
            scores.append(-1.0)
            continue

        # Recover the env_id and seed for this sample
        eid = env_ids[i] if i < len(env_ids) else _OPENENV.env_ids[0]
        eseed = episode_seeds[i] if i < len(episode_seeds) else 42

        try:
            result = _OPENENV.evaluate_completion(
                completion=extracted,
                env_id=eid,
                episode_seed=int(eseed),
            )
            env_score = result["reward"]

            if should_print:
                logger.info(
                    "[env_reward] env=%s seed=%d reward=%.4f correct=%s reason=%s breakdown=%s",
                    eid,
                    eseed,
                    env_score,
                    result["is_correct"],
                    result["termination_reason"],
                    {k: f"{v:.3f}" if isinstance(v, float) else v
                     for k, v in result.get("reward_breakdown", {}).items()
                     if k in ("r_task", "r_eff", "r_hall", "r_total")},
                )

            # Scale the env reward to dominate the combined signal
            scores.append(float(env_score) * 5.0)

        except Exception as exc:
            logger.warning("[env_reward] Exception: %s: %s", type(exc).__name__, exc)
            scores.append(-3.0)

    return scores


# ===================================================================
# Dataset builder
# ===================================================================

def build_dataset(
    openenv: ShopRLVEOpenEnv,
    tokenizer: Any,
    n_prompts: int = 1000,
) -> "Dataset":
    """Build a HuggingFace Dataset of prompts sampled from ShopRLVE-GYM.

    Each row has:
        - 'prompt':        list of chat messages for apply_chat_template()
        - 'env_id':        environment ID (e.g. "PD", "SUB", ...)
        - 'episode_seed':  deterministic seed for env.reset()

    TRL GRPOTrainer passes extra columns through to reward functions
    as **kwargs, allowing the reward function to reconstruct the exact
    same episode that generated each prompt.
    """
    from datasets import Dataset

    rows: list[dict[str, Any]] = []
    for i in range(n_prompts):
        messages, env_id, episode_seed = openenv.sample_prompt(tokenizer)
        rows.append({
            "prompt": messages,
            "env_id": env_id,
            "episode_seed": episode_seed,
        })

    dataset = Dataset.from_list(rows)
    logger.info(
        "Built dataset with %d prompts (envs: %s)",
        len(dataset),
        ", ".join(sorted(set(r["env_id"] for r in rows))),
    )
    return dataset


# ===================================================================
# Main training loop
# ===================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ShopRLVE-GYM OpenEnv RL Training with GRPO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Model
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-1.7B",
        help="HuggingFace model name or path (default: Qwen/Qwen3-1.7B)",
    )
    parser.add_argument(
        "--load_in_4bit", action="store_true", default=True,
        help="Load model in 4-bit quantization (default: True)",
    )
    parser.add_argument(
        "--load_in_16bit", action="store_true", default=False,
        help="Load model in 16-bit (overrides --load_in_4bit)",
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=2048,
        help="Maximum sequence length (default: 2048)",
    )
    parser.add_argument(
        "--lora_rank", type=int, default=16,
        help="LoRA rank (default: 16)",
    )

    # Environment
    parser.add_argument(
        "--collection", type=str, default="C1",
        choices=sorted(COLLECTIONS.keys()),
        help="Environment collection (default: C1)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--n_prompts", type=int, default=1000,
        help="Number of training prompts to generate (default: 1000)",
    )

    # Training
    parser.add_argument(
        "--max_steps", type=int, default=300,
        help="Maximum training steps (default: 300)",
    )
    parser.add_argument(
        "--num_generations", type=int, default=4,
        help="GRPO group size G (default: 4)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Per-device train batch size (default: 1)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1,
        help="Gradient accumulation steps (default: 1)",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5,
        help="Learning rate (default: 2e-5)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--max_prompt_length", type=int, default=None,
        help="Max prompt length in tokens (auto-detected if None)",
    )
    parser.add_argument(
        "--max_completion_length", type=int, default=512,
        help="Max completion length in tokens (default: 512)",
    )

    # Output
    parser.add_argument(
        "--output_dir", type=str, default="outputs/shoprlve_grpo",
        help="Checkpoint output directory (default: outputs/shoprlve_grpo)",
    )
    parser.add_argument(
        "--save_steps", type=int, default=50,
        help="Save checkpoint every N steps (default: 50)",
    )
    parser.add_argument(
        "--report_to", type=str, default="none",
        choices=["none", "wandb", "tensorboard", "trackio"],
        help="Experiment tracker (default: none)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("=" * 70)
    logger.info("ShopRLVE-GYM OpenEnv Training")
    logger.info("=" * 70)
    logger.info("Model:      %s", args.model)
    logger.info("Collection: %s -> %s", args.collection, get_collection(args.collection))
    logger.info("LoRA rank:  %d", args.lora_rank)
    logger.info("Max steps:  %d", args.max_steps)
    logger.info("Group size: %d (G)", args.num_generations)
    logger.info("LR:         %s", args.learning_rate)
    logger.info("Output:     %s", args.output_dir)
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # 1. Initialize ShopRLVE-GYM environment
    # ------------------------------------------------------------------
    logger.info("Initializing ShopRLVE-GYM environment (collection=%s)...", args.collection)
    global _OPENENV
    _OPENENV = ShopRLVEOpenEnv(collection=args.collection, seed=args.seed)
    logger.info(
        "Environment ready: %d envs, catalog loaded",
        len(_OPENENV.env_ids),
    )

    # ------------------------------------------------------------------
    # 2. Load model with Unsloth
    # ------------------------------------------------------------------
    logger.info("Loading model: %s ...", args.model)

    from unsloth import FastLanguageModel

    load_in_4bit = args.load_in_4bit and not args.load_in_16bit

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=load_in_4bit,
        fast_inference=True,       # Enable vLLM fast inference for GRPO
        max_lora_rank=args.lora_rank,
        gpu_memory_utilization=0.6,  # Reduce if OOM
    )

    logger.info("Model loaded. Adding LoRA adapters (rank=%d)...", args.lora_rank)

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_rank * 2,
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    # ------------------------------------------------------------------
    # 3. Build the training dataset
    # ------------------------------------------------------------------
    logger.info("Building training dataset (%d prompts)...", args.n_prompts)
    dataset = build_dataset(_OPENENV, tokenizer, n_prompts=args.n_prompts)

    # Compute max_prompt_length from a sample if not specified
    if args.max_prompt_length is None:
        sample_text = tokenizer.apply_chat_template(
            dataset[0]["prompt"],
            tokenize=False,
            add_generation_prompt=True,
        )
        sample_len = len(tokenizer.encode(sample_text))
        # Add 20% headroom
        args.max_prompt_length = int(sample_len * 1.2) + 1
        logger.info(
            "Auto-detected max_prompt_length=%d (sample=%d tokens)",
            args.max_prompt_length, sample_len,
        )

    max_completion_length = min(
        args.max_completion_length,
        args.max_seq_length - args.max_prompt_length,
    )
    if max_completion_length <= 0:
        logger.error(
            "max_seq_length (%d) is too small for max_prompt_length (%d). "
            "Increase --max_seq_length or decrease --max_prompt_length.",
            args.max_seq_length, args.max_prompt_length,
        )
        sys.exit(1)

    logger.info(
        "Sequence budget: prompt=%d + completion=%d = %d / %d",
        args.max_prompt_length,
        max_completion_length,
        args.max_prompt_length + max_completion_length,
        args.max_seq_length,
    )

    # ------------------------------------------------------------------
    # 4. Configure GRPO training
    # ------------------------------------------------------------------
    logger.info("Configuring GRPOTrainer...")

    from trl import GRPOConfig, GRPOTrainer

    training_args = GRPOConfig(
        # Generation
        temperature=args.temperature,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=max_completion_length,
        num_generations=args.num_generations,

        # Optimization
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",

        # Batching
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        # Schedule
        max_steps=args.max_steps,
        logging_steps=1,
        save_steps=args.save_steps,

        # Output
        output_dir=args.output_dir,
        report_to=args.report_to,

        # Misc
        seed=args.seed,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
    )

    # ------------------------------------------------------------------
    # 5. Create trainer with ShopRLVE reward functions
    # ------------------------------------------------------------------
    logger.info("Creating GRPOTrainer with 3 reward functions...")
    logger.info("  1. format_reward:     valid JSON action format check")
    logger.info("  2. tool_usage_reward: correct tool names & structure")
    logger.info("  3. env_reward:        ShopRLVE-GYM environment reward (×5)")

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            format_reward,
            tool_usage_reward,
            env_reward,
        ],
        args=training_args,
        train_dataset=dataset,
    )

    # ------------------------------------------------------------------
    # 6. Train!
    # ------------------------------------------------------------------
    logger.info("Starting GRPO training...")
    logger.info(
        "Expect slow initial progress — the model needs ~50-100 steps "
        "to learn JSON formatting before env rewards improve."
    )
    t0 = time.monotonic()

    trainer.train()

    elapsed = time.monotonic() - t0
    logger.info("Training completed in %.1f minutes (%d steps)", elapsed / 60, args.max_steps)

    # ------------------------------------------------------------------
    # 7. Save final model
    # ------------------------------------------------------------------
    final_dir = os.path.join(args.output_dir, "final")
    logger.info("Saving final LoRA adapters to %s ...", final_dir)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    # ------------------------------------------------------------------
    # 8. Quick inference test
    # ------------------------------------------------------------------
    logger.info("Running quick inference test...")
    test_messages, _, _ = _OPENENV.sample_prompt(tokenizer)
    text = tokenizer.apply_chat_template(
        test_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Switch from training mode to inference mode
    FastLanguageModel.for_inference(model)

    from transformers import TextStreamer
    _ = model.generate(
        **tokenizer(text, return_tensors="pt").to("cuda"),
        temperature=0.7,
        max_new_tokens=max_completion_length,
        streamer=TextStreamer(tokenizer, skip_prompt=True),
    )

    logger.info("=" * 70)
    logger.info("Training complete! Model saved to: %s", final_dir)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
