"""E_POLICY -- Policy QA environment.

Spec Section 5 / E6:
    Difficulty:
        B_branch(d) = 1 + floor(d/3)  conditions

    Generator:
        1. Sample a rule with B_branch(d) clauses from the policy KB.
        2. Provide a user question and context fields.

    Reward:
        Numeric:
            rho = (min(x,y) / max(x,y))^4    (beta=4)
            r_task = 2*rho - 1
            Parse failure -> r_task = -1
        Categorical:
            r_task = 1  if exact match else -1

    IsCorrect:
        r_task >= 0.95
"""

from __future__ import annotations

from typing import Any

from shop_rlve.data.schema import Product
from shop_rlve.difficulty.mapping import map_difficulty
from shop_rlve.rewards.verifiers import verify_policy
from shop_rlve.simulator.templates import render_template
from shop_rlve.tools.policy import (
    PolicyKB,
    PolicyRule,
    build_default_policy_kb,
    generate_policy_question,
)

from shop_rlve.envs.base import (
    BaseEnvironment,
    EpisodeResult,
    ProblemParams,
    register_env,
)


@register_env
class PolicyQAEnv(BaseEnvironment):
    """E_POLICY: Policy QA -- answer policy questions deterministically."""

    ENV_ID = "POLICY"

    # ------------------------------------------------------------------
    # P_d : Problem generator
    # ------------------------------------------------------------------

    def generate_problem(
        self,
        difficulty: int,
        catalog: list[Product],
        seed: int,
        **kwargs: Any,
    ) -> ProblemParams:
        """Sample a policy rule and generate a question.

        Spec Section 5 / E6:
            B_branch(d) = 1 + floor(d/3)
        """
        dp = map_difficulty(difficulty)

        kb: PolicyKB = kwargs.get("policy_kb") or build_default_policy_kb()

        rule, context, question_text = generate_policy_question(
            kb=kb,
            n_clauses=dp.B_branch_val,
            seed=seed,
        )

        return ProblemParams(
            env_id=self.ENV_ID,
            difficulty=difficulty,
            seed=seed,
            target_product_ids=[],
            constraints=[],
            extra={
                "T_max": dp.T_max_val,
                "p_missing": dp.p_missing_val,
                "p_noise": dp.p_noise_val,
                "rule": rule,
                "context": context,
                "question_text": question_text,
                "expected_answer": rule.answer,
                "answer_type": rule.answer_type,
                "n_clauses": len(rule.conditions),
            },
        )

    # ------------------------------------------------------------------
    # I : Input generator
    # ------------------------------------------------------------------

    def generate_input(self, params: ProblemParams) -> str:
        """Render policy question as user utterance."""
        question_text: str = params.extra["question_text"]

        # Add context information for multi-clause rules
        context: dict[str, Any] = params.extra.get("context", {})
        context_parts: list[str] = []
        for field, value in context.items():
            context_parts.append(f"{field}: {value}")
        context_str = ", ".join(context_parts)

        template_params: dict[str, Any] = {
            "policy_question": question_text,
        }

        return render_template(
            env_id=self.ENV_ID,
            params=template_params,
            p_missing=params.extra.get("p_missing", 0.0),
            p_noise=params.extra.get("p_noise", 0.0),
            seed=params.seed + 1,
        )

    # ------------------------------------------------------------------
    # R : Verifier
    # ------------------------------------------------------------------

    def verify(
        self,
        answer: dict[str, Any],
        params: ProblemParams,
        episode_state: dict[str, Any],
    ) -> EpisodeResult:
        """Compute r_task per Spec Section 5 / E6.

        Numeric:     rho = (min(x,y)/max(x,y))^4, r_task = 2*rho - 1
        Categorical: exact match = 1, else -1
        IsCorrect = r_task >= 0.95
        """
        answer_value = answer.get("policy_answer")
        expected_value = params.extra["expected_answer"]
        answer_type: str = params.extra["answer_type"]

        r_task, is_correct = verify_policy(
            answer_value=answer_value,
            expected_value=expected_value,
            answer_type=answer_type,
        )

        return EpisodeResult(
            r_task=r_task,
            is_correct=is_correct,
            details={
                "expected_answer": expected_value,
                "agent_answer": answer_value,
                "answer_type": answer_type,
                "n_clauses": params.extra.get("n_clauses", 0),
            },
        )
