"""Policy knowledge base with deterministic rule-based answers for ShopRLVE-GYM.

Spec Section 2.2 (Tool list C: Policy / knowledge retrieval):
    11. policy.search -- search the policy knowledge base by keyword

The policy KB is a collection of deterministic rules covering:
    - Return windows by product category
    - Return fees by method and membership tier
    - Free shipping thresholds by membership tier
    - Warranty durations by category and price tier
    - Price match windows and conditions
    - Membership benefits

Each rule has conditions (attribute checks) and a deterministic answer.
The difficulty parameter B_branch(d) controls how many clauses a policy
question requires the agent to evaluate.
"""

from __future__ import annotations

import logging
import random
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Policy data models
# ---------------------------------------------------------------------------


class PolicyRule(BaseModel):
    """A single policy rule in the knowledge base.

    Attributes:
        rule_id:     Unique rule identifier (e.g. "pol_001").
        title:       Human-readable rule title.
        category:    Policy category (returns, shipping, pricing, membership, warranty).
        conditions:  List of condition dicts, each with {"field", "op", "value"}.
                     Supported ops: eq, neq, gt, gte, lt, lte, in, not_in.
        answer_type: "numeric" for numbers, "categorical" for string answers.
        answer:      The deterministic answer (str, float, or int).
    """

    rule_id: str = Field(..., description="Unique rule identifier")
    title: str = Field(..., description="Human-readable rule title")
    category: str = Field(
        ...,
        description="Policy category: returns, shipping, pricing, membership, or warranty",
    )
    conditions: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Condition clauses: [{field, op, value}, ...]",
    )
    answer_type: str = Field(
        ..., description="Answer type: 'numeric' or 'categorical'"
    )
    answer: str | float | int = Field(..., description="The deterministic answer")


class PolicyKB(BaseModel):
    """Policy knowledge base: a collection of deterministic rules.

    Attributes:
        rules: List of PolicyRule objects.
    """

    rules: list[PolicyRule] = Field(default_factory=list, description="Policy rules")

    def get_rule(self, rule_id: str) -> PolicyRule | None:
        """Look up a rule by ID."""
        for rule in self.rules:
            if rule.rule_id == rule_id:
                return rule
        return None

    def get_rules_by_category(self, category: str) -> list[PolicyRule]:
        """Get all rules in a given category."""
        return [r for r in self.rules if r.category == category]

    def get_rules_by_clause_count(self, n_clauses: int) -> list[PolicyRule]:
        """Get rules with exactly n_clauses conditions."""
        return [r for r in self.rules if len(r.conditions) == n_clauses]


# ---------------------------------------------------------------------------
# Default policy KB builder (50+ rules)
# ---------------------------------------------------------------------------


def build_default_policy_kb() -> PolicyKB:
    """Build the default policy knowledge base with 50+ rules.

    Covers five categories:
        - returns:    Return windows, fees, eligibility by category/membership
        - shipping:   Free shipping thresholds, delivery times, express options
        - pricing:    Price match windows, discount rules, bulk pricing
        - membership: Tier benefits, upgrade requirements, perks
        - warranty:   Duration by category, extended warranty options

    Returns:
        PolicyKB with all default rules.
    """
    rules: list[PolicyRule] = []
    rule_num = 1

    def _rid() -> str:
        nonlocal rule_num
        rid = f"pol_{rule_num:03d}"
        rule_num += 1
        return rid

    # -----------------------------------------------------------------------
    # RETURNS category (rules 1-14)
    # -----------------------------------------------------------------------

    # Return windows by category (1 clause each)
    for cat, days in [
        ("electronics", 15),
        ("clothing", 30),
        ("furniture", 30),
        ("groceries", 0),
        ("jewelry", 30),
        ("books", 14),
        ("toys", 30),
        ("sports", 30),
        ("beauty", 14),
        ("automotive", 15),
    ]:
        rules.append(PolicyRule(
            rule_id=_rid(),
            title=f"Return window for {cat}",
            category="returns",
            conditions=[{"field": "cat", "op": "eq", "value": cat}],
            answer_type="numeric",
            answer=days,
        ))

    # Return fee by method (1 clause each)
    rules.append(PolicyRule(
        rule_id=_rid(),
        title="Return fee for mail returns",
        category="returns",
        conditions=[{"field": "return_method", "op": "eq", "value": "mail"}],
        answer_type="numeric",
        answer=5.99,
    ))
    rules.append(PolicyRule(
        rule_id=_rid(),
        title="Return fee for in-store returns",
        category="returns",
        conditions=[{"field": "return_method", "op": "eq", "value": "in_store"}],
        answer_type="numeric",
        answer=0.0,
    ))
    rules.append(PolicyRule(
        rule_id=_rid(),
        title="Return fee for pickup returns",
        category="returns",
        conditions=[{"field": "return_method", "op": "eq", "value": "pickup"}],
        answer_type="numeric",
        answer=0.0,
    ))

    # Free returns for premium members (2 clauses)
    rules.append(PolicyRule(
        rule_id=_rid(),
        title="Free mail returns for premium members",
        category="returns",
        conditions=[
            {"field": "return_method", "op": "eq", "value": "mail"},
            {"field": "membership_tier", "op": "eq", "value": "premium"},
        ],
        answer_type="numeric",
        answer=0.0,
    ))

    # -----------------------------------------------------------------------
    # SHIPPING category (rules 15-28)
    # -----------------------------------------------------------------------

    # Free shipping thresholds by membership tier (1 clause each)
    rules.append(PolicyRule(
        rule_id=_rid(),
        title="Free shipping threshold for non-members",
        category="shipping",
        conditions=[{"field": "membership_tier", "op": "eq", "value": "none"}],
        answer_type="numeric",
        answer=50.0,
    ))
    rules.append(PolicyRule(
        rule_id=_rid(),
        title="Free shipping threshold for basic members",
        category="shipping",
        conditions=[{"field": "membership_tier", "op": "eq", "value": "basic"}],
        answer_type="numeric",
        answer=35.0,
    ))
    rules.append(PolicyRule(
        rule_id=_rid(),
        title="Free shipping threshold for premium members",
        category="shipping",
        conditions=[{"field": "membership_tier", "op": "eq", "value": "premium"}],
        answer_type="numeric",
        answer=0.0,
    ))

    # Standard shipping costs (1 clause)
    rules.append(PolicyRule(
        rule_id=_rid(),
        title="Standard shipping cost",
        category="shipping",
        conditions=[{"field": "shipping_method", "op": "eq", "value": "standard"}],
        answer_type="numeric",
        answer=7.99,
    ))
    rules.append(PolicyRule(
        rule_id=_rid(),
        title="Express shipping cost",
        category="shipping",
        conditions=[{"field": "shipping_method", "op": "eq", "value": "express"}],
        answer_type="numeric",
        answer=14.99,
    ))
    rules.append(PolicyRule(
        rule_id=_rid(),
        title="Overnight shipping cost",
        category="shipping",
        conditions=[{"field": "shipping_method", "op": "eq", "value": "overnight"}],
        answer_type="numeric",
        answer=24.99,
    ))

    # Delivery time estimates (1 clause each)
    rules.append(PolicyRule(
        rule_id=_rid(),
        title="Standard shipping delivery time",
        category="shipping",
        conditions=[{"field": "shipping_method", "op": "eq", "value": "standard"}],
        answer_type="categorical",
        answer="5-7 business days",
    ))
    rules.append(PolicyRule(
        rule_id=_rid(),
        title="Express shipping delivery time",
        category="shipping",
        conditions=[{"field": "shipping_method", "op": "eq", "value": "express"}],
        answer_type="categorical",
        answer="2-3 business days",
    ))
    rules.append(PolicyRule(
        rule_id=_rid(),
        title="Overnight shipping delivery time",
        category="shipping",
        conditions=[{"field": "shipping_method", "op": "eq", "value": "overnight"}],
        answer_type="categorical",
        answer="Next business day",
    ))

    # Free express for premium + order over $100 (2 clauses)
    rules.append(PolicyRule(
        rule_id=_rid(),
        title="Free express shipping for premium members on orders over $100",
        category="shipping",
        conditions=[
            {"field": "membership_tier", "op": "eq", "value": "premium"},
            {"field": "order_total", "op": "gte", "value": 100.0},
        ],
        answer_type="numeric",
        answer=0.0,
    ))

    # Heavy item surcharge (2 clauses)
    rules.append(PolicyRule(
        rule_id=_rid(),
        title="Heavy item shipping surcharge for furniture over 50 lbs",
        category="shipping",
        conditions=[
            {"field": "cat", "op": "eq", "value": "furniture"},
            {"field": "weight_lbs", "op": "gt", "value": 50},
        ],
        answer_type="numeric",
        answer=29.99,
    ))

    # International shipping (2 clauses)
    rules.append(PolicyRule(
        rule_id=_rid(),
        title="International standard shipping cost",
        category="shipping",
        conditions=[
            {"field": "shipping_method", "op": "eq", "value": "standard"},
            {"field": "destination", "op": "eq", "value": "international"},
        ],
        answer_type="numeric",
        answer=19.99,
    ))

    # Alaska/Hawaii surcharge (2 clauses)
    rules.append(PolicyRule(
        rule_id=_rid(),
        title="Alaska/Hawaii shipping surcharge",
        category="shipping",
        conditions=[
            {"field": "destination", "op": "in", "value": ["alaska", "hawaii"]},
            {"field": "shipping_method", "op": "eq", "value": "standard"},
        ],
        answer_type="numeric",
        answer=9.99,
    ))

    # -----------------------------------------------------------------------
    # PRICING category (rules 29-40)
    # -----------------------------------------------------------------------

    # Price match window (1 clause)
    rules.append(PolicyRule(
        rule_id=_rid(),
        title="Price match window for in-store purchases",
        category="pricing",
        conditions=[{"field": "purchase_channel", "op": "eq", "value": "in_store"}],
        answer_type="numeric",
        answer=14,
    ))
    rules.append(PolicyRule(
        rule_id=_rid(),
        title="Price match window for online purchases",
        category="pricing",
        conditions=[{"field": "purchase_channel", "op": "eq", "value": "online"}],
        answer_type="numeric",
        answer=7,
    ))

    # Price match eligibility (2 clauses)
    rules.append(PolicyRule(
        rule_id=_rid(),
        title="Price match eligible for electronics from authorized retailers",
        category="pricing",
        conditions=[
            {"field": "cat", "op": "eq", "value": "electronics"},
            {"field": "competitor_type", "op": "eq", "value": "authorized_retailer"},
        ],
        answer_type="categorical",
        answer="eligible",
    ))
    rules.append(PolicyRule(
        rule_id=_rid(),
        title="Price match ineligible for marketplace sellers",
        category="pricing",
        conditions=[
            {"field": "competitor_type", "op": "eq", "value": "marketplace"},
            {"field": "cat", "op": "neq", "value": "electronics"},
        ],
        answer_type="categorical",
        answer="ineligible",
    ))

    # Bulk discount rules (2 clauses)
    rules.append(PolicyRule(
        rule_id=_rid(),
        title="5% bulk discount for 10+ identical items",
        category="pricing",
        conditions=[
            {"field": "quantity", "op": "gte", "value": 10},
            {"field": "quantity", "op": "lt", "value": 25},
        ],
        answer_type="numeric",
        answer=5,
    ))
    rules.append(PolicyRule(
        rule_id=_rid(),
        title="10% bulk discount for 25+ identical items",
        category="pricing",
        conditions=[
            {"field": "quantity", "op": "gte", "value": 25},
            {"field": "quantity", "op": "lt", "value": 100},
        ],
        answer_type="numeric",
        answer=10,
    ))
    rules.append(PolicyRule(
        rule_id=_rid(),
        title="15% bulk discount for 100+ identical items",
        category="pricing",
        conditions=[{"field": "quantity", "op": "gte", "value": 100}],
        answer_type="numeric",
        answer=15,
    ))

    # Coupon stacking rules (1 clause)
    rules.append(PolicyRule(
        rule_id=_rid(),
        title="Maximum coupons per order",
        category="pricing",
        conditions=[{"field": "order_type", "op": "eq", "value": "standard"}],
        answer_type="numeric",
        answer=2,
    ))

    # Sale price exclusion (2 clauses)
    rules.append(PolicyRule(
        rule_id=_rid(),
        title="Coupons not applicable on clearance items",
        category="pricing",
        conditions=[
            {"field": "item_status", "op": "eq", "value": "clearance"},
            {"field": "coupon_type", "op": "eq", "value": "percentage"},
        ],
        answer_type="categorical",
        answer="not_applicable",
    ))

    # Student discount (2 clauses)
    rules.append(PolicyRule(
        rule_id=_rid(),
        title="Student discount on electronics",
        category="pricing",
        conditions=[
            {"field": "customer_type", "op": "eq", "value": "student"},
            {"field": "cat", "op": "eq", "value": "electronics"},
        ],
        answer_type="numeric",
        answer=10,
    ))

    # Employee discount (1 clause)
    rules.append(PolicyRule(
        rule_id=_rid(),
        title="Employee discount percentage",
        category="pricing",
        conditions=[{"field": "customer_type", "op": "eq", "value": "employee"}],
        answer_type="numeric",
        answer=20,
    ))

    # Price adjustment for open-box items (1 clause)
    rules.append(PolicyRule(
        rule_id=_rid(),
        title="Open-box item discount",
        category="pricing",
        conditions=[{"field": "item_condition", "op": "eq", "value": "open_box"}],
        answer_type="numeric",
        answer=15,
    ))

    # -----------------------------------------------------------------------
    # MEMBERSHIP category (rules 41-50)
    # -----------------------------------------------------------------------

    # Membership tier requirements (1 clause each)
    rules.append(PolicyRule(
        rule_id=_rid(),
        title="Basic membership annual fee",
        category="membership",
        conditions=[{"field": "membership_tier", "op": "eq", "value": "basic"}],
        answer_type="numeric",
        answer=29.99,
    ))
    rules.append(PolicyRule(
        rule_id=_rid(),
        title="Premium membership annual fee",
        category="membership",
        conditions=[{"field": "membership_tier", "op": "eq", "value": "premium"}],
        answer_type="numeric",
        answer=99.99,
    ))

    # Points earning rates (2 clauses)
    rules.append(PolicyRule(
        rule_id=_rid(),
        title="Basic member points per dollar spent",
        category="membership",
        conditions=[
            {"field": "membership_tier", "op": "eq", "value": "basic"},
            {"field": "purchase_channel", "op": "eq", "value": "online"},
        ],
        answer_type="numeric",
        answer=1,
    ))
    rules.append(PolicyRule(
        rule_id=_rid(),
        title="Premium member points per dollar spent",
        category="membership",
        conditions=[
            {"field": "membership_tier", "op": "eq", "value": "premium"},
            {"field": "purchase_channel", "op": "eq", "value": "online"},
        ],
        answer_type="numeric",
        answer=3,
    ))

    # Premium benefits (1 clause)
    rules.append(PolicyRule(
        rule_id=_rid(),
        title="Premium member free returns",
        category="membership",
        conditions=[{"field": "membership_tier", "op": "eq", "value": "premium"}],
        answer_type="categorical",
        answer="Free returns on all items via any method",
    ))
    rules.append(PolicyRule(
        rule_id=_rid(),
        title="Premium member priority shipping",
        category="membership",
        conditions=[{"field": "membership_tier", "op": "eq", "value": "premium"}],
        answer_type="categorical",
        answer="Free 2-day shipping on all orders",
    ))

    # Upgrade requirement (2 clauses)
    rules.append(PolicyRule(
        rule_id=_rid(),
        title="Basic to premium upgrade spending requirement",
        category="membership",
        conditions=[
            {"field": "membership_tier", "op": "eq", "value": "basic"},
            {"field": "annual_spend", "op": "gte", "value": 500},
        ],
        answer_type="categorical",
        answer="Eligible for complimentary premium upgrade",
    ))

    # Cancellation policy (1 clause)
    rules.append(PolicyRule(
        rule_id=_rid(),
        title="Membership cancellation refund policy",
        category="membership",
        conditions=[{"field": "days_since_signup", "op": "lte", "value": 30}],
        answer_type="categorical",
        answer="Full refund within 30 days of signup",
    ))

    # Guest checkout limits (1 clause)
    rules.append(PolicyRule(
        rule_id=_rid(),
        title="Guest checkout order limit",
        category="membership",
        conditions=[{"field": "membership_tier", "op": "eq", "value": "none"}],
        answer_type="numeric",
        answer=500.0,
    ))

    # Birthday bonus (2 clauses)
    rules.append(PolicyRule(
        rule_id=_rid(),
        title="Premium member birthday bonus points",
        category="membership",
        conditions=[
            {"field": "membership_tier", "op": "eq", "value": "premium"},
            {"field": "is_birthday_month", "op": "eq", "value": True},
        ],
        answer_type="numeric",
        answer=500,
    ))

    # -----------------------------------------------------------------------
    # WARRANTY category (rules 51-62)
    # -----------------------------------------------------------------------

    # Standard warranty by category (1 clause each)
    for cat, months in [
        ("electronics", 12),
        ("furniture", 24),
        ("appliances", 12),
        ("clothing", 3),
        ("jewelry", 6),
        ("toys", 6),
        ("sports", 6),
        ("automotive", 12),
    ]:
        rules.append(PolicyRule(
            rule_id=_rid(),
            title=f"Standard warranty for {cat}",
            category="warranty",
            conditions=[{"field": "cat", "op": "eq", "value": cat}],
            answer_type="numeric",
            answer=months,
        ))

    # Extended warranty pricing by price tier (2 clauses)
    rules.append(PolicyRule(
        rule_id=_rid(),
        title="Extended warranty cost for electronics under $200",
        category="warranty",
        conditions=[
            {"field": "cat", "op": "eq", "value": "electronics"},
            {"field": "price", "op": "lt", "value": 200},
        ],
        answer_type="numeric",
        answer=19.99,
    ))
    rules.append(PolicyRule(
        rule_id=_rid(),
        title="Extended warranty cost for electronics $200-$500",
        category="warranty",
        conditions=[
            {"field": "cat", "op": "eq", "value": "electronics"},
            {"field": "price", "op": "gte", "value": 200},
        ],
        answer_type="numeric",
        answer=49.99,
    ))

    # Extended warranty for furniture (2 clauses)
    rules.append(PolicyRule(
        rule_id=_rid(),
        title="Extended warranty cost for furniture over $500",
        category="warranty",
        conditions=[
            {"field": "cat", "op": "eq", "value": "furniture"},
            {"field": "price", "op": "gte", "value": 500},
        ],
        answer_type="numeric",
        answer=79.99,
    ))

    # Premium member warranty extension (2 clauses)
    rules.append(PolicyRule(
        rule_id=_rid(),
        title="Premium member bonus warranty months for electronics",
        category="warranty",
        conditions=[
            {"field": "membership_tier", "op": "eq", "value": "premium"},
            {"field": "cat", "op": "eq", "value": "electronics"},
        ],
        answer_type="numeric",
        answer=6,
    ))

    # Accidental damage protection (3 clauses)
    rules.append(PolicyRule(
        rule_id=_rid(),
        title="Accidental damage protection for premium members on electronics over $100",
        category="warranty",
        conditions=[
            {"field": "membership_tier", "op": "eq", "value": "premium"},
            {"field": "cat", "op": "eq", "value": "electronics"},
            {"field": "price", "op": "gte", "value": 100},
        ],
        answer_type="categorical",
        answer="Included free for 12 months",
    ))

    return PolicyKB(rules=rules)


# ---------------------------------------------------------------------------
# Policy question generator
# ---------------------------------------------------------------------------

# Natural language templates for generating questions from rules
_QUESTION_TEMPLATES: dict[str, list[str]] = {
    "returns": [
        "What is the return window for {cat} products?",
        "How many days do I have to return a {cat} item?",
        "What is the return policy for {cat}?",
        "Can I return {cat} items and within what timeframe?",
    ],
    "shipping": [
        "What is the shipping cost for {shipping_method} delivery?",
        "How much does {shipping_method} shipping cost?",
        "What are the shipping options and how much do they cost?",
        "How long does {shipping_method} shipping take?",
    ],
    "pricing": [
        "What discount do I get for buying {quantity} items?",
        "Is there a bulk discount for ordering {quantity} units?",
        "What is the price match policy for {cat} from {competitor_type}?",
        "What discounts are available for {customer_type} customers?",
    ],
    "membership": [
        "What are the benefits of {membership_tier} membership?",
        "How much does {membership_tier} membership cost per year?",
        "What is the points earning rate for {membership_tier} members?",
        "What perks do {membership_tier} members get?",
    ],
    "warranty": [
        "What is the standard warranty for {cat} products?",
        "How long is the warranty on {cat} items?",
        "What does the extended warranty cost for {cat}?",
        "Is accidental damage covered for {cat}?",
    ],
}


def generate_policy_question(
    kb: PolicyKB,
    n_clauses: int,
    seed: int,
) -> tuple[PolicyRule, dict[str, Any], str]:
    """Generate a policy question from the knowledge base.

    Samples a rule with the specified clause count (or closest available),
    generates a matching context dict, and produces a natural language question.

    Args:
        kb:        PolicyKB to sample from.
        n_clauses: Desired number of condition clauses (B_branch(d)).
        seed:      Random seed for reproducibility.

    Returns:
        Tuple of (rule, context_dict, question_text):
            - rule: The PolicyRule that answers the question.
            - context_dict: Dict of field->value matching the rule's conditions.
            - question_text: Natural language question string.

    Raises:
        ValueError: If KB is empty.
    """
    if not kb.rules:
        raise ValueError("Cannot generate question from empty PolicyKB")

    rng = random.Random(seed)

    # Find rules with exactly n_clauses conditions
    matching_rules = kb.get_rules_by_clause_count(n_clauses)

    if not matching_rules:
        # Find closest clause count
        all_counts = sorted({len(r.conditions) for r in kb.rules})
        closest = min(all_counts, key=lambda c: abs(c - n_clauses))
        matching_rules = kb.get_rules_by_clause_count(closest)

    # Sample a rule
    rule = rng.choice(matching_rules)

    # Build context dict from conditions
    context: dict[str, Any] = {}
    for condition in rule.conditions:
        field = condition["field"]
        value = condition["value"]
        context[field] = value

    # Generate natural language question
    templates = _QUESTION_TEMPLATES.get(rule.category, [])
    if templates:
        # Try to find a template that can be filled with our context
        fillable_templates: list[str] = []
        for template in templates:
            try:
                # Check if all placeholders can be filled
                template.format(**context)
                fillable_templates.append(template)
            except KeyError:
                continue

        if fillable_templates:
            template = rng.choice(fillable_templates)
            question_text = template.format(**context)
        else:
            # Fallback: construct from rule title and conditions
            question_text = _build_fallback_question(rule, context)
    else:
        question_text = _build_fallback_question(rule, context)

    return rule, context, question_text


def _build_fallback_question(rule: PolicyRule, context: dict[str, Any]) -> str:
    """Build a fallback question from a rule's title and context.

    Used when no template matches the available context fields.

    Args:
        rule:    The PolicyRule.
        context: The context dict derived from conditions.

    Returns:
        A natural language question string.
    """
    # Convert conditions to readable string
    condition_parts: list[str] = []
    for cond in rule.conditions:
        field = cond["field"]
        op = cond["op"]
        value = cond["value"]

        if op == "eq":
            condition_parts.append(f"{field} is {value}")
        elif op == "neq":
            condition_parts.append(f"{field} is not {value}")
        elif op == "gt":
            condition_parts.append(f"{field} is greater than {value}")
        elif op == "gte":
            condition_parts.append(f"{field} is at least {value}")
        elif op == "lt":
            condition_parts.append(f"{field} is less than {value}")
        elif op == "lte":
            condition_parts.append(f"{field} is at most {value}")
        elif op == "in":
            condition_parts.append(f"{field} is one of {value}")
        elif op == "not_in":
            condition_parts.append(f"{field} is not one of {value}")

    conditions_str = " and ".join(condition_parts) if condition_parts else ""

    if conditions_str:
        return f"Given that {conditions_str}, what is the policy on: {rule.title}?"
    return f"What is the policy on: {rule.title}?"


# ---------------------------------------------------------------------------
# Internal: format a rule as readable text for search results
# ---------------------------------------------------------------------------


def _format_rule_text(rule: PolicyRule) -> str:
    """Format a policy rule as human-readable text for LLM consumption.

    Args:
        rule: PolicyRule to format.

    Returns:
        Formatted text string.
    """
    parts: list[str] = [f"Policy: {rule.title}"]
    parts.append(f"Category: {rule.category}")

    if rule.conditions:
        cond_strs: list[str] = []
        for cond in rule.conditions:
            field = cond["field"]
            op = cond["op"]
            value = cond["value"]
            cond_strs.append(f"{field} {op} {value}")
        parts.append(f"Conditions: {'; '.join(cond_strs)}")

    if rule.answer_type == "numeric":
        parts.append(f"Answer: {rule.answer}")
    else:
        parts.append(f"Answer: {rule.answer}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Internal: keyword matching scorer
# ---------------------------------------------------------------------------


def _keyword_score(query: str, rule: PolicyRule) -> float:
    """Compute a simple keyword-match score between a query and a rule.

    Tokenizes the query and checks for matches in the rule's title,
    category, conditions, and answer. Returns a normalized score in [0, 1].

    Args:
        query: Search query string.
        rule:  PolicyRule to score against.

    Returns:
        Score in [0.0, 1.0] representing match quality.
    """
    # Tokenize query into lowercase words
    query_tokens = set(query.lower().split())
    if not query_tokens:
        return 0.0

    # Build searchable text from rule fields
    searchable_parts: list[str] = [
        rule.title.lower(),
        rule.category.lower(),
        rule.rule_id.lower(),
    ]

    # Include condition fields and values
    for cond in rule.conditions:
        searchable_parts.append(str(cond.get("field", "")).lower())
        searchable_parts.append(str(cond.get("value", "")).lower())

    # Include answer
    searchable_parts.append(str(rule.answer).lower())

    searchable_text = " ".join(searchable_parts)
    searchable_tokens = set(searchable_text.split())

    # Count matching tokens
    matches = query_tokens & searchable_tokens
    if not matches:
        # Try substring matching as fallback
        substring_matches = 0
        for qt in query_tokens:
            if qt in searchable_text:
                substring_matches += 1
        return substring_matches / len(query_tokens) * 0.8  # slight discount for substring
    else:
        return len(matches) / len(query_tokens)


# ---------------------------------------------------------------------------
# Pydantic arg model for tool registration
# ---------------------------------------------------------------------------


class PolicySearchArgs(BaseModel):
    """Arguments for policy.search tool.

    Searches the policy knowledge base by keyword matching.
    """

    query: str = Field(..., min_length=1, description="Search query for policy rules")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return")


# ---------------------------------------------------------------------------
# State accessor helpers
# ---------------------------------------------------------------------------


def _get_policy_kb(state: Any) -> PolicyKB:
    """Extract PolicyKB from the episode state.

    Supports:
        - state.policy_kb (attribute)
        - state["policy_kb"] (dict)

    Falls back to building the default KB if not found.
    """
    if hasattr(state, "policy_kb"):
        kb = state.policy_kb
        if isinstance(kb, PolicyKB):
            return kb

    if isinstance(state, dict) and "policy_kb" in state:
        kb = state["policy_kb"]
        if isinstance(kb, PolicyKB):
            return kb

    # Fallback: build and cache the default KB
    logger.debug("PolicyKB not found in state; building default KB")
    default_kb = build_default_policy_kb()

    # Try to cache it on state for future calls
    if hasattr(state, "__dict__"):
        try:
            state.policy_kb = default_kb
        except AttributeError:
            pass
    elif isinstance(state, dict):
        state["policy_kb"] = default_kb

    return default_kb


# ---------------------------------------------------------------------------
# Tool handler: policy.search
# ---------------------------------------------------------------------------


def policy_search(
    query: str,
    top_k: int = 5,
    *,
    state: Any = None,
) -> list[dict[str, Any]]:
    """Search the policy knowledge base by keyword matching.

    Scores each rule against the query using token overlap and returns
    the top-k results sorted by relevance.

    Args:
        query: Search query string.
        top_k: Number of results to return (default: 5).
        state: Episode state with .policy_kb (optional).

    Returns:
        List of result dicts, each containing:
            - snippet_id: str (rule_id)
            - title: str
            - text: str (formatted rule)
            - category: str
            - score: float (match score)
    """
    kb = _get_policy_kb(state)

    # Score all rules
    scored: list[tuple[PolicyRule, float]] = []
    for rule in kb.rules:
        score = _keyword_score(query, rule)
        if score > 0.0:
            scored.append((rule, score))

    # Sort by score descending
    scored.sort(key=lambda x: x[1], reverse=True)

    # Build result list
    results: list[dict[str, Any]] = []
    for rule, score in scored[:top_k]:
        results.append({
            "snippet_id": rule.rule_id,
            "title": rule.title,
            "text": _format_rule_text(rule),
            "category": rule.category,
            "score": round(score, 3),
        })

    return results


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------


def register_policy_tools(registry: Any) -> None:
    """Register all policy tool handlers with a ToolRegistry instance.

    Registers:
        - policy.search

    Args:
        registry: ToolRegistry instance.
    """
    registry.register("policy.search", policy_search, PolicySearchArgs)

    logger.info("Registered 1 policy tool: search")
