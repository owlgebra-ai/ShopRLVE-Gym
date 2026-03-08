"""User simulator for ShopRLVE-GYM.

Provides persona-driven, template-based user simulation for multi-turn
e-commerce dialogues. Components:

    - persona:    Persona sampling (Dirichlet) and latent utility computation
    - templates:  Per-environment utterance templates with slot filling and noise
    - dialogue:   Deterministic dialogue manager with clarification and ragequit logic
"""

from shop_rlve.simulator.dialogue import DialogueState, UserSimulator
from shop_rlve.simulator.persona import (
    PersonaWeights,
    compute_utility,
    phi_brand,
    phi_price,
    phi_rating,
    phi_ship,
    phi_similarity,
    sample_persona_weights,
)
from shop_rlve.simulator.templates import (
    apply_noise,
    get_available_env_ids,
    get_template_count,
    render_clarification,
    render_template,
    render_template_deterministic,
)

__all__ = [
    # persona
    "PersonaWeights",
    "sample_persona_weights",
    "phi_price",
    "phi_rating",
    "phi_ship",
    "phi_brand",
    "phi_similarity",
    "compute_utility",
    # templates
    "render_template",
    "render_template_deterministic",
    "render_clarification",
    "apply_noise",
    "get_template_count",
    "get_available_env_ids",
    # dialogue
    "DialogueState",
    "UserSimulator",
]
