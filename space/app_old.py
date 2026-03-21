"""
EcomRLVE-GYM Interactive Demo — Hugging Face Space

An interactive environment where YOU play the role of the candidate LLM agent.
A persona-driven user simulator (powered by HF Inference API) plays the customer.
Your goal: use the available tools to find the right products and earn maximum reward.

Flow:
  1. Reset → FAISS index loaded, persona sampled, LLM generates initial customer message
  2. You see the customer message + available tool buttons
  3. Pick a tool, fill in args, submit → tool executes, results shown
  4. Write an assistant response → LLM generates next customer message
  5. Submit final answer → reward computed & displayed
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import gradio as gr
import numpy as np

# ---------------------------------------------------------------------------
# Add src to path
# ---------------------------------------------------------------------------
_app_dir = Path(__file__).resolve().parent          # directory containing app.py
# Local dev: app.py lives in space/ → ROOT is parent.parent
# HF Space:  app.py lives at repo root → ROOT is parent
if (_app_dir / "src" / "ecom_rlve").is_dir():
    ROOT = _app_dir           # HF Space layout (app.py + src/ at same level)
else:
    ROOT = _app_dir.parent    # local dev layout  (space/app.py, src/ one level up)
sys.path.insert(0, str(ROOT / "src"))

from ecom_rlve.data.catalog_loader import load_catalog
from ecom_rlve.data.embeddings import EmbeddingEngine
from ecom_rlve.data.index import VectorIndex
from ecom_rlve.server.openenv import EcomRLVEEnv
from ecom_rlve.simulator.persona import PersonaWeights
from ecom_rlve.tools.orders import generate_order_history
from ecom_rlve.tools.cart import CartState, CartLine
from ecom_rlve.tools.policy import build_default_policy_kb

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ecomrlve-space")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HF_TOKEN = os.getenv("HF_TOKEN", None)

# Remote HF dataset repo holding catalog + FAISS index
DATA_REPO_ID = os.getenv("DATA_REPO_ID", "thebajajra/ecomrlve-data")


def _ensure_data_downloaded() -> Path:
    """Download catalog + FAISS index from HF dataset repo if not present locally.

    Priority:
      1. If FAISS_INDEX_DIR / CATALOG_PATH env-vars point to existing dirs → use them (local dev).
      2. If ROOT/data/ already exists with the files → use it (previous download / git-lfs).
      3. Otherwise pull from ``DATA_REPO_ID`` via ``snapshot_download`` into a cache dir.

    Returns the base data directory that contains ``faiss-index/`` and ``amazebay-2M/``.
    """
    # --- fast path: local dirs already exist (set via env or from a prior run) ---
    env_faiss = os.getenv("FAISS_INDEX_DIR")
    env_catalog = os.getenv("CATALOG_PATH")
    if env_faiss and Path(env_faiss).exists() and env_catalog and Path(env_catalog).exists():
        logger.info("Using local data dirs (env-vars): faiss=%s  catalog=%s", env_faiss, env_catalog)
        return Path(env_faiss).parent  # assume sibling layout

    local_data = ROOT / "data"
    if (local_data / "faiss-index" / "index.faiss").exists() and \
       (local_data / "amazebay-2M").exists():
        logger.info("Using local data dir: %s", local_data)
        return local_data

    # --- download from HF ---
    from huggingface_hub import snapshot_download

    logger.info("Downloading data from HF dataset repo: %s ...", DATA_REPO_ID)
    cache_dir = snapshot_download(
        repo_id=DATA_REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN,
        allow_patterns=["faiss-index/*", "amazebay-2M/*"],
    )
    logger.info("Data downloaded to: %s", cache_dir)
    return Path(cache_dir)


# Run the download check once at import time so the Space shows progress immediately.
_DATA_DIR = _ensure_data_downloaded()

FAISS_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", str(_DATA_DIR / "faiss-index"))
CATALOG_PATH = os.getenv("CATALOG_PATH", str(_DATA_DIR / "amazebay-2M"))
CATALOG_MAX_ITEMS = int(os.getenv("CATALOG_MAX_ITEMS", "5000"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "thenlper/gte-small")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")  # cpu for HF Space
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")

# HF Inference Client (lazy)
_inference_client = None


def _get_inference_client():
    global _inference_client
    if _inference_client is None:
        from huggingface_hub import InferenceClient
        _inference_client = InferenceClient(
            model=LLM_MODEL,
            token=HF_TOKEN,
        )
    return _inference_client


# ---------------------------------------------------------------------------
# Global singletons (loaded once)
# ---------------------------------------------------------------------------
_env_singleton = None
_catalog_loaded = False


def _get_env() -> EcomRLVEEnv:
    """Lazy-load the environment with FAISS index + catalog."""
    global _env_singleton, _catalog_loaded

    if _env_singleton is not None:
        return _env_singleton

    logger.info("Loading catalog from %s (max_items=%d)...", CATALOG_PATH, CATALOG_MAX_ITEMS)
    products = load_catalog(CATALOG_PATH, max_items=CATALOG_MAX_ITEMS, seed=42)
    logger.info("Loaded %d products", len(products))

    # Use the pre-built 2M FAISS index if available.  openenv.py will
    # automatically call set_allowed_ids() to restrict FAISS search to
    # only the loaded products (IDSelectorBatch), so results are always
    # valid even though the index covers 2M products.
    faiss_path = FAISS_INDEX_DIR if Path(FAISS_INDEX_DIR).exists() else None
    config = {
        "embedding_model": EMBEDDING_MODEL,
        "embedding_debug": False,  # need real model to encode search queries
        "embedding_device": EMBEDDING_DEVICE,
    }
    if faiss_path:
        config["faiss_index_path"] = faiss_path

    logger.info(
        "Creating EcomRLVEEnv (faiss=%s, products=%d)...",
        faiss_path or "build-from-scratch", len(products),
    )
    env = EcomRLVEEnv(
        collection="C8",
        catalog=(products, []),
        config=config,
        seed=int(time.time()) % 100000,
    )
    _env_singleton = env
    _catalog_loaded = True
    logger.info("Environment ready!")
    return env


# ---------------------------------------------------------------------------
# LLM User Simulator via HF Inference
# ---------------------------------------------------------------------------
PERSONA_SYSTEM_PROMPT = """You are a customer shopping on an e-commerce platform. You must stay in character.

YOUR PERSONA:
{persona_desc}

YOUR SHOPPING GOAL:
{goal_desc}

BEHAVIOR RULES:
- You are a real customer with specific needs. Be natural and conversational.
- If the assistant asks clarifying questions, answer based on your persona.
- If shown products, evaluate them based on your preferences (price, brand, rating, shipping).
- If the assistant helps you well, express satisfaction.
- If the assistant is unhelpful or shows irrelevant items, express mild frustration.
- Keep responses concise (1-3 sentences).
- NEVER break character or reveal you are an AI.
"""


def _build_persona_description(weights: PersonaWeights) -> str:
    """Convert persona weights into a natural language description."""
    parts = []
    w = weights

    if w.w_price > 0.3:
        parts.append("You are very price-conscious and always look for the best deal")
    elif w.w_price > 0.15:
        parts.append("You care about price but it's not the only factor")

    if w.w_rating > 0.3:
        parts.append("You heavily rely on product ratings and reviews")
    elif w.w_rating > 0.15:
        parts.append("Good ratings are somewhat important to you")

    if w.w_ship > 0.3:
        parts.append("Fast shipping is critical for you")
    elif w.w_ship > 0.15:
        parts.append("You prefer reasonable shipping times")

    if w.w_brand > 0.3:
        parts.append("Brand name matters a lot to you")
    elif w.w_brand > 0.15:
        parts.append("You have some brand preferences")

    if w.w_similarity > 0.3:
        parts.append("You want products that exactly match your description")

    return ". ".join(parts) + "." if parts else "You are a typical shopper with balanced preferences."


def _build_goal_description(goal_params: dict) -> str:
    """Build goal description from extracted goal params."""
    parts = []
    cat = goal_params.get("category", "product")
    parts.append(f"You are looking for: {cat}")

    if "price_max" in goal_params:
        parts.append(f"Budget: under ${goal_params['price_max']}")
    if "brand_pref" in goal_params:
        parts.append(f"Preferred brand: {goal_params['brand_pref']}")
    if "rating_req" in goal_params:
        parts.append(f"Minimum rating: {goal_params['rating_req']} stars")
    if "ship_req" in goal_params:
        parts.append(f"Need delivery within: {goal_params['ship_req']} days")
    if "color_pref" in goal_params:
        parts.append(f"Color preference: {goal_params['color_pref']}")

    return "\n".join(parts)


def _llm_generate_user_message(
    system_prompt: str,
    conversation: list[dict],
) -> str:
    """Call HF Inference API to generate a user (customer) message."""
    try:
        client = _get_inference_client()
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation)

        response = client.chat_completion(
            messages=messages,
            max_tokens=150,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning("LLM inference failed: %s. Using fallback.", e)
        return None  # fallback to template-based


# ---------------------------------------------------------------------------
# Tool definitions for the UI
# ---------------------------------------------------------------------------
TOOLS = {
    # ── Catalog (4) ──────────────────────────────────────────────
    "catalog.search": {
        "description": "🔍 Search products by query + optional filters",
        "args": {
            "query": {"type": "str", "required": True, "description": "Search query"},
            "top_k": {"type": "int", "required": False, "default": 20, "description": "Number of results (1-500)"},
            "filters": {"type": "json", "required": False, "default": None, "description": 'Filters JSON e.g. {"price_max":50,"brand":"Nike"}'},
        },
    },
    "catalog.rerank": {
        "description": "📊 Re-rank candidates by query relevance",
        "args": {
            "query": {"type": "str", "required": True, "description": "Reranking query"},
            "candidate_product_ids": {"type": "list", "required": True, "description": "Comma-separated product IDs"},
            "top_k": {"type": "int", "required": False, "default": 10, "description": "Results to return (1-100)"},
        },
    },
    "catalog.get_product": {
        "description": "📦 Get full product details by ID",
        "args": {
            "product_id": {"type": "str", "required": True, "description": "Product ID (ASIN)"},
        },
    },
    "catalog.get_variants": {
        "description": "🎨 Get color/size variants for a product",
        "args": {
            "product_id": {"type": "str", "required": True, "description": "Product ID"},
        },
    },
    # ── Cart (4) ─────────────────────────────────────────────────
    "cart.view": {
        "description": "🛒 View current cart contents",
        "args": {},
    },
    "cart.add": {
        "description": "➕ Add a product to cart",
        "args": {
            "product_id": {"type": "str", "required": True, "description": "Product ID to add"},
            "variant_id": {"type": "str", "required": False, "default": None, "description": "Variant ID (optional)"},
            "quantity": {"type": "int", "required": False, "default": 1, "description": "Quantity (≥1)"},
        },
    },
    "cart.remove": {
        "description": "🗑️ Remove a line item from cart",
        "args": {
            "line_id": {"type": "str", "required": True, "description": "Cart line ID to remove"},
        },
    },
    "cart.set_quantity": {
        "description": "✏️ Update quantity for a cart line (0 = remove)",
        "args": {
            "line_id": {"type": "str", "required": True, "description": "Cart line ID"},
            "quantity": {"type": "int", "required": True, "description": "New quantity (0 = remove)"},
        },
    },
    # ── Orders (3) ───────────────────────────────────────────────
    "order.list": {
        "description": "📋 List recent orders",
        "args": {
            "days": {"type": "int", "required": False, "default": 30, "description": "Lookback period in days (1-365)"},
        },
    },
    "order.get_status": {
        "description": "📍 Get detailed order status & tracking",
        "args": {
            "order_id": {"type": "str", "required": True, "description": "Order ID"},
        },
    },
    "order.checkout": {
        "description": "💳 Checkout: create order from current cart",
        "args": {
            "address_id": {"type": "str", "required": True, "description": "Shipping address ID"},
            "payment_id": {"type": "str", "required": True, "description": "Payment method ID"},
        },
    },
    # ── Returns (3) ──────────────────────────────────────────────
    "return.check_eligibility": {
        "description": "✅ Check if an order line is eligible for return",
        "args": {
            "order_id": {"type": "str", "required": True, "description": "Order ID"},
            "line_id": {"type": "str", "required": True, "description": "Line ID within order"},
        },
    },
    "return.initiate": {
        "description": "↩️ Initiate a return for an order line",
        "args": {
            "order_id": {"type": "str", "required": True, "description": "Order ID"},
            "line_id": {"type": "str", "required": True, "description": "Line ID"},
            "reason": {"type": "str", "required": True, "description": "defective|wrong_item|not_as_described|changed_mind|too_large|too_small|arrived_late|better_price_found"},
            "method": {"type": "str", "required": True, "description": "mail|in_store|pickup"},
        },
    },
    "return.exchange": {
        "description": "🔄 Exchange an order line for a different product",
        "args": {
            "order_id": {"type": "str", "required": True, "description": "Order ID"},
            "line_id": {"type": "str", "required": True, "description": "Line ID"},
            "replacement_product_id": {"type": "str", "required": True, "description": "Replacement product ID"},
            "replacement_variant_id": {"type": "str", "required": False, "default": None, "description": "Replacement variant ID (optional)"},
        },
    },
    # ── Policy (1) ───────────────────────────────────────────────
    "policy.search": {
        "description": "📜 Search store policies & rules",
        "args": {
            "query": {"type": "str", "required": True, "description": "Policy search query"},
            "top_k": {"type": "int", "required": False, "default": 5, "description": "Number of results (1-20)"},
        },
    },
}


def _format_tool_result(result: dict) -> str:
    """Format a tool result dict for display."""
    name = result.get("name", "?")
    error = result.get("error")
    data = result.get("result")
    duration = result.get("duration_ms", 0)

    if error:
        return f"❌ **{name}** ({duration:.0f}ms): Error - {error}"

    if isinstance(data, list):
        lines = [f"✅ **{name}** ({duration:.0f}ms) — {len(data)} results:"]
        for i, item in enumerate(data[:8]):
            if isinstance(item, dict):
                title = item.get("title", "?")[:60]
                price = item.get("price", "?")
                rating = item.get("rating", "?")
                pid = item.get("product_id", item.get("id", "?"))
                lines.append(f"  `{i+1}.` **{title}** — ${price} ★{rating} `[{pid}]`")
        if len(data) > 8:
            lines.append(f"  ... and {len(data) - 8} more")
        return "\n".join(lines)
    elif isinstance(data, dict):
        title = data.get("title", data.get("id", ""))
        return f"✅ **{name}** ({duration:.0f}ms): {json.dumps(data, indent=2)[:500]}"
    else:
        return f"✅ **{name}** ({duration:.0f}ms): {str(data)[:300]}"


def _format_reward_breakdown(info: dict) -> str:
    """Format reward breakdown for display."""
    rb = info.get("reward_breakdown", {})
    if not rb:
        return ""

    lines = ["### 🏆 Reward Breakdown"]
    r_total = rb.get("r_total", 0)
    emoji = "🟢" if r_total > 0.5 else "🟡" if r_total > 0 else "🔴"
    lines.append(f"**Total Reward: {emoji} {r_total:.4f}**")
    lines.append("")
    lines.append(f"| Component | Value |")
    lines.append(f"|-----------|-------|")
    lines.append(f"| r_task (task quality) | {rb.get('r_task', 0):.4f} |")
    lines.append(f"| r_eff (efficiency) | {rb.get('r_eff', 0):.4f} |")
    lines.append(f"| r_hall (hallucination) | {rb.get('r_hall', 0):.4f} |")
    lines.append(f"| format_valid | {'✅' if rb.get('format_valid', True) else '❌'} |")
    lines.append(f"| tool_valid | {'✅' if rb.get('tool_valid', True) else '❌'} |")
    lines.append(f"| safety_valid | {'✅' if rb.get('safety_valid', True) else '❌'} |")
    lines.append(f"| is_correct | {'✅' if rb.get('is_correct', False) else '❌'} |")

    details = rb.get("details", {})
    if details:
        lines.append("")
        lines.append("**Details:**")
        for k, v in details.items():
            if isinstance(v, float):
                lines.append(f"- {k}: {v:.4f}")
            else:
                lines.append(f"- {k}: {v}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
# Environment labels for the UI dropdown
_ENV_LABELS: dict[str, str] = {
    "PD": "🔍 Product Discovery",
    "SUB": "🔄 Substitution",
    "CART": "🛒 Cart Building",
    "RETURN": "↩️ Return + Replacement",
    "STATUS": "📍 Order Tracking",
    "POLICY": "📜 Policy QA",
    "BUNDLE": "📦 Bundle Planning",
    "JOURNEY": "🗺️ Multi-Intent Journey",
}


class SessionState:
    """Holds episode state for a Gradio session."""

    def __init__(self):
        self.env: EcomRLVEEnv | None = None
        self.obs = None
        self.done = False
        self.reward = 0.0
        self.conversation: list[dict] = []
        self.tool_history: list[str] = []
        self.turn = 0
        self.env_id: str = "PD"  # current environment
        self.difficulty: int = 5
        self.persona_weights: PersonaWeights | None = None
        self.goal_params: dict = {}
        self.system_prompt: str = ""
        self.episode_info: dict = {}
        self.hidden_goal_display: str = ""


def _new_session() -> SessionState:
    return SessionState()


# ---------------------------------------------------------------------------
# Core Actions
# ---------------------------------------------------------------------------

def reset_episode(session_state, selected_env, selected_difficulty):
    """Reset the environment and start a new episode."""
    env = _get_env()
    session = _new_session()
    session.env = env

    # Parse env_id from the dropdown label (e.g., "PD" from "🔍 Product Discovery (PD)")
    chosen_env = selected_env.split("(")[-1].rstrip(")").strip() if "(" in selected_env else selected_env
    chosen_difficulty = int(selected_difficulty)
    session.env_id = chosen_env
    session.difficulty = chosen_difficulty

    # Reset environment
    obs = env.reset(env_id=chosen_env, difficulty=chosen_difficulty)
    session.obs = obs
    session.turn = obs.turn
    session.conversation = list(obs.conversation)

    # Get episode state for persona + goal info
    ep_state = env.get_episode_state()
    if ep_state:
        session.persona_weights = ep_state.persona_weights
        session.goal_params = env._extract_goal_params(ep_state.hidden_goal, ep_state.env_id)
        session.hidden_goal_display = _format_hidden_goal(ep_state)

        # --- Seed order history so order/return tools work ---
        if not ep_state.orders:
            import random as _rng
            seed = ep_state.seed if ep_state.seed else 42
            products_list = list(ep_state.products_by_id.values())
            if products_list:
                ep_state.orders = generate_order_history(
                    products=products_list,
                    n_orders=_rng.Random(seed).randint(3, 8),
                    seed=seed,
                )
                logger.info("Seeded %d orders for episode", len(ep_state.orders))

        # --- Pre-populate cart with 1-2 items so cart tools work ---
        if not ep_state.cart.lines:
            import random as _rng
            seed = (ep_state.seed if ep_state.seed else 42) + 7
            rng = _rng.Random(seed)
            products_list = list(ep_state.products_by_id.values())
            if products_list:
                n_items = rng.randint(1, 2)
                for p in rng.sample(products_list, min(n_items, len(products_list))):
                    line_id = ep_state.cart._generate_line_id()
                    ep_state.cart.lines.append(CartLine(
                        line_id=line_id,
                        product_id=p.id,
                        qty=rng.randint(1, 3),
                        unit_price=p.price,
                    ))
                logger.info("Seeded cart with %d item(s)", len(ep_state.cart.lines))

    # Build persona description
    persona_desc = _build_persona_description(session.persona_weights) if session.persona_weights else "A typical shopper."
    goal_desc = _build_goal_description(session.goal_params)
    session.system_prompt = PERSONA_SYSTEM_PROMPT.format(
        persona_desc=persona_desc,
        goal_desc=goal_desc,
    )

    # Try LLM-generated initial message, fall back to template
    user_msg = ""
    for msg in session.conversation:
        if msg.get("role") == "user":
            user_msg = msg.get("content", "")
            break

    # If we have an LLM endpoint, enhance the initial message
    if HF_TOKEN:
        llm_msg = _llm_generate_user_message(
            session.system_prompt,
            [],  # no prior conversation
        )
        if llm_msg:
            user_msg = llm_msg
            # Update conversation
            session.conversation = [{"role": "user", "content": user_msg}]

    # Format chat display
    chat_messages = []
    for msg in session.conversation:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            chat_messages.append({"role": "user", "content": f"🛒 **Customer:** {content}"})
        elif role == "assistant":
            chat_messages.append({"role": "assistant", "content": content})

    # Persona info
    persona_md = _format_persona(session.persona_weights, session.goal_params)

    # Episode info
    n_orders = len(ep_state.orders) if ep_state else 0
    n_cart = len(ep_state.cart.lines) if ep_state else 0
    env_label = _ENV_LABELS.get(session.env_id, session.env_id)
    info_md = f"""### 📋 Episode Info
- **Environment:** {env_label} ({session.env_id})
- **Difficulty:** {obs.difficulty or session.difficulty}
- **Turn:** {session.turn} / 14
- **Status:** 🟢 Active
- **Cart:** {n_cart} item(s) pre-loaded
- **Orders:** {n_orders} in history
- **Policies:** 62 rules loaded
"""

    # Hidden goal (debug)
    goal_md = session.hidden_goal_display

    return (
        session,           # state
        chat_messages,     # chatbot
        persona_md,        # persona display
        info_md,           # info display
        "",                # reward display (empty at start)
        goal_md,           # hidden goal display
        gr.update(interactive=True),   # tool buttons
        gr.update(interactive=True),   # assistant input
        gr.update(interactive=True),   # submit button
        gr.update(interactive=True),   # done button
    )


def execute_tool(session_state, tool_name, arg1, arg2, arg3, arg4, arg5):
    """Execute a tool call and display results."""
    if session_state is None or session_state.done:
        return session_state, [], "Reset first!", "", ""

    env = session_state.env or _get_env()

    # Build tool args
    args = {}
    tool_def = TOOLS.get(tool_name, {})
    arg_defs = tool_def.get("args", {})
    arg_values = [arg1, arg2, arg3, arg4, arg5]

    for (arg_name, arg_def), val in zip(arg_defs.items(), arg_values):
        if val is None or val == "":
            if arg_def.get("required"):
                return (
                    session_state,
                    _format_chat(session_state),
                    f"❌ Required argument `{arg_name}` is missing!",
                    _format_episode_info(session_state),
                    "",
                )
            continue

        atype = arg_def.get("type", "str")
        if atype == "int":
            try:
                args[arg_name] = int(val)
            except ValueError:
                args[arg_name] = arg_def.get("default", 10)
        elif atype == "json":
            try:
                args[arg_name] = json.loads(val) if val else None
            except json.JSONDecodeError:
                args[arg_name] = None
        elif atype == "list":
            args[arg_name] = [s.strip() for s in val.split(",") if s.strip()]
        else:
            args[arg_name] = str(val)

    # Build action JSON
    action = {
        "assistant_message": f"Let me use {tool_name} to help you.",
        "tool_calls": [{"name": tool_name, "args": args}],
    }
    action_json = json.dumps(action)

    # Execute step
    obs, reward, done, info = env.step(action_json)
    session_state.obs = obs
    session_state.turn = obs.turn
    session_state.done = done
    session_state.reward = reward
    session_state.conversation = list(obs.conversation)
    session_state.episode_info = info

    # Format tool results
    tool_output = ""
    if obs.tool_results:
        for tr in obs.tool_results:
            tool_output += _format_tool_result(tr) + "\n\n"
        session_state.tool_history.append(tool_output)

    # Reward display
    reward_md = ""
    if done:
        reward_md = _format_reward_breakdown(info)

    return (
        session_state,
        _format_chat(session_state),
        tool_output or "No results returned.",
        _format_episode_info(session_state),
        reward_md,
    )


def submit_response(session_state, assistant_message):
    """Submit an assistant response (no tool calls) and get next user message."""
    if session_state is None or session_state.done:
        return session_state, [], "", "", ""

    if not assistant_message.strip():
        return (
            session_state,
            _format_chat(session_state),
            "Please write a message to the customer.",
            _format_episode_info(session_state),
            "",
        )

    env = session_state.env or _get_env()

    action = {
        "assistant_message": assistant_message,
        "tool_calls": [],
    }
    action_json = json.dumps(action)

    obs, reward, done, info = env.step(action_json)
    session_state.obs = obs
    session_state.turn = obs.turn
    session_state.done = done
    session_state.reward = reward
    session_state.conversation = list(obs.conversation)
    session_state.episode_info = info

    # If we have LLM endpoint and not done, enhance the next user message
    if not done and HF_TOKEN and session_state.system_prompt:
        llm_msg = _llm_generate_user_message(
            session_state.system_prompt,
            session_state.conversation,
        )
        if llm_msg:
            # Replace the last user message with LLM-generated one
            if session_state.conversation and session_state.conversation[-1].get("role") == "user":
                session_state.conversation[-1]["content"] = llm_msg

    reward_md = _format_reward_breakdown(info) if done else ""

    return (
        session_state,
        _format_chat(session_state),
        "",
        _format_episode_info(session_state),
        reward_md,
    )


def submit_answer(session_state, assistant_message, product_ids_str):
    """Submit final answer with recommended product IDs."""
    if session_state is None or session_state.done:
        return session_state, [], "", "", "Episode already ended. Click Reset."

    env = session_state.env or _get_env()

    product_ids = [s.strip() for s in product_ids_str.split(",") if s.strip()] if product_ids_str else []

    current_env = session_state.env_id if session_state else "PD"
    action = {
        "assistant_message": assistant_message or "Here are my recommendations.",
        "tool_calls": [],
        "answer": {
            "env": current_env,
            "done": True,
            "recommended_product_ids": product_ids,
        },
    }
    action_json = json.dumps(action)

    obs, reward, done, info = env.step(action_json)
    session_state.obs = obs
    session_state.turn = obs.turn
    session_state.done = done
    session_state.reward = reward
    session_state.conversation = list(obs.conversation)
    session_state.episode_info = info

    reward_md = _format_reward_breakdown(info)

    return (
        session_state,
        _format_chat(session_state),
        "",
        _format_episode_info(session_state),
        reward_md,
    )


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _format_chat(session: SessionState) -> list[dict]:
    """Format conversation for Gradio chatbot (messages format)."""
    messages = []
    for msg in session.conversation:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            messages.append({"role": "user", "content": f"🛒 {content}"})
        elif role == "assistant":
            messages.append({"role": "assistant", "content": f"🤖 {content}"})
    return messages


def _format_persona(weights: PersonaWeights | None, goal: dict) -> str:
    if weights is None:
        return "No persona loaded."

    bars = {
        "💰 Price": weights.w_price,
        "⭐ Rating": weights.w_rating,
        "🚚 Shipping": weights.w_ship,
        "🏷️ Brand": weights.w_brand,
        "🎯 Relevance": weights.w_similarity,
    }

    lines = ["### 👤 Customer Persona", ""]
    for label, w in bars.items():
        bar_len = int(w * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        lines.append(f"**{label}:** `{bar}` {w:.2f}")

    # Natural language persona description
    desc = _build_persona_description(weights)
    lines.append("")
    lines.append(f"📝 *{desc}*")

    lines.append("")
    lines.append("### 🎯 Shopping Goal")
    for k, v in goal.items():
        lines.append(f"- **{k}:** {v}")

    # Natural language goal description
    goal_desc = _build_goal_description(goal)
    lines.append("")
    lines.append(f"📋 *{goal_desc}*")

    return "\n".join(lines)


def _format_hidden_goal(ep_state) -> str:
    """Format hidden goal info (visible to user for debugging)."""
    problem = ep_state.hidden_goal
    lines = ["**Target products:** " + ", ".join(problem.target_product_ids[:3])]
    if problem.constraints:
        lines.append(f"**Constraints:** {len(problem.constraints)}")
        for c in problem.constraints[:5]:
            lines.append(f"  - {c.get('attr', '?')} {c.get('op', '?')} {c.get('value', '?')}")
    return "\n".join(lines)


def _format_episode_info(session: SessionState) -> str:
    status = "🔴 Done" if session.done else "🟢 Active"
    reward_str = f"{session.reward:.4f}" if session.done else "—"
    env_label = _ENV_LABELS.get(session.env_id, session.env_id)
    return f"""### 📋 Episode Info
- **Environment:** {env_label} ({session.env_id})
- **Difficulty:** {session.difficulty}
- **Turn:** {session.turn} / 14
- **Status:** {status}
- **Reward:** {reward_str}
- **Tools used:** {len(session.tool_history)}
"""


def update_tool_args(tool_name):
    """Update argument input labels based on selected tool."""
    tool_def = TOOLS.get(tool_name, {})
    arg_defs = tool_def.get("args", {})
    arg_names = list(arg_defs.keys())

    updates = []
    for i in range(5):
        if i < len(arg_names):
            name = arg_names[i]
            info = arg_defs[name]
            req = " ✱" if info.get("required") else f" (default: {info.get('default', '')})"
            updates.append(
                gr.update(
                    label=f"{name}{req}",
                    placeholder=info.get("description", ""),
                    value="",
                    visible=True,
                )
            )
        else:
            updates.append(
                gr.update(label=f"arg{i+1} (unused)", placeholder="", value="", visible=False)
            )

    # Build a rich per-tool description with args
    desc = tool_def.get("description", "")
    arg_defs = tool_def.get("args", {})
    detail_lines = [f"**{tool_name}** — {desc}"]
    if arg_defs:
        detail_lines.append("")
        for aname, adef in arg_defs.items():
            req = "✱ required" if adef.get("required") else f"optional, default={adef.get('default')}"
            detail_lines.append(f"- `{aname}` ({req}) — {adef.get('description', '')}")
    else:
        detail_lines.append("\n*No arguments needed — just click Execute.*")
    return (*updates, gr.update(value="\n".join(detail_lines)))


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

DESCRIPTION = """
# 🛍️ EcomRLVE-GYM — Interactive Demo

**You are the AI agent.** A simulated customer will ask for help finding products.
Use the available tools to search the catalog, then submit your recommendation.

**Flow:** Reset → Read customer message → Use tools → Write response → Submit answer → See reward

The environment has **2M real Amazon products** indexed with FAISS HNSW + gte-small embeddings.
"""

CUSTOM_CSS = """
.tool-btn { min-width: 180px; }
.reward-box { border: 2px solid #4CAF50; border-radius: 8px; padding: 12px; }
"""


def _build_tools_reference() -> str:
    """Build a Markdown reference table of all tools from the TOOLS dict."""
    sections = {
        "catalog": ("🔍 Catalog", "Search, browse and inspect products"),
        "cart":    ("🛒 Cart", "Manage shopping cart"),
        "order":   ("📋 Orders", "List orders, check status, checkout"),
        "return":  ("↩️ Returns", "Check eligibility, initiate returns, exchanges"),
        "policy":  ("📜 Policy", "Search store policies and rules"),
    }
    lines = []
    for prefix, (heading, subtitle) in sections.items():
        tools_in_section = {k: v for k, v in TOOLS.items() if k.startswith(prefix + ".")}
        if not tools_in_section:
            continue
        lines.append(f"**{heading}** — {subtitle}")
        lines.append("")
        lines.append("| Tool | Description | Args |")
        lines.append("|------|-------------|------|")
        for name, tdef in tools_in_section.items():
            desc = tdef["description"]
            arg_defs = tdef.get("args", {})
            if not arg_defs:
                args_str = "*(none)*"
            else:
                parts = []
                for aname, adef in arg_defs.items():
                    marker = "**" if adef.get("required") else ""
                    default = f"={adef['default']}" if not adef.get("required") and adef.get("default") is not None else ""
                    parts.append(f"`{marker}{aname}{marker}`{default}")
                args_str = ", ".join(parts)
            lines.append(f"| `{name}` | {desc} | {args_str} |")
        lines.append("")
    return "\n".join(lines)


TOOLS_REFERENCE_MD = _build_tools_reference()


def _build_policies_reference() -> str:
    """Build a human-readable Markdown summary of all store policies."""
    kb = build_default_policy_kb()
    cat_titles = {
        "returns": "↩️ Returns",
        "shipping": "🚚 Shipping",
        "pricing": "💰 Pricing",
        "membership": "⭐ Membership",
        "warranty": "🛡️ Warranty",
    }
    lines = []
    for cat, heading in cat_titles.items():
        rules = kb.get_rules_by_category(cat)
        if not rules:
            continue
        lines.append(f"**{heading}** ({len(rules)} rules)")
        lines.append("")
        lines.append("| Rule | Conditions | Answer |")
        lines.append("|------|-----------|--------|")
        for r in rules:
            conds = ", ".join(f"`{c['field']}` {c['op']} `{c['value']}`" for c in r.conditions) or "—"
            ans = f"`{r.answer}`" if r.answer_type == "categorical" else f"**{r.answer}**"
            lines.append(f"| {r.title} | {conds} | {ans} |")
        lines.append("")
    return "\n".join(lines)


POLICIES_REFERENCE_MD = _build_policies_reference()


with gr.Blocks(title="EcomRLVE-GYM") as demo:

    # State
    session_state = gr.State(value=None)

    # Header
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Accordion("📖 Tools Reference", open=False):
            gr.Markdown(TOOLS_REFERENCE_MD)
        with gr.Accordion("📜 Store Policies", open=False):
            gr.Markdown(POLICIES_REFERENCE_MD)

    with gr.Row():
        # ===== LEFT COLUMN: Persona + Info =====
        with gr.Column(scale=1, min_width=280):
            persona_display = gr.Markdown(
                value="Click **Reset** to start a new episode.",
                label="Persona",
            )
            episode_info = gr.Markdown(value="", label="Episode")
            hidden_goal_display = gr.Markdown(value="", label="Hidden Goal (Debug)")

            env_selector = gr.Dropdown(
                choices=[f"{label} ({eid})" for eid, label in _ENV_LABELS.items()],
                value=f"{_ENV_LABELS['PD']} (PD)",
                label="Environment",
                interactive=True,
            )
            difficulty_slider = gr.Slider(
                minimum=1, maximum=12, step=1, value=5,
                label="Difficulty (1-12)",
                interactive=True,
            )

            reset_btn = gr.Button("🔄 Reset Episode", variant="primary", size="lg")

        # ===== CENTER COLUMN: Chat =====
        with gr.Column(scale=2, min_width=400):
            chatbot = gr.Chatbot(
                value=[],
                label="Conversation",
                height=400,
            )

            with gr.Group():
                assistant_input = gr.Textbox(
                    label="Your response (as the AI agent)",
                    placeholder="Type your message to the customer...",
                    lines=2,
                    interactive=False,
                )
                with gr.Row():
                    send_btn = gr.Button("💬 Send Response", interactive=False)
                    with gr.Column(scale=2):
                        answer_ids = gr.Textbox(
                            label="Product IDs to recommend",
                            placeholder="B01ABC123, B02DEF456",
                            lines=1,
                        )
                    done_btn = gr.Button("✅ Submit Answer", variant="stop", interactive=False)

        # ===== RIGHT COLUMN: Tools =====
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### 🔧 Tools")

            tool_selector = gr.Dropdown(
                choices=list(TOOLS.keys()),
                value="catalog.search",
                label="Select Tool",
                interactive=True,
            )

            tool_desc = gr.Markdown(value="**catalog.search** — 🔍 Search products by query + optional filters\n\n"
                                          "- `query` (✱ required) — Search query\n"
                                          "- `top_k` (optional, default=20) — Number of results (1-500)\n"
                                          "- `filters` (optional) — Filters JSON e.g. {\"price_max\":50}")

            tool_arg1 = gr.Textbox(label="query ✱", placeholder="Search query", interactive=True)
            tool_arg2 = gr.Textbox(label="top_k (default: 20)", placeholder="Number of results (1-500)", interactive=True, visible=True)
            tool_arg3 = gr.Textbox(label="filters (optional)", placeholder='{"price_max": 50}', interactive=True, visible=True)
            tool_arg4 = gr.Textbox(label="arg4 (unused)", placeholder="", interactive=True, visible=False)
            tool_arg5 = gr.Textbox(label="arg5 (unused)", placeholder="", interactive=True, visible=False)

            execute_btn = gr.Button("⚡ Execute Tool", variant="secondary", interactive=True)

            tool_output = gr.Markdown(value="", label="Tool Results")

    # ===== BOTTOM ROW: Reward (full width, always visible) =====
    with gr.Row():
        reward_display = gr.Markdown(value="", label="Reward")

    # ===== Event wiring =====

    # Reset
    reset_btn.click(
        fn=reset_episode,
        inputs=[session_state, env_selector, difficulty_slider],
        outputs=[
            session_state, chatbot, persona_display, episode_info,
            reward_display, hidden_goal_display,
            execute_btn, assistant_input, send_btn, done_btn,
        ],
    )

    # Tool selection changes arg labels
    tool_selector.change(
        fn=update_tool_args,
        inputs=[tool_selector],
        outputs=[tool_arg1, tool_arg2, tool_arg3, tool_arg4, tool_arg5, tool_desc],
    )

    # Execute tool
    execute_btn.click(
        fn=execute_tool,
        inputs=[session_state, tool_selector, tool_arg1, tool_arg2, tool_arg3, tool_arg4, tool_arg5],
        outputs=[session_state, chatbot, tool_output, episode_info, reward_display],
    )

    # Send response
    send_btn.click(
        fn=submit_response,
        inputs=[session_state, assistant_input],
        outputs=[session_state, chatbot, tool_output, episode_info, reward_display],
    ).then(
        fn=lambda: "",
        outputs=[assistant_input],
    )

    # Submit final answer
    done_btn.click(
        fn=submit_answer,
        inputs=[session_state, assistant_input, answer_ids],
        outputs=[session_state, chatbot, tool_output, episode_info, reward_display],
    )


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
    )
