"""
EcomRLVE-GYM Interactive Testing UI — Local Dev

YOU play the role of the candidate LLM agent.
A persona-driven user simulator (Ollama LLM) plays the customer.
Use tools, chat, then submit your answer — see reward breakdown.

Key features over the original app:
  - datetime.now tool for return window reasoning
  - RETURN answer fields (selected_order_id, selected_line_id)
  - Big visible reward banner with colour coding
  - LLM verbalization indicator panel
  - Env-specific hidden goal display (CART items w/ variants, RETURN order table)
  - p_missing / p_noise shown in episode info
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
# Path setup
# ---------------------------------------------------------------------------
_app_dir = Path(__file__).resolve().parent
if (_app_dir / "src" / "ecom_rlve").is_dir():
    ROOT = _app_dir
else:
    ROOT = _app_dir.parent
sys.path.insert(0, str(ROOT / "src"))

from ecom_rlve.data.catalog_loader import load_catalog
from ecom_rlve.server.openenv import EcomRLVEEnv
from ecom_rlve.simulator.persona import PersonaWeights
from ecom_rlve.tools.cart import CartLine

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ecomrlve-space")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FAISS_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", str(ROOT / "data" / "faiss-index"))
CATALOG_PATH = os.getenv("CATALOG_PATH", str(ROOT / "data" / "amazebay-2M"))
CATALOG_MAX_ITEMS = int(os.getenv("CATALOG_MAX_ITEMS", "5000"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "thenlper/gte-small")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------
_env_singleton = None


def _get_env() -> EcomRLVEEnv:
    global _env_singleton
    if _env_singleton is not None:
        return _env_singleton

    logger.info("Loading catalog from %s (max_items=%d)…", CATALOG_PATH, CATALOG_MAX_ITEMS)
    products = load_catalog(CATALOG_PATH, max_items=CATALOG_MAX_ITEMS, seed=42)
    logger.info("Loaded %d products", len(products))

    faiss_path = FAISS_INDEX_DIR if Path(FAISS_INDEX_DIR).exists() else None
    config = {
        "embedding_model": EMBEDDING_MODEL,
        "embedding_debug": False,
        "embedding_device": EMBEDDING_DEVICE,
    }
    if faiss_path:
        config["faiss_index_path"] = faiss_path

    env = EcomRLVEEnv(
        collection="C8",
        catalog=(products, []),
        config=config,
        seed=int(time.time()) % 100_000,
    )
    _env_singleton = env
    logger.info("Environment ready!")
    return env


# ---------------------------------------------------------------------------
# Tool definitions (with datetime.now)
# ---------------------------------------------------------------------------
TOOLS: dict[str, dict] = {
    "catalog.search": {
        "description": "🔍 Search products by query + optional filters",
        "args": {
            "query":   {"type": "str",  "required": True,  "description": "Search query"},
            "top_k":   {"type": "int",  "required": False, "default": 20,   "description": "Number of results (1-500)"},
            "filters": {"type": "json", "required": False, "default": None, "description": 'Filters JSON e.g. {"price_max":50}'},
        },
    },
    "catalog.rerank": {
        "description": "📊 Re-rank candidates by query relevance",
        "args": {
            "query":                 {"type": "str",  "required": True,  "description": "Reranking query"},
            "candidate_product_ids": {"type": "list", "required": True,  "description": "Comma-separated product IDs"},
            "top_k":                 {"type": "int",  "required": False, "default": 10,  "description": "Results to return"},
        },
    },
    "catalog.get_product": {
        "description": "📦 Get full product details by ID",
        "args": {"product_id": {"type": "str", "required": True, "description": "Product ID"}},
    },
    "catalog.get_variants": {
        "description": "🎨 Get colour/size variants for a product",
        "args": {"product_id": {"type": "str", "required": True, "description": "Product ID"}},
    },
    "cart.view":         {"description": "🛒 View current cart",         "args": {}},
    "cart.add": {
        "description": "➕ Add product to cart",
        "args": {
            "product_id": {"type": "str", "required": True,  "description": "Product ID"},
            "variant_id": {"type": "str", "required": False, "default": None, "description": "Variant ID"},
            "quantity":   {"type": "int", "required": False, "default": 1,    "description": "Quantity"},
        },
    },
    "cart.remove": {
        "description": "🗑️ Remove cart line",
        "args": {"line_id": {"type": "str", "required": True, "description": "Cart line ID"}},
    },
    "cart.set_quantity": {
        "description": "✏️ Set cart line quantity",
        "args": {
            "line_id":  {"type": "str", "required": True, "description": "Cart line ID"},
            "quantity": {"type": "int", "required": True, "description": "New quantity (0 = remove)"},
        },
    },
    "order.list": {
        "description": "📋 List recent orders",
        "args": {"days": {"type": "int", "required": False, "default": 30, "description": "Lookback days"}},
    },
    "order.get_status": {
        "description": "📍 Get order status & tracking",
        "args": {"order_id": {"type": "str", "required": True, "description": "Order ID"}},
    },
    "order.checkout": {
        "description": "💳 Checkout current cart",
        "args": {
            "address_id": {"type": "str", "required": True, "description": "Address ID"},
            "payment_id": {"type": "str", "required": True, "description": "Payment ID"},
        },
    },
    "return.check_eligibility": {
        "description": "✅ Check return eligibility",
        "args": {
            "order_id": {"type": "str", "required": True, "description": "Order ID"},
            "line_id":  {"type": "str", "required": True, "description": "Line ID"},
        },
    },
    "return.initiate": {
        "description": "↩️ Initiate a return",
        "args": {
            "order_id":    {"type": "str", "required": True, "description": "Order ID"},
            "line_id":     {"type": "str", "required": True, "description": "Line ID"},
            "reason_code": {"type": "str", "required": True, "description": "defective|wrong_item|not_as_described|changed_mind|too_large|too_small|arrived_late|better_price_found"},
            "method":      {"type": "str", "required": True, "description": "mail|in_store|pickup"},
        },
    },
    "return.exchange": {
        "description": "🔄 Exchange for a different product",
        "args": {
            "order_id":       {"type": "str", "required": True,  "description": "Order ID"},
            "line_id":        {"type": "str", "required": True,  "description": "Line ID"},
            "new_product_id": {"type": "str", "required": True,  "description": "Replacement product ID"},
            "new_variant_id": {"type": "str", "required": False, "default": None, "description": "Replacement variant ID"},
        },
    },
    "policy.search": {
        "description": "📜 Search store policies",
        "args": {
            "query": {"type": "str", "required": True,  "description": "Policy query"},
            "top_k": {"type": "int", "required": False, "default": 5, "description": "Results"},
        },
    },
    "datetime.now": {
        "description": "🕐 Get current date & time",
        "args": {},
    },
    "user.get_visit_history": {
        "description": "📜 Get customer's recently viewed products",
        "args": {},
    },
}

_ENV_LABELS: dict[str, str] = {
    "PD":      "🔍 Product Discovery",
    "SUB":     "🔄 Substitution",
    "CART":    "🛒 Cart Building",
    "RETURN":  "↩️ Return + Replacement",
    "STATUS":  "📍 Order Tracking",
    "POLICY":  "📜 Policy QA",
    "BUNDLE":  "📦 Bundle Planning",
    "JOURNEY": "🗺️ Multi-Intent Journey",
}


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------
class SessionState:
    def __init__(self):
        self.env: EcomRLVEEnv | None = None
        self.obs = None
        self.done = False
        self.reward = 0.0
        self.conversation: list[dict] = []
        self.tool_history: list[str] = []
        self.turn = 0
        self.env_id: str = "CART"
        self.difficulty: int = 3
        self.persona_weights: PersonaWeights | None = None
        self.goal_params: dict = {}
        self.episode_info: dict = {}
        self.hidden_goal_md: str = ""
        # Verbalization tracking
        self.initial_msg_source: str = "unknown"
        self.user_sim_log: list[str] = []


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_tool_result(result: dict) -> str:
    name = result.get("name", "?")
    error = result.get("error")
    data = result.get("result")
    ms = result.get("duration_ms", 0)

    if error:
        return f"❌ **{name}** ({ms:.0f}ms): {error}"

    if name == "datetime.now" and isinstance(data, dict):
        return (f"🕐 **{name}** ({ms:.0f}ms)\n"
                f"  📅 **{data.get('date','?')}** ({data.get('day_of_week','?')})"
                f"  ⏰ {data.get('time','?')}")

    if isinstance(data, list):
        lines = [f"✅ **{name}** ({ms:.0f}ms) — {len(data)} results:"]
        for i, item in enumerate(data[:12]):
            if isinstance(item, dict):
                title = item.get("title", item.get("product_title", "?"))[:55]
                pid   = item.get("product_id", item.get("id", item.get("order_id", "?")))
                price = item.get("price", item.get("unit_price", ""))
                rating = item.get("rating", "")
                status = item.get("status", "")
                qty    = item.get("qty", "")
                parts = [f"`{i+1}.` **{title}**"]
                if price:  parts.append(f"${price}")
                if rating: parts.append(f"★{rating}")
                if status: parts.append(f"[{status}]")
                if qty:    parts.append(f"×{qty}")
                parts.append(f"`{pid}`")
                lines.append("  " + " ".join(parts))
        if len(data) > 12:
            lines.append(f"  … +{len(data)-12} more")
        return "\n".join(lines)

    if isinstance(data, dict):
        return f"✅ **{name}** ({ms:.0f}ms):\n```json\n{json.dumps(data, indent=2, default=str)[:900]}\n```"

    return f"✅ **{name}** ({ms:.0f}ms): {str(data)[:300]}"


def _fmt_reward_banner(reward: float, info: dict) -> str:
    """Big visible HTML reward banner."""
    rb = info.get("reward_breakdown", {}) or {}
    is_correct = rb.get("is_correct", False)
    r_task = rb.get("r_task", 0)
    r_eff  = rb.get("r_eff", 0)
    r_hall = rb.get("r_hall", 0)
    fmt_ok = rb.get("format_valid", True)
    tool_ok = rb.get("tool_valid", True)

    if reward >= 0.8:   c, label = "#2ecc71", "🟢 EXCELLENT"
    elif reward >= 0.5: c, label = "#27ae60", "🟢 GOOD"
    elif reward >= 0.2: c, label = "#f39c12", "🟡 FAIR"
    elif reward >= 0:   c, label = "#e67e22", "🟠 POOR"
    else:               c, label = "#e74c3c", "🔴 FAILED"

    correct_txt = "✅ CORRECT" if is_correct else "❌ INCORRECT"

    html = f"""
<div style="background:linear-gradient(135deg,{c}22,{c}44);border:3px solid {c};
            border-radius:16px;padding:24px;margin:12px 0;text-align:center;">
  <div style="font-size:48px;font-weight:bold;color:{c};">{reward:+.4f}</div>
  <div style="font-size:20px;margin:6px 0;color:{c};">{label} &nbsp;|&nbsp; {correct_txt}</div>
  <hr style="border-color:{c}44;margin:12px 0;">
  <div style="display:flex;justify-content:center;gap:40px;flex-wrap:wrap;">
    <div><div style="font-size:11px;color:#888;">r_task (75%)</div>
         <div style="font-size:24px;font-weight:bold;">{r_task:+.4f}</div></div>
    <div><div style="font-size:11px;color:#888;">r_eff (15%)</div>
         <div style="font-size:24px;font-weight:bold;">{r_eff:+.4f}</div></div>
    <div><div style="font-size:11px;color:#888;">r_hall (10%)</div>
         <div style="font-size:24px;font-weight:bold;">{r_hall:+.4f}</div></div>
  </div>
  <div style="margin-top:10px;font-size:13px;color:#888;">
    Format: {'✅' if fmt_ok else '❌'} &nbsp;|&nbsp; Tools: {'✅' if tool_ok else '❌'}
  </div>
</div>"""

    details = rb.get("details", {})
    if details:
        html += "\n\n**Details:**\n\n| Key | Value |\n|-----|-------|\n"
        for k, v in details.items():
            html += f"| {k} | {v:.4f if isinstance(v, float) else v} |\n"
    return html


def _fmt_hidden_goal(ep_state) -> str:
    """Env-specific hidden goal formatting."""
    if not ep_state or not ep_state.hidden_goal:
        return ""
    problem = ep_state.hidden_goal
    env_id  = ep_state.env_id
    extra   = problem.extra or {}
    lines   = []

    if env_id == "CART":
        lines.append("### 🛒 CART Goal")
        items = extra.get("item_details", [])
        visit = extra.get("visit_history", [])
        if items:
            lines += ["", "| # | Description | Qty | Variant | PID |",
                          "|---|-------------|-----|---------|-----|"]
            for i, d in enumerate(items, 1):
                desc_parts = d.get('description', [])
                desc_str = ', '.join(str(p) for p in desc_parts[:3]) if desc_parts else '—'
                lines.append(
                    f"| {i} | {desc_str[:40]} | {d.get('qty',1)} "
                    f"| {d.get('variant_desc','—')} | `{d.get('product_id','?')}` |")
            lines.append(f"")
            lines.append(f"**Full titles** (hidden from user):")
            for i, d in enumerate(items, 1):
                lines.append(f"  {i}. {d.get('title','?')[:60]}")
        if visit:
            lines.append(f"")
            lines.append(f"**Visit history:** {len(visit)} items (targets + {len(visit) - len(items)} distractors)")

    elif env_id == "RETURN":
        lines.append("### ↩️ RETURN Goal")
        t_title = extra.get("target_product_title", "?")
        t_oid   = extra.get("target_order_id", "?")
        t_lid   = extra.get("target_line_id", "?")
        window  = extra.get("window_days", "?")
        t_days  = extra.get("t_days", "?")
        repl    = extra.get("replacement_required", False)
        p_edge  = extra.get("p_edge", 0)

        eligible = "✅" if isinstance(t_days, int) and isinstance(window, int) and t_days <= window else "❌ expired"

        lines += ["",
            "| Field | Value |", "|-------|-------|",
            f"| Product | **{t_title}** |",
            f"| Order | `{t_oid}` |",
            f"| Line | `{t_lid}` |",
            f"| Window | {window} days |",
            f"| Age | {t_days} days |",
            f"| Eligible | {eligible} |",
            f"| p_edge | {p_edge:.3f} |",
            f"| Replacement | {'✅' if repl else '❌'} |",
        ]
        if repl and problem.constraints:
            lines += ["", "**Replacement constraints:**"]
            for c in problem.constraints:
                lines.append(f"  - {c.get('attr')} {c.get('op')} {c.get('value')}")

        orders = extra.get("orders", [])
        if orders:
            lines += ["", f"**Orders ({len(orders)}):**"]
            for o in orders[:10]:
                titles = ", ".join(l.product_title[:25] for l in (o.lines if hasattr(o,'lines') else [])[:3])
                star = " ⭐" if getattr(o, 'order_id', '') == t_oid else ""
                lines.append(f"  `{o.order_id}` [{o.status}] {titles}{star}")

    else:
        lines.append(f"### {env_id} Goal")
        lines.append(f"**Targets:** {', '.join(f'`{p}`' for p in problem.target_product_ids[:5])}")
        if problem.constraints:
            lines.append(f"**Constraints ({len(problem.constraints)}):**")
            for c in problem.constraints[:8]:
                lines.append(f"  - {c.get('attr')} {c.get('op')} {c.get('value')}")

    return "\n".join(lines)


def _fmt_verbalization(session: SessionState) -> str:
    src = session.initial_msg_source
    icon = {"llm": "🟢 LLM (Ollama)", "template": "🟡 Template"}.get(src, "⚪ —")
    lines = [
        "### 🤖 Verbalization Tracker",
        "", f"**Initial message:** {icon}", "",
    ]
    if session.user_sim_log:
        lines.append("**Response log:**")
        for entry in session.user_sim_log[-8:]:
            lines.append(f"  {entry}")
    return "\n".join(lines)


def _fmt_episode(session: SessionState) -> str:
    ep = session.env.get_episode_state() if session.env else None
    extra = ep.hidden_goal.extra if ep and ep.hidden_goal else {}
    t_max = extra.get("T_max", 14)
    p_miss = extra.get("p_missing", 0)
    p_noise = extra.get("p_noise", 0)
    n_cart = len(ep.cart.lines) if ep else 0
    n_ord  = len(ep.orders) if ep else 0
    n_seen = len(ep.seen_product_ids) if ep else 0
    n_ret  = len(ep.initiated_returns) if ep else 0
    status = "🔴 Done" if session.done else "🟢 Active"
    env_lbl = _ENV_LABELS.get(session.env_id, session.env_id)

    return f"""### 📋 Episode
| | |
|---|---|
| **Env** | {env_lbl} |
| **Difficulty** | {session.difficulty} |
| **Turn** | {session.turn} / {t_max} |
| **Status** | {status} |
| **Cart** | {n_cart} items |
| **Orders** | {n_ord} |
| **Seen** | {n_seen} products |
| **Returns** | {n_ret} |
| **p_missing** | {p_miss:.2f} |
| **p_noise** | {p_noise:.2f} |
| **Tools** | {len(session.tool_history)} calls |
"""


def _fmt_chat(session: SessionState) -> list[dict]:
    out = []
    for m in session.conversation:
        r, c = m.get("role", "user"), m.get("content", "")
        if r == "user":
            out.append({"role": "user",      "content": f"🛒 {c}"})
        else:
            out.append({"role": "assistant", "content": f"🤖 {c}"})
    return out


def _fmt_persona(w: PersonaWeights | None) -> str:
    if w is None:
        return "Click **Reset**."
    bars = {"💰 Price": w.w_price, "⭐ Rating": w.w_rating,
            "🚚 Ship": w.w_ship, "🏷️ Brand": w.w_brand, "🎯 Sim": w.w_similarity}
    lines = ["### 👤 Persona", ""]
    for lab, val in bars.items():
        b = "█" * int(val*20) + "░" * (20 - int(val*20))
        lines.append(f"**{lab}:** `{b}` {val:.2f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core actions
# ---------------------------------------------------------------------------

def reset_episode(state, sel_env, sel_diff):
    env = _get_env()
    s = SessionState()
    s.env = env

    eid = sel_env.split("(")[-1].rstrip(")").strip() if "(" in sel_env else sel_env
    diff = int(sel_diff)
    s.env_id, s.difficulty = eid, diff

    obs = env.reset(env_id=eid, difficulty=diff)
    s.obs, s.turn, s.conversation = obs, obs.turn, list(obs.conversation)

    ep = env.get_episode_state()
    if ep:
        s.persona_weights = ep.persona_weights
        s.goal_params = env._extract_goal_params(ep.hidden_goal, ep.env_id)
        s.hidden_goal_md = _fmt_hidden_goal(ep)
        s.initial_msg_source = "llm" if eid in ("CART", "RETURN") else "template"
        s.user_sim_log.append(f"Initial: {'🟢 LLM' if s.initial_msg_source=='llm' else '🟡 template'}")

    return (
        s, _fmt_chat(s), _fmt_persona(s.persona_weights), _fmt_episode(s),
        "", s.hidden_goal_md, _fmt_verbalization(s),
        gr.update(interactive=True), gr.update(interactive=True),
        gr.update(interactive=True), gr.update(interactive=True),
    )


def execute_tool(state, tool_name, a1, a2, a3, a4, a5):
    if state is None or state.done:
        return state, [], "Reset first!", "", "", "", ""
    env = state.env or _get_env()

    args = {}
    tdef = TOOLS.get(tool_name, {})
    adefs = tdef.get("args", {})
    vals = [a1, a2, a3, a4, a5]
    for (aname, adef), v in zip(adefs.items(), vals):
        if v is None or v == "":
            if adef.get("required"):
                return (state, _fmt_chat(state),
                        f"❌ Required: `{aname}`", _fmt_episode(state),
                        "", _fmt_verbalization(state), state.hidden_goal_md)
            continue
        t = adef.get("type", "str")
        if t == "int":
            try:    args[aname] = int(v)
            except: args[aname] = adef.get("default", 10)
        elif t == "json":
            try:    args[aname] = json.loads(v) if v else None
            except: args[aname] = None
        elif t == "list":
            args[aname] = [x.strip() for x in v.split(",") if x.strip()]
        else:
            args[aname] = str(v)

    action = json.dumps({
        "assistant_message": f"[tool: {tool_name}]",
        "tool_calls": [{"name": tool_name, "args": args}],
    })
    obs, reward, done, info = env.step(action)
    state.obs, state.turn, state.done = obs, obs.turn, done
    state.reward, state.conversation = reward, list(obs.conversation)
    state.episode_info = info

    if not done and obs.conversation and obs.conversation[-1].get("role") == "user":
        state.user_sim_log.append(f"T{state.turn}: 🟢 LLM response")

    tout = ""
    if obs.tool_results:
        for tr in obs.tool_results:
            tout += _fmt_tool_result(tr) + "\n\n"
        state.tool_history.append(tool_name)

    rmd = _fmt_reward_banner(reward, info) if done else ""
    return (state, _fmt_chat(state), tout or "No output.",
            _fmt_episode(state), rmd, _fmt_verbalization(state), state.hidden_goal_md)


def submit_response(state, msg):
    if state is None or state.done:
        return state, [], "", "", "", "", ""
    if not msg.strip():
        return (state, _fmt_chat(state), "Write a message!",
                _fmt_episode(state), "", _fmt_verbalization(state), state.hidden_goal_md)

    env = state.env or _get_env()
    action = json.dumps({"assistant_message": msg, "tool_calls": []})
    obs, reward, done, info = env.step(action)
    state.obs, state.turn, state.done = obs, obs.turn, done
    state.reward, state.conversation = reward, list(obs.conversation)
    state.episode_info = info

    if not done and obs.conversation and obs.conversation[-1].get("role") == "user":
        state.user_sim_log.append(f"T{state.turn}: 🟢 LLM response")

    rmd = _fmt_reward_banner(reward, info) if done else ""
    return (state, _fmt_chat(state), "", _fmt_episode(state),
            rmd, _fmt_verbalization(state), state.hidden_goal_md)


def submit_answer(state, msg, pids, order_id, line_id):
    if state is None or state.done:
        return state, [], "", "", "Episode ended.", "", ""

    env = state.env or _get_env()
    ids = [x.strip() for x in pids.split(",") if x.strip()] if pids else []

    answer: dict = {"env": state.env_id, "done": True, "recommended_product_ids": ids}
    if state.env_id == "RETURN":
        if order_id and order_id.strip(): answer["selected_order_id"] = order_id.strip()
        if line_id and line_id.strip():   answer["selected_line_id"]  = line_id.strip()

    action = json.dumps({
        "assistant_message": msg or "Here is my answer.",
        "tool_calls": [], "answer": answer,
    })
    obs, reward, done, info = env.step(action)
    state.obs, state.turn, state.done = obs, obs.turn, done
    state.reward, state.conversation = reward, list(obs.conversation)
    state.episode_info = info

    return (state, _fmt_chat(state), "", _fmt_episode(state),
            _fmt_reward_banner(reward, info),
            _fmt_verbalization(state), state.hidden_goal_md)


def update_tool_args(tool_name):
    tdef = TOOLS.get(tool_name, {})
    adefs = tdef.get("args", {})
    names = list(adefs.keys())
    updates = []
    for i in range(5):
        if i < len(names):
            n = names[i]; d = adefs[n]
            req = " ✱" if d.get("required") else f" (={d.get('default','')})"
            updates.append(gr.update(label=f"{n}{req}", placeholder=d.get("description",""),
                                     value="", visible=True))
        else:
            updates.append(gr.update(label=f"arg{i+1}", value="", visible=False))

    desc = tdef.get("description", "")
    dl = [f"**{tool_name}** — {desc}"]
    if adefs:
        for an, ad in adefs.items():
            r = "✱" if ad.get("required") else f"opt={ad.get('default')}"
            dl.append(f"- `{an}` ({r}) — {ad.get('description','')}")
    else:
        dl.append("*No arguments — click Execute.*")
    return (*updates, gr.update(value="\n".join(dl)))


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

HEADER = """# 🛍️ EcomRLVE-GYM — Testing UI
**You = AI agent.** Simulated customer (Ollama LLM) asks for help.
Use tools → chat → submit answer → see reward.
"""

with gr.Blocks(title="EcomRLVE Testing") as demo:
    session_state = gr.State(value=None)
    gr.Markdown(HEADER)

    # ══════ REWARD BANNER (top, full width) ══════
    reward_display = gr.HTML(value="")

    with gr.Row():
        # ══════ LEFT: controls ══════
        with gr.Column(scale=1, min_width=290):
            with gr.Group():
                gr.Markdown("### ⚙️ Controls")
                env_sel = gr.Dropdown(
                    [f"{l} ({e})" for e, l in _ENV_LABELS.items()],
                    value=f"{_ENV_LABELS['CART']} (CART)", label="Environment")
                diff_sl = gr.Slider(0, 10, step=1, value=3, label="Difficulty")
                reset_btn = gr.Button("🔄 Reset Episode", variant="primary", size="lg")

            episode_md = gr.Markdown("")
            with gr.Accordion("👤 Persona", open=False):
                persona_md = gr.Markdown("Click Reset.")
            with gr.Accordion("🤖 Verbalization", open=True):
                verb_md = gr.Markdown("")

        # ══════ CENTRE: chat ══════
        with gr.Column(scale=2, min_width=450):
            chatbot = gr.Chatbot([], height=420)
            with gr.Group():
                asst_in = gr.Textbox(label="Your response", lines=2, interactive=False,
                                     placeholder="Type your message to the customer…")
                send_btn = gr.Button("💬 Send", interactive=False)

            with gr.Accordion("✅ Submit Final Answer", open=False):
                ans_pids = gr.Textbox(label="Product IDs", placeholder="B01…, B02…", lines=1)
                with gr.Row():
                    ans_oid = gr.Textbox(label="Order ID (RETURN)", placeholder="ord_001")
                    ans_lid = gr.Textbox(label="Line ID (RETURN)",  placeholder="ord_001_line_01")
                done_btn = gr.Button("✅ Submit Answer", variant="stop", interactive=False)

        # ══════ RIGHT: tools + goal ══════
        with gr.Column(scale=1, min_width=310):
            gr.Markdown("### 🔧 Tools")
            tool_sel = gr.Dropdown(list(TOOLS.keys()), value="catalog.search", label="Tool")
            tool_desc = gr.Markdown("")
            ta1 = gr.Textbox(label="arg1")
            ta2 = gr.Textbox(label="arg2", visible=True)
            ta3 = gr.Textbox(label="arg3", visible=True)
            ta4 = gr.Textbox(label="arg4", visible=False)
            ta5 = gr.Textbox(label="arg5", visible=False)
            exec_btn = gr.Button("⚡ Execute", variant="secondary")
            with gr.Accordion("📄 Output", open=True):
                tool_out = gr.Markdown("")
            with gr.Accordion("🎯 Hidden Goal", open=True):
                goal_md = gr.Markdown("")

    # ══════ Wiring ══════
    _out7 = [session_state, chatbot, tool_out, episode_md,
             reward_display, verb_md, goal_md]

    reset_btn.click(reset_episode, [session_state, env_sel, diff_sl],
                    [session_state, chatbot, persona_md, episode_md,
                     reward_display, goal_md, verb_md,
                     exec_btn, asst_in, send_btn, done_btn])

    tool_sel.change(update_tool_args, [tool_sel], [ta1, ta2, ta3, ta4, ta5, tool_desc])

    exec_btn.click(execute_tool, [session_state, tool_sel, ta1, ta2, ta3, ta4, ta5], _out7)

    send_btn.click(submit_response, [session_state, asst_in], _out7
                   ).then(lambda: "", outputs=[asst_in])

    done_btn.click(submit_answer,
                   [session_state, asst_in, ans_pids, ans_oid, ans_lid], _out7)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0",
                server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
                share=False, show_error=True,
                theme=gr.themes.Soft())
