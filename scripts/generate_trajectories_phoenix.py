#!/usr/bin/env python3
"""Generate conversation trajectories for E_CART with Phoenix observability.

Same logic as generate_trajectories.py but uses:
  - OpenAI SDK against Ollama's /v1 endpoint (auto-instrumented by Phoenix)
  - Arize Phoenix for trace collection and visualization
  - Custom spans for env.step, tool execution, and reward

Usage:
    uv run python scripts/generate_trajectories_phoenix.py [--output FILE] [--model MODEL]
    # Then open http://localhost:6006 to view traces in Phoenix UI

Produces 1 trajectory per difficulty level (d=0..10), saved as JSON.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from typing import Any

import openai
import requests
from opentelemetry import trace

from openinference.semconv.trace import (
    OpenInferenceSpanKindValues,
    SpanAttributes,
)

from ecom_rlve.server.openenv import EcomRLVEEnv

# ---------------------------------------------------------------------------
# Phoenix + OpenTelemetry setup
# ---------------------------------------------------------------------------

def _phoenix_is_running(port: int = 6006) -> bool:
    """Check if a Phoenix server is already running on the given port."""
    try:
        resp = requests.get(f"http://127.0.0.1:{port}/healthz", timeout=2)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def setup_phoenix() -> trace.Tracer:
    """Launch Phoenix server (if not already running) and configure tracing.

    Returns a tracer for creating custom spans.
    """
    from openinference.instrumentation.openai import OpenAIInstrumentor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk import trace as trace_sdk
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    if _phoenix_is_running():
        print("Phoenix already running at http://localhost:6006 — skipping launch")
    else:
        import phoenix as px
        px.launch_app()

    # Configure OTEL to send spans to Phoenix
    endpoint = "http://127.0.0.1:6006/v1/traces"
    tracer_provider = trace_sdk.TracerProvider()
    tracer_provider.add_span_processor(
        SimpleSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
    )

    # Set as the GLOBAL tracer provider so custom spans (AGENT, CHAIN, TOOL)
    # are also exported to Phoenix — not just auto-instrumented OpenAI spans
    trace.set_tracer_provider(tracer_provider)

    # Auto-instrument OpenAI SDK (which we point at Ollama)
    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

    return trace.get_tracer("ecomrlve-trajectory-gen")


# ---------------------------------------------------------------------------
# Ollama client (via OpenAI SDK — auto-instrumented by Phoenix)
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3.5:2b"
TIMEOUT = 60

# OpenAI client pointing at Ollama's OpenAI-compatible endpoint
_client: openai.OpenAI | None = None


def get_openai_client() -> openai.OpenAI:
    """Lazy-init OpenAI client targeting Ollama's /v1 endpoint."""
    global _client
    if _client is None:
        _client = openai.OpenAI(
            base_url=f"{OLLAMA_BASE_URL}/v1",
            api_key="ollama",  # Ollama doesn't need a key, but the SDK requires one
        )
    return _client


def ollama_chat(
    messages: list[dict[str, str]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    seed: int = 42,
) -> str | None:
    """Call Ollama via OpenAI-compatible API. Auto-traced by Phoenix."""
    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )
        text = (response.choices[0].message.content or "").strip()
        # Strip <think> blocks (qwen3 reasoning tokens)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return text if text else None
    except Exception as exc:
        print(f"  [WARN] Ollama call failed: {exc}")
        return None


def _record_tool_spans(
    tracer: trace.Tracer,
    tool_calls: list[dict],
    tool_results: list[dict],
) -> None:
    """Create an OpenInference TOOL span for each tool call / result pair.

    These child spans appear in Phoenix as first-class tool entities, showing
    tool name, input args, and output results in the trace tree.
    """
    result_by_name: dict[str, Any] = {}
    for tr in tool_results:
        result_by_name[tr.get("name", "")] = tr.get("result")

    for tc in tool_calls:
        tool_name = tc.get("name", "unknown_tool")
        tool_args = tc.get("args", {})
        result = result_by_name.get(tool_name)

        with tracer.start_as_current_span(tool_name) as tool_span:
            tool_span.set_attribute(
                SpanAttributes.OPENINFERENCE_SPAN_KIND,
                OpenInferenceSpanKindValues.TOOL.value,
            )
            tool_span.set_attribute(SpanAttributes.TOOL_NAME, tool_name)
            tool_span.set_attribute(
                SpanAttributes.TOOL_PARAMETERS,
                json.dumps(tool_args, default=str),
            )
            tool_span.set_attribute(
                SpanAttributes.INPUT_VALUE,
                json.dumps(tool_args, default=str),
            )
            if result is not None:
                out = json.dumps(result, default=str)
                # OTEL attributes have a size limit; truncate large results
                tool_span.set_attribute(
                    SpanAttributes.OUTPUT_VALUE,
                    out[:4000] if len(out) > 4000 else out,
                )


# ---------------------------------------------------------------------------
# System prompt (identical to non-Phoenix version)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a shopping cart assistant. Output ONLY valid JSON. No other text.

IMPORTANT RULE: NEVER mention product_id, variant_id, or any internal ID (like "syn_...") in your assistant_message. Always refer to products by their name/title only. The IDs are internal and must only appear inside tool_calls args.

DISPLAY RULE — Product cards:
When you find products and want the customer to review or choose from them,
include an HTML product card block in your assistant_message. Wrap it with
marker comments so it can be parsed later. Use ONLY customer-relevant fields
(title, brand, price, rating, key attributes). NEVER include product_id or
variant_id in the HTML.

Format for showing products the customer should review:
<!--PRODUCT_CARDS_START-->
<div class="product-card">
  <h4>{title}</h4>
  <p class="brand">{brand}</p>
  <p class="price">${price}</p>
  <p class="rating">★ {rating}/5</p>
  <p class="attrs">{key attribute: value, ...}</p>
</div>
<!--PRODUCT_CARDS_END-->

DISPLAY RULE — Cart confirmation:
After you add items to the cart via cart.add, show a cart confirmation block
so the customer can see what was added:
<!--CART_UPDATE_START-->
<div class="cart-item">
  <span class="item-name">{title}</span>
  <span class="item-variant">{variant detail if any, e.g. "Color: teal" or "Latency: Battery Saving Mode"}</span>
  <span class="item-qty">Qty: {quantity}</span>
  <span class="item-price">${unit_price}</span>
</div>
<!--CART_UPDATE_END-->

You may include multiple <div> elements inside one marker block.

STEP 1: Call user.get_visit_history to find products.
STEP 2: Match customer request to products from results. Show matching products as product cards. Use cart.add to add them.
STEP 3: If customer wants a variant, call catalog.get_variants first to get the variant_id. Show available variants as product cards.
STEP 4: After adding items, show a cart confirmation block. Then submit final answer.

Tools:
- user.get_visit_history args: {} — Returns recently viewed products with product_id, title, brand
- catalog.search args: {"query": "text", "top_k": 10} — Search catalog
- catalog.get_variants args: {"product_id": "pid"} — Get variants for a product
- cart.add args: {"product_id": "pid", "variant_id": null, "quantity": 1} — Add to cart
- cart.view args: {} — View cart

Example turn showing matching products:
{"assistant_message": "I found the following product in your browsing history:\\n<!--PRODUCT_CARDS_START-->\\n<div class=\\"product-card\\">\\n  <h4>PureBlend Smartphone 9021</h4>\\n  <p class=\\"brand\\">PureBlend</p>\\n  <p class=\\"price\\">$4.97</p>\\n  <p class=\\"rating\\">★ 4.2/5</p>\\n  <p class=\\"attrs\\">Color: silver, Connector: USB-C</p>\\n</div>\\n<!--PRODUCT_CARDS_END-->\\nWould you like me to add this to your cart?", "tool_calls": [], "answer": null}

Example after cart.add:
{"assistant_message": "Added to your cart:\\n<!--CART_UPDATE_START-->\\n<div class=\\"cart-item\\">\\n  <span class=\\"item-name\\">PureBlend Smartphone 9021</span>\\n  <span class=\\"item-variant\\"></span>\\n  <span class=\\"item-qty\\">Qty: 1</span>\\n  <span class=\\"item-price\\">$4.97</span>\\n</div>\\n<!--CART_UPDATE_END-->", "tool_calls": [{"name": "cart.add", "args": {"product_id": "syn_000123", "variant_id": null, "quantity": 1}}], "answer": null}

Example final answer:
{"assistant_message": "All done! Your cart is ready.", "tool_calls": [], "answer": {"env": "CART", "recommended_product_ids": [], "done": true}}
"""


# ---------------------------------------------------------------------------
# JSON extraction from LLM output
# ---------------------------------------------------------------------------

def extract_json(text: str) -> dict | None:
    """Extract a JSON object from LLM output, handling markdown fences."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    patterns = [
        r"```json\s*\n?(.*?)\n?```",
        r"```\s*\n?(.*?)\n?```",
        r"(\{.*\})",
    ]
    for pat in patterns:
        match = re.search(pat, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
    return None


def make_fallback_action(message: str = "Let me help you with that.") -> dict:
    """Create a safe fallback action when LLM output is unparseable."""
    return {
        "assistant_message": message,
        "tool_calls": [{"name": "user.get_visit_history", "args": {}}],
    }


def make_done_action() -> dict:
    """Create a done action to end the episode."""
    return {
        "assistant_message": "I've added everything to your cart. You're all set!",
        "tool_calls": [],
        "answer": {"env": "CART", "recommended_product_ids": [], "done": True},
    }


# ---------------------------------------------------------------------------
# HTML card generation from tool results (fallback enrichment)
# ---------------------------------------------------------------------------

def _product_card_html(product: dict) -> str:
    """Generate a single product card HTML div from a product dict."""
    title = product.get("title", "Unknown")
    brand = product.get("brand") or product.get("key_attrs", {}).get("brand", "")
    price = product.get("price", "")
    rating = product.get("rating", "")
    attrs = product.get("key_attrs", {})
    skip_keys = {"brand", "store", "features"}
    attr_parts = [f"{k}: {v}" for k, v in attrs.items()
                  if k not in skip_keys and v is not None]
    attrs_str = ", ".join(attr_parts) if attr_parts else ""
    lines = [
        '<div class="product-card">',
        f"  <h4>{title}</h4>",
    ]
    if brand:
        lines.append(f'  <p class="brand">{brand}</p>')
    if price:
        lines.append(f'  <p class="price">${price}</p>')
    if rating:
        lines.append(f'  <p class="rating">★ {rating}/5</p>')
    if attrs_str:
        lines.append(f'  <p class="attrs">{attrs_str}</p>')
    lines.append("</div>")
    return "\n".join(lines)


def _variant_card_html(variant: dict, parent_title: str) -> str:
    """Generate a product card for a variant option."""
    color = variant.get("color", "")
    size = variant.get("size", "")
    extra_attrs = variant.get("attrs", {})
    price_delta = variant.get("price_delta", 0)
    stock = variant.get("stock_qty", 0)

    desc_parts = []
    if color:
        desc_parts.append(f"Color: {color}")
    if size:
        desc_parts.append(f"Size: {size}")
    for k, v in extra_attrs.items():
        if v is not None:
            desc_parts.append(f"{k.replace('_', ' ').title()}: {v}")
    desc_str = ", ".join(desc_parts) if desc_parts else "Standard"

    sign = "+" if price_delta >= 0 else ""
    lines = [
        '<div class="product-card variant">',
        f"  <h4>{parent_title}</h4>",
        f'  <p class="attrs">{desc_str}</p>',
        f'  <p class="price">{sign}${price_delta:.2f}</p>',
        f'  <p class="stock">{"In stock" if stock > 0 else "Out of stock"}</p>',
        "</div>",
    ]
    return "\n".join(lines)


def _cart_item_html(title: str, variant_desc: str, qty: int, unit_price: float) -> str:
    """Generate a cart confirmation item div."""
    lines = [
        '<div class="cart-item">',
        f'  <span class="item-name">{title}</span>',
        f'  <span class="item-variant">{variant_desc}</span>',
        f'  <span class="item-qty">Qty: {qty}</span>',
        f'  <span class="item-price">${unit_price:.2f}</span>',
        "</div>",
    ]
    return "\n".join(lines)


def enrich_message_with_cards(
    message: str,
    tool_calls: list[dict],
    tool_results: list[dict],
    product_lookup: dict[str, dict],
) -> tuple[str, dict[str, bool]]:
    """Inject HTML product/cart cards into assistant_message if not already present."""
    tool_names = [tc.get("name", "") for tc in tool_calls]
    result_map = {tr["name"]: tr.get("result") for tr in tool_results}

    has_product_markers = ("PRODUCT_CARDS_START" in message or "PRODUCT_CARDS_END" in message)
    has_cart_markers = ("CART_UPDATE_START" in message or "CART_UPDATE_END" in message)

    card_flags = {
        "llm_generated_product_cards": has_product_markers,
        "llm_generated_cart_cards": has_cart_markers,
        "enrichment_added_product_cards": False,
        "enrichment_added_cart_cards": False,
    }

    # 1. Show product cards for browsing history / search / variant results
    if not has_product_markers:
        products_to_show = []

        if "user.get_visit_history" in result_map:
            results = result_map["user.get_visit_history"]
            if isinstance(results, list):
                products_to_show = results

        if "catalog.search" in result_map:
            results = result_map["catalog.search"]
            if isinstance(results, list):
                products_to_show = results[:5]

        if "catalog.get_variants" in result_map:
            variants = result_map["catalog.get_variants"]
            if isinstance(variants, list) and variants:
                parent_pid = variants[0].get("product_id", "")
                parent_title = product_lookup.get(parent_pid, {}).get("title", "Product")
                variant_cards = "\n".join(
                    _variant_card_html(v, parent_title) for v in variants[:6]
                )
                block = f"\n<!--PRODUCT_CARDS_START-->\n{variant_cards}\n<!--PRODUCT_CARDS_END-->"
                message = message + block
                has_product_markers = True
                card_flags["enrichment_added_product_cards"] = True

        if products_to_show and "cart.add" not in tool_names and not has_product_markers:
            html_cards = "\n".join(
                _product_card_html(p) for p in products_to_show[:5]
            )
            block = f"\n<!--PRODUCT_CARDS_START-->\n{html_cards}\n<!--PRODUCT_CARDS_END-->"
            message = message + block
            card_flags["enrichment_added_product_cards"] = True

    # 2. Show cart confirmation when cart.add was called
    if "cart.add" in tool_names and not has_cart_markers:
        cart_items_html = []
        for tc in tool_calls:
            if tc.get("name") != "cart.add":
                continue
            args = tc.get("args", {})
            pid = args.get("product_id", "")
            vid = args.get("variant_id")
            qty = args.get("quantity", 1)

            prod = product_lookup.get(pid, {})
            title = prod.get("title", "Item")
            price = prod.get("price", 0)

            variant_desc = ""
            if vid:
                parts = vid.split("_v_", 1)
                if len(parts) == 2:
                    raw = parts[1]
                    m = re.match(r"([a-z_]+?)_([A-Z0-9].*)", raw)
                    if m:
                        attr_name = m.group(1).replace("_", " ").title()
                        attr_val = m.group(2).replace("_", " ")
                        variant_desc = f"{attr_name}: {attr_val}"
                    else:
                        variant_desc = raw.replace("_", " ").title()
                elif "_v" in vid and not vid.endswith("_v"):
                    variant_desc = ""

            cart_items_html.append(
                _cart_item_html(title, variant_desc, qty, price)
            )

        if cart_items_html:
            items_block = "\n".join(cart_items_html)
            block = f"\n<!--CART_UPDATE_START-->\n{items_block}\n<!--CART_UPDATE_END-->"
            message = message + block
            card_flags["enrichment_added_cart_cards"] = True

    return message, card_flags


# ---------------------------------------------------------------------------
# Trajectory recording (with Phoenix spans)
# ---------------------------------------------------------------------------

def run_trajectory(
    env: EcomRLVEEnv,
    difficulty: int,
    seed: int,
    model: str,
    tracer: trace.Tracer,
) -> dict[str, Any]:
    """Run a single CART episode and record the full trajectory.

    Each episode is wrapped in a Phoenix trace span. Sub-spans are created
    for each LLM call (auto-instrumented via OpenAI SDK), env.step, and
    reward computation.
    """
    from ecom_rlve.difficulty.mapping import map_difficulty

    with tracer.start_as_current_span(
        "cart_episode",
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.AGENT.value,
            "episode.difficulty": difficulty,
            "episode.seed": seed,
            "episode.model": model,
            "episode.env_id": "CART",
        },
    ) as episode_span:
        t0 = time.time()
        obs = env.reset(env_id="CART", difficulty=difficulty, seed=seed)

        T_max = map_difficulty(difficulty).T_max_val
        user_message = obs.conversation[0]["content"]
        episode_span.set_attribute("episode.T_max", T_max)
        episode_span.set_attribute("episode.initial_user_message", user_message[:500])

        trajectory: dict[str, Any] = {
            "env_id": "CART",
            "difficulty": difficulty,
            "seed": seed,
            "model": model,
            "T_max": T_max,
            "initial_user_message": user_message,
            "steps": [],
            "total_turns": 0,
            "final_reward": None,
            "is_correct": None,
            "reward_breakdown": None,
            "termination_reason": None,
            "generation_time_s": None,
        }

        # Build the LLM conversation history
        llm_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        max_turns = T_max + 2
        turn = 0
        done = False
        consecutive_failures = 0
        has_added_items = False
        reward = 0.0
        info: dict = {}

        # Force first turn: always call user.get_visit_history
        turn = 1
        first_action = {
            "assistant_message": "Let me check your browsing history to find those items.",
            "tool_calls": [{"name": "user.get_visit_history", "args": {}}],
        }

        with tracer.start_as_current_span(
            "env_step",
            attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
                "step.turn": 1,
                "step.tool_calls": "user.get_visit_history",
                "step.is_forced_first_turn": True,
            },
        ):
            obs, reward, done, info = env.step(json.dumps(first_action))
            # Record individual TOOL spans so Phoenix shows them
            _record_tool_spans(
                tracer,
                first_action["tool_calls"],
                obs.tool_results,
            )

        safe_info = {k: v for k, v in info.items() if k != "state"} if info else {}
        first_step: dict[str, Any] = {
            "turn": 1,
            "assistant_message": first_action["assistant_message"],
            "user_response": None,
            "_meta": {
                "llm_input_messages": 2,
                "raw_llm_output": json.dumps(first_action),
                "parsed_action": first_action,
                "parse_success": True,
                "tool_calls": first_action["tool_calls"],
                "tool_results": obs.tool_results,
                "submitted_answer": None,
                "reward": reward,
                "done": done,
                "info": safe_info,
            },
        }

        # Build product_lookup from visit history for enrichment
        product_lookup: dict[str, dict] = {}
        for tr in obs.tool_results:
            if tr.get("name") == "user.get_visit_history":
                for p in (tr.get("result") or []):
                    if isinstance(p, dict) and "product_id" in p:
                        product_lookup[p["product_id"]] = p

        trajectory["steps"].append(first_step)

        # Build LLM messages with the visit history results
        llm_messages.append({"role": "assistant", "content": json.dumps(first_action)})
        visit_results = json.dumps(obs.tool_results, indent=2, default=str)
        llm_messages.append({
            "role": "user",
            "content": (
                f"Tool results:\n{visit_results}\n\n"
                "Now match the customer's request to products above. "
                "Use cart.add to add matching items. If they want a variant, "
                "call catalog.get_variants first."
            ),
        })

        while not done and turn < max_turns:
            turn += 1
            step_record: dict[str, Any] = {
                "turn": turn,
                "assistant_message": None,
                "user_response": None,
                "_meta": {
                    "llm_input_messages": len(llm_messages),
                    "raw_llm_output": None,
                    "parsed_action": None,
                    "parse_success": False,
                    "tool_calls": [],
                    "tool_results": [],
                    "submitted_answer": None,
                    "reward": None,
                    "done": False,
                    "info": None,
                },
            }

            # Call LLM — auto-traced by Phoenix via OpenAI instrumentation
            msgs_to_send = list(llm_messages)
            if turn >= T_max - 1 and not has_added_items:
                msgs_to_send.append({
                    "role": "user",
                    "content": (
                        "URGENT: This is your last turn. You must add items to cart and submit now. "
                        "Call cart.add for the best matching product_id, then submit your answer."
                    ),
                })
            elif turn >= T_max - 1 and has_added_items:
                msgs_to_send.append({
                    "role": "user",
                    "content": (
                        "URGENT: This is your last turn. Submit your final answer now: "
                        '{"assistant_message": "Done!", "tool_calls": [], '
                        '"answer": {"env": "CART", "recommended_product_ids": [], "done": true}}'
                    ),
                })

            # The OpenAI SDK call is auto-instrumented — Phoenix captures
            # input/output messages, tokens, latency automatically
            raw = ollama_chat(
                msgs_to_send,
                model=model,
                seed=seed + turn,
                temperature=0.7,
                max_tokens=1024,
            )
            step_record["_meta"]["raw_llm_output"] = raw

            if raw is None:
                action_dict = make_fallback_action()
                step_record["_meta"]["parsed_action"] = action_dict
                consecutive_failures += 1
            else:
                action_dict = extract_json(raw)
                if action_dict is None:
                    print(f"  [d={difficulty} t={turn}] JSON parse failed, using fallback")
                    action_dict = make_fallback_action()
                    consecutive_failures += 1
                else:
                    step_record["_meta"]["parse_success"] = True
                    consecutive_failures = 0

            step_record["_meta"]["parsed_action"] = action_dict

            if consecutive_failures >= 3:
                print(f"  [d={difficulty}] 3 consecutive failures, submitting done")
                action_dict = make_done_action()
                step_record["_meta"]["parsed_action"] = action_dict

            # Ensure required fields
            if "assistant_message" not in action_dict or not action_dict["assistant_message"]:
                action_dict["assistant_message"] = "Processing your request."
            if "tool_calls" not in action_dict:
                action_dict["tool_calls"] = []

            # Scrub product IDs from assistant_message
            msg = action_dict["assistant_message"]
            msg = re.sub(r"\bsyn_\d+\S*", "", msg)
            msg = re.sub(r"\s{2,}", " ", msg).strip()
            msg = re.sub(r"\(\s*\)", "", msg)
            msg = re.sub(r"\s+([,.])", r"\1", msg)
            # Sanitize malformed card markers
            for tag in ("PRODUCT_CARDS", "CART_UPDATE"):
                start_tag = f"<!--{tag}_START-->"
                end_tag = f"<!--{tag}_END-->"
                n_start = msg.count(start_tag)
                n_end = msg.count(end_tag)
                if n_start != n_end:
                    msg = msg.replace(start_tag, "").replace(end_tag, "")
                    msg = re.sub(r"\s{2,}", " ", msg).strip()
            action_dict["assistant_message"] = msg

            step_record["assistant_message"] = msg
            step_record["_meta"]["tool_calls"] = action_dict.get("tool_calls", [])
            step_record["_meta"]["submitted_answer"] = action_dict.get("answer")

            for tc in action_dict.get("tool_calls", []):
                if tc.get("name") == "cart.add":
                    has_added_items = True

            # Send to environment — wrapped in a CHAIN span with TOOL children
            tool_names_str = ", ".join(
                tc.get("name", "?") for tc in action_dict.get("tool_calls", [])
            ) or "none"
            with tracer.start_as_current_span(
                "env_step",
                attributes={
                    SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
                    "step.turn": turn,
                    "step.tool_calls": tool_names_str,
                    "step.has_answer": bool(action_dict.get("answer")),
                    "step.parse_success": step_record["_meta"]["parse_success"],
                },
            ) as step_span:
                action_json = json.dumps(action_dict)
                try:
                    obs, reward, done, info = env.step(action_json)
                except Exception as exc:
                    print(f"  [d={difficulty} t={turn}] env.step error: {exc}")
                    action_dict = make_done_action()
                    action_json = json.dumps(action_dict)
                    obs, reward, done, info = env.step(action_json)

                # Record individual TOOL spans so Phoenix shows them
                _record_tool_spans(
                    tracer,
                    action_dict.get("tool_calls", []),
                    obs.tool_results,
                )

                step_span.set_attribute("step.reward", reward)
                step_span.set_attribute("step.done", done)
                if done and info:
                    rb = info.get("reward_breakdown", {})
                    step_span.set_attribute("step.is_correct", rb.get("is_correct", False))
                    step_span.set_attribute("step.r_task", rb.get("r_task", 0.0))

                # Update product_lookup from new tool results (inside span)
                for tr in obs.tool_results:
                    name = tr.get("name", "")
                    result = tr.get("result")
                    if name in ("user.get_visit_history", "catalog.search") and isinstance(result, list):
                        for p in result:
                            if isinstance(p, dict) and "product_id" in p:
                                product_lookup[p["product_id"]] = p

                # Enrich assistant_message with HTML product/cart cards
                enriched_msg, card_flags = enrich_message_with_cards(
                    message=step_record["assistant_message"],
                    tool_calls=action_dict.get("tool_calls", []),
                    tool_results=obs.tool_results,
                    product_lookup=product_lookup,
                )
                step_record["assistant_message"] = enriched_msg
                action_dict["assistant_message"] = enriched_msg
                step_record["_meta"]["card_flags"] = card_flags

                # Record enriched message + card flags on the Phoenix span
                step_span.set_attribute(
                    SpanAttributes.OUTPUT_VALUE,
                    enriched_msg[:8000] if len(enriched_msg) > 8000 else enriched_msg,
                )
                if any(card_flags.values()):
                    step_span.set_attribute("step.has_product_cards", card_flags.get("llm_generated_product_cards", False) or card_flags.get("enrichment_added_product_cards", False))
                    step_span.set_attribute("step.has_cart_cards", card_flags.get("llm_generated_cart_cards", False) or card_flags.get("enrichment_added_cart_cards", False))
                    step_span.set_attribute("step.card_flags", json.dumps(card_flags))

            step_record["_meta"]["reward"] = reward
            step_record["_meta"]["done"] = done
            step_record["_meta"]["tool_results"] = obs.tool_results
            safe_info = {k: v for k, v in info.items() if k != "state"} if info else {}
            step_record["_meta"]["info"] = safe_info

            # If there's a user response in conversation (user sim reply)
            if len(obs.conversation) > 0:
                last_msg = obs.conversation[-1]
                if last_msg.get("role") == "user":
                    step_record["user_response"] = last_msg["content"]

            trajectory["steps"].append(step_record)

            # Update LLM conversation for next turn
            llm_messages.append({"role": "assistant", "content": raw or json.dumps(action_dict)})

            feedback_parts = []
            if obs.tool_results:
                trimmed_results = []
                for tr in obs.tool_results:
                    result = tr.get("result")
                    if isinstance(result, list) and len(result) > 5:
                        result = result[:5]
                    trimmed_results.append({
                        "name": tr["name"],
                        "args": tr.get("args", {}),
                        "result": result,
                    })
                feedback_parts.append("Tool results:\n" + json.dumps(trimmed_results, indent=2, default=str))

                tool_names = [tr["name"] for tr in obs.tool_results]
                if "user.get_visit_history" in tool_names and turn == 1:
                    feedback_parts.append(
                        "IMPORTANT: Now match the customer's request to these products. "
                        "If they want a variant, call catalog.get_variants first, then cart.add. "
                        "If no variant needed, call cart.add directly with the matching product_id."
                    )

            if step_record["user_response"]:
                feedback_parts.append("Customer: " + step_record["user_response"])

            if feedback_parts and not done:
                llm_messages.append({"role": "user", "content": "\n\n".join(feedback_parts)})

            if done:
                break

        # Fill trajectory summary
        trajectory["total_turns"] = turn
        trajectory["final_reward"] = reward if done else None
        if done and info:
            safe_info = {k: v for k, v in info.items() if k != "state"}
            trajectory["reward_breakdown"] = safe_info.get("reward_breakdown")
            trajectory["is_correct"] = safe_info.get("reward_breakdown", {}).get("is_correct")
            trajectory["termination_reason"] = safe_info.get("termination_reason", "agent_done")

        trajectory["generation_time_s"] = round(time.time() - t0, 2)

        # Annotate steps with user_act
        user_acts = (info or {}).get("user_act_history", [])
        act_idx = 0
        for step in trajectory["steps"]:
            if step.get("user_response") is not None and act_idx < len(user_acts):
                step["user_act"] = user_acts[act_idx]
                act_idx += 1
            else:
                step["user_act"] = None

        # Build conversation view
        conversation_view = []
        conversation_view.append({"role": "user", "content": user_message})
        for step in trajectory["steps"]:
            if step["assistant_message"]:
                conversation_view.append({"role": "assistant", "content": step["assistant_message"]})
            if step.get("user_response"):
                conversation_view.append({"role": "user", "content": step["user_response"]})
        trajectory["conversation"] = conversation_view

        # Set final attributes on the episode span
        episode_span.set_attribute("episode.total_turns", turn)
        episode_span.set_attribute("episode.final_reward", trajectory["final_reward"] or 0.0)
        episode_span.set_attribute("episode.is_correct", trajectory.get("is_correct", False))
        episode_span.set_attribute(
            "episode.termination_reason",
            trajectory.get("termination_reason", "unknown"),
        )
        episode_span.set_attribute(
            "episode.generation_time_s",
            trajectory["generation_time_s"],
        )

        return trajectory


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate CART trajectories with Ollama agent + Phoenix observability"
    )
    parser.add_argument("--output", default="data/cart_trajectories_phoenix.json", help="Output JSON file")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--min-d", type=int, default=0, help="Min difficulty")
    parser.add_argument("--max-d", type=int, default=10, help="Max difficulty")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    parser.add_argument("--catalog-size", type=int, default=1000, help="Synthetic catalog size")
    parser.add_argument("--no-phoenix", action="store_true", help="Disable Phoenix (run without tracing)")
    args = parser.parse_args()

    # Setup Phoenix tracing
    if args.no_phoenix:
        tracer = trace.get_tracer("ecomrlve-trajectory-gen")
        print("Phoenix disabled — running without tracing")
    else:
        tracer = setup_phoenix()
        print("Phoenix UI available at http://localhost:6006")

    # Check Ollama
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        if not any(args.model in m for m in models):
            print(f"ERROR: Model '{args.model}' not found. Available: {models}")
            return
        print(f"Ollama ready, model: {args.model}")
    except requests.RequestException:
        print("ERROR: Ollama not available at", OLLAMA_BASE_URL)
        return

    # Build environment
    env = EcomRLVEEnv(
        collection="C4",
        seed=args.seed,
        config={
            "n_synthetic_products": args.catalog_size,
            "disclose_env_id": True,
            "disclose_difficulty": True,
        },
    )
    print(f"Environment ready (catalog: {args.catalog_size} products)")

    # Generate trajectories
    results = {
        "metadata": {
            "env_id": "CART",
            "model": args.model,
            "min_difficulty": args.min_d,
            "max_difficulty": args.max_d,
            "base_seed": args.seed,
            "catalog_size": args.catalog_size,
            "phoenix_enabled": not args.no_phoenix,
        },
        "trajectories": [],
    }

    total_t0 = time.time()

    for d in range(args.min_d, args.max_d + 1):
        print(f"\n{'='*60}")
        print(f"  Difficulty {d}")
        print(f"{'='*60}")

        traj_seed = args.seed * 10000 + d * 100
        traj = run_trajectory(env, difficulty=d, seed=traj_seed, model=args.model, tracer=tracer)
        results["trajectories"].append(traj)

        # Summary
        n_steps = len(traj["steps"])
        n_tools = sum(len(s["_meta"]["tool_calls"]) for s in traj["steps"])
        n_parse_ok = sum(1 for s in traj["steps"] if s["_meta"]["parse_success"])
        traj_reward = traj["final_reward"]
        correct = traj["is_correct"]
        gen_time = traj["generation_time_s"]

        print(f"  Steps: {n_steps}, Tool calls: {n_tools}, Parse OK: {n_parse_ok}/{n_steps}")
        print(f"  Reward: {traj_reward}, IsCorrect: {correct}")
        print(f"  Time: {gen_time}s")

        for s in traj["steps"]:
            tc_names = [tc["name"] for tc in s["_meta"]["tool_calls"]]
            answer = " [ANSWER]" if s["_meta"]["submitted_answer"] else ""
            print(f"    t={s['turn']}: {', '.join(tc_names) or 'no tools'}{answer} → r={s['_meta']['reward']}")

    total_time = round(time.time() - total_t0, 1)
    results["metadata"]["total_generation_time_s"] = total_time

    # Save
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n{'='*60}")
    print(f"  Saved {len(results['trajectories'])} trajectories to {args.output}")
    print(f"  Total time: {total_time}s")
    if not args.no_phoenix:
        print(f"  View traces at http://localhost:6006")
    print(f"{'='*60}")

    # Keep the process alive so Phoenix UI stays accessible
    if not args.no_phoenix:
        print("\nPhoenix UI running at http://localhost:6006 — press Ctrl+C to stop")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down.")


if __name__ == "__main__":
    main()
