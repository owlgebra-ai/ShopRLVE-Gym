#!/usr/bin/env python3
"""End-to-end flow test: pre-built FAISS HNSW index + gte-small + OpenEnv.

Tests:
  1. Load FAISS HNSW index (2M vectors) from disk
  2. Load gte-small embedding model
  3. Raw search: encode query → FAISS → top-K product IDs + scores
  4. Load a small catalog subset → wire into OpenEnv with pre-built index
  5. Run reset() → step() for a Product Discovery episode
"""

import sys
import time
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("test_flow")

# ---------------------------------------------------------------------------
# Step 1: Load FAISS index
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 1: Loading FAISS HNSW index (2M vectors)")
print("=" * 70)

t0 = time.time()

sys.path.insert(0, "src")
from ecom_rlve.data.index import VectorIndex

vi = VectorIndex(dim=384)
vi.load_from_dir("data/faiss-index")

t_index = time.time() - t0
print(f"  ✅ Loaded {len(vi):,} vectors in {t_index:.1f}s")

# ---------------------------------------------------------------------------
# Step 2: Load gte-small
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: Loading gte-small embedding model")
print("=" * 70)

t0 = time.time()

from ecom_rlve.data.embeddings import EmbeddingEngine
engine = EmbeddingEngine(model_name="thenlper/gte-small", device="cuda:0", debug_mode=False)

# Force model load
_ = engine.encode_query("warmup")
t_model = time.time() - t0
print(f"  ✅ gte-small loaded in {t_model:.1f}s")

# ---------------------------------------------------------------------------
# Step 3: Raw FAISS search (no catalog needed)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: Raw FAISS search (query → encode → search → IDs)")
print("=" * 70)

test_queries = [
    "wireless bluetooth headphones with noise cancelling",
    "organic face moisturizer for dry skin",
    "kids dinosaur toys educational",
    "gaming mechanical keyboard rgb backlit",
    "stainless steel water bottle insulated",
]

for q in test_queries:
    t0 = time.time()
    q_emb = engine.encode_query(q)
    results = vi.search(q_emb, top_k=5)
    elapsed = (time.time() - t0) * 1000

    print(f"\n  Query: '{q}' ({elapsed:.1f}ms)")
    for rank, (pid, score) in enumerate(results, 1):
        print(f"    [{rank}] {pid}  score={score:.4f}")

# ---------------------------------------------------------------------------
# Step 4: Load small catalog subset + OpenEnv
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4: Loading catalog subset + creating OpenEnv")
print("=" * 70)

t0 = time.time()

# Get the IDs returned by FAISS so we can load those specific products
# First, do a broader search to collect some real IDs
sample_queries = [
    "wireless headphones", "laptop computer", "running shoes",
    "kitchen knife set", "yoga mat", "face cream", "toy for kids",
    "phone case", "gaming keyboard", "water bottle",
]
faiss_ids: set[str] = set()
for q in sample_queries:
    q_emb = engine.encode_query(q)
    results = vi.search(q_emb, top_k=100)
    faiss_ids.update(pid for pid, _ in results)
print(f"  Collected {len(faiss_ids)} unique IDs from sample FAISS queries")

# Load products from local dataset, filtering to IDs found by FAISS
from datasets import load_from_disk

ds = load_from_disk("data/amazebay-2M")
print(f"  Dataset loaded: {len(ds)} rows")

# Build a lookup of parent_asin -> row index for the IDs we need
# First 5000 rows to also have some extra products for the env
from ecom_rlve.data.catalog_loader import load_catalog, _map_hf_row_to_product

products = []
columns = set(ds.column_names)
n_loaded = 0
n_faiss_match = 0

# Load products that match FAISS results + some extras
for i in range(len(ds)):
    row = ds[i]
    asin = str(row.get("parent_asin", ""))
    if asin in faiss_ids or n_loaded < 3000:
        try:
            product = _map_hf_row_to_product(row, columns)
            products.append(product)
            n_loaded += 1
            if asin in faiss_ids:
                n_faiss_match += 1
        except Exception:
            continue
    if n_faiss_match >= len(faiss_ids) and n_loaded >= 3000:
        break
    if n_loaded >= 20000:  # safety cap
        break

t_cat = time.time() - t0
print(f"  ✅ Loaded {len(products)} products ({n_faiss_match} matching FAISS results) in {t_cat:.1f}s")

# Quick stats
cats = set(p.cat for p in products)
brands = set(p.brand for p in products)
print(f"  Categories: {len(cats)}, Brands: {len(brands)}")

# Create OpenEnv with pre-built FAISS index
print("\n  Creating EcomRLVEEnv with pre-built FAISS index...")
t0 = time.time()

from ecom_rlve.server.openenv import EcomRLVEEnv

env = EcomRLVEEnv(
    collection="C1",  # PD only
    catalog=(products, []),  # products, no variants
    config={
        "faiss_index_path": "data/faiss-index",
        "embedding_model": "thenlper/gte-small",
        "embedding_debug": False,
        "embedding_device": "cuda:0",
    },
    seed=42,
)
t_env = time.time() - t0
print(f"  ✅ EcomRLVEEnv created in {t_env:.1f}s")

# ---------------------------------------------------------------------------
# Step 5: Run reset() → step() for a PD episode
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 5: Running PD episode (reset → step)")
print("=" * 70)

t0 = time.time()
obs = env.reset(env_id="PD", difficulty=5)
t_reset = time.time() - t0

print(f"\n  reset() in {t_reset*1000:.0f}ms")
print(f"  Turn:       {obs.turn}")
print(f"  Env:        {obs.env_id}")
print(f"  Difficulty: {obs.difficulty}")

# Get first user message from conversation
user_msg = ""
for msg in obs.conversation:
    if msg.get("role") == "user":
        user_msg = msg.get("content", "")
        break
print(f"  User msg:   {user_msg[:200]}...")

# Step 1: Agent does a catalog.search
# Extract a reasonable search query from user message
search_query = user_msg[:100] if user_msg else "headphones"

search_action = json.dumps({
    "assistant_message": "Let me search for that for you.",
    "tool_calls": [
        {
            "name": "catalog.search",
            "args": {
                "query": search_query,
                "top_k": 10,
            },
        }
    ],
})

print(f"\n  Agent action: catalog.search(query='{search_query[:60]}...', top_k=10)")
t0 = time.time()
obs2, reward, done, info = env.step(search_action)
t_step = time.time() - t0

print(f"  step() in {t_step*1000:.0f}ms")
print(f"  Turn:    {obs2.turn}")
print(f"  Reward:  {reward}")
print(f"  Done:    {done}")
print(f"  Tool results count: {len(obs2.tool_results)}")

# Show tool results (list of dicts)
found_pid = None
if obs2.tool_results:
    for tr in obs2.tool_results:
        tool_name = tr.get("tool_name", "?") if isinstance(tr, dict) else getattr(tr, "tool_name", "?")
        status = tr.get("status", "?") if isinstance(tr, dict) else getattr(tr, "status", "?")
        result = tr.get("result", None) if isinstance(tr, dict) else getattr(tr, "result", None)
        print(f"\n  Tool: {tool_name} (status={status})")
        if status == "success" and isinstance(result, list):
            print(f"  Returned {len(result)} products:")
            for i, card in enumerate(result[:5]):
                if isinstance(card, dict):
                    print(f"    [{i+1}] {card.get('title', '?')[:70]}")
                    print(f"         ${card.get('price', '?')} | ★{card.get('rating', '?')} | {card.get('ship_days', '?')}d shipping")
                    if i == 0:
                        found_pid = card.get("product_id")
        elif isinstance(result, str):
            print(f"  Result: {result[:300]}")
        elif status != "success":
            print(f"  Error: {result}")
else:
    print("  (no tool results in observation)")
    # Check conversation for tool outputs
    for msg in obs2.conversation:
        if msg.get("role") == "tool":
            print(f"  [conversation tool msg]: {str(msg.get('content', ''))[:200]}")

# Step 2: Agent gives a final answer
print("\n  Agent: giving final answer...")
if found_pid:
    answer_action = json.dumps({
        "assistant_message": f"Based on your needs, I recommend product {found_pid}.",
        "tool_calls": [],
        "answer": {
            "done": True,
            "recommended_product_ids": [found_pid],
        },
    })
else:
    answer_action = json.dumps({
        "assistant_message": "I couldn't find matching products.",
        "tool_calls": [],
        "answer": {"done": True},
    })

t0 = time.time()
obs3, reward, done, info = env.step(answer_action)
t_ans = time.time() - t0

print(f"\n  Final step() in {t_ans*1000:.0f}ms")
print(f"  Done:   {done}")
print(f"  Reward: {reward}")

if info:
    rb = info.get("reward_breakdown")
    if rb:
        print(f"\n  Reward breakdown:")
        for k, v in rb.items():
            if isinstance(v, (int, float)):
                print(f"    {k}: {v:.4f}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  FAISS index:   {len(vi):,} vectors, loaded in {t_index:.1f}s")
print(f"  gte-small:     loaded in {t_model:.1f}s")
print(f"  Catalog:       {len(products)} products")
print(f"  OpenEnv init:  {t_env:.1f}s")
print(f"  Episode:       reset={t_reset*1000:.0f}ms, search_step={t_step*1000:.0f}ms, answer_step={t_ans*1000:.0f}ms")
print(f"  Final reward:  {reward}")
print(f"  ✅ End-to-end flow test complete!")
