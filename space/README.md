---
title: EcomRLVE-GYM
emoji: 🛍️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "6.9.0"
app_file: app.py
pinned: false
license: mit
short_description: Interactive RL environment for e-commerce shopping agents
---

# 🛍️ EcomRLVE-GYM — Interactive Demo

An interactive Reinforcement Learning environment for training and evaluating e-commerce shopping agents.

## What is this?

**You play as the AI agent.** A simulated customer (powered by HF Inference API) asks for help finding products. You use tools like `catalog.search`, `catalog.get_product`, etc. to search a real **2M product catalog** indexed with FAISS HNSW + GTE-Small embeddings. Then you submit your product recommendation and get a reward score.

## Architecture

- **Catalog:** 2M real Amazon products (Amazebay dataset)
- **Embeddings:** `thenlper/gte-small` (384-dim, 33M params)
- **Index:** FAISS HNSW (3.4GB, ~10ms search)
- **User Simulator:** HF Inference Provider (Mistral-7B-Instruct) with persona-driven prompting
- **Reward:** `r_total = w_task × r_task + w_eff × r_eff + w_hall × r_hall`
  - `r_task = clip(0.55 × r_rank + 0.35 × r_constraints + 0.10 × r_oos, -1, 1)`

## Tools Available

| Tool | Description |
|------|-------------|
| `catalog.search` | Search products by query + filters |
| `catalog.get_product` | Get full product details |
| `catalog.rerank` | Re-rank candidates by relevance |
| `catalog.get_variants` | Get product variants |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_TOKEN` | HuggingFace API token (for LLM user simulator) | — |
| `LLM_MODEL` | Model ID for user simulator | `mistralai/Mistral-7B-Instruct-v0.3` |
| `FAISS_INDEX_DIR` | Path to pre-built FAISS index | `../data/faiss-index` |
| `CATALOG_PATH` | Path to product catalog | `../data/amazebay-2M` |
| `CATALOG_MAX_ITEMS` | Max products to load in memory | `5000` |
| `EMBEDDING_MODEL` | Sentence transformer model | `thenlper/gte-small` |
| `EMBEDDING_DEVICE` | Compute device | `cpu` |

## Running Locally

```bash
cd ecom-rlve/space
pip install -r requirements.txt
export HF_TOKEN="hf_..."
python app.py
```

Open http://localhost:7860

## References

- [EcomRLVE paper](https://arxiv.org/abs/xxxx.xxxxx)
- [OpenEnv framework](https://github.com/facebookresearch/openenv)
- [RLVE-Gym](https://huggingface.co/spaces/ZhiyuanZeng/RLVE_Gym)
