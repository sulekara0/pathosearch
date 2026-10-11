# PathoSearch

Multimodal histopathology image search over Quilt-1M: text→image and image→image retrieval with CLIP and QuiltNet encoders, FAISS indexes, a domain gate and caption-based reranking. Streamlit demo.

> **For research and educational use only. Not a diagnostic tool.**

## Status

Work in progress. This repository currently holds the **v0 prototype** (tag `v0-prototype`). The plan for turning it into a measured, tested and deployed system is in [`docs/PLAN.md`](docs/PLAN.md).

## What's in v0

- Streamlit app (`app.py`) with three modes: single model, CLIP vs QuiltNet comparison, image→image
- Embedding and FAISS index scripts (`scripts/`)
- Domain gate that rejects out-of-domain queries, and a caption-based reranker
- Query generation and comparison reports (`reports/`)

## Roadmap

- [ ] Packaging, tests, linting and CI
- [ ] Reproducible evaluation with an independent golden set (Recall@k, MRR, nDCG)
- [ ] LoRA fine-tuning of CLIP, compared against QuiltNet
- [ ] Hybrid search (dense + BM25), cross-encoder reranking, Qdrant benchmark
- [ ] FastAPI + PostgreSQL service
- [ ] MCP server and a citation-grounded agent with guardrails
- [ ] Docker, deployment and live demo

## Data

Quilt-1M images and captions are **not** included in this repository. Check the dataset's own page for access and license terms. Embeddings and FAISS indexes are generated locally with the scripts in `scripts/`.

## Setup

Installation and usage instructions will be added after the packaging step.
