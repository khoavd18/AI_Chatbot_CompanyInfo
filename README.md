# Nguyen Vo Dang Khoa Architects — Hybrid RAG Chatbot

> A domain-specific Retrieval-Augmented Generation (RAG) chatbot for an architecture & interior design company, built with **FastAPI + Streamlit + Qdrant + hybrid retrieval (dense + BM25-style sparse) + local LLM**.

---

## Overview

This project is a **local-first AI assistant** designed to answer questions about a company’s:

- company profile and contact information
- projects and project categories
- architecture types and interior styles
- news articles and hero content

Instead of relying on generic web search or a remote closed API, the system uses a **custom knowledge base**, domain-specific chunking, hybrid retrieval, reranking, and a local LLM generation pipeline.

The main goal is to build a chatbot that is:

- **more accurate on company-specific content** than a general-purpose LLM
- **structured like a production AI feature** rather than a single notebook/demo
- **easy to extend** with more data sources, better retrieval logic, and cleaner evaluation

---

## Why this project matters

From an engineering perspective, this project demonstrates end-to-end AI application development across:

- **data ingestion and normalization**
- **domain-aware chunking**
- **dense + sparse retrieval**
- **reranking**
- **API serving**
- **UI integration**
- **debugging and iterative quality improvement**

This is not just a “call an LLM” demo. The repo is structured around the same concerns that appear in real AI product work:

- data quality affects retrieval quality
- chunk design affects answer quality
- retrieval errors propagate to generation
- generation quality depends on both prompt design and context quality
- vector database lifecycle and indexing strategy matter in practice

---

## Key features

- **Domain-specific chunking** for each entity type:
  - `companyInfo`
  - `projects`
  - `projectCategories`
  - `news`
  - `newsCategories`
  - `architectureTypes`
  - `interiorStyles`
  - `heroSlides`
- **Hybrid retrieval** using:
  - dense embeddings with `intfloat/multilingual-e5-small`
  - sparse lexical scoring with a custom BM25-style sparse embedder
- **Cross-encoder reranking** for better top-k document selection
- **Qdrant** as vector database with hybrid vectors (dense + sparse)
- **FastAPI backend** for serving chatbot endpoints
- **Streamlit UI** for quick interaction and demo usage
- **Local LLM generation** using Hugging Face models (e.g. Qwen Instruct)
- **Structured pipeline** for chunking, embedding, and indexing

---

## System architecture

```text
Raw / processed company data
        ↓
Domain-aware chunking per entity type
        ↓
Dense embedding + sparse representation
        ↓
Qdrant hybrid index
        ↓
Dense retrieval + sparse/BM25 scoring
        ↓
Cross-encoder reranking
        ↓
Context builder
        ↓
Local LLM generation
        ↓
FastAPI response / Streamlit UI
```

---

## Project structure

```text
src/
├── api/
│   └── routes/
│       ├── app.py
│       ├── chat.py
│       └── health.py
├── config/
│   ├── logging.yaml
│   └── settings.yaml
├── core/
│   ├── logging_setup.py
│   ├── schema.py
│   ├── setting_loader.py
│   └── startup.py
├── llm/
│   ├── generator.py
│   └── prompt.py
├── rag/
│   ├── chunking/
│   │   ├── architectureType.py
│   │   ├── companyInfo.py
│   │   ├── heroSlides.py
│   │   ├── interiorStyles.py
│   │   ├── news.py
│   │   ├── newsCategories.py
│   │   ├── projectCategories.py
│   │   ├── projects.py
│   │   └── helpers/
│   ├── embedding/
│   │   ├── batch_embed_text.py
│   │   ├── embed_text.py
│   │   └── sparse_embeder.py
│   ├── retrieval/
│   │   ├── context_builder.py
│   │   ├── hybrid_retriever.py
│   │   ├── retriever.py
│   │   ├── scoring/
│   │   └── reranking/
│   └── vectorstore/
│       ├── hybrid_index.py
│       ├── index.py
│       ├── qdrant.py
│       └── upsert.py
├── pipeline.py
└── ui/
    └── chatbot.py
```

---

## Technical stack

### Backend
- **Python 3.10+**
- **FastAPI**
- **Uvicorn**

### Retrieval / ML
- **Transformers / Hugging Face**
- **Qwen/Qwen2.5-1.5B-Instruct** (local generation)
- **intfloat/multilingual-e5-small** (dense embeddings)
- **cross-encoder/ms-marco-MiniLM-L-6-v2** (reranker)
- custom **SparseEmbedder** + **BM25-style scoring**

### Vector DB
- **Qdrant**

### UI
- **Streamlit**

---

## Data model

The project uses a structured business dataset centered around an architecture/interior design company.

Main entity groups:

- `companyInfo`
- `projects`
- `projectCategories`
- `news`
- `newsCategories`
- `architectureTypes`
- `interiorStyles`
- `heroSlides`

A major design decision in this project is that **each entity type is chunked differently**, instead of using a generic “split every document by length” strategy.

### Example
A `project` is not treated as a single blob. It can be chunked into:

- `overview`
- `full_content`
- `context`
- `specs`
- `seo`
- `media`

This improves retrieval precision for questions like:

- “Which project is in Bình Tân?”
- “What is the area of project X?”
- “Show full information for project Y.”
- “Which projects match Japandi style?”

---

## Retrieval design

### 1. Dense retrieval
Dense embeddings are generated with `multilingual-e5-small`, which is suitable for Vietnamese and short semantic search queries.

### 2. Sparse retrieval
A custom sparse embedding / BM25-like scorer is built over the indexed corpus to preserve exact-keyword behavior.

### 3. Hybrid retrieval
The system combines:

- semantic relevance from dense embeddings
- lexical relevance from sparse/BM25 scoring

This is especially important for company/product/project names, addresses, and style names.

### 4. Reranking
A cross-encoder reranker is used to improve final document ordering before context is passed to the LLM.

### 5. Context building
Retrieved chunks are transformed into a bounded, structured context block before answer generation.

---

## Engineering decisions

### Why hybrid retrieval instead of dense-only?
Dense retrieval is good for semantic similarity, but company/project queries often depend on:

- exact names
- addresses
- phone numbers
- category labels
- style labels

Adding sparse/BM25 behavior improves robustness for these cases.

### Why custom chunking instead of naive text splitting?
Because the dataset is highly structured.

For example:

- `companyInfo` is better split into contact / overview / brand / stats
- `projects` benefit from content-aware chunks
- `news` benefit from overview + full_content + meta
- `styles` benefit from definition + description + SEO metadata

### Why local LLM?
A local model allows:

- offline / low-cost experimentation
- better control of prompting and inference
- a realistic AI engineering workflow for local development

---

## What I improved during development

This project was iteratively refined instead of being built as a one-shot demo.

Examples of improvements made during debugging:

- reduced noisy or weak chunks
- improved `projects` chunking to prioritize rich `content`
- improved `news` chunking to avoid overly long overview chunks
- added metadata consistency across chunkers
- added `heroSlides` into the indexing pipeline
- reduced retrieval noise by improving chunk type design
- improved vector index rebuild behavior to avoid duplicated points
- improved context construction to reduce LLM echoing raw metadata
- improved generation guardrails to reduce repetition and malformed output

---

## How to run

### 1. Create environment

```bash
python -m venv env
```

**Windows PowerShell**
```bash
.\env\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

If your project uses `pyproject.toml`, use your preferred package manager instead.

### 3. Start Qdrant

```bash
docker compose up -d
```

### 4. Build / rebuild the index

```bash
python -m src.pipeline
```

### 5. Start the API

```bash
uvicorn src.api.routes.app:app --reload
```

### 6. Start the UI

```bash
streamlit run src/ui/chatbot.py
```

---

## Example questions

- `Cho tôi thông tin công ty`
- `Hotline công ty là gì?`
- `Địa chỉ công ty ở đâu?`
- `Công ty có những phong cách nội thất nào?`
- `Phong cách Japandi là gì?`
- `Dự án nào ở Bình Tân?`
- `Cho tôi thông tin dự án Văn phòng sáng tạo d-one studio`
- `Có bài viết nào về xu hướng thiết kế không?`

---

## Example API endpoints

### Health check
```http
GET /health
```

### Chat endpoint
```http
POST /chat
Content-Type: application/json

{
  "query": "Cho tôi thông tin công ty"
}
```

---

## Current limitations

This project is already functional, but there are still areas for improvement:

- evaluation set and automated retrieval benchmarking are still limited
- generation quality depends heavily on prompt and context cleanliness
- local LLM output can still degrade on ambiguous or poorly specified questions
- query disambiguation for vague requests can be improved further
- data quality still matters a lot when synthetic/fallback content exists
- deployment is local/dev-oriented rather than production-hardened

---

## Next steps

Planned improvements that would make this project even stronger:

- add a formal evaluation suite for retrieval and answer quality
- add query rewriting / query classification
- add metadata-based filtering and better source balancing
- improve ambiguous project selection flow
- cache model loading and optimize generation latency further
- add screenshots / demo GIFs to improve portfolio presentation
- add CI checks and a cleaner public repo structure

---

## What this project shows about my engineering profile

This project reflects how I approach AI systems:

- I do not treat LLMs as magic black boxes
- I care about data quality, retrieval design, and system structure
- I can debug issues across the full stack:
  - data
  - chunking
  - vector indexing
  - retrieval
  - reranking
  - prompting
  - inference behavior
- I can build AI features in a way that is closer to product engineering than notebook experimentation

---

## Notes for recruiters / interviewers

If you are reviewing this repository for an AI Engineer / Applied AI / ML Engineer role, the most relevant technical sections are:

- `src/rag/chunking/` → domain-aware data modeling and chunk design
- `src/rag/retrieval/` → retrieval, BM25-style scoring, reranking, context building
- `src/rag/vectorstore/` → Qdrant integration and indexing
- `src/api/routes/chat.py` → serving layer and orchestration
- `src/llm/` → generation pipeline and answer control

This repo is best understood as an **engineering-focused RAG application**, not just a model demo.

---

## Repository hygiene recommendation

For a public GitHub version, do **not** push:

- local virtual environments
- Hugging Face caches
- vector database storage
- raw private company data
- `.env` files or secrets

Typical `.gitignore` should include things like:

```gitignore
env/
.venv/
__pycache__/
*.log
.env
qdrant_storage/
storage/
data/raw/
data/processed/
```

---

## Contact

If needed, this section can be customized with:

- your name
- LinkedIn
- GitHub
- portfolio
- email

For a portfolio-ready version, I recommend replacing this section with your real professional links.

---

## License

Add a license if you plan to make this repository public.

Suggested options:

- MIT
- Apache-2.0

---
