# Multi-Source RAG Agent

![CI](https://github.com/shalu-2804/ai-multi-source-agent/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A production-oriented RAG agent that answers questions across heterogeneous document collections — PDFs, Word documents, and structured CSV data — grounding every answer in cited source evidence.

Built to demonstrate the core engineering patterns behind enterprise AI agent deployment: multi-source retrieval, hybrid structured/unstructured querying, tool orchestration, and citation-backed responses.

---

## What it does

The agent ingests a corpus of mixed-format documents and exposes a conversational interface where every answer is grounded in retrieved evidence with explicit source citations. It handles two retrieval modes simultaneously:

- **Semantic retrieval** over unstructured documents (PDF, DOCX) via Chroma vector store + Sentence Transformers embeddings
- **Structured querying** over tabular CSV data with filter and aggregation logic

The agent selects the appropriate tool (or combines both) based on query intent, then synthesises a cited response using a local LLM (Ollama `llama3.1:8b`) — zero API cost, fully offline.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────┐
│  BPSSAgent  │  ← orchestration layer: decides which tools to call
└──────┬──────┘
       │
  ┌────┴────┐
  │         │
  ▼         ▼
AgentTools (tool registry)
  │         │
  ▼         ▼
VectorRetriever     StructuredQueryer
(ChromaDB +         (pandas over CSV:
 SentenceTransf.)    filter, aggregate,
                     candidate lookup)
  │
  ▼
DocumentLoader
(PDF → pypdf,
 DOCX → python-docx,
 CSV → pandas)
```

**Key design decisions:**

- **Chroma over FAISS** — chosen for persistent storage across sessions; avoids re-indexing on every run, which matters in production deployments where document corpora are stable but queries are continuous
- **Sentence Transformers locally** — `all-MiniLM-L6-v2` runs without an API key; embeddings are deterministic and reproducible across environments
- **Tool routing in the agent layer** — the agent decides whether to call the vector retriever, the structured queryer, or both, depending on whether the question is about document content, record data, or a combination
- **Citation injection** — every response includes source document names and chunk references; hallucinated answers that lack grounding are structurally penalised by the prompt design

---

## Project structure

```
ai-multi-source-agent/
├── src/
│   ├── document_loader.py      # PDF/DOCX/CSV ingestion and chunking
│   ├── vector_retriever.py     # Chroma vector store: index, query, stats
│   ├── structured_queryer.py   # Pandas-based CSV querying with filter logic
│   ├── agent_tools.py          # Tool registry exposed to the agent
│   └── bpss_agent.py           # Core agent: tool selection + cited response generation
├── config/
│   └── settings.py             # All paths and constants in one place
├── bpss_agentic_dataset/       # Sample document corpus (anonymised)
├── main.py                     # Entry point — interactive chat loop
├── requirements.txt
└── USAGE.md                    # Extended usage examples
```

---

## Quickstart

**Prerequisites:** Python 3.9+, [Ollama](https://ollama.ai) with `llama3.1:8b` pulled

```bash
# Clone
git clone https://github.com/shalu-2804/ai-multi-source-agent.git
cd ai-multi-source-agent

# Install dependencies
pip install -r requirements.txt

# Pull the local LLM (one-time)
ollama pull llama3.1:8b

# Run
python main.py
```

---

## Example interaction

```
You: Which candidates have outstanding document gaps?

Agent: Based on the screening tracker (tracker.csv) and document
inventory (document_inventory.csv), two candidates have incomplete
submissions:

- CAND-103: Missing employment reference letter [source: tracker.csv,
  row 12; document_inventory.csv, row 8]
- CAND-107: Right-to-work evidence not verified [source: tracker.csv,
  row 19]

Recommend escalating CAND-103 first given the 72-hour SLA breach.
[source: policies/screening_policy.pdf, page 4]
```

---

## Retrieval design

Documents are chunked at ingestion time with overlap to preserve context across chunk boundaries. Chunk size and overlap are configurable in `config/settings.py`. The vector store persists to disk so re-indexing is only triggered when the corpus changes (via `reload=True`).

For structured data, the queryer supports:
- Candidate lookup by ID
- Policy compliance filtering
- Document inventory status checks
- Cross-referencing structured records against unstructured evidence

---

## Debugging and observability

```python
# Inspect which tools were called for a given query
agent.get_tool_call_history()

# Check vector store statistics
vector_retriever.get_collection_stats()
# → {"document_count": 47, "collection_name": "bpss_docs"}
```

---

## Extending to new domains

The BPSS screening domain is the sample dataset — the agent architecture is domain-agnostic. To adapt to a new use case:

1. Replace documents in `bpss_agentic_dataset/` with your corpus
2. Update `config/settings.py` paths
3. Adjust the system prompt in `bpss_agent.py` for your domain
4. Run `python main.py` — the vector store rebuilds automatically on first run

---

## Tech stack

| Component | Library | Version |
|-----------|---------|---------|
| Vector store | chromadb | 0.4.24 |
| Embeddings | sentence-transformers | 2.5.1 |
| PDF parsing | pypdf | 4.0.1 |
| DOCX parsing | python-docx | 0.8.11 |
| Structured data | pandas | 2.2.0 |
| LLM (local) | Ollama llama3.1:8b | — |

---

