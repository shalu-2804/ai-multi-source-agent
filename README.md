# AI-POWERED AGENT

A Retrieval-Augmented Generation (RAG) system for analyzing BPSS (Baseline Personnel Security Standard) screening data.

## Quick Start

```bash
# Activate virtual environment
source venv/bin/activate

# Run interactive chat
python main.py
```

## Requirements

- Python 3.9+
- Ollama with llama3.1:8b model
- Virtual environment (already set up)

## How to Use

Simply run `python main.py` and ask questions:

```
Which records violate policies?
Which candidates are high risk?
What compliance issues exist?
Tell me about CAND-101
```

Each answer includes citations from source documents.

## Architecture

- **Vector DB**: Chroma (12 documents indexed)
- **Embeddings**: Sentence Transformers
- **LLM**: Ollama (llama3.1:8b) - local, no API costs
- **Data**: 6 candidates with CSV tracking + Word documents

## Project Structure

```
src/
  ├── document_loader.py      # PDF/DOCX/CSV loading
  ├── vector_retriever.py     # Chroma vector database
  ├── structured_queryer.py   # CSV data queries
  ├── agent_tools.py          # Tool collection
  └── bpss_agent.py           # RAG agent with citations

config/
  └── settings.py             # Configuration constants

main.py                        # Entry point (interactive mode)
```

## Features

- Interactive Q&A with document citations
- Structured data querying (CSV-based)
- Policy violation detection
- Compliance issue analysis
- Zero API costs (local LLM)
- Clean, maintainable code

For detailed usage, see [USAGE.md](USAGE.md).

This system processes confidential screening data. Ensure:
- Proper access controls on API keys
- Secure storage of responses
- Audit logging of all queries
- Compliance with GDPR/local privacy laws

## Support

For issues or questions:
1. Check `Troubleshooting` section above
2. Review code comments in `src/` modules
3. Inspect tool call history: `agent.get_tool_call_history()`
4. Check vector DB stats: `vector_retriever.get_collection_stats()`
