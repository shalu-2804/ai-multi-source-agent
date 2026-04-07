# BPSS RAG System - Usage Guide

## Quick Start

```bash
python main.py
```

This launches an interactive chat where you can ask questions about BPSS screening data.

## Example Questions

### Policy & Compliance
- "Which records violate policies?"
- "What compliance issues exist?"
- "Which candidates are high risk?"

### Candidate-Specific
- "What is CAND-101's status?"
- "What documents are missing for CAND-105?"
- "Tell me about CAND-102"

### General Queries
- "Which candidates have incomplete checks?"
- "What are the screening procedures?"
- "Show me all candidates"

## How It Works

**RAG Architecture:**
1. **Retrieval** - Chroma vector database searches for relevant documents
2. **Augmentation** - Retrieved content provides context
3. **Generation** - Ollama LLM generates answers based on context

**Smart Question Routing:**
- Policy violations → Queries structured data (CSV)
- Compliance issues → Analyzes candidate risk levels
- Specific candidates → Retrieves candidate details
- General questions → Semantic search with LLM

## Output Format

Each answer includes:
- **Direct answer** - Specific findings or data
- **Sources** - Which documents/files were consulted

Example:
```
**Policy Violations Found:**
- CAND-102: Status 'Ready to Join' but marked Ready-to-Join
- CAND-105: Status 'Pending' but marked Ready-to-Join

Sources: bpps_tracker_export.csv
```

## Data Available

- **Candidates**: CAND-101 through CAND-106
- **Documents**: 12 total (4 PDFs, 8 Word docs)
- **Structured Data**: CSV files with candidate tracker, documents, employment history

## System Requirements

- Python 3.9+
- Ollama running locally (`ollama serve`)
- Model: llama3.1:8b (already installed)

## Exit

Type `exit`, `quit`, or `q` to end the chat.
