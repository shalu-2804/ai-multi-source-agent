import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "bpss_agentic_dataset"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Data subdirectories
CANDIDATE_PACK_DIR = DATA_DIR / "candidate_pack"
EVIDENCE_DIR = DATA_DIR / "evidence"
POLICIES_DIR = DATA_DIR / "policies"
REFERENCE_DIR = DATA_DIR / "reference"
STRUCTURED_DIR = DATA_DIR / "structured"
EXPECTED_OUTPUTS_DIR = DATA_DIR / "expected_outputs"

# LLM Configuration
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"
OLLAMA_MODEL = "llama3.1:8b"  # Using llama3.1:8b (available locally)
OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chroma DB Configuration
CHROMA_PERSIST_DIR = PROJECT_ROOT / ".chroma_db"
CHROMA_COLLECTION_NAME = "bpss_documents"

# Agent configuration
AGENT_VERBOSE = True
TEMPERATURE = 0.3  # Lower temperature for factual accuracy
MAX_ITERATIONS = 10

# CSV files
TRACKER_CSV = STRUCTURED_DIR / "bpps_tracker_export.csv"
DOCUMENT_INVENTORY_CSV = STRUCTURED_DIR / "document_inventory.csv"
EMPLOYMENT_HISTORY_CSV = STRUCTURED_DIR / "employment_history.csv"
