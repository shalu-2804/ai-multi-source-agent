"""
Smoke tests for ai-multi-source-agent.
These tests verify that all modules import correctly and core classes
instantiate without errors — no Ollama or data files required.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_config_imports():
    """Config module loads without errors."""
    from config import settings
    assert hasattr(settings, "CHROMA_PERSIST_DIR")


def test_document_loader_imports():
    """DocumentLoader class is importable and instantiable."""
    from document_loader import DocumentLoader
    loader = DocumentLoader()
    assert loader is not None


def test_vector_retriever_imports():
    """VectorRetriever class is importable."""
    from vector_retriever import VectorRetriever
    assert VectorRetriever is not None


def test_structured_queryer_imports():
    """StructuredDataQueryer class is importable and instantiable."""
    from structured_queryer import StructuredDataQueryer
    queryer = StructuredDataQueryer()
    assert queryer is not None


def test_agent_tools_imports():
    """AgentToolkit class is importable."""
    from agent_tools import AgentToolkit
    assert AgentToolkit is not None


def test_bpss_agent_imports():
    """BPSSAgent class is importable."""
    from bpss_agent import BPSSAgent
    assert BPSSAgent is not None


def test_main_module_imports():
    """BPSSRAGSystem class in main.py is importable."""
    from main import BPSSRAGSystem
    system = BPSSRAGSystem()
    assert system is not None
    assert system.loader is not None
    assert system.vector_retriever is not None
    assert system.structured_queryer is not None
