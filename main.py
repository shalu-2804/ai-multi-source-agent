#!/usr/bin/env python3
"""
BPSS RAG System - Interactive Chat
Simple question-answering with document citations
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config.settings import (
    DATA_DIR, CANDIDATE_PACK_DIR, EVIDENCE_DIR, POLICIES_DIR,
    REFERENCE_DIR, STRUCTURED_DIR, TRACKER_CSV, DOCUMENT_INVENTORY_CSV,
    EMPLOYMENT_HISTORY_CSV, CHROMA_PERSIST_DIR
)
from document_loader import DocumentLoader
from vector_retriever import VectorRetriever
from structured_queryer import StructuredDataQueryer
from agent_tools import AgentToolkit
from bpss_agent import BPSSAgent

class BPSSRAGSystem:
    """Simple RAG system for BPSS screening analysis"""
    
    def __init__(self):
        self.loader = DocumentLoader()
        self.vector_retriever = VectorRetriever(persist_dir=CHROMA_PERSIST_DIR)
        self.structured_queryer = StructuredDataQueryer()
        self.toolkit = None
        self.agent = None
        
    def initialize(self):
        """Initialize all system components"""
        print("\n" + "=" * 60)
        print("BPSS RAG System - Initializing")
        print("=" * 60)
        
        # Check data directory
        if not DATA_DIR.exists():
            print(f"ERROR: Dataset directory not found at {DATA_DIR}")
            return False
        
        print(f"✓ Dataset directory found")
        
        # Load documents
        print("Loading documents...")
        self._load_all_documents()
        
        # Load structured data
        print("Loading structured data...")
        self._load_structured_data()
        
        # Initialize toolkit
        print("Initializing toolkit...")
        self.toolkit = AgentToolkit(
            vector_retriever=self.vector_retriever,
            structured_queryer=self.structured_queryer,
            documents_metadata=self.loader.get_metadata()
        )
        
        # Initialize agent
        print("Initializing agent...")
        self.agent = BPSSAgent(self.toolkit)
        
        print("\n" + "=" * 60)
        print("✓ System ready! Type your questions below.")
        print("✓ Type 'exit' or 'quit' to end the chat.")
        print("=" * 60 + "\n")
        
        return True
    
    def _load_all_documents(self):
        """Load all document types"""
        # Load PDFs
        pdfs = {}
        pdfs.update(self.loader.load_pdfs(POLICIES_DIR))
        pdfs.update(self.loader.load_pdfs(EVIDENCE_DIR))
        pdfs.update(self.loader.load_pdfs(REFERENCE_DIR))
        
        # Load DOCX files
        docx_files = self.loader.load_docx_files(CANDIDATE_PACK_DIR)
        docx_files.update(self.loader.load_docx_files(EVIDENCE_DIR))
        
        # Combine all documents
        all_documents = {**pdfs, **docx_files}
        
        # Add to vector store
        if all_documents:
            self.vector_retriever.add_documents(all_documents, self.loader.get_metadata())
            stats = self.vector_retriever.get_collection_stats()
            print(f"✓ Loaded {len(all_documents)} documents ({stats.get('document_count', 0)} chunks indexed)")
    
    def _load_structured_data(self):
        """Load structured data files"""
        self.structured_queryer.load_data(
            tracker_csv=TRACKER_CSV,
            document_inv_csv=DOCUMENT_INVENTORY_CSV,
            employment_csv=EMPLOYMENT_HISTORY_CSV
        )
        all_candidates = self.structured_queryer.get_all_candidates()
        print(f"✓ Loaded screening data for {len(all_candidates)} candidates")
        
        print("\n" + "=" * 60)
        print("System initialized successfully!")
        print("=" * 60)
        
        return True
    
    def _load_all_documents(self, reload: bool = False):
        """Load all document types"""
        if reload:
            self.vector_retriever.clear_collection()
            print("✓ Cleared existing vector store")
        
        # Load PDFs
        print("  Loading PDFs...")
        pdfs = {}
        pdfs.update(self.loader.load_pdfs(POLICIES_DIR))
        pdfs.update(self.loader.load_pdfs(EVIDENCE_DIR))
        pdfs.update(self.loader.load_pdfs(REFERENCE_DIR))
        
        # Load DOCX files
        print("  Loading Word documents...")
        docx_files = self.loader.load_docx_files(CANDIDATE_PACK_DIR)
        docx_files.update(self.loader.load_docx_files(EVIDENCE_DIR))
        
        # Combine all documents
        all_documents = {**pdfs, **docx_files}
        
        # Add to vector store
        if all_documents:
            self.vector_retriever.add_documents(all_documents, self.loader.get_metadata())
            print(f"✓ Added {len(all_documents)} documents to vector store")
        
        # Print collection stats
        stats = self.vector_retriever.get_collection_stats()
        print(f"  Vector store: {stats.get('document_count', 0)} chunks indexed")
    
    def _load_structured_data(self):
        """Load structured data files"""
        self.structured_queryer.load_data(
            tracker_csv=TRACKER_CSV,
            document_inv_csv=DOCUMENT_INVENTORY_CSV,
            employment_csv=EMPLOYMENT_HISTORY_CSV
        )
        all_candidates = self.structured_queryer.get_all_candidates()
        print(f"✓ Loaded screening data for {len(all_candidates)} candidates")
    
    def start_interactive_chat(self):
        """Start interactive chat loop"""
        print("You: ", end="", flush=True)
        
        try:
            while True:
                question = input().strip()
                
                if not question:
                    print("You: ", end="", flush=True)
                    continue
                
                if question.lower() in ['exit', 'quit', 'bye', 'q']:
                    print("\nThank you for using BPSS RAG System. Goodbye!")
                    break
                
                print()
                answer = self.agent.ask_with_citations(question)
                print(f"{answer}\n")
                print("You: ", end="", flush=True)
                
        except KeyboardInterrupt:
            print("\n\nSystem interrupted. Goodbye!")
            sys.exit(0)
        except EOFError:
            print("\nThank you for using BPSS RAG System. Goodbye!")


def main():
    """Main entry point - interactive mode only"""
    # Initialize system
    system = BPSSRAGSystem()
    
    if not system.initialize():
        sys.exit(1)
    
    # Start interactive chat
    system.start_interactive_chat()


if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()
