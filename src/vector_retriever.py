from typing import List, Dict, Tuple
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer
import json
import logging
import warnings

# Suppress Chroma verbose logging and warnings
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("chromadb.telemetry").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

class VectorRetriever:
    """Manage vector embeddings and retrieval using Chroma"""
    
    def __init__(self, persist_dir: Path = None, collection_name: str = "bpss_documents"):
        self.persist_dir = persist_dir or Path.home() / ".chroma_db"
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.embedding_model = None
        self._initialize_client()
        
    def _initialize_client(self):
        """Initialize Chroma client with persistence"""
        # Use new Chroma client API
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        
        try:
            # Try to get existing collection
            self.collection = self.client.get_collection(name=self.collection_name)
            print(f"Loaded existing collection: {self.collection_name}")
        except:
            # Create new collection if it doesn't exist
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Created new collection: {self.collection_name}")
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Initialized sentence transformer model")
    
    def add_documents(self, documents: Dict[str, str], metadata_dict: Dict = None):
        """Add documents to vector store with metadata"""
        if not documents:
            print("No documents to add")
            return
        
        ids = []
        texts = []
        metadatas = []
        
        for doc_name, content in documents.items():
            # Split content into manageable chunks
            chunks = self._chunk_text(content, chunk_size=1000)
            
            for i, chunk in enumerate(chunks):
                doc_id = f"{doc_name}_{i}"
                ids.append(doc_id)
                texts.append(chunk)
                
                meta = {
                    "source": doc_name,
                    "chunk_idx": i,
                    "chunk_count": len(chunks)
                }
                if metadata_dict and doc_name in metadata_dict:
                    meta.update(metadata_dict[doc_name])
                metadatas.append(meta)
        
        if ids:
            # Get embeddings
            embeddings = self.embedding_model.encode(texts)
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas
            )
            
            print(f"Added {len(ids)} document chunks to collection")
    
    def search(self, query: str, n_results: int = 5, filters: Dict = None) -> List[Dict]:
        """Search for relevant documents"""
        if not query or not query.strip():
            return []
        
        try:
            # Get embedding for query
            query_embedding = self.embedding_model.encode([query])[0].tolist()
            
            # Search collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filters if filters else None,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results and results.get("documents"):
                for i, doc in enumerate(results["documents"][0]):
                    formatted_results.append({
                        "text": doc,
                        "source": results["metadatas"][0][i].get("source", "unknown"),
                        "distance": results["distances"][0][i],
                        "metadata": results["metadatas"][0][i]
                    })
            
            return formatted_results
        except Exception as e:
            print(f"Error during search: {e}")
            return []
    
    def _chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Split text into overlapping chunks"""
        sentences = text.split('.')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > chunk_size:
                if current_chunk:
                    chunks.append(". ".join(current_chunk) + ".")
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(". ".join(current_chunk) + ".")
        
        return chunks if chunks else [text]
    
    def clear_collection(self):
        """Clear all documents from the collection"""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Cleared collection: {self.collection_name}")
        except Exception as e:
            print(f"Error clearing collection: {e}")
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": str(self.persist_dir)
            }
        except Exception as e:
            print(f"Error getting collection stats: {e}")
            return {}
