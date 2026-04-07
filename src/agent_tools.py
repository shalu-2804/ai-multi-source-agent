from typing import List, Dict, Any, Optional
from datetime import datetime
import json

class AgentToolkit:
    """Collection of tools for the BPSS Agentic AI system"""
    
    def __init__(self, vector_retriever=None, structured_queryer=None, 
                 documents_metadata=None):
        self.vector_retriever = vector_retriever
        self.structured_queryer = structured_queryer
        self.documents_metadata = documents_metadata or {}
        self.tool_calls = []  # Track all tool calls for transparency
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """Return list of available tools for agent"""
        return [
            {
                "name": "policy_lookup",
                "description": "Search BPSS policies and screening procedures to find relevant policy requirements and procedures",
                "function": self.policy_lookup
            },
            {
                "name": "candidate_evidence_search",
                "description": "Search candidate evidence documents to find specific information about a candidate",
                "function": self.candidate_evidence_search
            },
            {
                "name": "get_candidate_status",
                "description": "Get screening status and completion status for a candidate from structured data",
                "function": self.get_candidate_status_tool
            },
            {
                "name": "check_document_validity",
                "description": "Check if specific documents are present and valid for a candidate",
                "function": self.check_document_validity_tool
            },
            {
                "name": "identify_contradictions",
                "description": "Identify contradictions between tracker status and analyst notes for a candidate",
                "function": self.identify_contradictions
            },
            {
                "name": "find_missing_information",
                "description": "Identify missing or incomplete information required for screening decision",
                "function": self.find_missing_information
            },
            {
                "name": "get_candidate_summary",
                "description": "Get comprehensive summary of candidate screening status and documents",
                "function": self.get_candidate_summary_tool
            },
            {
                "name": "search_candidates_by_criteria",
                "description": "Search candidates by risk level, status, or other criteria",
                "function": self.search_candidates_by_criteria
            }
        ]
    
    def policy_lookup(self, query: str) -> Dict[str, Any]:
        """Search for relevant policy information"""
        if not self.vector_retriever:
            return {"error": "Vector retriever not initialized"}
        
        results = self.vector_retriever.search(query, n_results=3)
        
        self._record_tool_call("policy_lookup", {"query": query}, results)
        
        formatted_results = {
            "query": query,
            "results": [],
            "sources": set()
        }
        
        for result in results:
            formatted_results["results"].append({
                "text": result["text"][:500] + "..." if len(result["text"]) > 500 else result["text"],
                "source": result["source"],
                "relevance_score": 1 - result["distance"]
            })
            formatted_results["sources"].add(result["source"])
        
        formatted_results["sources"] = list(formatted_results["sources"])
        
        return formatted_results
    
    def candidate_evidence_search(self, candidate_query: str, candidate_id: str = None) -> Dict[str, Any]:
        """Search candidate-related documents"""
        if not self.vector_retriever:
            return {"error": "Vector retriever not initialized"}
        
        # If candidate_id provided, filter search
        filters = None
        if candidate_id:
            filters = {"source": {"$contains": candidate_id}}
        
        results = self.vector_retriever.search(candidate_query, n_results=3, filters=filters)
        
        self._record_tool_call("candidate_evidence_search", 
                              {"query": candidate_query, "candidate_id": candidate_id}, 
                              results)
        
        formatted_results = {
            "query": candidate_query,
            "candidate_id": candidate_id,
            "results": [],
            "sources": set()
        }
        
        for result in results:
            formatted_results["results"].append({
                "text": result["text"][:500] + "..." if len(result["text"]) > 500 else result["text"],
                "source": result["source"],
                "relevance_score": 1 - result["distance"]
            })
            formatted_results["sources"].add(result["source"])
        
        formatted_results["sources"] = list(formatted_results["sources"])
        
        return formatted_results
    
    def get_candidate_status_tool(self, candidate_id: str) -> Dict[str, Any]:
        """Get candidate status from structured data"""
        if not self.structured_queryer:
            return {"error": "Structured queryer not initialized"}
        
        result = self.structured_queryer.get_candidate_status(candidate_id)
        
        self._record_tool_call("get_candidate_status", {"candidate_id": candidate_id}, result)
        
        return result
    
    def check_document_validity_tool(self, candidate_id: str, doc_type: str) -> Dict[str, Any]:
        """Check if document is present and valid"""
        if not self.structured_queryer:
            return {"error": "Structured queryer not initialized"}
        
        result = self.structured_queryer.check_document_validity(candidate_id, doc_type)
        
        self._record_tool_call("check_document_validity", 
                              {"candidate_id": candidate_id, "doc_type": doc_type}, 
                              result)
        
        return result
    
    def identify_contradictions(self, candidate_id: str) -> Dict[str, Any]:
        """Identify contradictions in candidate records"""
        if not self.structured_queryer or not self.vector_retriever:
            return {"error": "Required components not initialized"}
        
        # Get status from tracker
        tracker_status = self.structured_queryer.get_candidate_status(candidate_id)
        
        # Search for analyst notes
        analyst_search = self.vector_retriever.search(
            f"analyst notes {candidate_id}", n_results=3
        )
        
        contradictions = {
            "candidate_id": candidate_id,
            "contradictions_found": False,
            "details": []
        }
        
        # Basic contradiction detection logic
        if tracker_status and not isinstance(tracker_status, dict) or "error" not in tracker_status:
            # Check if status vs ready_to_join mismatch
            status = tracker_status.get("status_tracker", "")
            ready = tracker_status.get("ready_to_join", "")
            
            if status == "Clear" and ready != "Yes":
                contradictions["contradictions_found"] = True
                contradictions["details"].append({
                    "type": "status_mismatch",
                    "description": f"Status is '{status}' but ready_to_join is '{ready}'",
                    "severity": "high"
                })
        
        # Check analyst notes
        if analyst_search:
            for result in analyst_search:
                if "contradiction" in result["text"].lower() or \
                   "inconsistent" in result["text"].lower() or \
                   "discrepancy" in result["text"].lower():
                    contradictions["contradictions_found"] = True
                    contradictions["details"].append({
                        "type": "analyst_note",
                        "description": result["text"][:200],
                        "source": result["source"],
                        "severity": "medium"
                    })
        
        self._record_tool_call("identify_contradictions", {"candidate_id": candidate_id}, contradictions)
        
        return contradictions
    
    def find_missing_information(self, candidate_id: str) -> Dict[str, Any]:
        """Find missing or incomplete information"""
        if not self.structured_queryer:
            return {"error": "Structured queryer not initialized"}
        
        missing_docs = self.structured_queryer.get_missing_documents(candidate_id)
        status = self.structured_queryer.get_candidate_status(candidate_id)
        
        missing_info = {
            "candidate_id": candidate_id,
            "missing_documents": len(missing_docs),
            "documents": [],
            "incomplete_checks": []
        }
        
        # List missing documents
        for doc in missing_docs:
            missing_info["documents"].append({
                "doc_type": doc.get("doc_type"),
                "remarks": doc.get("remarks")
            })
        
        # Check for incomplete checks
        if status and "error" not in status:
            checks = {
                "identity": status.get("identity_complete"),
                "rtw": status.get("rtw_complete"),
                "employment": status.get("employment_complete"),
                "criminality": status.get("criminality_complete")
            }
            
            for check_name, check_status in checks.items():
                if check_status != "Yes":
                    missing_info["incomplete_checks"].append(check_name)
        
        self._record_tool_call("find_missing_information", {"candidate_id": candidate_id}, missing_info)
        
        return missing_info
    
    def get_candidate_summary_tool(self, candidate_id: str) -> Dict[str, Any]:
        """Get comprehensive candidate summary"""
        if not self.structured_queryer:
            return {"error": "Structured queryer not initialized"}
        
        result = self.structured_queryer.generate_candidate_summary(candidate_id)
        
        self._record_tool_call("get_candidate_summary", {"candidate_id": candidate_id}, result)
        
        return result
    
    def search_candidates_by_criteria(self, criteria_type: str, criteria_value: str) -> Dict[str, Any]:
        """Search candidates by criteria like risk_level or status"""
        if not self.structured_queryer:
            return {"error": "Structured queryer not initialized"}
        
        if criteria_type.lower() == "risk_level":
            candidates = self.structured_queryer.search_candidates_by_risk(criteria_value)
        elif criteria_type.lower() == "status":
            candidates = self.structured_queryer.get_candidates_by_status(criteria_value)
        else:
            return {"error": f"Unknown criteria type: {criteria_type}"}
        
        result = {
            "criteria_type": criteria_type,
            "criteria_value": criteria_value,
            "count": len(candidates),
            "candidates": candidates
        }
        
        self._record_tool_call("search_candidates_by_criteria", 
                              {"criteria_type": criteria_type, "criteria_value": criteria_value}, 
                              result)
        
        return result
    
    def _record_tool_call(self, tool_name: str, args: Dict, result: Any):
        """Record tool calls for transparency and debugging"""
        self.tool_calls.append({
            "timestamp": datetime.now().isoformat(),
            "tool": tool_name,
            "args": args,
            "result_summary": str(result)[:100] if isinstance(result, dict) else str(result)[:100]
        })
    
    def get_tool_call_history(self) -> List[Dict]:
        """Get history of tool calls"""
        return self.tool_calls
