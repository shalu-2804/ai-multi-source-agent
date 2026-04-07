from typing import Dict, List, Any, Optional
from pathlib import Path
import pandas as pd
import json

class StructuredDataQueryer:
    """Query structured data (CSV/Excel) for BPSS screening information"""
    
    def __init__(self):
        self.tracker_df = None
        self.document_inventory_df = None
        self.employment_history_df = None
        self.case_tracker_df = None
    
    def load_data(self, tracker_csv: Path, document_inv_csv: Path, 
                  employment_csv: Path, case_tracker_xlsx: Path = None):
        """Load all structured data files"""
        try:
            # Load tracker
            if tracker_csv.exists():
                self.tracker_df = pd.read_csv(tracker_csv)
                print(f"Loaded tracker: {len(self.tracker_df)} candidates")
            
            # Load document inventory
            if document_inv_csv.exists():
                self.document_inventory_df = pd.read_csv(document_inv_csv)
                print(f"Loaded document inventory: {len(self.document_inventory_df)} records")
            
            # Load employment history
            if employment_csv.exists():
                self.employment_history_df = pd.read_csv(employment_csv)
                print(f"Loaded employment history: {len(self.employment_history_df)} records")
            
            # Load case tracker Excel (if available)
            if case_tracker_xlsx and case_tracker_xlsx.exists():
                self.case_tracker_df = pd.read_excel(case_tracker_xlsx, sheet_name=0)
                print(f"Loaded case tracker: {len(self.case_tracker_df)} records")
                
        except Exception as e:
            print(f"Error loading structured data: {e}")
    
    def get_candidate_status(self, candidate_id: str) -> Dict[str, Any]:
        """Get candidate screening status from tracker"""
        if self.tracker_df is None:
            return {"error": "Tracker data not loaded"}
        
        candidate_records = self.tracker_df[self.tracker_df['candidate_id'].astype(str) == candidate_id]
        
        if candidate_records.empty:
            return {"error": f"Candidate {candidate_id} not found"}
        
        record = candidate_records.iloc[0]
        return {
            "candidate_id": record.get("candidate_id"),
            "candidate_name": record.get("candidate_name"),
            "role_code": record.get("role_code"),
            "status_tracker": record.get("status_tracker"),
            "ready_to_join": record.get("ready_to_join"),
            "identity_complete": record.get("identity_complete"),
            "rtw_complete": record.get("rtw_complete"),
            "employment_complete": record.get("employment_complete"),
            "criminality_complete": record.get("criminality_complete"),
            "risk_level": record.get("risk_level"),
            "analyst_review_date": record.get("analyst_review_date"),
            "notes": record.get("notes")
        }
    
    def get_candidate_documents(self, candidate_id: str) -> List[Dict[str, Any]]:
        """Get all documents associated with a candidate"""
        if self.document_inventory_df is None:
            return []
        
        docs = self.document_inventory_df[
            self.document_inventory_df['candidate_id'].astype(str) == candidate_id
        ]
        
        return docs.to_dict('records')
    
    def get_missing_documents(self, candidate_id: str) -> List[Dict[str, Any]]:
        """Get missing or invalid documents for a candidate"""
        if self.document_inventory_df is None:
            return []
        
        docs = self.document_inventory_df[
            self.document_inventory_df['candidate_id'].astype(str) == candidate_id
        ]
        
        # Filter for missing documents
        missing = docs[docs['present_in_folder'].astype(str).str.lower() != 'yes']
        return missing.to_dict('records')
    
    def get_employment_history(self, candidate_id: str) -> List[Dict[str, Any]]:
        """Get employment history for a candidate"""
        if self.employment_history_df is None:
            return []
        
        employment = self.employment_history_df[
            self.employment_history_df['candidate_id'].astype(str) == candidate_id
        ]
        
        return employment.to_dict('records')
    
    def get_all_candidates(self) -> List[Dict[str, Any]]:
        """Get all candidates from tracker"""
        if self.tracker_df is None:
            return []
        
        return self.tracker_df.to_dict('records')
    
    def check_compliance_status(self, candidate_id: str) -> Dict[str, Any]:
        """Check screening completion and compliance"""
        status = self.get_candidate_status(candidate_id)
        
        if "error" in status:
            return status
        
        compliance = {
            "candidate_id": candidate_id,
            "candidate_name": status.get("candidate_name"),
            "checks_complete": {
                "identity": status.get("identity_complete") == "Yes",
                "rtw": status.get("rtw_complete") == "Yes",
                "employment": status.get("employment_complete") == "Yes",
                "criminality": status.get("criminality_complete") == "Yes"
            },
            "status": status.get("status_tracker"),
            "ready_to_join": status.get("ready_to_join"),
            "risk_level": status.get("risk_level"),
            "notes": status.get("notes")
        }
        
        # Determine if all checks are complete
        checks_complete = all(compliance["checks_complete"].values())
        compliance["all_checks_complete"] = checks_complete
        
        return compliance
    
    def search_candidates_by_risk(self, risk_level: str) -> List[Dict[str, Any]]:
        """Find candidates with specific risk level"""
        if self.tracker_df is None:
            return []
        
        candidates = self.tracker_df[
            self.tracker_df['risk_level'].astype(str).str.lower() == risk_level.lower()
        ]
        
        return candidates.to_dict('records')
    
    def get_candidates_by_status(self, status: str) -> List[Dict[str, Any]]:
        """Find candidates with specific status"""
        if self.tracker_df is None:
            return []
        
        candidates = self.tracker_df[
            self.tracker_df['status_tracker'].astype(str).str.lower() == status.lower()
        ]
        
        return candidates.to_dict('records')
    
    def check_document_validity(self, candidate_id: str, doc_type: str) -> Dict[str, Any]:
        """Check if specific document type is valid for candidate"""
        docs = self.get_candidate_documents(candidate_id)
        
        for doc in docs:
            if doc.get('doc_type', '').lower() == doc_type.lower():
                return {
                    "candidate_id": candidate_id,
                    "doc_type": doc_type,
                    "present": doc.get('present_in_folder') == 'Yes',
                    "document_date": doc.get('document_date'),
                    "valid_to": doc.get('valid_to'),
                    "remarks": doc.get('remarks')
                }
        
        return {
            "candidate_id": candidate_id,
            "doc_type": doc_type,
            "present": False,
            "error": f"Document type {doc_type} not found for candidate"
        }
    
    def generate_candidate_summary(self, candidate_id: str) -> Dict[str, Any]:
        """Generate comprehensive summary for a candidate"""
        status = self.get_candidate_status(candidate_id)
        docs = self.get_candidate_documents(candidate_id)
        missing_docs = self.get_missing_documents(candidate_id)
        employment = self.get_employment_history(candidate_id)
        
        return {
            "candidate_id": candidate_id,
            "status": status,
            "documents": {
                "total": len(docs),
                "present": len([d for d in docs if d.get('present_in_folder') == 'Yes']),
                "missing": len(missing_docs),
                "details": docs
            },
            "employment": {
                "records": len(employment),
                "details": employment
            },
            "missing_documents": missing_docs
        }
