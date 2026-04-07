import json
import os
from typing import Dict, Any, List
import requests
from config.settings import OLLAMA_MODEL, OLLAMA_URL

class BPSSAgent:
    """Main BPSS Agentic AI system powered by Local LLM (Ollama)"""
    
    def __init__(self, toolkit, model: str = None, ollama_url: str = None):
        self.model = model or OLLAMA_MODEL
        self.ollama_url = ollama_url or OLLAMA_URL
        self.toolkit = toolkit
        self.conversation_history = []
        
        # Verify Ollama is running
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
            if response.status_code != 200:
                raise ValueError("Ollama server not responding properly")
        except requests.ConnectionError:
            raise ValueError(
                f"Ollama not running at {self.ollama_url}\n"
                "Run: ollama serve\n"
                f"Model configured: {self.model}"
            )
        except Exception as e:
            raise ValueError(f"Error connecting to Ollama: {e}")
        
    def get_available_tools(self) -> List[Dict]:
        """Get available tools (simplified for Ollama)"""
        return self.toolkit.get_tools()
    
    def ask(self, question: str, system_prompt: str = None) -> str:
        """Ask the agent a question with actual tool execution (ReAct loop)"""
        if system_prompt is None:
            system_prompt = self._get_default_system_prompt()
        
        print("🤔 Thinking... (using local Ollama model)")
        
        # Execute tools based on question intent
        tool_results = self._execute_tools_for_question(question)
        
        # Now use Ollama to reason over the tool results
        tools_context = self._get_tools_context()
        
        full_prompt = f"""{system_prompt}

QUESTION: {question}

RELEVANT DATA FROM TOOLS:
{tool_results}

Based on the data above, provide a clear, specific answer to the question."""
        
        try:
            # Call Ollama with the tool results as context
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "temperature": 0.3
                },
                timeout=60
            )
            
            if response.status_code != 200:
                return f"Error from Ollama: {response.text}"
            
            result = response.json()
            answer = result.get("response", "No response generated")
            
            # Add to conversation history
            self.conversation_history.append({
                "role": "user",
                "content": question
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": answer
            })
            
            return answer
            
        except requests.ConnectionError:
            return ("Ollama is not running. Please start it with: ollama serve\n"
                   "And ensure the model is downloaded: ollama pull mistral")
        except Exception as e:
            return f"Error: {str(e)}"
    
    def ask_with_citations(self, question: str) -> str:
        """Ask a question and return answer with source citations"""
        question_lower = question.lower()
        answer_parts = []
        citations = set()
        
        # Route to specific query types
        if "violate" in question_lower or "violation" in question_lower or "policy" in question_lower:
            answer_parts, citations = self._answer_policy_violations(question)
        
        elif "compliance" in question_lower or "comply" in question_lower:
            answer_parts, citations = self._answer_compliance_issues(question)
        
        elif "high risk" in question_lower or "risk" in question_lower:
            answer_parts, citations = self._answer_high_risk(question)
        
        elif "missing" in question_lower or "incomplete" in question_lower:
            answer_parts, citations = self._answer_missing_info(question)
        
        elif "cand-" in question_lower:
            answer_parts, citations = self._answer_candidate_specific(question)
        
        else:
            # Default: semantic search with LLM
            answer_parts, citations = self._answer_semantic(question)
        
        answer_text = "\n".join(answer_parts)
        citation_text = "\n\nSources: " + ", ".join(sorted(citations)) if citations else ""
        
        return f"{answer_text}{citation_text}"
    
    def _answer_policy_violations(self, question: str) -> tuple:
        """Find policy violations in structured data"""
        findings = []
        citations = set()
        
        try:
            if self.toolkit.structured_queryer and self.toolkit.structured_queryer.tracker_df is not None:
                df = self.toolkit.structured_queryer.tracker_df
                
                # Check for contradictions (status vs ready-to-join)
                violations = []
                for _, row in df.iterrows():
                    if row['status_tracker'] == 'Clear' and not row['ready_to_join']:
                        violations.append(f"- {row['candidate_id']}: Status 'Clear' but Ready-to-Join = False")
                        citations.add("bpps_tracker_export.csv")
                    elif row['status_tracker'] != 'Clear' and row['ready_to_join']:
                        violations.append(f"- {row['candidate_id']}: Status '{row['status_tracker']}' but marked Ready-to-Join")
                        citations.add("bpps_tracker_export.csv")
                
                if violations:
                    findings.append("**Policy Violations Found:**")
                    findings.extend(violations)
                    findings.append("\nThese records have inconsistencies between tracker status and ready-to-join flags, potentially violating BPSS approval procedures.")
                else:
                    findings.append("No clear policy violations detected in the tracker data. All candidates with 'Clear' status are appropriately flagged.")
                    citations.add("bpps_tracker_export.csv")
        except Exception as e:
            findings.append(f"Error analyzing violations: {str(e)}")
        
        return findings, citations
    
    def _answer_compliance_issues(self, question: str) -> tuple:
        """Find compliance issues for entities"""
        findings = []
        citations = set()
        
        try:
            if self.toolkit.structured_queryer and self.toolkit.structured_queryer.tracker_df is not None:
                df = self.toolkit.structured_queryer.tracker_df
                
                # Find candidates with incomplete checks or high risk
                issues = []
                for _, row in df.iterrows():
                    incomplete = []
                    if not row['identity_complete']:
                        incomplete.append("Identity")
                    if not row['rtw_complete']:
                        incomplete.append("Right-to-Work")
                    if not row['employment_complete']:
                        incomplete.append("Employment")
                    if not row['criminality_complete']:
                        incomplete.append("Criminality")
                    
                    if incomplete or row['risk_level'].lower() in ['high', 'medium']:
                        issue_text = f"- {row['candidate_id']} ({row['candidate_name']})"
                        if incomplete:
                            issue_text += f": Missing {', '.join(incomplete)}"
                        if row['risk_level'].lower() in ['high', 'medium']:
                            issue_text += f" [Risk: {row['risk_level']}]"
                        issues.append(issue_text)
                        citations.add("bpps_tracker_export.csv")
                
                if issues:
                    findings.append("**Compliance Issues Identified:**")
                    findings.extend(issues)
                else:
                    findings.append("All candidates are compliant with BPSS requirements.")
                    citations.add("bpps_tracker_export.csv")
        except Exception as e:
            findings.append(f"Error: {str(e)}")
        
        return findings, citations
    
    def _answer_high_risk(self, question: str) -> tuple:
        """Find high-risk candidates"""
        findings = []
        citations = set()
        
        try:
            if self.toolkit.structured_queryer and self.toolkit.structured_queryer.tracker_df is not None:
                df = self.toolkit.structured_queryer.tracker_df
                high_risk = df[df['risk_level'].str.lower() == 'high']
                
                if len(high_risk) > 0:
                    findings.append(f"**{len(high_risk)} High-Risk Candidates:**")
                    for _, row in high_risk.iterrows():
                        findings.append(f"- {row['candidate_id']}: {row['candidate_name']} - {row['status_tracker']} (Note: {row['notes']})")
                    citations.add("bpps_tracker_export.csv")
                else:
                    findings.append("No high-risk candidates identified.")
                    citations.add("bpps_tracker_export.csv")
        except Exception as e:
            findings.append(f"Error: {str(e)}")
        
        return findings, citations
    
    def _answer_missing_info(self, question: str) -> tuple:
        """Find missing information"""
        findings = []
        citations = set()
        
        try:
            if self.toolkit.structured_queryer and self.toolkit.structured_queryer.tracker_df is not None:
                df = self.toolkit.structured_queryer.tracker_df
                
                incomplete_list = []
                for _, row in df.iterrows():
                    missing = []
                    if not row['identity_complete']:
                        missing.append("Identity")
                    if not row['rtw_complete']:
                        missing.append("Right-to-Work")
                    if not row['employment_complete']:
                        missing.append("Employment")
                    if not row['criminality_complete']:
                        missing.append("Criminality")
                    
                    if missing:
                        incomplete_list.append(f"- {row['candidate_id']}: {', '.join(missing)}")
                        citations.add("bpps_tracker_export.csv")
                
                if incomplete_list:
                    findings.append("**Incomplete Checks:**")
                    findings.extend(incomplete_list)
                else:
                    findings.append("All candidates have complete screening checks.")
                    citations.add("bpps_tracker_export.csv")
        except Exception as e:
            findings.append(f"Error: {str(e)}")
        
        return findings, citations
    
    def _answer_candidate_specific(self, question: str) -> tuple:
        """Answer questions about specific candidates"""
        findings = []
        citations = set()
        
        try:
            if self.toolkit.structured_queryer and self.toolkit.structured_queryer.tracker_df is not None:
                df = self.toolkit.structured_queryer.tracker_df
                
                # Extract candidate ID from question
                for cand_id in ["CAND-101", "CAND-102", "CAND-103", "CAND-104", "CAND-105", "CAND-106"]:
                    if cand_id in question:
                        candidate = df[df['candidate_id'] == cand_id]
                        if not candidate.empty:
                            row = candidate.iloc[0]
                            findings.append(f"**{cand_id} - {row['candidate_name']}**")
                            findings.append(f"Status: {row['status_tracker']}")
                            findings.append(f"Risk Level: {row['risk_level']}")
                            findings.append(f"Ready to Join: {row['ready_to_join']}")
                            findings.append(f"Notes: {row['notes']}")
                            citations.add("bpps_tracker_export.csv")
                        break
                else:
                    findings.append("Candidate not found in database.")
        except Exception as e:
            findings.append(f"Error: {str(e)}")
        
        return findings, citations
    
    def _answer_semantic(self, question: str) -> tuple:
        """Default semantic search with LLM reasoning"""
        findings = []
        citations = set()
        
        # Get relevant documents from vector store
        search_results = self.toolkit.vector_retriever.search(question, n_results=3)
        
        # Extract citations
        for result in search_results:
            source = result.get("source", "Unknown")
            citations.add(source)
        
        context = "RELEVANT DOCUMENTS:\n"
        for result in search_results:
            source = result.get("source", "Unknown")
            context += f"- {source}: {result.get('text', '')[:200]}...\n"
        
        # Build prompt with context
        system_prompt = """You are a BPSS screening analyst. Answer questions clearly based on the provided documents.
Keep your answer brief (2-3 sentences max) and factual. If you cannot find information, say so."""
        
        full_prompt = f"""{system_prompt}

{context}

QUESTION: {question}

Provide a brief, factual answer."""
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "temperature": 0.3
                },
                timeout=60
            )
            
            if response.status_code != 200:
                findings.append(f"Error: {response.text}")
            else:
                answer = response.json().get("response", "No response generated").strip()
                findings.append(answer)
        except Exception as e:
            findings.append(f"Error: {str(e)}")
        
        return findings, citations
    
    def _execute_tools_for_question(self, question: str) -> str:
        """Execute relevant tools based on question intent"""
        results = []
        question_lower = question.lower()
        
        # Detect question intent and execute appropriate tools
        
        # 1. High-risk candidates
        if "high risk" in question_lower or "risk level" in question_lower:
            results.append("=== HIGH-RISK CANDIDATES ===")
            try:
                # Get tracker data directly
                if self.toolkit.structured_queryer and self.toolkit.structured_queryer.tracker_df is not None:
                    df = self.toolkit.structured_queryer.tracker_df
                    high_risk = df[df['risk_level'].str.lower() == 'high']
                    if len(high_risk) > 0:
                        results.append(f"Found {len(high_risk)} high-risk candidates:")
                        for _, row in high_risk.iterrows():
                            results.append(f"- {row['candidate_id']}: {row['candidate_name']} (Status: {row['status_tracker']}, Notes: {row['notes']})")
                    else:
                        results.append("No high-risk candidates found")
                else:
                    results.append("Tracker data not available")
            except Exception as e:
                results.append(f"Error retrieving high-risk candidates: {str(e)}")
        
        # 2. Compliance issues / missing documents
        if "compliance" in question_lower or "comply" in question_lower or "missing" in question_lower or "incomplete" in question_lower:
            results.append("\n=== COMPLIANCE & MISSING INFORMATION ===")
            try:
                if self.toolkit.structured_queryer and self.toolkit.structured_queryer.tracker_df is not None:
                    df = self.toolkit.structured_queryer.tracker_df
                    results.append("Candidate Screening Status:")
                    for _, row in df.iterrows():
                        missing = []
                        if not row['identity_complete']:
                            missing.append("Identity")
                        if not row['rtw_complete']:
                            missing.append("Right-to-Work")
                        if not row['employment_complete']:
                            missing.append("Employment History")
                        if not row['criminality_complete']:
                            missing.append("Criminality")
                        
                        missing_str = ", ".join(missing) if missing else "All complete"
                        results.append(f"{row['candidate_id']}: Missing checks: {missing_str}")
            except Exception as e:
                results.append(f"Error: {str(e)}")
        
        # 3. Contradictions
        if "contradict" in question_lower or "discrepanc" in question_lower or "conflict" in question_lower:
            results.append("\n=== CONTRADICTIONS DETECTED ===")
            try:
                if self.toolkit.structured_queryer and self.toolkit.structured_queryer.tracker_df is not None:
                    df = self.toolkit.structured_queryer.tracker_df
                    for _, row in df.iterrows():
                        # Check for contradictions: status vs ready_to_join
                        if row['status_tracker'] == 'Clear' and not row['ready_to_join']:
                            results.append(f"{row['candidate_id']}: Status says 'Clear' but not marked Ready-to-Join")
                        elif row['status_tracker'] != 'Clear' and row['ready_to_join']:
                            results.append(f"{row['candidate_id']}: Status says '{row['status_tracker']}' but marked Ready-to-Join")
                    if len(results) == 1:  # Only header
                        results.append("No obvious contradictions found between status and ready-to-join flags")
            except Exception as e:
                results.append(f"Error: {str(e)}")
        
        # 4. Specific candidate
        if "cand-" in question_lower or "candidate" in question_lower:
            for cand_id in ["CAND-101", "CAND-102", "CAND-103", "CAND-104", "CAND-105", "CAND-106"]:
                if cand_id.lower() in question_lower:
                    results.append(f"\n=== {cand_id} DETAILS ===")
                    try:
                        if self.toolkit.structured_queryer and self.toolkit.structured_queryer.tracker_df is not None:
                            df = self.toolkit.structured_queryer.tracker_df
                            candidate = df[df['candidate_id'] == cand_id]
                            if not candidate.empty:
                                row = candidate.iloc[0]
                                results.append(f"Name: {row['candidate_name']}")
                                results.append(f"Role: {row['role_code']}")
                                results.append(f"Status: {row['status_tracker']}")
                                results.append(f"Risk Level: {row['risk_level']}")
                                results.append(f"Identity Complete: {row['identity_complete']}")
                                results.append(f"Right-to-Work Complete: {row['rtw_complete']}")
                                results.append(f"Employment Complete: {row['employment_complete']}")
                                results.append(f"Criminality Complete: {row['criminality_complete']}")
                                results.append(f"Ready to Join: {row['ready_to_join']}")
                                results.append(f"Notes: {row['notes']}")
                    except Exception as e:
                        results.append(f"Error: {str(e)}")
                    break
        
        # 5. Policy lookup
        if "policy" in question_lower or "requirement" in question_lower or "procedure" in question_lower:
            results.append("\n=== POLICY INFORMATION ===")
            try:
                policy_results = self.toolkit.policy_lookup(question)
                if "results" in policy_results:
                    for i, res in enumerate(policy_results["results"][:2], 1):
                        results.append(f"Result {i}: {res['text']}")
                        results.append(f"Source: {res['source']}")
            except Exception as e:
                results.append(f"Error: {str(e)}")
        
        # 6. General candidate status if no specific match
        if not results or len(results) < 2:
            results.append("=== ALL CANDIDATES STATUS ===")
            try:
                if self.toolkit.structured_queryer and self.toolkit.structured_queryer.tracker_df is not None:
                    df = self.toolkit.structured_queryer.tracker_df
                    for _, row in df.iterrows():
                        results.append(f"{row['candidate_id']}: {row['candidate_name']} - Status: {row['status_tracker']}, Risk: {row['risk_level']}")
            except Exception as e:
                results.append(f"Error: {str(e)}")
        
        return "\n".join(results)
    
    def _get_tools_context(self) -> str:
        """Generate a text description of available tools"""
        tools_text = ""
        for tool in self.toolkit.get_tools():
            tools_text += f"- {tool['name']}: {tool['description']}\n"
        return tools_text
    
    def _get_default_system_prompt(self) -> str:
        return """You are an expert BPSS (Baseline Personnel Security Standard) screening analyst. 
Your role is to help analyze and answer questions about candidate screening cases using available tools.

You have access to:
1. Policy documents (BPSS screening policies and SOPs)
2. Candidate evidence documents (candidate packs, analyst notes)
3. Structured screening data (candidate tracker, document inventory)

When answering questions:
- Use the available tools to search for relevant information
- Always cite your sources with specific documents or data fields
- Identify contradictions between different sources
- Flag missing information that affects screening decisions
- Provide clear, evidence-based answers
- Consider policy requirements when making assessments

Be thorough in your research, using multiple tools to cross-check information.
Always ground your answers in specific evidence."""
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
    
    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history"""
        return self.conversation_history
