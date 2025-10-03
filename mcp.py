"""
FastMCP-based AI Agent for Error Classification and Code Debugging
Uses Ollama LLMs and Hybrid Retrieval for intelligent debugging
"""

import asyncio
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass
import json
from mcp.server.fastmcp import FastMCP
import httpx
from datetime import datetime

# Assuming hybrid retrieval system is available
from hybrid_retrieval_chroma import build_hybrid_system, HybridRetriever


class ErrorType(Enum):
    """Classification of error types"""
    DEBUG = "debug"  # Code logic errors, bugs
    ENVIRONMENT = "environment"  # Missing dependencies, config issues
    AUTHENTICATION = "authentication"  # Auth/permission errors
    NETWORK = "network"  # Connection, timeout errors
    OTHER = "other"  # Unclassified errors


@dataclass
class ErrorReport:
    """Structured error report"""
    error_message: str
    error_type: str
    stack_trace: Optional[str] = None
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    timestamp: str = None
    context: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class DebugSolution:
    """Debugging solution with code fixes"""
    error_classification: str
    confidence: float
    explanation: str
    relevant_code: Optional[List[Dict]] = None
    suggested_fixes: Optional[List[str]] = None
    additional_steps: Optional[List[str]] = None


class OllamaClient:
    """Client for interacting with Ollama LLM"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2"):
        self.base_url = base_url
        self.model = model
        self.client = httpx.AsyncClient(timeout=120.0)
    
    async def generate(self, prompt: str, system: str = None, 
                      temperature: float = 0.1) -> str:
        """Generate response from Ollama"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        
        if system:
            payload["system"] = system
        
        try:
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except Exception as e:
            return f"Error calling Ollama: {str(e)}"
    
    async def chat(self, messages: List[Dict[str, str]], 
                   temperature: float = 0.1) -> str:
        """Chat with Ollama using conversation history"""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        
        try:
            response = await self.client.post(
                f"{self.base_url}/api/chat",
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            return result.get("message", {}).get("content", "")
        except Exception as e:
            return f"Error calling Ollama: {str(e)}"
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()


class ErrorClassifier:
    """Classifies errors using Ollama LLM"""
    
    def __init__(self, ollama_client: OllamaClient):
        self.ollama = ollama_client
    
    async def classify_error(self, error_report: ErrorReport) -> Dict[str, Any]:
        """Classify error into categories"""
        
        classification_prompt = f"""You are an expert error classifier. Analyze the following error and classify it into ONE of these categories:

1. DEBUG - Code logic errors, bugs, incorrect implementations, runtime errors in code
2. ENVIRONMENT - Missing dependencies, configuration issues, package installation problems
3. AUTHENTICATION - Authentication failures, permission denied, token/credential issues
4. NETWORK - Connection errors, timeouts, DNS issues, API unavailability
5. OTHER - Errors that don't fit the above categories

Error Details:
Error Message: {error_report.error_message}
{f"Stack Trace: {error_report.stack_trace}" if error_report.stack_trace else ""}
{f"File: {error_report.file_path}" if error_report.file_path else ""}
{f"Line: {error_report.line_number}" if error_report.line_number else ""}
{f"Context: {error_report.context}" if error_report.context else ""}

Respond ONLY with a valid JSON object in this exact format:
{{
    "classification": "debug|environment|authentication|network|other",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation",
    "key_indicators": ["indicator1", "indicator2"]
}}"""

        system_prompt = "You are an expert error classifier. Always respond with valid JSON only."
        
        response = await self.ollama.generate(
            prompt=classification_prompt,
            system=system_prompt,
            temperature=0.1
        )
        
        try:
            # Extract JSON from response
            response = response.strip()
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            
            classification = json.loads(response)
            return classification
        except json.JSONDecodeError:
            # Fallback classification
            return {
                "classification": "other",
                "confidence": 0.5,
                "reasoning": "Failed to parse classification response",
                "key_indicators": []
            }


class DebugAgent:
    """Main agent that handles error classification and debugging"""
    
    def __init__(self, ollama_client: OllamaClient, 
                 hybrid_retriever: Optional[HybridRetriever] = None):
        self.ollama = ollama_client
        self.classifier = ErrorClassifier(ollama_client)
        self.retriever = hybrid_retriever
    
    async def process_error(self, error_report: ErrorReport) -> DebugSolution:
        """Process error: classify and provide solution"""
        
        # Step 1: Classify the error
        classification = await self.classifier.classify_error(error_report)
        error_type = classification.get("classification", "other")
        confidence = classification.get("confidence", 0.5)
        
        print(f"\n{'='*80}")
        print(f"ERROR CLASSIFICATION")
        print(f"{'='*80}")
        print(f"Type: {error_type.upper()}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Reasoning: {classification.get('reasoning', 'N/A')}")
        print(f"Key Indicators: {', '.join(classification.get('key_indicators', []))}")
        print(f"{'='*80}\n")
        
        # Step 2: If DEBUG error, use hybrid retrieval for code fixes
        if error_type == "debug" and self.retriever:
            return await self._handle_debug_error(error_report, classification)
        else:
            return await self._handle_non_debug_error(error_report, classification)
    
    async def _handle_debug_error(self, error_report: ErrorReport, 
                                  classification: Dict) -> DebugSolution:
        """Handle debug errors with code retrieval and fixes"""
        
        print(f"\n{'='*80}")
        print(f"DEBUG ERROR DETECTED - Performing Hybrid Retrieval")
        print(f"{'='*80}\n")
        
        # Create search query from error
        search_query = self._create_search_query(error_report)
        print(f"Search Query: {search_query}\n")
        
        # Retrieve relevant code
        retrieval_results = self.retriever.retrieve(
            query=search_query,
            kg_top_k=3,
            chroma_top_k=5,
            n_hops=2,
            final_top_k=5,
            kg_weight=0.6,
            chroma_weight=0.4
        )
        
        # Format context for LLM
        code_context = self.retriever.format_context_for_llm(retrieval_results)
        
        # Generate fix suggestions using LLM
        fix_prompt = f"""You are an expert debugger. Analyze this error and the retrieved code context to provide fixes.

ERROR DETAILS:
{error_report.error_message}
{f"Stack Trace: {error_report.stack_trace}" if error_report.stack_trace else ""}
{f"File: {error_report.file_path}:{error_report.line_number}" if error_report.file_path else ""}

RETRIEVED CODE CONTEXT:
{code_context}

Provide a detailed debugging solution with:
1. Root cause analysis
2. Specific code fixes (with line numbers if possible)
3. Step-by-step implementation guide
4. Prevention tips

Be specific and actionable."""

        fix_response = await self.ollama.generate(
            prompt=fix_prompt,
            system="You are an expert software debugger. Provide clear, actionable solutions.",
            temperature=0.2
        )
        
        # Parse suggested fixes
        suggested_fixes = self._extract_fixes(fix_response)
        
        return DebugSolution(
            error_classification="debug",
            confidence=classification.get("confidence", 0.8),
            explanation=fix_response,
            relevant_code=[r for r in retrieval_results['results']],
            suggested_fixes=suggested_fixes,
            additional_steps=classification.get("key_indicators", [])
        )
    
    async def _handle_non_debug_error(self, error_report: ErrorReport, 
                                     classification: Dict) -> DebugSolution:
        """Handle non-debug errors (environment, auth, network, other)"""
        
        error_type = classification.get("classification", "other")
        
        print(f"\n{'='*80}")
        print(f"{error_type.upper()} ERROR DETECTED - Providing General Solution")
        print(f"{'='*80}\n")
        
        # Generate appropriate solution based on error type
        solution_prompt = f"""You are an expert system administrator and DevOps engineer.

ERROR TYPE: {error_type.upper()}
ERROR DETAILS:
{error_report.error_message}
{f"Context: {error_report.context}" if error_report.context else ""}

Provide a solution guide including:
1. Likely causes
2. Step-by-step resolution steps
3. Verification methods
4. Prevention tips

Be specific to {error_type} errors."""

        solution_response = await self.ollama.generate(
            prompt=solution_prompt,
            system=f"You are an expert in resolving {error_type} errors.",
            temperature=0.2
        )
        
        # Extract steps
        steps = self._extract_steps(solution_response)
        
        return DebugSolution(
            error_classification=error_type,
            confidence=classification.get("confidence", 0.7),
            explanation=solution_response,
            relevant_code=None,
            suggested_fixes=None,
            additional_steps=steps
        )
    
    def _create_search_query(self, error_report: ErrorReport) -> str:
        """Create effective search query from error report"""
        query_parts = []
        
        # Add error message keywords
        error_msg = error_report.error_message.lower()
        
        # Extract key terms
        if "exception" in error_msg or "error" in error_msg:
            # Extract exception type
            for word in error_msg.split():
                if "exception" in word or "error" in word:
                    query_parts.append(word)
        
        # Add file/function context if available
        if error_report.file_path:
            query_parts.append(error_report.file_path.split('/')[-1].replace('.py', ''))
        
        # Add context
        if error_report.context:
            query_parts.append(error_report.context[:50])
        
        return " ".join(query_parts[:5])  # Limit query length
    
    def _extract_fixes(self, response: str) -> List[str]:
        """Extract actionable fixes from LLM response"""
        fixes = []
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for numbered lists, bullet points, or "fix" keywords
            if any([
                line.startswith(('1.', '2.', '3.', '4.', '5.')),
                line.startswith(('-', '*', '•')),
                'fix:' in line.lower(),
                'change:' in line.lower(),
                'update:' in line.lower()
            ]):
                if len(line) > 10:  # Avoid empty items
                    fixes.append(line.lstrip('0123456789.-*• '))
        
        return fixes[:10]  # Limit to top 10 fixes
    
    def _extract_steps(self, response: str) -> List[str]:
        """Extract step-by-step instructions"""
        steps = []
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if any([
                line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')),
                line.startswith(('-', '*', '•')),
                line.startswith(('Step', 'STEP'))
            ]):
                if len(line) > 10:
                    steps.append(line.lstrip('0123456789.-*• '))
        
        return steps


# FastMCP Server Setup
mcp = FastMCP("Error Debug Agent")

# Global instances (will be initialized)
ollama_client: Optional[OllamaClient] = None
debug_agent: Optional[DebugAgent] = None
hybrid_retriever: Optional[HybridRetriever] = None


@mcp.tool()
async def initialize_agent(
    repo_path: str,
    kg_path: str = "my_kg.pkl.gz",
    chroma_dir: str = "./chroma_db",
    ollama_url: str = "http://localhost:11434",
    ollama_model: str = "llama3.2"
) -> str:
    """
    Initialize the debug agent with hybrid retrieval system
    
    Args:
        repo_path: Path to code repository for building knowledge graph
        kg_path: Path to save/load knowledge graph
        chroma_dir: Directory for ChromaDB persistence
        ollama_url: URL of Ollama server
        ollama_model: Ollama model name to use
    """
    global ollama_client, debug_agent, hybrid_retriever
    
    try:
        # Initialize Ollama client
        ollama_client = OllamaClient(base_url=ollama_url, model=ollama_model)
        
        # Build hybrid retrieval system
        print(f"Building hybrid retrieval system from {repo_path}...")
        kg, vector_store, retriever = build_hybrid_system(
            repo_path=repo_path,
            kg_path=kg_path,
            chroma_dir=chroma_dir,
            collection_name="code_chunks",
            force_rebuild=False
        )
        hybrid_retriever = retriever
        
        # Initialize debug agent
        debug_agent = DebugAgent(ollama_client, hybrid_retriever)
        
        return json.dumps({
            "status": "success",
            "message": "Debug agent initialized successfully",
            "kg_nodes": len(kg.elements),
            "vector_chunks": len(vector_store.chunks),
            "ollama_model": ollama_model
        })
    
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Failed to initialize agent: {str(e)}"
        })


@mcp.tool()
async def classify_error(
    error_message: str,
    stack_trace: str = None,
    file_path: str = None,
    line_number: int = None,
    context: str = None
) -> str:
    """
    Classify an error into categories: debug, environment, authentication, network, or other
    
    Args:
        error_message: The error message
        stack_trace: Stack trace if available
        file_path: File where error occurred
        line_number: Line number where error occurred
        context: Additional context about the error
    """
    global debug_agent
    
    if debug_agent is None:
        return json.dumps({
            "status": "error",
            "message": "Agent not initialized. Call initialize_agent first."
        })
    
    try:
        error_report = ErrorReport(
            error_message=error_message,
            error_type="unknown",
            stack_trace=stack_trace,
            file_path=file_path,
            line_number=line_number,
            context=context
        )
        
        classification = await debug_agent.classifier.classify_error(error_report)
        
        return json.dumps({
            "status": "success",
            "classification": classification
        })
    
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Classification failed: {str(e)}"
        })


@mcp.tool()
async def debug_error(
    error_message: str,
    stack_trace: str = None,
    file_path: str = None,
    line_number: int = None,
    context: str = None
) -> str:
    """
    Analyze and debug an error. If it's a debug error, provides code fixes using hybrid retrieval.
    
    Args:
        error_message: The error message
        stack_trace: Stack trace if available
        file_path: File where error occurred
        line_number: Line number where error occurred
        context: Additional context about the error
    """
    global debug_agent
    
    if debug_agent is None:
        return json.dumps({
            "status": "error",
            "message": "Agent not initialized. Call initialize_agent first."
        })
    
    try:
        error_report = ErrorReport(
            error_message=error_message,
            error_type="unknown",
            stack_trace=stack_trace,
            file_path=file_path,
            line_number=line_number,
            context=context
        )
        
        solution = await debug_agent.process_error(error_report)
        
        # Format response
        response = {
            "status": "success",
            "error_classification": solution.error_classification,
            "confidence": solution.confidence,
            "explanation": solution.explanation,
        }
        
        if solution.relevant_code:
            response["relevant_code_count"] = len(solution.relevant_code)
            response["relevant_files"] = list(set([
                code['file_path'] for code in solution.relevant_code
            ]))
        
        if solution.suggested_fixes:
            response["suggested_fixes"] = solution.suggested_fixes
        
        if solution.additional_steps:
            response["additional_steps"] = solution.additional_steps
        
        return json.dumps(response, indent=2)
    
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Debug failed: {str(e)}"
        })


@mcp.tool()
async def get_code_context(
    query: str,
    top_k: int = 5
) -> str:
    """
    Retrieve relevant code context using hybrid retrieval
    
    Args:
        query: Natural language query about the code
        top_k: Number of results to return
    """
    global hybrid_retriever
    
    if hybrid_retriever is None:
        return json.dumps({
            "status": "error",
            "message": "Retrieval system not initialized. Call initialize_agent first."
        })
    
    try:
        results = hybrid_retriever.retrieve(
            query=query,
            kg_top_k=top_k,
            chroma_top_k=top_k,
            final_top_k=top_k
        )
        
        formatted_results = []
        for result in results['results']:
            formatted_results.append({
                "name": result['name'],
                "type": result['type'],
                "file": result['file_path'],
                "score": result['combined_score'],
                "summary": result.get('summary', ''),
                "content_preview": result['content'][:300] + "..."
            })
        
        return json.dumps({
            "status": "success",
            "results": formatted_results,
            "total_results": len(formatted_results)
        }, indent=2)
    
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Retrieval failed: {str(e)}"
        })


if __name__ == "__main__":
    # Run the MCP server
    mcp.run(transport="stdio")
