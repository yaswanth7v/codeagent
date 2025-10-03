"""
Hybrid Retrieval System combining Knowledge Graph and FAISS Vector Search
- KG: Structured, relationship-aware retrieval with n-hop traversal
- FAISS: Chunk-based semantic search respecting code boundaries
"""

import faiss
import pickle
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
from sentence_transformers import SentenceTransformer
import json

@dataclass
class CodeChunk:
    """Represents a semantically meaningful chunk of code"""
    chunk_id: str
    content: str
    chunk_type: str  # 'class', 'function', 'method', 'file_section', 'mixed'
    file_path: str
    start_line: int
    end_line: int
    element_names: List[str]  # Names of classes/functions in this chunk
    parent_context: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BoundaryAwareChunker:
    """
    Creates chunks that respect code element boundaries
    Keeps classes, functions, and methods as coherent units
    """
    
    def __init__(self, max_chunk_size: int = 1500, overlap: int = 100):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
    
    def chunk_from_knowledge_graph(self, kg) -> List[CodeChunk]:
        """
        Create boundary-aware chunks from knowledge graph elements
        
        Strategy:
        1. Each class = 1 chunk (with all its methods)
        2. Each standalone function = 1 chunk
        3. If class too large, split by methods
        4. Group small functions together
        """
        chunks = []
        chunk_id = 0
        
        # Group elements by file
        files_map = {}
        for elem_id, elem in kg.elements.items():
            if elem.file_path not in files_map:
                files_map[elem.file_path] = []
            files_map[elem.file_path].append((elem_id, elem))
        
        for file_path, elements in files_map.items():
            # Process classes (with their methods)
            classes = [e for _, e in elements if e.type == 'class']
            for class_elem in classes:
                chunk = self._create_class_chunk(class_elem, kg, chunk_id)
                if chunk:
                    chunks.append(chunk)
                    chunk_id += 1
            
            # Process standalone functions
            functions = [e for _, e in elements if e.type == 'function']
            if functions:
                func_chunks = self._group_functions_into_chunks(functions, file_path, chunk_id)
                chunks.extend(func_chunks)
                chunk_id += len(func_chunks)
            
            # Process file-level content (imports, module docstring, etc.)
            file_elem = next((e for _, e in elements if e.type == 'file'), None)
            if file_elem and file_elem.docstring:
                chunk = CodeChunk(
                    chunk_id=f"chunk_{chunk_id}",
                    content=self._create_file_header_content(file_elem),
                    chunk_type='file_section',
                    file_path=file_path,
                    start_line=1,
                    end_line=50,
                    element_names=['module'],
                    metadata={'file_name': file_elem.name}
                )
                chunks.append(chunk)
                chunk_id += 1
        
        return chunks
    
    def _create_class_chunk(self, class_elem, kg, chunk_id: int) -> Optional[CodeChunk]:
        """Create a chunk for a class with all its context"""
        if not class_elem.full_context:
            return None
        
        content = class_elem.full_context
        
        # If class is too large, it's already been split in full_context
        # But we keep it as one logical unit
        if len(content) > self.max_chunk_size * 2:
            # For very large classes, create summary chunk
            content = self._create_class_summary(class_elem, kg)
        
        return CodeChunk(
            chunk_id=f"chunk_{chunk_id}",
            content=content,
            chunk_type='class',
            file_path=class_elem.file_path,
            start_line=class_elem.line_number,
            end_line=class_elem.line_number + len(class_elem.source_code.split('\n')),
            element_names=[class_elem.name],
            parent_context=None,
            metadata={
                'class_name': class_elem.name,
                'num_methods': class_elem.metadata.get('num_methods', 0),
                'docstring': class_elem.docstring
            }
        )
    
    def _create_class_summary(self, class_elem, kg) -> str:
        """Create a summary for very large classes"""
        summary = f"# Class: {class_elem.name}\n\n"
        
        if class_elem.docstring:
            summary += f'"""{class_elem.docstring}"""\n\n'
        
        summary += f"## Overview:\n"
        summary += f"- Methods: {class_elem.metadata.get('num_methods', 0)}\n"
        summary += f"- Attributes: {class_elem.metadata.get('num_attributes', 0)}\n\n"
        
        if class_elem.metadata.get('method_names'):
            summary += f"## Methods:\n"
            for method_name in class_elem.metadata['method_names']:
                # Find method and add signature
                method_elem = next((e for e in kg.elements.values() 
                                  if e.type == 'method' and e.name == method_name 
                                  and e.parent == class_elem.name), None)
                if method_elem:
                    summary += f"- {method_elem.signature or f'def {method_name}(...)'}\n"
                    if method_elem.docstring:
                        summary += f"  {method_elem.docstring[:100]}...\n"
        
        return summary
    
    def _group_functions_into_chunks(self, functions: List, file_path: str, 
                                     start_chunk_id: int) -> List[CodeChunk]:
        """Group standalone functions into reasonably-sized chunks"""
        chunks = []
        current_chunk_content = []
        current_chunk_names = []
        current_size = 0
        chunk_id = start_chunk_id
        
        for func_elem in functions:
            func_content = func_elem.full_context or func_elem.source_code
            func_size = len(func_content)
            
            # If single function is too large, make it its own chunk
            if func_size > self.max_chunk_size:
                if current_chunk_content:
                    # Save accumulated chunk
                    chunks.append(self._create_function_group_chunk(
                        current_chunk_content, current_chunk_names, file_path, chunk_id
                    ))
                    chunk_id += 1
                    current_chunk_content = []
                    current_chunk_names = []
                    current_size = 0
                
                # Add large function as its own chunk
                chunks.append(CodeChunk(
                    chunk_id=f"chunk_{chunk_id}",
                    content=func_content,
                    chunk_type='function',
                    file_path=file_path,
                    start_line=func_elem.line_number,
                    end_line=func_elem.line_number + len(func_elem.source_code.split('\n')),
                    element_names=[func_elem.name],
                    metadata={'function_name': func_elem.name, 'docstring': func_elem.docstring}
                ))
                chunk_id += 1
            
            # If adding this function exceeds max size, start new chunk
            elif current_size + func_size > self.max_chunk_size:
                if current_chunk_content:
                    chunks.append(self._create_function_group_chunk(
                        current_chunk_content, current_chunk_names, file_path, chunk_id
                    ))
                    chunk_id += 1
                
                current_chunk_content = [func_content]
                current_chunk_names = [func_elem.name]
                current_size = func_size
            
            # Add to current chunk
            else:
                current_chunk_content.append(func_content)
                current_chunk_names.append(func_elem.name)
                current_size += func_size
        
        # Add remaining chunk
        if current_chunk_content:
            chunks.append(self._create_function_group_chunk(
                current_chunk_content, current_chunk_names, file_path, chunk_id
            ))
        
        return chunks
    
    def _create_function_group_chunk(self, contents: List[str], names: List[str], 
                                    file_path: str, chunk_id: int) -> CodeChunk:
        """Create a chunk from grouped functions"""
        combined_content = "\n\n".join(contents)
        
        return CodeChunk(
            chunk_id=f"chunk_{chunk_id}",
            content=combined_content,
            chunk_type='function' if len(names) == 1 else 'mixed',
            file_path=file_path,
            start_line=0,
            end_line=0,
            element_names=names,
            metadata={'function_names': names}
        )
    
    def _create_file_header_content(self, file_elem) -> str:
        """Create content for file-level information"""
        content = f"# File: {file_elem.name}\n\n"
        
        if file_elem.docstring:
            content += f'"""{file_elem.docstring}"""\n\n'
        
        content += f"Summary: {file_elem.summary}\n"
        
        return content


class FAISSVectorStore:
    """
    FAISS-based vector store for code chunks
    """
    
    def __init__(self, encoder_model: str = "all-MiniLM-L6-v2", dimension: int = 384):
        self.encoder = SentenceTransformer(encoder_model)
        self.dimension = dimension
        self.index = None
        self.chunks: List[CodeChunk] = []
        self.chunk_id_to_idx: Dict[str, int] = {}
    
    def build_index(self, chunks: List[CodeChunk]) -> None:
        """Build FAISS index from chunks"""
        print(f"Building FAISS index for {len(chunks)} chunks...")
        
        self.chunks = chunks
        texts = [chunk.content for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.encoder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        embeddings = embeddings.astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        self.index.add(embeddings)
        
        # Build mapping
        for idx, chunk in enumerate(chunks):
            self.chunk_id_to_idx[chunk.chunk_id] = idx
        
        print(f"FAISS index built with {self.index.ntotal} vectors")
    
    def search(self, query: str, top_k: int = 5, 
              filter_types: Optional[List[str]] = None) -> List[Tuple[CodeChunk, float]]:
        """Search for relevant chunks"""
        if self.index is None:
            return []
        
        # Encode query
        query_embedding = self.encoder.encode([query], convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, min(top_k * 2, len(self.chunks)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                
                # Apply type filter if specified
                if filter_types and chunk.chunk_type not in filter_types:
                    continue
                
                results.append((chunk, float(score)))
                
                if len(results) >= top_k:
                    break
        
        return results
    
    def save(self, filepath: str) -> None:
        """Save FAISS index and chunks"""
        base_path = Path(filepath).with_suffix('')
        
        # Save FAISS index
        faiss.write_index(self.index, f"{base_path}.faiss")
        
        # Save chunks and metadata
        metadata = {
            'chunks': [
                {
                    'chunk_id': c.chunk_id,
                    'content': c.content,
                    'chunk_type': c.chunk_type,
                    'file_path': c.file_path,
                    'start_line': c.start_line,
                    'end_line': c.end_line,
                    'element_names': c.element_names,
                    'parent_context': c.parent_context,
                    'metadata': c.metadata
                }
                for c in self.chunks
            ],
            'chunk_id_to_idx': self.chunk_id_to_idx,
            'dimension': self.dimension
        }
        
        with open(f"{base_path}_metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"FAISS index saved to {base_path}.faiss")
    
    @staticmethod
    def load(filepath: str, encoder_model: str = "all-MiniLM-L6-v2") -> 'FAISSVectorStore':
        """Load FAISS index and chunks"""
        base_path = Path(filepath).with_suffix('')
        
        # Load FAISS index
        index = faiss.read_index(f"{base_path}.faiss")
        
        # Load metadata
        with open(f"{base_path}_metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
        
        # Reconstruct chunks
        chunks = [
            CodeChunk(
                chunk_id=c['chunk_id'],
                content=c['content'],
                chunk_type=c['chunk_type'],
                file_path=c['file_path'],
                start_line=c['start_line'],
                end_line=c['end_line'],
                element_names=c['element_names'],
                parent_context=c.get('parent_context'),
                metadata=c.get('metadata')
            )
            for c in metadata['chunks']
        ]
        
        # Create vector store
        vector_store = FAISSVectorStore(encoder_model, metadata['dimension'])
        vector_store.index = index
        vector_store.chunks = chunks
        vector_store.chunk_id_to_idx = metadata['chunk_id_to_idx']
        
        print(f"FAISS index loaded from {base_path}.faiss")
        return vector_store


class HybridRetriever:
    """
    Hybrid retrieval combining Knowledge Graph and FAISS
    
    Strategy:
    1. Query both KG and FAISS in parallel
    2. Merge and deduplicate results
    3. Rerank by relevance
    4. Return enriched context
    """
    
    def __init__(self, kg, vector_store: FAISSVectorStore):
        self.kg = kg
        self.vector_store = vector_store
        self.encoder = vector_store.encoder
    
    def retrieve(self, query: str, 
                kg_top_k: int = 5,
                faiss_top_k: int = 5,
                n_hops: int = 2,
                final_top_k: int = 10,
                kg_weight: float = 0.5,
                faiss_weight: float = 0.5) -> Dict[str, Any]:
        """
        Perform hybrid retrieval
        
        Args:
            query: Natural language query
            kg_top_k: Top-k for KG semantic search
            faiss_top_k: Top-k for FAISS search
            n_hops: Number of hops for KG expansion
            final_top_k: Final number of results
            kg_weight: Weight for KG scores
            faiss_weight: Weight for FAISS scores
        
        Returns:
            Dictionary with merged results
        """
        print(f"\nHybrid Retrieval for: '{query}'")
        print("="*60)
        
        # 1. KG Retrieval
        print("Retrieving from Knowledge Graph...")
        kg_results = self.kg.hybrid_retrieve(
            query_text=query,
            semantic_top_k=kg_top_k,
            n_hops=n_hops,
            final_top_k=kg_top_k
        )
        
        # 2. FAISS Retrieval
        print("Retrieving from FAISS vector store...")
        faiss_results = self.vector_store.search(query, top_k=faiss_top_k)
        
        # 3. Merge results
        print("Merging and deduplicating results...")
        merged_results = self._merge_results(
            kg_results, faiss_results, query, kg_weight, faiss_weight
        )
        
        # 4. Rerank and select top_k
        top_results = merged_results[:final_top_k]
        
        # 5. Enrich context
        enriched_results = self._enrich_results(top_results)
        
        print(f"Final results: {len(enriched_results)} items")
        print("="*60)
        
        return {
            'results': enriched_results,
            'kg_count': len(kg_results.get('nodes', [])),
            'faiss_count': len(faiss_results),
            'merged_count': len(merged_results),
            'final_count': len(enriched_results)
        }
    
    def _merge_results(self, kg_results: Dict, faiss_results: List, 
                      query: str, kg_weight: float, faiss_weight: float) -> List[Dict]:
        """
        Merge and deduplicate results from KG and FAISS
        """
        merged = {}
        
        # Add KG results
        for node in kg_results.get('nodes', []):
            key = f"{node['file_path']}:{node['name']}"
            merged[key] = {
                'source': 'kg',
                'type': node['type'],
                'name': node['name'],
                'file_path': node['file_path'],
                'content': node.get('full_context') or node.get('source_code', ''),
                'summary': node.get('summary', ''),
                'kg_score': 1.0,  # KG results are already filtered
                'faiss_score': 0.0,
                'metadata': node.get('metadata', {})
            }
        
        # Add FAISS results
        for chunk, score in faiss_results:
            # Try to match with KG results
            matched = False
            for elem_name in chunk.element_names:
                key = f"{chunk.file_path}:{elem_name}"
                if key in merged:
                    # Update existing entry
                    merged[key]['faiss_score'] = score
                    merged[key]['source'] = 'both'
                    matched = True
                    break
            
            if not matched:
                # Add as new entry
                key = f"{chunk.file_path}:{chunk.chunk_id}"
                merged[key] = {
                    'source': 'faiss',
                    'type': chunk.chunk_type,
                    'name': ', '.join(chunk.element_names),
                    'file_path': chunk.file_path,
                    'content': chunk.content,
                    'summary': f"{chunk.chunk_type}: {', '.join(chunk.element_names)}",
                    'kg_score': 0.0,
                    'faiss_score': score,
                    'metadata': chunk.metadata or {}
                }
        
        # Calculate combined scores and sort
        results = []
        for key, result in merged.items():
            combined_score = (kg_weight * result['kg_score'] + 
                            faiss_weight * result['faiss_score'])
            result['combined_score'] = combined_score
            results.append(result)
        
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return results
    
    def _enrich_results(self, results: List[Dict]) -> List[Dict]:
        """
        Enrich results with additional context
        """
        enriched = []
        
        for result in results:
            enriched_result = result.copy()
            
            # Add line numbers if available
            if result['source'] in ['kg', 'both']:
                # Try to find element in KG for additional metadata
                for elem in self.kg.elements.values():
                    if (elem.name == result['name'] or result['name'] in elem.name) and \
                       elem.file_path == result['file_path']:
                        enriched_result['line_number'] = elem.line_number
                        enriched_result['docstring'] = elem.docstring
                        enriched_result['signature'] = elem.signature
                        break
            
            enriched.append(enriched_result)
        
        return enriched
    
    def format_context_for_llm(self, retrieval_results: Dict) -> str:
        """
        Format retrieval results as context for LLM
        """
        context = "# Retrieved Code Context\n\n"
        
        for i, result in enumerate(retrieval_results['results'], 1):
            context += f"## Result {i} (Score: {result['combined_score']:.3f})\n"
            context += f"**Source:** {result['source']} | "
            context += f"**Type:** {result['type']} | "
            context += f"**Name:** {result['name']}\n"
            context += f"**File:** {result['file_path']}\n\n"
            
            if result.get('summary'):
                context += f"**Summary:** {result['summary']}\n\n"
            
            context += f"```python\n{result['content'][:1000]}\n```\n\n"
            context += "-" * 60 + "\n\n"
        
        return context


def build_hybrid_system(repo_path: str = None, 
                       kg_path: str = "my_kg.pkl.gz",
                       faiss_path: str = "faiss_index",
                       force_rebuild: bool = False) -> Tuple:
    """
    Build complete hybrid retrieval system
    
    Args:
        repo_path: Path to repository (only needed if building from scratch)
        kg_path: Path to saved knowledge graph file
        faiss_path: Path to FAISS index (without extension)
        force_rebuild: Force rebuild even if files exist
    
    Returns:
        (kg, vector_store, hybrid_retriever)
    """
    from knowledge_graph import KnowledgeGraphConstructor
    
    # Load or build KG
    if Path(kg_path).exists() and not force_rebuild:
        print(f"Loading existing knowledge graph from {kg_path}...")
        kg = KnowledgeGraphConstructor.load(kg_path)
        print(f"  Loaded {len(kg.elements)} nodes, {len(kg.relationships)} relationships")
    else:
        if repo_path is None:
            raise ValueError("repo_path is required when KG file doesn't exist")
        
        print(f"Building knowledge graph from {repo_path}...")
        from knowledge_graph import create_knowledge_graph_from_repo
        kg = create_knowledge_graph_from_repo(repo_path)
        kg.save(kg_path)
        print(f"  KG saved to {kg_path}")
    
    # Load or build FAISS index
    if Path(f"{faiss_path}.faiss").exists() and not force_rebuild:
        print(f"Loading existing FAISS index from {faiss_path}...")
        vector_store = FAISSVectorStore.load(faiss_path)
        print(f"  Loaded {len(vector_store.chunks)} chunks")
    else:
        print("Building FAISS index from knowledge graph...")
        chunker = BoundaryAwareChunker(max_chunk_size=1500, overlap=100)
        chunks = chunker.chunk_from_knowledge_graph(kg)
        print(f"  Created {len(chunks)} boundary-aware chunks")
        
        vector_store = FAISSVectorStore()
        vector_store.build_index(chunks)
        vector_store.save(faiss_path)
        print(f"  FAISS index saved to {faiss_path}")
    
    # Create hybrid retriever
    hybrid_retriever = HybridRetriever(kg, vector_store)
    print("\nHybrid retrieval system ready!")
    
    return kg, vector_store, hybrid_retriever


# Example usage
if __name__ == "__main__":
    # Build the system
    kg, vector_store, retriever = build_hybrid_system(
        repo_path="./codebase",
        kg_path="my_kg.pkl.gz",
        faiss_path="faiss_index"
    )
    
    # Perform hybrid retrieval
    query = "How does the authentication system work?"
    results = retriever.retrieve(
        query=query,
        kg_top_k=5,
        faiss_top_k=5,
        n_hops=2,
        final_top_k=10,
        kg_weight=0.5,
        faiss_weight=0.5
    )
    
    # Format for LLM
    context = retriever.format_context_for_llm(results)
    print("\n" + context)
    
    # Statistics
    print(f"\nRetrieval Statistics:")
    print(f"  KG results: {results['kg_count']}")
    print(f"  FAISS results: {results['faiss_count']}")
    print(f"  Merged results: {results['merged_count']}")
    print(f"  Final results: {results['final_count']}")
