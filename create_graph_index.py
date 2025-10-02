import ast
import os
import json
from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
from sentence_transformers import SentenceTransformer
import numpy as np

@dataclass
class CodeElement:
    """Represents a code element extracted from AST with enhanced context for RAG"""
    name: str
    type: str  # 'class', 'method', 'function', 'attribute', 'file'
    file_path: str
    line_number: int
    docstring: Optional[str] = None
    source_code: Optional[str] = None
    parent: Optional[str] = None
    dependencies: List[str] = None
    
    # Enhanced fields for better RAG context
    full_context: Optional[str] = None  # Complete code with surrounding context
    summary: Optional[str] = None  # Human-readable summary
    signature: Optional[str] = None  # Function/method signature
    parent_context: Optional[str] = None  # Parent class/module context
    related_elements: List[str] = field(default_factory=list)  # IDs of related elements
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

class KnowledgeGraphConstructor:
    """
    Constructs an enhanced knowledge graph optimized for RAG retrieval
    """
    
    def __init__(self, encoder_model: str = "all-MiniLM-L6-v2"):
        try:
            self.encoder = SentenceTransformer(encoder_model)
            print(f"Loaded embedding model: {encoder_model}")
        except Exception as e:
            print(f"Warning: Could not load embedding model {encoder_model}: {e}")
            self.encoder = None
        
        self.elements: Dict[str, CodeElement] = {}
        self.relationships: List[Dict[str, Any]] = []
        self.schema = self._define_schema()
        self.imports_map: Dict[str, Dict[str, str]] = {}
        self.repo_path: Optional[Path] = None
        self.file_contents: Dict[str, str] = {}  # Cache file contents
        
    def _define_schema(self) -> Dict[str, Any]:
        """Define the knowledge graph schema"""
        return {
            "node_types": {
                "File": {"properties": ["path", "name", "description", "full_context", "summary"]},
                "Class": {"properties": ["name", "docstring", "file_path", "signature", "full_context", "summary"]},
                "Method": {"properties": ["name", "docstring", "class_name", "signature", "full_context", "parent_context"]},
                "Function": {"properties": ["name", "docstring", "signature", "full_context", "summary"]},
                "Attribute": {"properties": ["name", "class_name", "parent_context"]},
                "Generated_Description": {"properties": ["description", "embedding"]}
            },
            "relation_types": [
                "defines_class", "defines_function", "has_method", 
                "used_in", "has_attribute", "has_description", "imports",
                "calls", "inherits_from", "related_to"
            ]
        }
    
    def parse_repository(self, repo_path: str, exclude_patterns: List[str] = None) -> None:
        """Parse the entire repository with enhanced context extraction"""
        if exclude_patterns is None:
            exclude_patterns = ['__pycache__', '.git', '.pytest_cache', 'test_', 'tests']
            
        self.repo_path = Path(repo_path)
        
        # First pass: parse all files and collect elements
        python_files = self._find_python_files(self.repo_path, exclude_patterns)
        for py_file in python_files:
            try:
                self._parse_file(py_file)
            except Exception as e:
                print(f"Error parsing {py_file}: {e}")
        
        # Second pass: enhance context for all elements
        print("Enhancing context for RAG...")
        self._enhance_all_contexts()
        
        # Third pass: resolve imports and create relationships
        print("Resolving imports and cross-file references...")
        for py_file in python_files:
            try:
                self._resolve_imports_for_file(py_file)
            except Exception as e:
                print(f"Error resolving imports for {py_file}: {e}")
    
    def _find_python_files(self, repo_path: Path, exclude_patterns: List[str]) -> List[Path]:
        """Find all Python files in the repository"""
        python_files = []
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if not any(pattern in d for pattern in exclude_patterns)]
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    if not any(pattern in str(file_path) for pattern in exclude_patterns):
                        python_files.append(file_path)
        return python_files
    
    def _parse_file(self, file_path: Path) -> None:
        """Parse a single Python file using AST"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            # Cache file contents for context enhancement
            self.file_contents[str(file_path)] = source_code
            
            tree = ast.parse(source_code)
            
            # Extract module-level docstring and summary
            module_doc = ast.get_docstring(tree)
            
            # Create file element with enhanced context
            file_element = CodeElement(
                name=file_path.name,
                type="file",
                file_path=str(file_path),
                line_number=1,
                docstring=module_doc,
                source_code=source_code,
                full_context=source_code,  # Full file content
                summary=self._create_file_summary(file_path.name, module_doc, source_code),
                metadata={
                    "file_size": len(source_code),
                    "num_lines": len(source_code.split('\n')),
                    "relative_path": str(file_path.relative_to(self.repo_path)) if self.repo_path else str(file_path)
                }
            )
            
            file_id = self._generate_id(file_element)
            self.elements[file_id] = file_element
            
            # Parse file contents
            self._parse_ast_node(tree, str(file_path), source_code.split('\n'))
            
        except Exception as e:
            print(f"Error parsing file {file_path}: {e}")
    
    def _create_file_summary(self, filename: str, docstring: Optional[str], source_code: str) -> str:
        """Create a summary of the file for better RAG context"""
        lines = source_code.split('\n')
        num_lines = len(lines)
        
        # Count different element types
        num_classes = source_code.count('class ')
        num_functions = source_code.count('def ')
        
        summary = f"File: {filename} ({num_lines} lines)"
        if docstring:
            summary += f"\nDescription: {docstring[:200]}"
        summary += f"\nContains: {num_classes} classes, {num_functions} functions/methods"
        
        return summary
    
    def _enhance_all_contexts(self) -> None:
        """Enhance context for all elements after initial parsing"""
        for element_id, element in list(self.elements.items()):
            if element.type == "class":
                self._enhance_class_context(element_id, element)
            elif element.type in ["method", "function"]:
                self._enhance_function_context(element_id, element)
            elif element.type == "attribute":
                self._enhance_attribute_context(element_id, element)
    
    def _enhance_class_context(self, class_id: str, class_element: CodeElement) -> None:
        """Enhance class node with complete context including all methods"""
        # Find all methods belonging to this class
        class_methods = []
        class_attributes = []
        
        for elem_id, elem in self.elements.items():
            if elem.parent == class_element.name and elem.file_path == class_element.file_path:
                if elem.type == "method":
                    class_methods.append(elem)
                elif elem.type == "attribute":
                    class_attributes.append(elem)
        
        # Build full context
        full_context = f"# Class: {class_element.name}\n"
        if class_element.docstring:
            full_context += f'"""{class_element.docstring}"""\n\n'
        
        full_context += f"## Class Definition:\n{class_element.source_code}\n\n"
        
        if class_attributes:
            full_context += f"## Attributes ({len(class_attributes)}):\n"
            for attr in class_attributes:
                full_context += f"- {attr.name}\n"
            full_context += "\n"
        
        if class_methods:
            full_context += f"## Methods ({len(class_methods)}):\n"
            for method in class_methods:
                full_context += f"\n### {method.name}:\n"
                if method.docstring:
                    full_context += f'"""{method.docstring}"""\n'
                full_context += f"{method.source_code}\n"
        
        # Create summary
        summary = f"Class '{class_element.name}' with {len(class_methods)} methods and {len(class_attributes)} attributes"
        if class_element.docstring:
            summary += f". {class_element.docstring[:150]}"
        
        # Update element
        class_element.full_context = full_context
        class_element.summary = summary
        class_element.related_elements = [self._generate_id(m) for m in class_methods]
        class_element.metadata = {
            "num_methods": len(class_methods),
            "num_attributes": len(class_attributes),
            "method_names": [m.name for m in class_methods]
        }
    
    def _enhance_function_context(self, func_id: str, func_element: CodeElement) -> None:
        """Enhance function/method with signature and parent context"""
        # Extract signature
        try:
            source_lines = func_element.source_code.split('\n')
            signature_lines = []
            for line in source_lines:
                signature_lines.append(line)
                if ':' in line and 'def ' in line:
                    break
            func_element.signature = '\n'.join(signature_lines)
        except:
            func_element.signature = f"def {func_element.name}(...)"
        
        # Build full context
        full_context = ""
        
        # Add parent context if it's a method
        if func_element.parent and func_element.type == "method":
            parent_class = self._find_class_element(func_element.file_path, func_element.parent)
            if parent_class:
                parent_context = f"# Parent Class: {parent_class.name}\n"
                if parent_class.docstring:
                    parent_context += f'"""{parent_class.docstring}"""\n\n'
                func_element.parent_context = parent_context
                full_context += parent_context
        
        # Add function itself
        full_context += f"# {'Method' if func_element.type == 'method' else 'Function'}: {func_element.name}\n"
        if func_element.docstring:
            full_context += f'"""{func_element.docstring}"""\n\n'
        full_context += func_element.source_code
        
        func_element.full_context = full_context
        
        # Create summary
        summary = f"{'Method' if func_element.type == 'method' else 'Function'} '{func_element.name}'"
        if func_element.parent:
            summary += f" in class '{func_element.parent}'"
        if func_element.docstring:
            summary += f". {func_element.docstring[:150]}"
        func_element.summary = summary
    
    def _enhance_attribute_context(self, attr_id: str, attr_element: CodeElement) -> None:
        """Enhance attribute with parent class context"""
        if attr_element.parent:
            parent_class = self._find_class_element(attr_element.file_path, attr_element.parent)
            if parent_class:
                parent_context = f"# Class: {parent_class.name}\n"
                if parent_class.docstring:
                    parent_context += f'"""{parent_class.docstring}"""\n\n'
                parent_context += f"Attribute: {attr_element.name}\n"
                parent_context += attr_element.source_code
                
                attr_element.full_context = parent_context
                attr_element.parent_context = parent_context
    
    def _parse_ast_node(self, node: ast.AST, file_path: str, source_lines: List[str], 
                       parent_class: Optional[str] = None) -> None:
        """Recursively parse AST nodes to extract code elements"""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                self._extract_class(child, file_path, source_lines)
                self._parse_ast_node(child, file_path, source_lines, child.name)
            elif isinstance(child, ast.FunctionDef):
                self._extract_function(child, file_path, source_lines, parent_class)
                self._extract_usage_relationships(child, file_path, parent_class)
            elif isinstance(child, ast.AsyncFunctionDef):
                self._extract_function(child, file_path, source_lines, parent_class)
                self._extract_usage_relationships(child, file_path, parent_class)
            elif isinstance(child, (ast.Import, ast.ImportFrom)):
                self._extract_imports(child, file_path)
            elif isinstance(child, ast.Assign) and parent_class:
                self._extract_attributes(child, file_path, parent_class)
            else:
                self._parse_ast_node(child, file_path, source_lines, parent_class)
    
    def _extract_imports(self, node: ast.AST, file_path: str) -> None:
        """Extract import statements"""
        if file_path not in self.imports_map:
            self.imports_map[file_path] = {}
        
        if isinstance(node, ast.Import):
            for alias in node.names:
                import_name = alias.asname if alias.asname else alias.name
                self.imports_map[file_path][import_name] = alias.name
                
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            for alias in node.names:
                import_name = alias.asname if alias.asname else alias.name
                full_name = f"{module_name}.{alias.name}" if module_name else alias.name
                self.imports_map[file_path][import_name] = full_name
    
    def _resolve_imports_for_file(self, file_path: Path) -> None:
        """Resolve imports and create relationships"""
        str_file_path = str(file_path)
        if str_file_path not in self.imports_map:
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            tree = ast.parse(source_code)
            self._resolve_usage_in_node(tree, str_file_path, None)
        except Exception as e:
            print(f"Error resolving imports for {file_path}: {e}")
    
    def _resolve_usage_in_node(self, node: ast.AST, file_path: str, current_element_id: Optional[str] = None) -> None:
        """Recursively resolve usage relationships"""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                class_element_id = self._find_element_id_by_name_and_line(file_path, child.name, child.lineno, "class")
                if class_element_id:
                    self._resolve_usage_in_node(child, file_path, class_element_id)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                element_type = "method" if current_element_id and self._is_class_element(current_element_id) else "function"
                func_element_id = self._find_element_id_by_name_and_line(file_path, child.name, child.lineno, element_type)
                if func_element_id:
                    self._analyze_function_usage(child, file_path, func_element_id)
                    self._resolve_usage_in_node(child, file_path, func_element_id)
            else:
                self._resolve_usage_in_node(child, file_path, current_element_id)
    
    def _analyze_function_usage(self, func_node: ast.AST, file_path: str, func_element_id: str) -> None:
        """Analyze function for usage patterns"""
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    self._create_usage_relationship_resolved(func_element_id, node.func.id, file_path)
                elif isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        base_name = node.func.value.id
                        attr_name = node.func.attr
                        self._create_cross_file_relationship(func_element_id, base_name, attr_name, file_path)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                self._create_usage_relationship_resolved(func_element_id, node.id, file_path)
            elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                if isinstance(node.value, ast.Name):
                    base_name = node.value.id
                    attr_name = node.attr
                    self._create_cross_file_relationship(func_element_id, base_name, attr_name, file_path)
    
    # [Additional helper methods from original code would continue here...]
    # Including: _create_usage_relationship_resolved, _create_cross_file_relationship,
    # _extract_class, _extract_function, _extract_attributes, etc.
    
    def _extract_class(self, node: ast.ClassDef, file_path: str, source_lines: List[str]) -> None:
        """Extract class information"""
        class_element = CodeElement(
            name=node.name,
            type="class",
            file_path=file_path,
            line_number=node.lineno,
            docstring=ast.get_docstring(node),
            source_code=self._get_source_code(node, source_lines)
        )
        
        class_id = self._generate_id(class_element)
        self.elements[class_id] = class_element
        
        file_element = self._find_file_element(file_path)
        if file_element:
            file_id = self._generate_id(file_element)
            self._add_relationship("defines_class", file_id, class_id)
    
    def _extract_function(self, node: ast.FunctionDef, file_path: str, 
                         source_lines: List[str], parent_class: Optional[str] = None) -> None:
        """Extract function information"""
        element_type = "method" if parent_class else "function"
        
        function_element = CodeElement(
            name=node.name,
            type=element_type,
            file_path=file_path,
            line_number=node.lineno,
            docstring=ast.get_docstring(node),
            source_code=self._get_source_code(node, source_lines),
            parent=parent_class
        )
        
        function_id = self._generate_id(function_element)
        self.elements[function_id] = function_element
        
        if parent_class:
            class_element = self._find_class_element(file_path, parent_class)
            if class_element:
                class_id = self._generate_id(class_element)
                self._add_relationship("has_method", class_id, function_id)
        else:
            file_element = self._find_file_element(file_path)
            if file_element:
                file_id = self._generate_id(file_element)
                self._add_relationship("defines_function", file_id, function_id)
    
    def _extract_attributes(self, node: ast.Assign, file_path: str, class_name: str) -> None:
        """Extract class attributes"""
        for target in node.targets:
            attr_name = None
            if isinstance(target, ast.Name):
                attr_name = target.id
            elif isinstance(target, ast.Attribute):
                if isinstance(target.value, ast.Name) and target.value.id == 'self':
                    attr_name = target.attr
            
            if attr_name:
                attr_element = CodeElement(
                    name=attr_name,
                    type="attribute",
                    file_path=file_path,
                    line_number=node.lineno,
                    parent=class_name,
                    source_code=self._get_source_code(node, [])
                )
                
                attr_id = self._generate_id(attr_element)
                self.elements[attr_id] = attr_element
                
                class_element = self._find_class_element(file_path, class_name)
                if class_element:
                    class_id = self._generate_id(class_element)
                    self._add_relationship("has_attribute", class_id, attr_id)
    
    def _extract_usage_relationships(self, node: ast.AST, file_path: str, parent_class: Optional[str] = None) -> None:
        """Placeholder - handled in second pass"""
        pass
    
    def _get_source_code(self, node: ast.AST, source_lines: List[str]) -> str:
        """Extract source code for AST node"""
        try:
            if not source_lines:
                return ""
            start_line = node.lineno - 1
            end_line = getattr(node, 'end_lineno', start_line + 1)
            return '\n'.join(source_lines[start_line:end_line])
        except:
            return ""
    
    def _find_file_element(self, file_path: str) -> Optional[CodeElement]:
        """Find file element"""
        for element in self.elements.values():
            if element.type == "file" and element.file_path == file_path:
                return element
        return None
    
    def _find_class_element(self, file_path: str, class_name: str) -> Optional[CodeElement]:
        """Find class element"""
        for element in self.elements.values():
            if (element.type == "class" and 
                element.file_path == file_path and 
                element.name == class_name):
                return element
        return None
    
    def _find_element_id_by_name_and_line(self, file_path: str, name: str, line_no: int, element_type: str) -> Optional[str]:
        """Find element ID"""
        for element_id, element in self.elements.items():
            if (element.file_path == file_path and 
                element.name == name and 
                element.line_number == line_no and
                element.type == element_type):
                return element_id
        return None
    
    def _is_class_element(self, element_id: str) -> bool:
        """Check if element is a class"""
        return element_id in self.elements and self.elements[element_id].type == "class"
    
    def _create_usage_relationship_resolved(self, source_element_id: str, used_name: str, file_path: str) -> None:
        """Create usage relationship with import resolution"""
        target_element_id = self._find_local_element_by_name(file_path, used_name)
        if target_element_id:
            self._add_relationship("used_in", source_element_id, target_element_id)
            return
        
        if file_path in self.imports_map and used_name in self.imports_map[file_path]:
            imported_full_name = self.imports_map[file_path][used_name]
            target_element_id = self._find_element_by_import_path(imported_full_name)
            if target_element_id:
                self._add_relationship("used_in", source_element_id, target_element_id)
    
    def _create_cross_file_relationship(self, source_element_id: str, base_name: str, attr_name: str, file_path: str) -> None:
        """Create cross-file relationships"""
        if file_path not in self.imports_map or base_name not in self.imports_map[file_path]:
            return
        
        imported_module = self.imports_map[file_path][base_name]
        target_element_id = self._find_element_by_module_and_name(imported_module, attr_name)
        if target_element_id:
            self._add_relationship("used_in", source_element_id, target_element_id)
    
    def _find_local_element_by_name(self, file_path: str, name: str) -> Optional[str]:
        """Find local element by name"""
        for element_id, element in self.elements.items():
            if (element.file_path == file_path and 
                element.name == name and
                element.type in ['class', 'function', 'method']):
                return element_id
        return None
    
    def _find_element_by_import_path(self, import_path: str) -> Optional[str]:
        """Find element by import path"""
        # Simplified implementation
        parts = import_path.split('.')
        element_name = parts[-1]
        for element_id, element in self.elements.items():
            if element.name == element_name and element.type in ['class', 'function']:
                return element_id
        return None
    
    def _find_element_by_module_and_name(self, module_path: str, element_name: str) -> Optional[str]:
        """Find element by module and name"""
        return self._find_element_by_import_path(f"{module_path}.{element_name}")
    
    def _generate_id(self, element: CodeElement) -> str:
        """Generate unique ID"""
        normalized_path = element.file_path.replace('\\', '/')
        if element.type == "file":
            identifier = f"file:{normalized_path}"
        elif element.type == "class":
            identifier = f"class:{normalized_path}:{element.name}"
        elif element.type in ["method", "attribute"] and element.parent:
            identifier = f"{element.type}:{normalized_path}:{element.parent}:{element.name}"
        else:
            identifier = f"{element.type}:{normalized_path}:{element.name}"
        identifier += f":{element.line_number}"
        return hashlib.md5(identifier.encode()).hexdigest()
    
    def _add_relationship(self, relation_type: str, source: str, target: str) -> None:
        """Add relationship"""
        self.relationships.append({
            "type": relation_type,
            "source": source,
            "target": target
        })
    
    def generate_embeddings(self) -> None:
        """Generate embeddings for full_context field for better RAG retrieval"""
        if self.encoder is None:
            print("No embedding model available")
            return
        
        texts_to_embed = []
        element_ids = []
        
        for element_id, element in list(self.elements.items()):
            # Embed the full_context instead of just docstring
            text = element.full_context or element.summary or element.docstring or ""
            
            if text.strip():
                texts_to_embed.append(text)
                element_ids.append(element_id)
        
        if texts_to_embed:
            print(f"Generating embeddings for {len(texts_to_embed)} elements...")
            embeddings = self.encoder.encode(texts_to_embed)
            
            for i, element_id in enumerate(element_ids):
                if element_id in self.elements:
                    self.elements[element_id].dependencies = embeddings[i].tolist()
            print("Embeddings generated successfully")
    
    def export_to_neo4j_format(self, output_dir: str = "knowledge_graph_export") -> None:
        """Export with enhanced context fields"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        nodes_data = []
        for element_id, element in self.elements.items():
            node_data = {
                "id": element_id,
                "name": element.name,
                "type": element.type,
                "file_path": element.file_path.replace('\\', '/'),
                "line_number": element.line_number,
                "docstring": element.docstring,
                "source_code": element.source_code,
                "full_context": element.full_context,  # Enhanced context
                "summary": element.summary,  # Summary for quick reference
                "signature": element.signature,  # Function signatures
                "parent": element.parent,
                "parent_context": element.parent_context,  # Parent context
                "related_elements": element.related_elements,  # Related IDs
                "metadata": element.metadata,  # Additional metadata
                "embedding": element.dependencies if element.dependencies else []
            }
            nodes_data.append(node_data)
        
        with open(output_path / "nodes.json", 'w', encoding='utf-8') as f:
            json.dump(nodes_data, f, indent=2, ensure_ascii=False)
        
        # Export relationships
        with open(output_path / "relationships.json", 'w', encoding='utf-8') as f:
            json.dump(self.relationships, f, indent=2, ensure_ascii=False)
        
        # Export schema
        with open(output_path / "schema.json", 'w', encoding='utf-8') as f:
            json.dump(self.schema, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"Knowledge graph exported to {output_path}")
        print(f"{'='*60}")
        print(f"Total nodes: {len(self.elements)}")
        print(f"Total relationships: {len(self.relationships)}")
        print(f"\nEnhanced RAG fields included:")
        print(f"  ✓ full_context - Complete code with surrounding context")
        print(f"  ✓ summary - Human-readable summaries")
        print(f"  ✓ signature - Function/method signatures")
        print(f"  ✓ parent_context - Parent class/module context")
        print(f"  ✓ related_elements - IDs of related code elements")
        print(f"  ✓ metadata - Additional contextual information")
    
    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about the knowledge graph"""
        stats = {}
        for element in self.elements.values():
            stats[element.type] = stats.get(element.type, 0) + 1
        
        stats['total_nodes'] = len(self.elements)
        stats['total_relationships'] = len(self.relationships)
        
        # Relationship statistics
        rel_stats = {}
        for rel in self.relationships:
            rel_type = rel['type']
            rel_stats[rel_type] = rel_stats.get(rel_type, 0) + 1
        
        # Count enhanced context coverage
        enhanced_count = sum(1 for e in self.elements.values() if e.full_context)
        stats['elements_with_full_context'] = enhanced_count
        
        summary_count = sum(1 for e in self.elements.values() if e.summary)
        stats['elements_with_summary'] = summary_count
        
        embedded_count = sum(1 for e in self.elements.values() if e.dependencies)
        stats['elements_with_embeddings'] = embedded_count
        
        stats.update(rel_stats)
        return stats
    
    def get_element_for_rag(self, element_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a code element optimized for RAG retrieval with full context
        
        Args:
            element_id: The ID of the element to retrieve
            
        Returns:
            Dictionary with complete context for RAG or None if not found
        """
        if element_id not in self.elements:
            return None
        
        element = self.elements[element_id]
        
        # Build comprehensive context for RAG
        rag_context = {
            "id": element_id,
            "name": element.name,
            "type": element.type,
            "summary": element.summary,
            "full_context": element.full_context,
            "source_code": element.source_code,
            "docstring": element.docstring,
            "signature": element.signature,
            "file_path": element.file_path,
            "line_number": element.line_number,
            "metadata": element.metadata,
        }
        
        # Add parent context if available
        if element.parent_context:
            rag_context["parent_context"] = element.parent_context
        
        # Add related elements with their summaries
        if element.related_elements:
            related_info = []
            for rel_id in element.related_elements:
                if rel_id in self.elements:
                    rel_elem = self.elements[rel_id]
                    related_info.append({
                        "id": rel_id,
                        "name": rel_elem.name,
                        "type": rel_elem.type,
                        "summary": rel_elem.summary or f"{rel_elem.type}: {rel_elem.name}"
                    })
            rag_context["related_elements"] = related_info
        
        # Add relationships
        incoming_rels = [r for r in self.relationships if r['target'] == element_id]
        outgoing_rels = [r for r in self.relationships if r['source'] == element_id]
        
        rag_context["relationships"] = {
            "incoming": incoming_rels,
            "outgoing": outgoing_rels
        }
        
        return rag_context
    
    def search_by_embedding(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant code elements using semantic similarity
        
        Args:
            query_text: Natural language query
            top_k: Number of top results to return
            
        Returns:
            List of relevant code elements with full context
        """
        if self.encoder is None:
            print("No embedding model available for search")
            return []
        
        # Generate query embedding
        query_embedding = self.encoder.encode([query_text])[0]
        
        # Find elements with embeddings
        elements_with_embeddings = [
            (elem_id, elem) for elem_id, elem in self.elements.items()
            if elem.dependencies
        ]
        
        if not elements_with_embeddings:
            print("No elements with embeddings found")
            return []
        
        # Calculate similarities
        similarities = []
        for elem_id, elem in elements_with_embeddings:
            elem_embedding = np.array(elem.dependencies)
            similarity = np.dot(query_embedding, elem_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(elem_embedding)
            )
            similarities.append((elem_id, similarity))
        
        # Sort by similarity and get top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = similarities[:top_k]
        
        # Build results with full context
        results = []
        for elem_id, score in top_results:
            rag_context = self.get_element_for_rag(elem_id)
            if rag_context:
                rag_context['similarity_score'] = float(score)
                results.append(rag_context)
        
        return results
    
    def generate_descriptions(self, llm_generate_func=None) -> None:
        """Generate descriptions for code elements"""
        if llm_generate_func is None:
            llm_generate_func = self._simple_description_generator
        
        elements_snapshot = list(self.elements.items())
        
        for element_id, element in elements_snapshot:
            if element.source_code and element.type in ['class', 'method', 'function']:
                try:
                    description = llm_generate_func(element.source_code, element.type, element.name)
                    
                    desc_name = f"{element.name}_description_{element.line_number}"
                    desc_element = CodeElement(
                        name=desc_name,
                        type="generated_description",
                        file_path=element.file_path,
                        line_number=element.line_number,
                        docstring=description,
                        parent=element.name
                    )
                    
                    desc_id = self._generate_id(desc_element)
                    self.elements[desc_id] = desc_element
                    self._add_relationship("has_description", element_id, desc_id)
                    
                except Exception as e:
                    print(f"Error generating description for {element.name}: {e}")
    
    def _simple_description_generator(self, source_code: str, element_type: str, name: str) -> str:
        """Simple fallback description generator"""
        return f"This {element_type} named '{name}' contains {len(source_code.split())} words of code."


def create_knowledge_graph_from_repo(repo_path: str, output_dir: str = "kg_export"):
    """
    Create an enhanced knowledge graph optimized for RAG retrieval
    
    Args:
        repo_path: Path to the Python repository
        output_dir: Directory to export the knowledge graph
    
    Returns:
        KnowledgeGraphConstructor instance
    """
    print("\n" + "="*60)
    print("Creating Enhanced Knowledge Graph for RAG")
    print("="*60 + "\n")
    
    kg_constructor = KnowledgeGraphConstructor()
    
    print("📂 Parsing repository...")
    kg_constructor.parse_repository(repo_path)
    
    print("\n🔤 Generating embeddings...")
    kg_constructor.generate_embeddings()
    
    print("\n💾 Exporting knowledge graph...")
    kg_constructor.export_to_neo4j_format(output_dir)
    
    # Print comprehensive statistics
    stats = kg_constructor.get_statistics()
    print("\n" + "="*60)
    print("Knowledge Graph Statistics")
    print("="*60)
    for key, value in sorted(stats.items()):
        print(f"  {key}: {value}")
    print("="*60 + "\n")
    
    return kg_constructor


def example_rag_usage(kg: KnowledgeGraphConstructor):
    """
    Example of how to use the knowledge graph for RAG
    """
    print("\n" + "="*60)
    print("Example RAG Usage")
    print("="*60 + "\n")
    
    # Example 1: Search by natural language query
    query = "find functions that handle file operations"
    print(f"Query: {query}\n")
    results = kg.search_by_embedding(query, top_k=3)
    
    for i, result in enumerate(results, 1):
        print(f"\nResult {i} (Score: {result['similarity_score']:.4f}):")
        print(f"  Name: {result['name']}")
        print(f"  Type: {result['type']}")
        print(f"  Summary: {result['summary']}")
        print(f"  File: {result['file_path']}:{result['line_number']}")
        
        if result.get('related_elements'):
            print(f"  Related: {len(result['related_elements'])} elements")
    
    # Example 2: Get full context for a specific element
    if results:
        element_id = results[0]['id']
        print(f"\n\nFull Context for Top Result:")
        print("-" * 60)
        full_context = kg.get_element_for_rag(element_id)
        print(full_context['full_context'][:500] + "...")


if __name__ == "__main__":
    # Example usage
    repo_path = "codebase"
    kg = create_knowledge_graph_from_repo(repo_path)
    
    # Demonstrate RAG usage
    example_rag_usage(kg)
