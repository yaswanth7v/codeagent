"""
Enhanced Knowledge Graph Constructor for Code Repositories
Following methodology from the research paper on Graph-based Code Retrieval
"""

import ast
import os
import json
import pickle
import gzip
from typing import Dict, List, Set, Any, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
from sentence_transformers import SentenceTransformer
import numpy as np

@dataclass
class CodeElement:
    """Represents a code element extracted from AST with enhanced context for RAG"""
    name: str
    type: str  # 'class', 'method', 'function', 'file', 'generated_description'
    file_path: str
    line_number: int
    docstring: Optional[str] = None
    source_code: Optional[str] = None
    parent: Optional[str] = None
    dependencies: List[float] = None  # Embeddings stored here
    
    # Enhanced fields for RAG context
    full_context: Optional[str] = None
    summary: Optional[str] = None
    signature: Optional[str] = None
    parent_context: Optional[str] = None
    related_elements: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Additional fields for paper's methodology
    attributes: List[str] = field(default_factory=list)  # For classes
    llm_description: Optional[str] = None  # LLM-generated functional description
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

class KnowledgeGraphConstructor:
    """
    Constructs knowledge graph following the research paper methodology:
    - AST-based parsing for Files, Classes, Methods, Functions, Attributes
    - Schema-based graph construction
    - LLM-generated descriptions for functional context
    - Vector embeddings for semantic search
    - Neo4j export format with full-text and vector indexes
    """
    
    def __init__(self, encoder_model: str = "all-MiniLM-L6-v2"):
        try:
            self.encoder = SentenceTransformer(encoder_model)
            self.encoder_model_name = encoder_model
            print(f"Loaded embedding model: {encoder_model}")
        except Exception as e:
            print(f"Warning: Could not load embedding model {encoder_model}: {e}")
            self.encoder = None
            self.encoder_model_name = encoder_model
        
        self.elements: Dict[str, CodeElement] = {}
        self.relationships: List[Dict[str, Any]] = []
        self.schema = self._define_schema()
        self.imports_map: Dict[str, Dict[str, str]] = {}
        self.repo_path: Optional[Path] = None
        self.file_contents: Dict[str, str] = {}
        
    def _define_schema(self) -> Dict[str, Any]:
        """
        Define the knowledge graph schema as per paper:
        Node types: File, Class, Method, Function, Attribute, Generated Description
        Relations: defines_class, defines_function, has_method, used_in, 
                   has_attribute, has_description
        """
        return {
            "node_types": {
                "File": {
                    "properties": ["path", "name", "description", "full_context", "summary"],
                    "indexes": ["full_text"]
                },
                "Class": {
                    "properties": ["name", "docstring", "file_path", "signature", 
                                 "full_context", "summary", "attributes"],
                    "indexes": ["full_text", "vector"]
                },
                "Method": {
                    "properties": ["name", "docstring", "class_name", "signature", 
                                 "full_context", "parent_context"],
                    "indexes": ["full_text", "vector"]
                },
                "Function": {
                    "properties": ["name", "docstring", "signature", "full_context", "summary"],
                    "indexes": ["full_text", "vector"]
                },
                "Attribute": {
                    "properties": ["name", "class_name", "parent_context"],
                    "indexes": ["full_text"]
                },
                "Generated_Description": {
                    "properties": ["description", "embedding", "element_type"],
                    "indexes": ["vector"]
                }
            },
            "relation_types": [
                "defines_class",      # File -> Class
                "defines_function",   # File -> Function
                "has_method",         # Class -> Method
                "used_in",           # Element -> Element (usage)
                "has_attribute",     # Class -> Attribute
                "has_description",   # Element -> Generated_Description
                "imports",           # File -> File/Element
                "calls",             # Function/Method -> Function/Method
                "inherits_from",     # Class -> Class
                "related_to"         # General relationship
            ]
        }
    
    def parse_repository(self, repo_path: str, exclude_patterns: List[str] = None) -> None:
        """
        Parse repository following paper's methodology:
        1. Parse files with AST
        2. Identify code components (Classes, Methods, Functions, Attributes)
        3. Align with schema
        4. Extract documentation and comments
        """
        if exclude_patterns is None:
            exclude_patterns = ['__pycache__', '.git', '.pytest_cache', 'test_', 'tests', 'venv', 'env']
            
        self.repo_path = Path(repo_path)
        
        # Step 1: Parse files with AST
        python_files = self._find_python_files(self.repo_path, exclude_patterns)
        print(f"Found {len(python_files)} Python files to parse")
        
        # Step 2: Identify code components
        print("Parsing files and extracting code elements...")
        for py_file in python_files:
            try:
                self._parse_file(py_file)
            except Exception as e:
                print(f"Error parsing {py_file}: {e}")
        
        # Step 3: Enhance context for all elements
        print("Enhancing context for RAG...")
        self._enhance_all_contexts()
        
        # Step 4: Resolve imports and create relationships
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
            
            self.file_contents[str(file_path)] = source_code
            tree = ast.parse(source_code)
            module_doc = ast.get_docstring(tree)
            
            # Create file node
            file_element = CodeElement(
                name=file_path.name,
                type="file",
                file_path=str(file_path),
                line_number=1,
                docstring=module_doc,
                source_code=source_code,
                full_context=source_code,
                summary=self._create_file_summary(file_path.name, module_doc, source_code),
                metadata={
                    "file_size": len(source_code),
                    "num_lines": len(source_code.split('\n')),
                    "relative_path": str(file_path.relative_to(self.repo_path)) if self.repo_path else str(file_path)
                }
            )
            
            file_id = self._generate_id(file_element)
            self.elements[file_id] = file_element
            self._parse_ast_node(tree, str(file_path), source_code.split('\n'))
            
        except Exception as e:
            print(f"Error parsing file {file_path}: {e}")
    
    def _create_file_summary(self, filename: str, docstring: Optional[str], source_code: str) -> str:
        """Create a summary of the file"""
        lines = source_code.split('\n')
        num_lines = len(lines)
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
        """
        Enhance class node with complete context
        Extracts attributes as per paper's methodology
        """
        class_methods = []
        class_attributes = set()
        
        # Find methods
        for elem_id, elem in self.elements.items():
            if elem.parent == class_element.name and elem.file_path == class_element.file_path:
                if elem.type == "method":
                    class_methods.append(elem)
                elif elem.type == "attribute":
                    class_attributes.add(elem.name)
        
        # Extract attributes from class AST
        try:
            tree = ast.parse(class_element.source_code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Attribute):
                            if isinstance(target.value, ast.Name) and target.value.id == 'self':
                                class_attributes.add(target.attr)
                        elif isinstance(target, ast.Name):
                            class_attributes.add(target.id)
        except:
            pass
        
        # Build full context
        full_context = f"# Class: {class_element.name}\n"
        if class_element.docstring:
            full_context += f'"""{class_element.docstring}"""\n\n'
        
        full_context += f"## Class Definition:\n{class_element.source_code}\n\n"
        
        if class_attributes:
            full_context += f"## Attributes ({len(class_attributes)}):\n"
            for attr in sorted(class_attributes):
                full_context += f"- {attr}\n"
            full_context += "\n"
        
        if class_methods:
            full_context += f"## Methods ({len(class_methods)}):\n"
            for method in class_methods:
                full_context += f"\n### {method.name}:\n"
                if method.docstring:
                    full_context += f'"""{method.docstring}"""\n'
                full_context += f"{method.source_code}\n"
        
        summary = f"Class '{class_element.name}' with {len(class_methods)} methods and {len(class_attributes)} attributes"
        if class_element.docstring:
            summary += f". {class_element.docstring[:150]}"
        
        # Update element
        class_element.full_context = full_context
        class_element.summary = summary
        class_element.attributes = sorted(list(class_attributes))
        class_element.related_elements = [self._generate_id(m) for m in class_methods]
        class_element.metadata = {
            "num_methods": len(class_methods),
            "num_attributes": len(class_attributes),
            "method_names": [m.name for m in class_methods],
            "attribute_names": sorted(list(class_attributes))
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
                if attr_element.source_code:
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
            elif isinstance(child, ast.AsyncFunctionDef):
                self._extract_function(child, file_path, source_lines, parent_class)
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
            pass
    
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
        """Analyze function for usage patterns and create 'used_in' relationships"""
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    self._create_usage_relationship_resolved(func_element_id, node.func.id, file_path)
                elif isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        base_name = node.func.value.id
                        attr_name = node.func.attr
                        self._create_cross_file_relationship(func_element_id, base_name, attr_name, file_path)
    
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
        
        # Create relationship: File defines_class Class
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
            # Create relationship: Class has_method Method
            class_element = self._find_class_element(file_path, parent_class)
            if class_element:
                class_id = self._generate_id(class_element)
                self._add_relationship("has_method", class_id, function_id)
        else:
            # Create relationship: File defines_function Function
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
                
                # Create relationship: Class has_attribute Attribute
                class_element = self._find_class_element(file_path, class_name)
                if class_element:
                    class_id = self._generate_id(class_element)
                    self._add_relationship("has_attribute", class_id, attr_id)
    
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
        # Avoid duplicate relationships
        existing = any(
            r['type'] == relation_type and r['source'] == source and r['target'] == target
            for r in self.relationships
        )
        if not existing:
            self.relationships.append({
                "type": relation_type,
                "source": source,
                "target": target
            })
    
    def generate_llm_descriptions(self, llm_generate_func: Optional[Callable] = None) -> None:
        """
        Generate LLM descriptions for code elements as per paper methodology
        This captures functional meaning and context
        
        Args:
            llm_generate_func: Custom LLM function that takes (source_code, element_type, name)
                             and returns description string
        """
        if llm_generate_func is None:
            llm_generate_func = self._simple_description_generator
        
        print("\nGenerating LLM descriptions for code elements...")
        elements_snapshot = list(self.elements.items())
        description_count = 0
        
        for element_id, element in elements_snapshot:
            if element.source_code and element.type in ['class', 'method', 'function']:
                try:
                    # Generate functional description
                    description = llm_generate_func(
                        element.source_code, 
                        element.type, 
                        element.name
                    )
                    
                    # Store in element
                    element.llm_description = description
                    
                    # Create separate Generated_Description node as per schema
                    desc_element = CodeElement(
                        name=f"{element.name}_description",
                        type="generated_description",
                        file_path=element.file_path,
                        line_number=element.line_number,
                        docstring=description,
                        parent=element.name,
                        metadata={
                            "element_type": element.type,
                            "element_id": element_id
                        }
                    )
                    
                    desc_id = self._generate_id(desc_element)
                    self.elements[desc_id] = desc_element
                    
                    # Create has_description relationship
                    self._add_relationship("has_description", element_id, desc_id)
                    description_count += 1
                    
                except Exception as e:
                    print(f"Error generating description for {element.name}: {e}")
        
        print(f"Generated {description_count} LLM descriptions")
    
    def _simple_description_generator(self, source_code: str, element_type: str, name: str) -> str:
        """Simple fallback description generator"""
        lines = source_code.split('\n')
        num_lines = len(lines)
        
        description = f"This {element_type} named '{name}' contains {num_lines} lines of code. "
        
        # Add basic heuristics
        if 'return' in source_code:
            description += "It returns a value. "
        if 'self' in source_code and element_type == "method":
            description += "It operates on instance data. "
        if 'def __init__' in source_code:
            description += "This is a constructor method. "
        
        return description
    
    def generate_embeddings(self) -> None:
        """
        Generate embeddings for documentation and descriptions as per paper methodology
        Uses all-MiniLM-L6-v2 encoder
        """
        if self.encoder is None:
            print("No embedding model available")
            return
        
        texts_to_embed = []
        element_ids = []
        
        for element_id, element in list(self.elements.items()):
            # Embed full_context for better retrieval
            text = None
            
            if element.type == "generated_description":
                # For generated descriptions, embed the description itself
                text = element.docstring
            elif element.type in ['class', 'method', 'function']:
                # For code elements, embed full_context or summary
                text = element.full_context or element.summary or element.docstring
            elif element.type == "file":
                # For files, embed summary
                text = element.summary or element.docstring
            
            if text and text.strip():
                texts_to_embed.append(text)
                element_ids.append(element_id)
        
        if texts_to_embed:
            print(f"Generating embeddings for {len(texts_to_embed)} elements...")
            embeddings = self.encoder.encode(texts_to_embed, show_progress_bar=True)
            
            for i, element_id in enumerate(element_ids):
                if element_id in self.elements:
                    self.elements[element_id].dependencies = embeddings[i].tolist()
            print("Embeddings generated successfully")
    
    def export_to_neo4j_format(self, output_dir: str = "neo4j_export") -> None:
        """
        Export knowledge graph in Neo4j-compatible format
        Includes nodes, relationships, and index configurations
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Export nodes
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
                "full_context": element.full_context,
                "summary": element.summary,
                "signature": element.signature,
                "parent": element.parent,
                "parent_context": element.parent_context,
                "related_elements": element.related_elements,
                "metadata": element.metadata,
                "attributes": element.attributes if hasattr(element, 'attributes') else [],
                "llm_description": element.llm_description if hasattr(element, 'llm_description') else None,
                "embedding": element.dependencies if element.dependencies else []
            }
            nodes_data.append(node_data)
        
        with open(output_path / "nodes.json", 'w', encoding='utf-8') as f:
            json.dump(nodes_data, f, indent=2, ensure_ascii=False)
        
        # Export relationships
        with open(output_path / "relationships.json", 'w', encoding='utf-8') as f:
            json.dump(self.relationships, f, indent=2, ensure_ascii=False)
        
        # Export schema with index configurations
        with open(output_path / "schema.json", 'w', encoding='utf-8') as f:
            json.dump(self.schema, f, indent=2, ensure_ascii=False)
        
        # Export index configuration for Neo4j
        index_config = self._generate_index_configuration()
        with open(output_path / "index_config.json", 'w', encoding='utf-8') as f:
            json.dump(index_config, f, indent=2, ensure_ascii=False)
        
        # Export Cypher import script
        self._generate_cypher_import_script(output_path)
        
        print(f"\n{'='*60}")
        print(f"Knowledge graph exported to {output_path}")
        print(f"{'='*60}")
        print(f"Total nodes: {len(self.elements)}")
        print(f"Total relationships: {len(self.relationships)}")
        print(f"\nNode type breakdown:")
        node_stats = {}
        for elem in self.elements.values():
            node_stats[elem.type] = node_stats.get(elem.type, 0) + 1
        for node_type, count in sorted(node_stats.items()):
            print(f"  {node_type}: {count}")
        
        print(f"\nRelationship type breakdown:")
        rel_stats = {}
        for rel in self.relationships:
            rel_stats[rel['type']] = rel_stats.get(rel['type'], 0) + 1
        for rel_type, count in sorted(rel_stats.items()):
            print(f"  {rel_type}: {count}")
    
    def _generate_index_configuration(self) -> Dict[str, Any]:
        """Generate Neo4j index configuration"""
        return {
            "full_text_indexes": [
                {
                    "name": "function_names_index",
                    "node_label": "Function",
                    "property": "name"
                },
                {
                    "name": "class_names_index",
                    "node_label": "Class",
                    "property": "name"
                },
                {
                    "name": "method_names_index",
                    "node_label": "Method",
                    "property": "name"
                },
                {
                    "name": "file_names_index",
                    "node_label": "File",
                    "property": "name"
                }
            ],
            "vector_indexes": [
                {
                    "name": "documentation_embeddings",
                    "node_label": "Class",
                    "property": "embedding",
                    "dimension": 384,
                    "similarity": "cosine"
                },
                {
                    "name": "function_embeddings",
                    "node_label": "Function",
                    "property": "embedding",
                    "dimension": 384,
                    "similarity": "cosine"
                },
                {
                    "name": "method_embeddings",
                    "node_label": "Method",
                    "property": "embedding",
                    "dimension": 384,
                    "similarity": "cosine"
                },
                {
                    "name": "description_embeddings",
                    "node_label": "Generated_Description",
                    "property": "embedding",
                    "dimension": 384,
                    "similarity": "cosine"
                }
            ]
        }
    
    def _generate_cypher_import_script(self, output_path: Path) -> None:
        """Generate Cypher script for importing into Neo4j"""
        script_lines = [
            "// Neo4j Import Script",
            "// Run this in Neo4j Browser or using cypher-shell\n",
            "// Clear existing data (optional)",
            "// MATCH (n) DETACH DELETE n;\n",
            "// Create constraints",
            "CREATE CONSTRAINT node_id IF NOT EXISTS FOR (n:CodeElement) REQUIRE n.id IS UNIQUE;\n",
            "// Import nodes from JSON",
            "// Note: Adjust file path as needed",
            "CALL apoc.load.json('file:///nodes.json') YIELD value",
            "CREATE (n:CodeElement)",
            "SET n = value",
            "SET n:$(value.type);\n",
            "// Import relationships",
            "CALL apoc.load.json('file:///relationships.json') YIELD value",
            "MATCH (source:CodeElement {id: value.source})",
            "MATCH (target:CodeElement {id: value.target})",
            "CALL apoc.create.relationship(source, value.type, {}, target) YIELD rel",
            "RETURN count(rel);\n",
            "// Create full-text indexes",
            "CALL db.index.fulltext.createNodeIndex('function_names', ['Function'], ['name']);",
            "CALL db.index.fulltext.createNodeIndex('class_names', ['Class'], ['name']);",
            "CALL db.index.fulltext.createNodeIndex('method_names', ['Method'], ['name']);",
            "CALL db.index.fulltext.createNodeIndex('file_names', ['File'], ['name']);\n",
            "// Create vector indexes (requires Neo4j 5.x with vector support)",
            "// CREATE VECTOR INDEX class_embeddings FOR (n:Class) ON (n.embedding) OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}};",
            "// CREATE VECTOR INDEX function_embeddings FOR (n:Function) ON (n.embedding) OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}};",
            "// CREATE VECTOR INDEX method_embeddings FOR (n:Method) ON (n.embedding) OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}};",
            "// CREATE VECTOR INDEX description_embeddings FOR (n:Generated_Description) ON (n.embedding) OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}};"
        ]
        
        with open(output_path / "import_script.cypher", 'w', encoding='utf-8') as f:
            f.write('\n'.join(script_lines))
    
    def get_element_for_rag(self, element_id: str) -> Optional[Dict[str, Any]]:
        """Get a code element optimized for RAG retrieval with full context"""
        if element_id not in self.elements:
            return None
        
        element = self.elements[element_id]
        
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
            "llm_description": element.llm_description if hasattr(element, 'llm_description') else None
        }
        
        if element.parent_context:
            rag_context["parent_context"] = element.parent_context
        
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
        
        incoming_rels = [r for r in self.relationships if r['target'] == element_id]
        outgoing_rels = [r for r in self.relationships if r['source'] == element_id]
        
        rag_context["relationships"] = {
            "incoming": incoming_rels,
            "outgoing": outgoing_rels
        }
        
        return rag_context
    
    def search_by_embedding(self, query_text: str, top_k: int = 5, 
                          filter_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Search for relevant code elements using semantic similarity
        This implements the paper's semantic search component
        
        Args:
            query_text: Natural language query
            top_k: Number of top results to return
            filter_types: Optional list of element types to filter by
        """
        if self.encoder is None:
            print("No embedding model available for search")
            return []
        
        query_embedding = self.encoder.encode([query_text])[0]
        
        elements_with_embeddings = [
            (elem_id, elem) for elem_id, elem in self.elements.items()
            if elem.dependencies and (filter_types is None or elem.type in filter_types)
        ]
        
        if not elements_with_embeddings:
            print("No elements with embeddings found")
            return []
        
        similarities = []
        for elem_id, elem in elements_with_embeddings:
            elem_embedding = np.array(elem.dependencies)
            similarity = np.dot(query_embedding, elem_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(elem_embedding)
            )
            similarities.append((elem_id, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = similarities[:top_k]
        
        results = []
        for elem_id, score in top_results:
            rag_context = self.get_element_for_rag(elem_id)
            if rag_context:
                rag_context['similarity_score'] = float(score)
                results.append(rag_context)
        
        return results
    
    def get_n_hop_subgraph(self, start_node_ids: List[str], n_hops: int = 2) -> Dict[str, Any]:
        """
        Retrieve n-hop subgraph from starting nodes
        This implements the paper's graph traversal for expanding initial retrieval results
        
        Args:
            start_node_ids: List of starting node IDs
            n_hops: Number of hops to traverse
            
        Returns:
            Dictionary containing nodes and relationships in the subgraph
        """
        visited_nodes = set(start_node_ids)
        current_frontier = set(start_node_ids)
        subgraph_relationships = []
        
        for hop in range(n_hops):
            next_frontier = set()
            
            for rel in self.relationships:
                if rel['source'] in current_frontier:
                    next_frontier.add(rel['target'])
                    if rel not in subgraph_relationships:
                        subgraph_relationships.append(rel)
                elif rel['target'] in current_frontier:
                    next_frontier.add(rel['source'])
                    if rel not in subgraph_relationships:
                        subgraph_relationships.append(rel)
            
            visited_nodes.update(next_frontier)
            current_frontier = next_frontier
        
        # Get node data for all visited nodes
        subgraph_nodes = []
        for node_id in visited_nodes:
            if node_id in self.elements:
                node_context = self.get_element_for_rag(node_id)
                if node_context:
                    subgraph_nodes.append(node_context)
        
        return {
            "nodes": subgraph_nodes,
            "relationships": subgraph_relationships,
            "node_count": len(subgraph_nodes),
            "relationship_count": len(subgraph_relationships)
        }
    
    def filter_subgraph_by_relevance(self, subgraph: Dict[str, Any], 
                                    query_text: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Filter subgraph nodes by semantic relevance to query
        This implements the paper's reranking/filtering step
        
        Args:
            subgraph: Subgraph dictionary from get_n_hop_subgraph
            query_text: Original query
            top_k: Number of top nodes to keep
        """
        if self.encoder is None:
            return subgraph
        
        query_embedding = self.encoder.encode([query_text])[0]
        
        node_scores = []
        for node in subgraph['nodes']:
            node_id = node['id']
            if node_id in self.elements:
                elem = self.elements[node_id]
                if elem.dependencies:
                    elem_embedding = np.array(elem.dependencies)
                    similarity = np.dot(query_embedding, elem_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(elem_embedding)
                    )
                    node_scores.append((node, similarity))
        
        # Sort by relevance and keep top_k
        node_scores.sort(key=lambda x: x[1], reverse=True)
        filtered_nodes = [node for node, score in node_scores[:top_k]]
        filtered_node_ids = set(node['id'] for node in filtered_nodes)
        
        # Filter relationships to only include those between kept nodes
        filtered_relationships = [
            rel for rel in subgraph['relationships']
            if rel['source'] in filtered_node_ids and rel['target'] in filtered_node_ids
        ]
        
        return {
            "nodes": filtered_nodes,
            "relationships": filtered_relationships,
            "node_count": len(filtered_nodes),
            "relationship_count": len(filtered_relationships)
        }
    
    def hybrid_retrieve(self, query_text: str, identified_entities: Optional[List[str]] = None,
                       semantic_top_k: int = 5, n_hops: int = 2, 
                       final_top_k: int = 10) -> Dict[str, Any]:
        """
        Perform hybrid retrieval as described in the paper:
        1. Full-text search for identified entities
        2. Semantic search on embeddings
        3. Expand to n-hop subgraph
        4. Filter by relevance
        
        Args:
            query_text: Natural language query
            identified_entities: Entities identified from query (e.g., by LLM)
            semantic_top_k: Top-k for initial semantic search
            n_hops: Number of hops for graph expansion
            final_top_k: Final number of nodes to return after filtering
        """
        initial_node_ids = set()
        
        # Step 1: Full-text search for identified entities
        if identified_entities:
            for entity in identified_entities:
                for elem_id, elem in self.elements.items():
                    if elem.name == entity:
                        initial_node_ids.add(elem_id)
        
        # Step 2: Semantic search
        semantic_results = self.search_by_embedding(query_text, top_k=semantic_top_k)
        for result in semantic_results:
            initial_node_ids.add(result['id'])
        
        if not initial_node_ids:
            return {"nodes": [], "relationships": [], "node_count": 0, "relationship_count": 0}
        
        # Step 3: Expand to n-hop subgraph
        subgraph = self.get_n_hop_subgraph(list(initial_node_ids), n_hops=n_hops)
        
        # Step 4: Filter by relevance
        filtered_subgraph = self.filter_subgraph_by_relevance(subgraph, query_text, top_k=final_top_k)
        
        return filtered_subgraph
    
    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about the knowledge graph"""
        stats = {}
        for element in self.elements.values():
            stats[element.type] = stats.get(element.type, 0) + 1
        
        stats['total_nodes'] = len(self.elements)
        stats['total_relationships'] = len(self.relationships)
        
        rel_stats = {}
        for rel in self.relationships:
            rel_type = rel['type']
            rel_stats[rel_type] = rel_stats.get(rel_type, 0) + 1
        
        enhanced_count = sum(1 for e in self.elements.values() if e.full_context)
        stats['elements_with_full_context'] = enhanced_count
        
        summary_count = sum(1 for e in self.elements.values() if e.summary)
        stats['elements_with_summary'] = summary_count
        
        embedded_count = sum(1 for e in self.elements.values() if e.dependencies)
        stats['elements_with_embeddings'] = embedded_count
        
        llm_desc_count = sum(1 for e in self.elements.values() if hasattr(e, 'llm_description') and e.llm_description)
        stats['elements_with_llm_descriptions'] = llm_desc_count
        
        stats.update(rel_stats)
        return stats
    
    def save(self, filepath: str = "knowledge_graph.pkl.gz", compress: bool = True) -> None:
        """
        Save the KnowledgeGraphConstructor instance to disk
        
        Args:
            filepath: Path to save file
            compress: Whether to use gzip compression
        """
        try:
            # Temporarily remove encoder to avoid serialization issues
            encoder = self.encoder
            self.encoder = None
            
            if compress:
                with gzip.open(filepath, 'wb') as f:
                    pickle.dump(self, f)
            else:
                with open(filepath, 'wb') as f:
                    pickle.dump(self, f)
            
            # Restore encoder
            self.encoder = encoder
            
            print(f"Knowledge graph saved to {filepath}")
        except Exception as e:
            self.encoder = encoder  # Restore on error
            print(f"Error saving knowledge graph: {e}")
            raise
    
    @staticmethod
    def load(filepath: str = "knowledge_graph.pkl.gz", 
            encoder_model: str = "all-MiniLM-L6-v2") -> 'KnowledgeGraphConstructor':
        """
        Load a KnowledgeGraphConstructor instance from disk
        
        Args:
            filepath: Path to saved file
            encoder_model: Model name to reinitialize encoder
            
        Returns:
            Loaded KnowledgeGraphConstructor instance
        """
        try:
            if filepath.endswith('.gz'):
                with gzip.open(filepath, 'rb') as f:
                    kg = pickle.load(f)
            else:
                with open(filepath, 'rb') as f:
                    kg = pickle.load(f)
            
            # Reinitialize encoder
            try:
                kg.encoder = SentenceTransformer(encoder_model)
                print(f"Reloaded encoder: {encoder_model}")
            except Exception as e:
                print(f"Warning: Could not reload encoder: {e}")
                kg.encoder = None
            
            print(f"Knowledge graph loaded from {filepath}")
            return kg
        except Exception as e:
            print(f"Error loading knowledge graph: {e}")
            raise


def create_knowledge_graph_from_repo(repo_path: str, 
                                     output_dir: str = "neo4j_export",
                                     generate_llm_descriptions: bool = False,
                                     llm_func: Optional[Callable] = None) -> KnowledgeGraphConstructor:
    """
    Create knowledge graph following the research paper methodology
    
    Args:
        repo_path: Path to Python repository
        output_dir: Directory for Neo4j export
        generate_llm_descriptions: Whether to generate LLM descriptions
        llm_func: Optional custom LLM function for generating descriptions
    
    Returns:
        KnowledgeGraphConstructor instance
    """
    print("\n" + "="*60)
    print("Creating Knowledge Graph (Research Paper Methodology)")
    print("="*60 + "\n")
    
    kg = KnowledgeGraphConstructor()
    
    print("Step 1: Parsing repository with AST...")
    kg.parse_repository(repo_path)
    
    if generate_llm_descriptions:
        print("\nStep 2: Generating LLM descriptions...")
        kg.generate_llm_descriptions(llm_func)
    
    print("\nStep 3: Generating embeddings...")
    kg.generate_embeddings()
    
    print("\nStep 4: Exporting to Neo4j format...")
    kg.export_to_neo4j_format(output_dir)
    
    # Print statistics
    stats = kg.get_statistics()
    print("\n" + "="*60)
    print("Knowledge Graph Statistics")
    print("="*60)
    for key, value in sorted(stats.items()):
        print(f"  {key}: {value}")
    print("="*60 + "\n")
    
    return kg


# Example usage with save/load
if __name__ == "__main__":
    # Build knowledge graph
    repo_path = "./codebase"
    kg = create_knowledge_graph_from_repo(
        repo_path,
        output_dir="neo4j_export",
        generate_llm_descriptions=False  # Set to True if you have LLM function
    )
    
    # Save the knowledge graph instance
    kg.save("my_kg.pkl.gz")
    
    # Later, load it for retrieval
    # loaded_kg = KnowledgeGraphConstructor.load("my_kg.pkl.gz")
    
    # Example hybrid retrieval
    # query = "find functions that handle file operations"
    # results = loaded_kg.hybrid_retrieve(query, n_hops=2, final_top_k=10)
    # print(f"Retrieved {results['node_count']} nodes and {results['relationship_count']} relationships")
