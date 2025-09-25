import ast
import os
import json
from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import hashlib
from sentence_transformers import SentenceTransformer
import numpy as np

@dataclass
class CodeElement:
    """Represents a code element extracted from AST"""
    name: str
    type: str  # 'class', 'method', 'function', 'attribute', 'file'
    file_path: str
    line_number: int
    docstring: Optional[str] = None
    source_code: Optional[str] = None
    parent: Optional[str] = None
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

class KnowledgeGraphConstructor:
    """
    Constructs a knowledge graph from Python codebase following the paper's methodology
    """
    
    def __init__(self, encoder_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the Knowledge Graph Constructor
        
        Args:
            encoder_model: Name of the sentence transformer model for embeddings
        """
        try:
            self.encoder = SentenceTransformer(encoder_model)
            print(f"Loaded embedding model: {encoder_model}")
        except Exception as e:
            print(f"Warning: Could not load embedding model {encoder_model}: {e}")
            print("Continuing without embeddings...")
            self.encoder = None
        
        self.elements: Dict[str, CodeElement] = {}
        self.relationships: List[Dict[str, Any]] = []
        self.schema = self._define_schema()
        self.imports_map: Dict[str, Dict[str, str]] = {}  # file_path -> {import_name: actual_element_id}
        self.repo_path: Optional[Path] = None
        
    def _define_schema(self) -> Dict[str, Any]:
        """Define the knowledge graph schema as per the paper"""
        return {
            "node_types": {
                "File": {"properties": ["path", "name", "description"]},
                "Class": {"properties": ["name", "docstring", "file_path", "line_number"]},
                "Method": {"properties": ["name", "docstring", "class_name", "file_path", "line_number"]},
                "Function": {"properties": ["name", "docstring", "file_path", "line_number"]},
                "Attribute": {"properties": ["name", "class_name", "file_path", "line_number"]},
                "Generated_Description": {"properties": ["description", "embedding"]}
            },
            "relation_types": [
                "defines_class", "defines_function", "has_method", 
                "used_in", "has_attribute", "has_description", "imports"
            ]
        }
    
    def parse_repository(self, repo_path: str, exclude_patterns: List[str] = None) -> None:
        """
        Parse the entire repository and extract code elements
        
        Args:
            repo_path: Path to the repository root
            exclude_patterns: List of patterns to exclude (e.g., ['test_', '__pycache__'])
        """
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
        
        # Second pass: resolve imports and create cross-file relationships
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
            # Filter out excluded directories
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
            
            tree = ast.parse(source_code)
            
            # Create file element
            file_element = CodeElement(
                name=file_path.name,
                type="file",
                file_path=str(file_path),
                line_number=1,
                docstring=ast.get_docstring(tree),
                source_code=source_code
            )
            
            file_id = self._generate_id(file_element)
            self.elements[file_id] = file_element
            
            # Parse file contents
            self._parse_ast_node(tree, str(file_path), source_code.split('\n'))
            
        except Exception as e:
            print(f"Error parsing file {file_path}: {e}")
    
    def _parse_ast_node(self, node: ast.AST, file_path: str, source_lines: List[str], 
                       parent_class: Optional[str] = None) -> None:
        """Recursively parse AST nodes to extract code elements"""
        
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                self._extract_class(child, file_path, source_lines)
                # Continue parsing inside the class with the class name as parent
                self._parse_ast_node(child, file_path, source_lines, child.name)
            elif isinstance(child, ast.FunctionDef):
                self._extract_function(child, file_path, source_lines, parent_class)
                # Parse function body for usage relationships
                self._extract_usage_relationships(child, file_path, parent_class)
            elif isinstance(child, ast.AsyncFunctionDef):
                self._extract_function(child, file_path, source_lines, parent_class)
                # Parse function body for usage relationships
                self._extract_usage_relationships(child, file_path, parent_class)
            elif isinstance(child, (ast.Import, ast.ImportFrom)):
                self._extract_imports(child, file_path)
            elif isinstance(child, ast.Assign) and parent_class:
                # Extract class-level attributes
                self._extract_attributes(child, file_path, parent_class)
            else:
                # Recursively process nested nodes
                self._parse_ast_node(child, file_path, source_lines, parent_class)
    
    def _extract_imports(self, node: ast.AST, file_path: str) -> None:
        """Extract import statements and store them for later resolution"""
        if not hasattr(self, 'imports_map'):
            self.imports_map = {}
        
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
        """Resolve imports for a specific file and create relationships with actual elements"""
        str_file_path = str(file_path)
        
        if str_file_path not in self.imports_map:
            return
        
        # Parse the file again to analyze usage
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            tree = ast.parse(source_code)
            self._resolve_usage_in_node(tree, str_file_path, None)
            
        except Exception as e:
            print(f"Error resolving imports for {file_path}: {e}")
    
    def _resolve_usage_in_node(self, node: ast.AST, file_path: str, current_element_id: Optional[str] = None) -> None:
        """Recursively resolve usage relationships in AST nodes"""
        
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                # Find the class element ID
                class_element_id = self._find_element_id_by_name_and_line(file_path, child.name, child.lineno, "class")
                if class_element_id:
                    self._resolve_usage_in_node(child, file_path, class_element_id)
            
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Find the function/method element ID
                element_type = "method" if current_element_id and self._is_class_element(current_element_id) else "function"
                func_element_id = self._find_element_id_by_name_and_line(file_path, child.name, child.lineno, element_type)
                if func_element_id:
                    # Analyze the function body for usage
                    self._analyze_function_usage(child, file_path, func_element_id)
                    self._resolve_usage_in_node(child, file_path, func_element_id)
            else:
                self._resolve_usage_in_node(child, file_path, current_element_id)
    
    def _analyze_function_usage(self, func_node: ast.AST, file_path: str, func_element_id: str) -> None:
        """Analyze a function for usage of imported or local elements"""
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                # Handle function calls
                if isinstance(node.func, ast.Name):
                    # Direct function call: func_name()
                    self._create_usage_relationship_resolved(func_element_id, node.func.id, file_path)
                elif isinstance(node.func, ast.Attribute):
                    # Method call: obj.method() or module.func()
                    if isinstance(node.func.value, ast.Name):
                        # Check if this is an imported module/function
                        base_name = node.func.value.id
                        attr_name = node.func.attr
                        self._create_cross_file_relationship(func_element_id, base_name, attr_name, file_path)
            
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                # Variable/function reference
                self._create_usage_relationship_resolved(func_element_id, node.id, file_path)
            
            elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                # Attribute access: obj.attr
                if isinstance(node.value, ast.Name):
                    base_name = node.value.id
                    attr_name = node.attr
                    self._create_cross_file_relationship(func_element_id, base_name, attr_name, file_path)
    
    def _create_usage_relationship_resolved(self, source_element_id: str, used_name: str, file_path: str) -> None:
        """Create usage relationship with proper import resolution"""
        
        # First, check if it's a local element in the same file
        target_element_id = self._find_local_element_by_name(file_path, used_name)
        
        if target_element_id:
            self._add_relationship("used_in", source_element_id, target_element_id)
            return
        
        # Check if it's an imported element
        if file_path in self.imports_map and used_name in self.imports_map[file_path]:
            imported_full_name = self.imports_map[file_path][used_name]
            target_element_id = self._find_element_by_import_path(imported_full_name)
            
            if target_element_id:
                self._add_relationship("used_in", source_element_id, target_element_id)
                # Also add import relationship
                file_element_id = self._find_file_element_id(file_path)
                if file_element_id:
                    self._add_relationship("imports", file_element_id, target_element_id)
    
    def _create_cross_file_relationship(self, source_element_id: str, base_name: str, attr_name: str, file_path: str) -> None:
        """Create relationships for cross-file references like module.function()"""
        
        if file_path not in self.imports_map:
            return
        
        # Check if base_name is an imported module
        if base_name in self.imports_map[file_path]:
            imported_module = self.imports_map[file_path][base_name]
            
            # Try to find the actual element (function/class) in the imported module
            target_element_id = self._find_element_by_module_and_name(imported_module, attr_name)
            
            if target_element_id:
                self._add_relationship("used_in", source_element_id, target_element_id)
                
                # Add import relationship between file and the imported element
                file_element_id = self._find_file_element_id(file_path)
                if file_element_id:
                    self._add_relationship("imports", file_element_id, target_element_id)
    
    def _find_element_id_by_name_and_line(self, file_path: str, name: str, line_no: int, element_type: str) -> Optional[str]:
        """Find element ID by name, line number, and type"""
        for element_id, element in self.elements.items():
            if (element.file_path == file_path and 
                element.name == name and 
                element.line_number == line_no and
                element.type == element_type):
                return element_id
        return None
    
    def _find_local_element_by_name(self, file_path: str, name: str) -> Optional[str]:
        """Find a local element in the same file by name"""
        for element_id, element in self.elements.items():
            if (element.file_path == file_path and 
                element.name == name and
                element.type in ['class', 'function', 'method']):
                return element_id
        return None
    
    def _find_element_by_import_path(self, import_path: str) -> Optional[str]:
        """Find element by its import path (e.g., 'utils.helper_function')"""
        
        if not self.repo_path:
            return None
        
        parts = import_path.split('.')
        if len(parts) == 1:
            # Simple import, look for function/class with this name
            element_name = parts[0]
            for element_id, element in self.elements.items():
                if (element.name == element_name and 
                    element.type in ['class', 'function']):
                    return element_id
        else:
            # Module.element format
            module_parts = parts[:-1]
            element_name = parts[-1]
            
            # Try to find the module file
            module_file = self._find_module_file(module_parts)
            if module_file:
                for element_id, element in self.elements.items():
                    if (element.file_path == str(module_file) and
                        element.name == element_name and
                        element.type in ['class', 'function']):
                        return element_id
        
        return None
    
    def _find_element_by_module_and_name(self, module_path: str, element_name: str) -> Optional[str]:
        """Find element by module path and element name"""
        return self._find_element_by_import_path(f"{module_path}.{element_name}")
    
    def _find_module_file(self, module_parts: List[str]) -> Optional[Path]:
        """Find the file corresponding to a module path"""
        if not self.repo_path:
            return None
        
        # Try different combinations to find the module file
        current_path = self.repo_path
        
        # First, try to find as nested directories
        for part in module_parts:
            current_path = current_path / part
        
        # Check if it's a file
        py_file = current_path.with_suffix('.py')
        if py_file.exists():
            return py_file
        
        # Check if it's a package with __init__.py
        init_file = current_path / '__init__.py'
        if init_file.exists():
            return init_file
        
        # Try looking for the file directly in the repo
        module_file = self.repo_path / f"{module_parts[-1]}.py"
        if module_file.exists():
            return module_file
        
        return None
    
    def _find_file_element_id(self, file_path: str) -> Optional[str]:
        """Find the file element ID for a given file path"""
        for element_id, element in self.elements.items():
            if element.type == "file" and element.file_path == file_path:
                return element_id
        return None
    
    def _is_class_element(self, element_id: str) -> bool:
        """Check if an element is a class"""
        return element_id in self.elements and self.elements[element_id].type == "class"
    
    def _extract_usage_relationships(self, node: ast.AST, file_path: str, parent_class: Optional[str] = None) -> None:
        """Extract usage relationships by analyzing function/method bodies - now handled in _resolve_imports_for_file"""
        # This method is now replaced by the more sophisticated import resolution system
        # The actual usage analysis is done in _analyze_function_usage during the second pass
        pass
    
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
        
        # Add relationship: File defines Class
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
        
        # Add relationships using actual element IDs
        if parent_class:
            # Method belongs to class
            class_element = self._find_class_element(file_path, parent_class)
            if class_element:
                class_id = self._generate_id(class_element)
                self._add_relationship("has_method", class_id, function_id)
        else:
            # Function belongs to file
            file_element = self._find_file_element(file_path)
            if file_element:
                file_id = self._generate_id(file_element)
                self._add_relationship("defines_function", file_id, function_id)
    
    def _extract_attributes(self, node: ast.Assign, file_path: str, class_name: str) -> None:
        """Extract class attributes"""
        for target in node.targets:
            if isinstance(target, ast.Name):
                attr_element = CodeElement(
                    name=target.id,
                    type="attribute",
                    file_path=file_path,
                    line_number=node.lineno,
                    parent=class_name,
                    source_code=self._get_source_code(node, [])  # Get assignment source
                )
                
                attr_id = self._generate_id(attr_element)
                self.elements[attr_id] = attr_element
                
                # Add relationship using actual element IDs
                class_element = self._find_class_element(file_path, class_name)
                if class_element:
                    class_id = self._generate_id(class_element)
                    self._add_relationship("has_attribute", class_id, attr_id)
            elif isinstance(target, ast.Attribute):
                # Handle self.attribute assignments
                if isinstance(target.value, ast.Name) and target.value.id == 'self':
                    attr_element = CodeElement(
                        name=target.attr,
                        type="attribute",
                        file_path=file_path,
                        line_number=node.lineno,
                        parent=class_name,
                        source_code=self._get_source_code(node, [])
                    )
                    
                    attr_id = self._generate_id(attr_element)
                    self.elements[attr_id] = attr_element
                    
                    # Add relationship using actual element IDs
                    class_element = self._find_class_element(file_path, class_name)
                    if class_element:
                        class_id = self._generate_id(class_element)
                        self._add_relationship("has_attribute", class_id, attr_id)
    
    def _get_source_code(self, node: ast.AST, source_lines: List[str]) -> str:
        """Extract source code for a given AST node"""
        try:
            if not source_lines:
                return ""
            start_line = node.lineno - 1
            end_line = getattr(node, 'end_lineno', start_line + 1)
            return '\n'.join(source_lines[start_line:end_line])
        except:
            return ""
    
    def _find_file_element(self, file_path: str) -> Optional[CodeElement]:
        """Find the file element for a given file path"""
        for element in self.elements.values():
            if element.type == "file" and element.file_path == file_path:
                return element
        return None
    
    def _find_class_element(self, file_path: str, class_name: str) -> Optional[CodeElement]:
        """Find the class element for a given file path and class name"""
        for element in self.elements.values():
            if (element.type == "class" and 
                element.file_path == file_path and 
                element.name == class_name):
                return element
        return None
    
    def _generate_id(self, element: CodeElement) -> str:
        """Generate unique ID for code element"""
        # Normalize file path
        normalized_path = element.file_path.replace('\\', '/')
        
        # Create identifier based on element type
        if element.type == "file":
            identifier = f"file:{normalized_path}"
        elif element.type == "class":
            identifier = f"class:{normalized_path}:{element.name}"
        elif element.type in ["method", "attribute"] and element.parent:
            identifier = f"{element.type}:{normalized_path}:{element.parent}:{element.name}"
        else:
            identifier = f"{element.type}:{normalized_path}:{element.name}"
        
        # Add line number to ensure uniqueness for elements with same name
        identifier += f":{element.line_number}"
        
        return hashlib.md5(identifier.encode()).hexdigest()
    
    def _add_relationship(self, relation_type: str, source: str, target: str) -> None:
        """Add a relationship between two nodes"""
        self.relationships.append({
            "type": relation_type,
            "source": source,
            "target": target
        })
    
    def generate_descriptions(self, llm_generate_func=None) -> None:
        """
        Generate descriptions for code elements using LLM
        
        Args:
            llm_generate_func: Function that takes code and returns description
        """
        if llm_generate_func is None:
            # Simple fallback description generation
            llm_generate_func = self._simple_description_generator
        
        # Create a snapshot of elements to avoid dictionary modification during iteration
        elements_snapshot = list(self.elements.items())
        
        for element_id, element in elements_snapshot:
            if element.source_code and element.type in ['class', 'method', 'function']:
                try:
                    description = llm_generate_func(element.source_code, element.type, element.name)
                    
                    # Create description element with unique name
                    desc_name = f"{element.name}_description_{element.line_number}"
                    desc_element = CodeElement(
                        name=desc_name,
                        type="generated_description",
                        file_path=element.file_path,
                        line_number=element.line_number,
                        docstring=description,
                        parent=element.name  # Link to original element
                    )
                    
                    desc_id = self._generate_id(desc_element)
                    self.elements[desc_id] = desc_element
                    
                    # Add relationship
                    self._add_relationship("has_description", element_id, desc_id)
                    
                except Exception as e:
                    print(f"Error generating description for {element.name}: {e}")
    
    def _simple_description_generator(self, source_code: str, element_type: str, name: str) -> str:
        """Simple fallback description generator"""
        return f"This {element_type} named '{name}' contains {len(source_code.split())} words of code."
    
    def generate_embeddings(self) -> None:
        """Generate embeddings for descriptions and docstrings"""
        if self.encoder is None:
            print("No embedding model available, skipping embeddings generation")
            return
        
        texts_to_embed = []
        element_ids = []
        
        # Create a snapshot to avoid iteration issues
        elements_snapshot = list(self.elements.items())
        
        for element_id, element in elements_snapshot:
            text = ""
            if element.docstring:
                text = element.docstring
            elif element.type == "generated_description":
                text = element.docstring or ""
            
            if text.strip():  # Only process non-empty text
                texts_to_embed.append(text)
                element_ids.append(element_id)
        
        if texts_to_embed:
            try:
                print(f"Generating embeddings for {len(texts_to_embed)} text items...")
                embeddings = self.encoder.encode(texts_to_embed)
                
                for i, element_id in enumerate(element_ids):
                    if element_id in self.elements:  # Check if element still exists
                        self.elements[element_id].dependencies = embeddings[i].tolist()
                
                print("Embeddings generated successfully")
            except Exception as e:
                print(f"Error generating embeddings: {e}")
                # Continue without embeddings if there's an error
        else:
            print("No text found for embedding generation")
    
    def export_to_neo4j_format(self, output_dir: str = "knowledge_graph_export") -> None:
        """Export knowledge graph in Neo4j import format"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Export nodes
        nodes_data = []
        for element_id, element in self.elements.items():
            # Normalize file path for consistency
            normalized_path = element.file_path.replace('\\', '/')
            
            node_data = {
                "id": element_id,
                "name": element.name,
                "type": element.type,
                "file_path": normalized_path,
                "line_number": element.line_number,
                "docstring": element.docstring,
                "source_code": element.source_code,
                "parent": element.parent,
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
        
        print(f"Knowledge graph exported to {output_path}")
        print(f"Total nodes: {len(self.elements)}")
        print(f"Total relationships: {len(self.relationships)}")
    
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
        
        stats.update(rel_stats)
        return stats

# Example usage function
def create_knowledge_graph_from_repo(repo_path: str, output_dir: str = "kg_export"):
    """
    Create knowledge graph from a Python repository
    
    Args:
        repo_path: Path to the Python repository
        output_dir: Directory to export the knowledge graph
    """
    # Initialize constructor
    kg_constructor = KnowledgeGraphConstructor()
    
    print("Parsing repository...")
    kg_constructor.parse_repository(repo_path)
    
    print("Generating descriptions...")
    kg_constructor.generate_descriptions()
    
    print("Generating embeddings...")
    kg_constructor.generate_embeddings()
    
    print("Exporting to Neo4j format...")
    kg_constructor.export_to_neo4j_format(output_dir)
    
    # Print statistics
    stats = kg_constructor.get_statistics()
    print("\nKnowledge Graph Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return kg_constructor

if __name__ == "__main__":
    # Example usage
    repo_path = "codebase"
    kg = create_knowledge_graph_from_repo(repo_path)