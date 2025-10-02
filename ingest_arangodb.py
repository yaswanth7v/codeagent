import json
import os
from pathlib import Path
from arango import ArangoClient
from typing import List, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArangoKnowledgeGraphImporter:
    """
    Import knowledge graph data into ArangoDB database
    """
    
    def __init__(self, host: str = "http://localhost:8529", 
                 username: str = "root", 
                 password: str = "password",
                 db_name: str = "knowledge_graph"):
        """
        Initialize ArangoDB connection
        
        Args:
            host: ArangoDB server URL
            username: Database username
            password: Database password
            db_name: Name of the database to use
        """
        try:
            # Initialize the ArangoDB client
            self.client = ArangoClient(hosts=host)
            
            # Connect to "_system" database first
            sys_db = self.client.db('_system', username=username, password=password)
            
            # Create database if it doesn't exist
            if not sys_db.has_database(db_name):
                sys_db.create_database(db_name)
                logger.info(f"Created database: {db_name}")
            
            # Connect to the knowledge graph database
            self.db = self.client.db(db_name, username=username, password=password)
            self.db_name = db_name
            
            logger.info(f"Connected to ArangoDB database '{db_name}' successfully")
        except Exception as e:
            logger.error(f"Failed to connect to ArangoDB: {e}")
            raise
    
    def clear_database(self):
        """Clear all collections from the database"""
        try:
            # Get all collections (excluding system collections)
            collections = [c['name'] for c in self.db.collections() 
                         if not c['name'].startswith('_')]
            
            for collection_name in collections:
                self.db.delete_collection(collection_name)
                logger.info(f"Deleted collection: {collection_name}")
            
            logger.info("Database cleared")
        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            raise
    
    def create_collections(self):
        """Create necessary document and edge collections"""
        # Document collections (vertex collections)
        doc_collections = ['File', 'Class', 'Method', 'Function', 'Attribute', 'Generated_Description']
        
        # Edge collections
        edge_collections = ['Contains', 'Has_Method', 'Has_Function', 'Has_Attribute', 
                          'Calls', 'Inherits', 'Uses', 'Describes']
        
        logger.info("Creating document collections...")
        for collection_name in doc_collections:
            if not self.db.has_collection(collection_name):
                self.db.create_collection(collection_name)
                logger.info(f"Created document collection: {collection_name}")
            else:
                logger.info(f"Collection already exists: {collection_name}")
        
        logger.info("Creating edge collections...")
        for collection_name in edge_collections:
            if not self.db.has_collection(collection_name):
                self.db.create_collection(collection_name, edge=True)
                logger.info(f"Created edge collection: {collection_name}")
            else:
                logger.info(f"Edge collection already exists: {collection_name}")
    
    def create_indexes(self):
        """Create necessary indexes for performance"""
        indexes_config = {
            'File': ['unique_key', 'file_path', 'name'],
            'Class': ['unique_key', 'name', 'file_path'],
            'Method': ['unique_key', 'name', 'parent'],
            'Function': ['unique_key', 'name', 'file_path'],
            'Attribute': ['unique_key', 'name', 'parent'],
            'Generated_Description': ['unique_key', 'name']
        }
        
        logger.info("Creating indexes...")
        for collection_name, fields in indexes_config.items():
            if self.db.has_collection(collection_name):
                collection = self.db.collection(collection_name)
                
                # Create hash index for unique_key
                try:
                    collection.add_hash_index(fields=['unique_key'], unique=True)
                    logger.info(f"Created unique hash index on {collection_name}.unique_key")
                except Exception as e:
                    if "duplicate" not in str(e).lower():
                        logger.warning(f"Error creating unique index on {collection_name}: {e}")
                
                # Create persistent indexes for other fields
                for field in fields[1:]:  # Skip unique_key as it's already indexed
                    try:
                        collection.add_persistent_index(fields=[field], sparse=True)
                        logger.info(f"Created index on {collection_name}.{field}")
                    except Exception as e:
                        if "duplicate" not in str(e).lower():
                            logger.warning(f"Error creating index on {collection_name}.{field}: {e}")
                
                # Create fulltext index for searchable fields
                if collection_name in ['Class', 'Method', 'Function']:
                    try:
                        collection.add_fulltext_index(fields=['docstring'])
                        collection.add_fulltext_index(fields=['source_code'])
                        logger.info(f"Created fulltext indexes on {collection_name}")
                    except Exception as e:
                        logger.warning(f"Error creating fulltext index on {collection_name}: {e}")
    
    def create_graph(self):
        """Create a named graph for visualization"""
        graph_name = "code_knowledge_graph"
        
        if self.db.has_graph(graph_name):
            logger.info(f"Graph '{graph_name}' already exists")
            self.graph = self.db.graph(graph_name)
            return
        
        # Define edge definitions
        edge_definitions = [
            {
                'edge_collection': 'Contains',
                'from_vertex_collections': ['File'],
                'to_vertex_collections': ['Class', 'Function']
            },
            {
                'edge_collection': 'Has_Method',
                'from_vertex_collections': ['Class'],
                'to_vertex_collections': ['Method']
            },
            {
                'edge_collection': 'Has_Function',
                'from_vertex_collections': ['File', 'Class'],
                'to_vertex_collections': ['Function']
            },
            {
                'edge_collection': 'Has_Attribute',
                'from_vertex_collections': ['Class'],
                'to_vertex_collections': ['Attribute']
            },
            {
                'edge_collection': 'Calls',
                'from_vertex_collections': ['Method', 'Function'],
                'to_vertex_collections': ['Method', 'Function']
            },
            {
                'edge_collection': 'Inherits',
                'from_vertex_collections': ['Class'],
                'to_vertex_collections': ['Class']
            },
            {
                'edge_collection': 'Uses',
                'from_vertex_collections': ['Method', 'Function', 'Class'],
                'to_vertex_collections': ['Class', 'Function']
            },
            {
                'edge_collection': 'Describes',
                'from_vertex_collections': ['Generated_Description'],
                'to_vertex_collections': ['Class', 'Method', 'Function']
            }
        ]
        
        self.graph = self.db.create_graph(graph_name, edge_definitions=edge_definitions)
        logger.info(f"Created graph: {graph_name}")
    
    def _create_unique_key(self, node_data: Dict[str, Any]) -> str:
        """
        Create a unique key for a node based on its type and properties
        
        Args:
            node_data: Node data dictionary
            
        Returns:
            Unique key string
        """
        node_type = node_data.get('type', 'unknown')
        
        # Use the exported ID as the primary unique key (most reliable)
        if node_data.get('id'):
            return node_data['id']
        
        # Fallback: construct key from components with proper null handling
        file_path = (node_data.get('file_path') or '').strip()
        name = (node_data.get('name') or '').strip()
        parent = (node_data.get('parent') or '').strip()
        line_number = node_data.get('line_number', 0)
        
        # Normalize file path
        if file_path:
            file_path = file_path.replace('\\', '/')
        
        # Create unique key based on node type
        if node_type == 'file':
            return f"file:{file_path}" if file_path else f"file:unknown_{hash(str(node_data))}"
        elif node_type == 'class':
            return f"class:{file_path}:{name}" if file_path and name else f"class:unknown_{hash(str(node_data))}"
        elif node_type == 'method':
            if file_path and parent and name:
                return f"method:{file_path}:{parent}:{name}:{line_number}"
            else:
                return f"method:unknown_{hash(str(node_data))}"
        elif node_type == 'function':
            return f"function:{file_path}:{name}:{line_number}" if file_path and name else f"function:unknown_{hash(str(node_data))}"
        elif node_type == 'attribute':
            if file_path and parent and name:
                return f"attribute:{file_path}:{parent}:{name}:{line_number}"
            else:
                return f"attribute:unknown_{hash(str(node_data))}"
        elif node_type == 'generated_description':
            return f"desc:{file_path}:{name}:{line_number}" if file_path and name else f"desc:unknown_{hash(str(node_data))}"
        else:
            return f"{node_type}:unknown_{hash(str(node_data))}"
    
    def _validate_node_data(self, node_data: Dict[str, Any]) -> bool:
        """
        Validate node data before import
        
        Args:
            node_data: Node data dictionary
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(node_data, dict):
            logger.warning(f"Invalid node data type: {type(node_data)}")
            return False
        
        if not node_data.get('type'):
            logger.warning(f"Node missing type: {node_data}")
            return False
        
        if not node_data.get('name'):
            logger.warning(f"Node missing name: {node_data}")
            return False
        
        return True
    
    def import_nodes(self, nodes_file: str):
        """
        Import nodes from JSON file
        
        Args:
            nodes_file: Path to nodes JSON file
        """
        if not os.path.exists(nodes_file):
            logger.error(f"Nodes file not found: {nodes_file}")
            return
        
        with open(nodes_file, 'r', encoding='utf-8') as f:
            nodes_data = json.load(f)
        
        if not nodes_data:
            logger.warning("No nodes data found")
            return
        
        logger.info(f"Importing {len(nodes_data)} nodes...")
        
        # Group nodes by type
        nodes_by_type = {}
        for node_data in nodes_data:
            if not self._validate_node_data(node_data):
                continue
                
            node_type = node_data['type']
            if node_type not in nodes_by_type:
                nodes_by_type[node_type] = []
            nodes_by_type[node_type].append(node_data)
        
        # Import each node type
        for node_type, nodes in nodes_by_type.items():
            logger.info(f"Creating {len(nodes)} {node_type} nodes...")
            try:
                self._create_nodes_batch(node_type, nodes)
                logger.info(f"Created {len(nodes)} {node_type} nodes")
            except Exception as e:
                logger.error(f"Error creating {node_type} nodes: {e}")
                continue
    
    def _create_nodes_batch(self, node_type: str, nodes: List[Dict[str, Any]]):
        """
        Create nodes in batch for a specific type
        
        Args:
            node_type: Type of nodes to create
            nodes: List of node data
        """
        if not nodes:
            return
        
        # Map node type to collection name
        collection_mapping = {
            'file': 'File',
            'class': 'Class',
            'method': 'Method',
            'function': 'Function',
            'attribute': 'Attribute',
            'generated_description': 'Generated_Description'
        }
        
        collection_name = collection_mapping.get(node_type, node_type.title())
        
        if not self.db.has_collection(collection_name):
            logger.error(f"Collection {collection_name} does not exist")
            return
        
        collection = self.db.collection(collection_name)
        
        # Prepare documents
        documents = []
        seen_keys = set()
        
        for node_data in nodes:
            try:
                # Create unique key
                unique_key = self._create_unique_key(node_data)
                
                # Skip duplicates
                if unique_key in seen_keys:
                    logger.warning(f"Duplicate unique key found, skipping: {unique_key}")
                    continue
                seen_keys.add(unique_key)
                
                # Prepare document
                doc = {
                    '_key': unique_key.replace(':', '_').replace('/', '_').replace('.', '_'),
                    'unique_key': unique_key,
                    'name': node_data.get('name', '').strip(),
                    'file_path': (node_data.get('file_path') or '').replace('\\', '/'),
                    'line_number': node_data.get('line_number', 0),
                    'type': node_type
                }
                
                # Add optional properties
                if node_data.get('docstring'):
                    doc['docstring'] = node_data['docstring']
                
                if node_data.get('source_code'):
                    doc['source_code'] = node_data['source_code']
                
                if node_data.get('parent'):
                    doc['parent'] = node_data['parent']
                
                if node_data.get('embedding') and isinstance(node_data['embedding'], list):
                    doc['embedding'] = node_data['embedding']
                
                # Special handling for generated descriptions
                if node_type == 'generated_description':
                    doc['description'] = node_data.get('docstring', '')
                
                documents.append(doc)
                
            except Exception as e:
                logger.error(f"Error processing node {node_data.get('name', 'unknown')}: {e}")
                continue
        
        if not documents:
            logger.warning(f"No valid {node_type} documents to create")
            return
        
        # Bulk insert documents
        try:
            result = collection.import_bulk(documents, on_duplicate='ignore')
            logger.info(f"Imported {result['created']} documents, "
                       f"ignored {result.get('ignored', 0)} duplicates, "
                       f"errors: {result.get('errors', 0)}")
        except Exception as e:
            logger.error(f"Error bulk importing {node_type} nodes: {e}")
            # Try individual inserts as fallback
            success_count = 0
            for doc in documents:
                try:
                    collection.insert(doc, overwrite_mode='ignore')
                    success_count += 1
                except Exception as individual_error:
                    logger.error(f"Failed to insert individual document: {individual_error}")
            logger.info(f"Successfully inserted {success_count}/{len(documents)} documents individually")
    
    def import_relationships(self, relationships_file: str):
        """
        Import relationships from JSON file
        
        Args:
            relationships_file: Path to relationships JSON file
        """
        if not os.path.exists(relationships_file):
            logger.error(f"Relationships file not found: {relationships_file}")
            return
        
        with open(relationships_file, 'r', encoding='utf-8') as f:
            relationships_data = json.load(f)
        
        if not relationships_data:
            logger.warning("No relationships data found")
            return
        
        logger.info(f"Importing {len(relationships_data)} relationships...")
        
        # Group relationships by type
        relationships_by_type = {}
        for rel_data in relationships_data:
            rel_type = rel_data.get('type', 'UNKNOWN')
            if rel_type not in relationships_by_type:
                relationships_by_type[rel_type] = []
            relationships_by_type[rel_type].append(rel_data)
        
        # Import each relationship type
        for rel_type, relationships in relationships_by_type.items():
            logger.info(f"Creating {len(relationships)} {rel_type} relationships...")
            try:
                self._create_relationships_batch(rel_type, relationships)
                logger.info(f"Created {len(relationships)} {rel_type} relationships")
            except Exception as e:
                logger.error(f"Error creating {rel_type} relationships: {e}")
                continue
    
    def _create_relationships_batch(self, rel_type: str, relationships: List[Dict[str, Any]]):
        """
        Create relationships in batch for a specific type
        
        Args:
            rel_type: Type of relationship
            relationships: List of relationship data
        """
        if not relationships:
            return
        
        # Map relationship type to collection name (capitalize first letter of each word)
        edge_collection_name = '_'.join(word.capitalize() for word in rel_type.split('_'))
        
        # Create edge collection if it doesn't exist
        if not self.db.has_collection(edge_collection_name):
            self.db.create_collection(edge_collection_name, edge=True)
            logger.info(f"Created edge collection: {edge_collection_name}")
        
        edge_collection = self.db.collection(edge_collection_name)
        
        # Prepare edge documents
        edges = []
        
        for rel_data in relationships:
            source = rel_data.get('source')
            target = rel_data.get('target')
            
            if not source or not target:
                logger.warning(f"Invalid relationship data: {rel_data}")
                continue
            
            # Extract unique keys and map to collection/key format
            source_info = self._get_document_id(source)
            target_info = self._get_document_id(target)
            
            if not source_info or not target_info:
                logger.warning(f"Could not resolve relationship: {source} -> {target}")
                continue
            
            edge_doc = {
                '_from': source_info,
                '_to': target_info,
                'type': rel_type
            }
            
            edges.append(edge_doc)
        
        if not edges:
            logger.warning(f"No valid {rel_type} edges to create")
            return
        
        # Bulk insert edges
        try:
            result = edge_collection.import_bulk(edges, on_duplicate='ignore')
            logger.info(f"Imported {result['created']} edges, "
                       f"ignored {result.get('ignored', 0)} duplicates, "
                       f"errors: {result.get('errors', 0)}")
        except Exception as e:
            logger.error(f"Error bulk importing {rel_type} relationships: {e}")
            # Try individual inserts
            success_count = 0
            for edge in edges:
                try:
                    edge_collection.insert(edge, overwrite_mode='ignore')
                    success_count += 1
                except Exception as individual_error:
                    logger.error(f"Failed to insert edge: {individual_error}")
            logger.info(f"Successfully inserted {success_count}/{len(edges)} edges individually")
    
    def _get_document_id(self, unique_key: str) -> str:
        """
        Get ArangoDB document ID from unique key
        
        Args:
            unique_key: Unique key of the node
            
        Returns:
            ArangoDB document ID in format "collection/_key"
        """
        # Determine collection from unique key prefix
        collection_map = {
            'file:': 'File',
            'class:': 'Class',
            'method:': 'Method',
            'function:': 'Function',
            'attribute:': 'Attribute',
            'desc:': 'Generated_Description'
        }
        
        collection_name = None
        for prefix, coll in collection_map.items():
            if unique_key.startswith(prefix):
                collection_name = coll
                break
        
        if not collection_name:
            logger.warning(f"Could not determine collection for key: {unique_key}")
            return None
        
        # Convert unique_key to valid _key format
        doc_key = unique_key.replace(':', '_').replace('/', '_').replace('.', '_')
        
        return f"{collection_name}/{doc_key}"
    
    def import_knowledge_graph(self, export_dir: str):
        """
        Import complete knowledge graph from export directory
        
        Args:
            export_dir: Directory containing exported knowledge graph files
        """
        export_path = Path(export_dir)
        
        if not export_path.exists():
            logger.error(f"Export directory not found: {export_dir}")
            return
        
        nodes_file = export_path / "nodes.json"
        relationships_file = export_path / "relationships.json"
        
        logger.info("Starting knowledge graph import...")
        
        # Import nodes first
        if nodes_file.exists():
            self.import_nodes(str(nodes_file))
        else:
            logger.error("nodes.json file not found")
            return
        
        # Then import relationships
        if relationships_file.exists():
            self.import_relationships(str(relationships_file))
        else:
            logger.warning("relationships.json file not found, skipping relationships")
        
        logger.info("Knowledge graph import completed")
        
        # Print final statistics
        self._print_import_statistics()
    
    def _print_import_statistics(self):
        """Print import statistics"""
        try:
            logger.info("=== IMPORT STATISTICS ===")
            
            # Count documents by collection
            collections = [c['name'] for c in self.db.collections() 
                         if not c['name'].startswith('_')]
            
            logger.info("Document counts:")
            total_docs = 0
            for coll_name in collections:
                if not coll_name.endswith('_'):  # Skip edge collections for now
                    collection = self.db.collection(coll_name)
                    count = collection.count()
                    if count > 0:
                        logger.info(f"  {coll_name}: {count}")
                        total_docs += count
            
            logger.info("\nEdge counts:")
            total_edges = 0
            for coll_name in collections:
                collection = self.db.collection(coll_name)
                if collection.properties().get('type') == 3:  # Edge collection
                    count = collection.count()
                    if count > 0:
                        logger.info(f"  {coll_name}: {count}")
                        total_edges += count
            
            logger.info(f"\nTotal documents: {total_docs}")
            logger.info(f"Total edges: {total_edges}")
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")


def import_knowledge_graph_to_arango(export_dir: str,
                                    arango_host: str = "http://localhost:8529",
                                    arango_user: str = "root",
                                    arango_password: str = "",
                                    db_name: str = "knowledge_graph",
                                    clear_existing: bool = True):
    """
    Complete function to import knowledge graph to ArangoDB
    
    Args:
        export_dir: Directory containing exported knowledge graph
        arango_host: ArangoDB server URL
        arango_user: ArangoDB username
        arango_password: ArangoDB password
        db_name: Name of database to create/use
        clear_existing: Whether to clear existing data
    """
    try:
        # Initialize importer
        importer = ArangoKnowledgeGraphImporter(arango_host, arango_user, arango_password, db_name)
        
        # Clear existing data if requested
        if clear_existing:
            importer.clear_database()
        
        # Create collections and indexes
        importer.create_collections()
        importer.create_indexes()
        
        # Create named graph for visualization
        importer.create_graph()
        
        # Import the knowledge graph
        importer.import_knowledge_graph(export_dir)
        
        logger.info("Knowledge graph import completed successfully!")
        logger.info(f"\nAccess ArangoDB Web UI at: {arango_host}")
        logger.info(f"Database: {db_name}")
        logger.info(f"Username: {arango_user}")
        
    except Exception as e:
        logger.error(f"Error importing knowledge graph: {e}")
        raise


if __name__ == "__main__":
    # Example usage
    export_directory = "kg_export"  # Directory containing nodes.json and relationships.json
    
    # Import with default ArangoDB settings
    # Note: Default ArangoDB password is empty string for fresh installations
    import_knowledge_graph_to_arango(
        export_dir=export_directory,
        arango_host="http://localhost:8529",
        arango_user="root",
        arango_password="",  # Change this if you set a password
        db_name="knowledge_graph",
        clear_existing=True
    )
