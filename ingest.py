import json
import os
from pathlib import Path
from arango import ArangoClient
from typing import List, Dict, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedArangoKnowledgeGraphImporter:
    """
    Import enhanced RAG-optimized knowledge graph data into ArangoDB database
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
            self.client = ArangoClient(hosts=host)
            sys_db = self.client.db('_system', username=username, password=password)
            
            if not sys_db.has_database(db_name):
                sys_db.create_database(db_name)
                logger.info(f"Created database: {db_name}")
            
            self.db = self.client.db(db_name, username=username, password=password)
            self.db_name = db_name
            
            logger.info(f"Connected to ArangoDB database '{db_name}' successfully")
        except Exception as e:
            logger.error(f"Failed to connect to ArangoDB: {e}")
            raise
    
    def clear_database(self):
        """Clear all collections from the database"""
        try:
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
        doc_collections = ['File', 'Class', 'Method', 'Function', 'Attribute', 'Generated_Description']
        edge_collections = ['Contains', 'Has_Method', 'Has_Function', 'Has_Attribute', 
                          'Calls', 'Inherits', 'Uses', 'Describes', 'Related_To']
        
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
        """Create necessary indexes for performance including RAG-specific fields"""
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
                
                # Create unique hash index on unique_key
                try:
                    collection.add_index({
                        'type': 'hash',
                        'fields': ['unique_key'],
                        'unique': True
                    })
                    logger.info(f"Created unique hash index on {collection_name}.unique_key")
                except Exception as e:
                    if "duplicate" not in str(e).lower():
                        logger.warning(f"Error creating unique index on {collection_name}: {e}")
                
                # Create hash indexes for other fields (faster than persistent for equality checks)
                for field in fields[1:]:
                    try:
                        collection.add_index({
                            'type': 'hash',
                            'fields': [field],
                            'sparse': True
                        })
                        logger.info(f"Created hash index on {collection_name}.{field}")
                    except Exception as e:
                        if "duplicate" not in str(e).lower():
                            logger.warning(f"Error creating index on {collection_name}.{field}: {e}")
                
                # Create fulltext indexes for RAG fields
                if collection_name in ['Class', 'Method', 'Function', 'File']:
                    fulltext_fields = ['docstring', 'source_code', 'full_context', 'summary']
                    for field in fulltext_fields:
                        try:
                            collection.add_index({
                                'type': 'fulltext',
                                'fields': [field],
                                'minLength': 2
                            })
                            logger.info(f"Created fulltext index on {collection_name}.{field}")
                        except Exception as e:
                            if "duplicate" not in str(e).lower():
                                logger.warning(f"Error creating fulltext index on {collection_name}.{field}: {e}")
    
    def create_graph(self):
        """Create a named graph for visualization"""
        graph_name = "code_knowledge_graph"
        
        if self.db.has_graph(graph_name):
            logger.info(f"Graph '{graph_name}' already exists")
            self.graph = self.db.graph(graph_name)
            return
        
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
            },
            {
                'edge_collection': 'Related_To',
                'from_vertex_collections': ['Class', 'Method', 'Function'],
                'to_vertex_collections': ['Class', 'Method', 'Function', 'Attribute']
            }
        ]
        
        self.graph = self.db.create_graph(graph_name, edge_definitions=edge_definitions)
        logger.info(f"Created graph: {graph_name}")
    
    def _create_unique_key(self, node_data: Dict[str, Any]) -> str:
        """Create a unique key for a node based on its type and properties"""
        node_type = node_data.get('type', 'unknown')
        
        if node_data.get('id'):
            return node_data['id']
        
        file_path = (node_data.get('file_path') or '').strip()
        name = (node_data.get('name') or '').strip()
        parent = (node_data.get('parent') or '').strip()
        line_number = node_data.get('line_number', 0)
        
        if file_path:
            file_path = file_path.replace('\\', '/')
        
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
        """Validate node data before import"""
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
        """Import nodes from JSON file with enhanced RAG fields"""
        if not os.path.exists(nodes_file):
            logger.error(f"Nodes file not found: {nodes_file}")
            return
        
        with open(nodes_file, 'r', encoding='utf-8') as f:
            nodes_data = json.load(f)
        
        if not nodes_data:
            logger.warning("No nodes data found")
            return
        
        logger.info(f"Importing {len(nodes_data)} nodes with enhanced RAG fields...")
        
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
        """Create nodes in batch with all enhanced RAG fields"""
        if not nodes:
            return
        
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
        
        documents = []
        seen_keys = set()
        
        # Store mapping of original IDs to unique keys for relationship resolution
        if not hasattr(self, '_id_mapping'):
            self._id_mapping = {}
        
        for node_data in nodes:
            try:
                unique_key = self._create_unique_key(node_data)
                
                if unique_key in seen_keys:
                    logger.warning(f"Duplicate unique key found, skipping: {unique_key}")
                    continue
                seen_keys.add(unique_key)
                
                # Base document with core fields
                doc = {
                    '_key': unique_key.replace(':', '_').replace('/', '_').replace('.', '_'),
                    'unique_key': unique_key,
                    'name': node_data.get('name', '').strip(),
                    'file_path': (node_data.get('file_path') or '').replace('\\', '/'),
                    'line_number': node_data.get('line_number', 0),
                    'type': node_type
                }
                
                # Store original ID mapping if present
                if node_data.get('id'):
                    original_id = node_data['id']
                    self._id_mapping[original_id] = unique_key
                    doc['original_id'] = original_id
                
                # Add standard optional properties
                if node_data.get('docstring'):
                    doc['docstring'] = node_data['docstring']
                
                if node_data.get('source_code'):
                    doc['source_code'] = node_data['source_code']
                
                if node_data.get('parent'):
                    doc['parent'] = node_data['parent']
                
                if node_data.get('embedding') and isinstance(node_data['embedding'], list):
                    doc['embedding'] = node_data['embedding']
                
                # ========== ADD ENHANCED RAG FIELDS ==========
                
                # Full context - complete code with surrounding context
                if node_data.get('full_context'):
                    doc['full_context'] = node_data['full_context']
                
                # Summary - human-readable summary for quick understanding
                if node_data.get('summary'):
                    doc['summary'] = node_data['summary']
                
                # Signature - function/method signature
                if node_data.get('signature'):
                    doc['signature'] = node_data['signature']
                
                # Parent context - parent class/module context
                if node_data.get('parent_context'):
                    doc['parent_context'] = node_data['parent_context']
                
                # Related elements - IDs of related code elements
                if node_data.get('related_elements'):
                    if isinstance(node_data['related_elements'], list):
                        doc['related_elements'] = node_data['related_elements']
                    else:
                        doc['related_elements'] = []
                
                # Metadata - additional contextual information
                if node_data.get('metadata'):
                    if isinstance(node_data['metadata'], dict):
                        doc['metadata'] = node_data['metadata']
                    else:
                        doc['metadata'] = {}
                
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
            
            # Log enhanced field coverage
            enhanced_count = sum(1 for d in documents if d.get('full_context'))
            summary_count = sum(1 for d in documents if d.get('summary'))
            logger.info(f"  Enhanced fields: {enhanced_count} with full_context, "
                       f"{summary_count} with summary")
            
        except Exception as e:
            logger.error(f"Error bulk importing {node_type} nodes: {e}")
            # Fallback to individual inserts
            success_count = 0
            for doc in documents:
                try:
                    collection.insert(doc, overwrite_mode='ignore')
                    success_count += 1
                except Exception as individual_error:
                    logger.error(f"Failed to insert individual document: {individual_error}")
            logger.info(f"Successfully inserted {success_count}/{len(documents)} documents individually")
    
    def import_relationships(self, relationships_file: str):
        """Import relationships from JSON file"""
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
        """Create relationships in batch for a specific type"""
        if not relationships:
            return
        
        edge_collection_name = '_'.join(word.capitalize() for word in rel_type.split('_'))
        
        if not self.db.has_collection(edge_collection_name):
            self.db.create_collection(edge_collection_name, edge=True)
            logger.info(f"Created edge collection: {edge_collection_name}")
        
        edge_collection = self.db.collection(edge_collection_name)
        
        edges = []
        unresolved_count = 0
        
        for rel_data in relationships:
            source = rel_data.get('source')
            target = rel_data.get('target')
            
            if not source or not target:
                logger.warning(f"Invalid relationship data: {rel_data}")
                continue
            
            source_info = self._get_document_id(source)
            target_info = self._get_document_id(target)
            
            if not source_info or not target_info:
                unresolved_count += 1
                continue
            
            edge_doc = {
                '_from': source_info,
                '_to': target_info,
                'type': rel_type
            }
            
            edges.append(edge_doc)
        
        if unresolved_count > 0:
            logger.info(f"  Could not resolve {unresolved_count}/{len(relationships)} {rel_type} relationships (nodes may not exist)")
        
        if not edges:
            logger.warning(f"No valid {rel_type} edges to create")
            return
        
        try:
            result = edge_collection.import_bulk(edges, on_duplicate='ignore')
            logger.info(f"Imported {result['created']} edges, "
                       f"ignored {result.get('ignored', 0)} duplicates, "
                       f"errors: {result.get('errors', 0)}")
        except Exception as e:
            logger.error(f"Error bulk importing {rel_type} relationships: {e}")
            success_count = 0
            for edge in edges:
                try:
                    edge_collection.insert(edge, overwrite_mode='ignore')
                    success_count += 1
                except Exception as individual_error:
                    logger.error(f"Failed to insert edge: {individual_error}")
            logger.info(f"Successfully inserted {success_count}/{len(edges)} edges individually")
    
    def _get_document_id(self, unique_key: str) -> Optional[str]:
        """Get ArangoDB document ID from unique key"""
        # First check if this is an original ID that was mapped
        if hasattr(self, '_id_mapping') and unique_key in self._id_mapping:
            unique_key = self._id_mapping[unique_key]
        
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
        
        # If we found a collection with prefix, use direct mapping
        if collection_name:
            doc_key = unique_key.replace(':', '_').replace('/', '_').replace('.', '_')
            return f"{collection_name}/{doc_key}"
        
        # No prefix found - this might be an original hash ID
        # Try to find the document across all collections
        for coll_name in ['File', 'Class', 'Method', 'Function', 'Attribute', 'Generated_Description']:
            try:
                collection = self.db.collection(coll_name)
                # Search by unique_key field or _key or original_id
                doc_key = unique_key.replace(':', '_').replace('/', '_').replace('.', '_')
                
                # Try direct _key lookup
                if collection.has(doc_key):
                    return f"{coll_name}/{doc_key}"
                
                # Try searching by unique_key field or original_id
                aql = f"""
                FOR doc IN {coll_name}
                    FILTER doc.unique_key == @unique_key OR doc._key == @doc_key OR doc.original_id == @unique_key
                    LIMIT 1
                    RETURN doc._id
                """
                cursor = self.db.aql.execute(aql, bind_vars={'unique_key': unique_key, 'doc_key': doc_key})
                results = list(cursor)
                if results:
                    return results[0]
            except Exception:
                # Silently continue to next collection
                continue
        
        # Document not found - this is normal for external references
        return None
    
    def import_knowledge_graph(self, export_dir: str):
        """Import complete knowledge graph from export directory"""
        export_path = Path(export_dir)
        
        if not export_path.exists():
            logger.error(f"Export directory not found: {export_dir}")
            return
        
        nodes_file = export_path / "nodes.json"
        relationships_file = export_path / "relationships.json"
        
        logger.info("="*60)
        logger.info("Starting Enhanced Knowledge Graph Import")
        logger.info("="*60)
        
        if nodes_file.exists():
            self.import_nodes(str(nodes_file))
        else:
            logger.error("nodes.json file not found")
            return
        
        if relationships_file.exists():
            self.import_relationships(str(relationships_file))
        else:
            logger.warning("relationships.json file not found, skipping relationships")
        
        logger.info("="*60)
        logger.info("Knowledge graph import completed")
        logger.info("="*60)
        
        self._print_import_statistics()
    
    def _print_import_statistics(self):
        """Print import statistics including RAG field coverage"""
        try:
            logger.info("\n" + "="*60)
            logger.info("IMPORT STATISTICS")
            logger.info("="*60)
            
            collections = [c['name'] for c in self.db.collections() 
                         if not c['name'].startswith('_')]
            
            logger.info("\nDocument counts:")
            total_docs = 0
            for coll_name in collections:
                collection = self.db.collection(coll_name)
                if collection.properties().get('type') != 3:  # Not edge collection
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
            
            # Check RAG field coverage
            logger.info("\nRAG Field Coverage:")
            for coll_name in ['Class', 'Method', 'Function', 'File']:
                if self.db.has_collection(coll_name):
                    collection = self.db.collection(coll_name)
                    
                    try:
                        # Simple count queries without cursor count
                        aql_full_context = f"""
                        FOR doc IN {coll_name}
                            FILTER doc.full_context != null
                            RETURN 1
                        """
                        full_context_results = list(self.db.aql.execute(aql_full_context))
                        full_context_count = len(full_context_results)
                        
                        aql_summary = f"""
                        FOR doc IN {coll_name}
                            FILTER doc.summary != null
                            RETURN 1
                        """
                        summary_results = list(self.db.aql.execute(aql_summary))
                        summary_count = len(summary_results)
                        
                        total = collection.count()
                        if total > 0:
                            logger.info(f"  {coll_name}:")
                            logger.info(f"    - with full_context: {full_context_count}/{total}")
                            logger.info(f"    - with summary: {summary_count}/{total}")
                    except Exception as e:
                        logger.warning(f"Could not get RAG field stats for {coll_name}: {e}")
            
            logger.info(f"\nTotal documents: {total_docs}")
            logger.info(f"Total edges: {total_edges}")
            logger.info("="*60 + "\n")
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")


def import_knowledge_graph_to_arango(export_dir: str,
                                    arango_host: str = "http://localhost:8529",
                                    arango_user: str = "root",
                                    arango_password: str = "",
                                    db_name: str = "knowledge_graph",
                                    clear_existing: bool = True):
    """
    Complete function to import enhanced RAG-optimized knowledge graph to ArangoDB
    
    Args:
        export_dir: Directory containing exported knowledge graph
        arango_host: ArangoDB server URL
        arango_user: ArangoDB username
        arango_password: ArangoDB password
        db_name: Name of database to create/use
        clear_existing: Whether to clear existing data
    """
    try:
        importer = EnhancedArangoKnowledgeGraphImporter(
            arango_host, arango_user, arango_password, db_name
        )
        
        if clear_existing:
            importer.clear_database()
        
        importer.create_collections()
        importer.create_indexes()
        importer.create_graph()
        importer.import_knowledge_graph(export_dir)
        
        logger.info("\n" + "="*60)
        logger.info("SUCCESS: Enhanced Knowledge Graph Import Completed!")
        logger.info("="*60)
        logger.info(f"\nAccess ArangoDB Web UI at: {arango_host}")
        logger.info(f"Database: {db_name}")
        logger.info(f"Username: {arango_user}")
        logger.info("\nEnhanced RAG fields imported:")
        logger.info("  ✓ full_context - Complete code with context")
        logger.info("  ✓ summary - Human-readable summaries")
        logger.info("  ✓ signature - Function/method signatures")
        logger.info("  ✓ parent_context - Parent class/module context")
        logger.info("  ✓ related_elements - Related code element IDs")
        logger.info("  ✓ metadata - Additional contextual information")
        logger.info("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"Error importing knowledge graph: {e}")
        raise


if __name__ == "__main__":
    # Example usage
    export_directory = "kg_export"
    
    import_knowledge_graph_to_arango(
        export_dir=export_directory,
        arango_host="http://localhost:8529",
        arango_user="root",
        arango_password="",
        db_name="knowledge_graph",
        clear_existing=True
    )
