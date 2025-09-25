import json
import os
from pathlib import Path
from py2neo import Graph, Node, Relationship
from py2neo.bulk import create_nodes, create_relationships
from typing import List, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeGraphImporter:
    """
    Import knowledge graph data into Neo4j database
    """
    
    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "password"):
        """
        Initialize Neo4j connection
        
        Args:
            uri: Neo4j database URI
            user: Database username
            password: Database password
        """
        try:
            self.graph = Graph(uri, auth=(user, password))
            logger.info("Connected to Neo4j database successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def clear_database(self):
        """Clear all nodes and relationships from the database"""
        try:
            self.graph.run("MATCH (n) DETACH DELETE n")
            logger.info("Database cleared")
        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            raise
    
    def create_constraints_and_indexes(self):
        """Create necessary constraints and indexes"""
        constraints = [
            # Unique constraints for each node type
            "CREATE CONSTRAINT file_unique IF NOT EXISTS FOR (f:File) REQUIRE f.unique_key IS UNIQUE",
            "CREATE CONSTRAINT class_unique IF NOT EXISTS FOR (c:Class) REQUIRE c.unique_key IS UNIQUE", 
            "CREATE CONSTRAINT method_unique IF NOT EXISTS FOR (m:Method) REQUIRE m.unique_key IS UNIQUE",
            "CREATE CONSTRAINT function_unique IF NOT EXISTS FOR (f:Function) REQUIRE f.unique_key IS UNIQUE",
            "CREATE CONSTRAINT attribute_unique IF NOT EXISTS FOR (a:Attribute) REQUIRE a.unique_key IS UNIQUE",
            "CREATE CONSTRAINT description_unique IF NOT EXISTS FOR (d:Generated_Description) REQUIRE d.unique_key IS UNIQUE"
        ]
        
        indexes = [
            # Performance indexes
            "CREATE INDEX file_path_index IF NOT EXISTS FOR (f:File) ON (f.file_path)",
            "CREATE INDEX class_name_index IF NOT EXISTS FOR (c:Class) ON (c.name)",
            "CREATE INDEX method_name_index IF NOT EXISTS FOR (m:Method) ON (m.name)",
            "CREATE INDEX function_name_index IF NOT EXISTS FOR (f:Function) ON (f.name)"
        ]
        
        # Full-text search indexes (separate handling due to syntax variations)
        fulltext_indexes = [
            "CALL db.index.fulltext.createNodeIndex('code_search', ['Class', 'Method', 'Function'], ['name', 'docstring', 'source_code'])",
            "CALL db.index.fulltext.createNodeIndex('description_search', ['Generated_Description'], ['description'])"
        ]
        
        logger.info("Creating constraints...")
        for constraint in constraints:
            try:
                self.graph.run(constraint)
                logger.info(f"Created constraint: {constraint.split('(')[1].split(')')[0]}")
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.warning(f"Error creating constraint: {e}")
        
        logger.info("Creating indexes...")
        for index in indexes:
            try:
                self.graph.run(index)
                logger.info(f"Created index: {index.split('IF NOT EXISTS')[0].split()[-1]}")
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.warning(f"Error creating index: {e}")
        
        # Try to create fulltext indexes with fallback
        for fulltext_index in fulltext_indexes:
            try:
                self.graph.run(fulltext_index)
                index_name = fulltext_index.split("'")[1]
                logger.info(f"Created fulltext index: {index_name}")
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.warning(f"Error creating fulltext index: {e}")
                    # Try alternative syntax for newer Neo4j versions
                    if "code_search" in fulltext_index:
                        try:
                            alt_query = "CREATE FULLTEXT INDEX code_search IF NOT EXISTS FOR (n:Class|Method|Function) ON EACH [n.name, n.docstring, n.source_code]"
                            self.graph.run(alt_query)
                            logger.info("Created fulltext index: code_search (alternative syntax)")
                        except:
                            logger.warning("Could not create code_search fulltext index")
                    elif "description_search" in fulltext_index:
                        try:
                            alt_query = "CREATE FULLTEXT INDEX description_search IF NOT EXISTS FOR (d:Generated_Description) ON EACH [d.description]"
                            self.graph.run(alt_query)
                            logger.info("Created fulltext index: description_search (alternative syntax)")
                        except:
                            logger.warning("Could not create description_search fulltext index")
        
        logger.info("Constraints and indexes created successfully")
    
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
                # Continue with other node types
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
        
        # Prepare node data with unique keys
        processed_nodes = []
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
                
                # Prepare node properties
                properties = {
                    'unique_key': unique_key,
                    'name': node_data.get('name', '').strip(),
                    'file_path': (node_data.get('file_path') or '').replace('\\', '/'),
                    'line_number': node_data.get('line_number', 0)
                }
                
                # Add type-specific properties
                if node_data.get('docstring'):
                    properties['docstring'] = node_data['docstring']
                
                if node_data.get('source_code'):
                    properties['source_code'] = node_data['source_code']
                
                if node_data.get('parent'):
                    properties['parent'] = node_data['parent']
                
                # Handle embeddings
                if node_data.get('embedding') and isinstance(node_data['embedding'], list):
                    properties['embedding'] = node_data['embedding']
                
                # Special handling for generated descriptions
                if node_type == 'generated_description':
                    properties['description'] = node_data.get('docstring', '')
                
                processed_nodes.append(properties)
                
            except Exception as e:
                logger.error(f"Error processing node {node_data.get('name', 'unknown')}: {e}")
                continue
        
        if not processed_nodes:
            logger.warning(f"No valid {node_type} nodes to create")
            return
        
        # Map node type to Neo4j label
        label_mapping = {
            'file': 'File',
            'class': 'Class',
            'method': 'Method',
            'function': 'Function',
            'attribute': 'Attribute',
            'generated_description': 'Generated_Description'
        }
        
        label = label_mapping.get(node_type, node_type.title())
        
        # Use different transaction handling based on py2neo version
        try:
            # Try modern transaction handling first
            with self.graph.begin() as tx:
                try:
                    create_nodes(tx, processed_nodes, labels={label}, keys=['unique_key'])
                except Exception as e:
                    logger.error(f"Bulk create failed for {node_type}, trying individual creates...")
                    # Fallback: create nodes individually
                    for node_props in processed_nodes:
                        try:
                            query = f"CREATE (n:{label} $props)"
                            tx.run(query, props=node_props)
                        except Exception as individual_error:
                            logger.error(f"Failed to create individual {node_type} node: {individual_error}")
                            logger.error(f"Node data: {node_props}")
                            
        except (AttributeError, TypeError) as tx_error:
            # Fallback for older py2neo versions
            logger.info(f"Using direct graph operations for {node_type} nodes...")
            tx = self.graph.begin()
            try:
                try:
                    create_nodes(tx, processed_nodes, labels={label}, keys=['unique_key'])
                    tx.commit()
                except Exception as e:
                    logger.error(f"Bulk create failed for {node_type}, trying individual creates...")
                    # Rollback and try individual creates
                    tx.rollback()
                    tx = self.graph.begin()
                    
                    for node_props in processed_nodes:
                        try:
                            query = f"CREATE (n:{label} $props)"
                            tx.run(query, props=node_props)
                        except Exception as individual_error:
                            logger.error(f"Failed to create individual {node_type} node: {individual_error}")
                            logger.error(f"Node data: {node_props}")
                    tx.commit()
                    
            except Exception as final_error:
                logger.error(f"All transaction methods failed for {node_type}: {final_error}")
                if hasattr(tx, 'rollback'):
                    tx.rollback()
    
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
        
        valid_relationships = []
        
        for rel_data in relationships:
            source = rel_data.get('source')
            target = rel_data.get('target')
            
            if not source or not target:
                logger.warning(f"Invalid relationship data: {rel_data}")
                continue
            
            valid_relationships.append({
                'source_key': self._extract_unique_key_from_reference(source),
                'target_key': self._extract_unique_key_from_reference(target)
            })
        
        if not valid_relationships:
            logger.warning(f"No valid {rel_type} relationships to create")
            return
        
        # Create relationships using Cypher query with proper transaction handling
        try:
            # Try modern transaction handling first
            with self.graph.begin() as tx:
                for rel in valid_relationships:
                    try:
                        query = f"""
                        MATCH (source {{unique_key: $source_key}})
                        MATCH (target {{unique_key: $target_key}})
                        CREATE (source)-[:{rel_type.upper()}]->(target)
                        """
                        tx.run(query, source_key=rel['source_key'], target_key=rel['target_key'])
                    except Exception as e:
                        logger.error(f"Error creating relationship {rel_type}: {e}")
                        logger.error(f"Source: {rel['source_key']}, Target: {rel['target_key']}")
                        
        except (AttributeError, TypeError) as tx_error:
            # Fallback for older py2neo versions
            logger.info(f"Using direct graph operations for {rel_type} relationships...")
            tx = self.graph.begin()
            try:
                for rel in valid_relationships:
                    try:
                        query = f"""
                        MATCH (source {{unique_key: $source_key}})
                        MATCH (target {{unique_key: $target_key}})
                        CREATE (source)-[:{rel_type.upper()}]->(target)
                        """
                        tx.run(query, source_key=rel['source_key'], target_key=rel['target_key'])
                    except Exception as e:
                        logger.error(f"Error creating relationship {rel_type}: {e}")
                        logger.error(f"Source: {rel['source_key']}, Target: {rel['target_key']}")
                tx.commit()
            except Exception as final_error:
                logger.error(f"Transaction failed for {rel_type} relationships: {final_error}")
                if hasattr(tx, 'rollback'):
                    tx.rollback()
    
    def _extract_unique_key_from_reference(self, reference: str) -> str:
        """
        Extract unique key from node reference
        
        Args:
            reference: Node reference string (e.g., "class:file.py:ClassName")
            
        Returns:
            Unique key for the node
        """
        # If it looks like a hash ID, use it directly
        if len(reference) == 32 and reference.isalnum():
            return reference
        
        # Otherwise, use the reference as is (it should be a unique key format)
        return reference
    
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
            # Count nodes by type
            node_counts = {}
            result = self.graph.run("MATCH (n) RETURN labels(n)[0] as label, count(n) as count")
            for record in result:
                label = record['label']
                count = record['count']
                node_counts[label] = count
            
            # Count relationships by type
            rel_counts = {}
            result = self.graph.run("MATCH ()-[r]->() RETURN type(r) as rel_type, count(r) as count")
            for record in result:
                rel_type = record['rel_type']
                count = record['count']
                rel_counts[rel_type] = count
            
            logger.info("=== IMPORT STATISTICS ===")
            logger.info("Node counts:")
            for label, count in node_counts.items():
                logger.info(f"  {label}: {count}")
            
            logger.info("Relationship counts:")
            for rel_type, count in rel_counts.items():
                logger.info(f"  {rel_type}: {count}")
            
            total_nodes = sum(node_counts.values())
            total_rels = sum(rel_counts.values())
            logger.info(f"Total nodes: {total_nodes}")
            logger.info(f"Total relationships: {total_rels}")
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")


def import_knowledge_graph_to_neo4j(export_dir: str, 
                                   neo4j_uri: str = "bolt://localhost:7687",
                                   neo4j_user: str = "neo4j", 
                                   neo4j_password: str = "password",
                                   clear_existing: bool = True):
    """
    Complete function to import knowledge graph to Neo4j
    
    Args:
        export_dir: Directory containing exported knowledge graph
        neo4j_uri: Neo4j database URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        clear_existing: Whether to clear existing data
    """
    try:
        # Initialize importer
        importer = KnowledgeGraphImporter(neo4j_uri, neo4j_user, neo4j_password)
        
        # Clear existing data if requested
        if clear_existing:
            importer.clear_database()
        
        # Create constraints and indexes
        importer.create_constraints_and_indexes()
        
        # Import the knowledge graph
        importer.import_knowledge_graph(export_dir)
        
        logger.info("Knowledge graph import completed successfully!")
        
    except Exception as e:
        logger.error(f"Error importing knowledge graph: {e}")
        raise


if __name__ == "__main__":
    # Example usage
    export_directory = "kg_export"  # Directory containing nodes.json and relationships.json
    
    # Import with default Neo4j settings
    import_knowledge_graph_to_neo4j(
        export_dir=export_directory,
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
        clear_existing=True
    )