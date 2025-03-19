import json
import os
import networkx as nx
import matplotlib.pyplot as plt
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer, util
import torch

logger = logging.getLogger(__name__)

class KnowledgeGraph:
    def __init__(self, model: Optional[SentenceTransformer] = None):
        self.entities = {}  # id -> entity
        self.relations = {}  # id -> relation
        self.triples = []   # list of (head, relation, tail)
        self.documents = {}  # entity_id -> document
        self.graph = nx.DiGraph()
        self.llm = None  # LLM instance for embedding computation
        self.model = model  # SentenceTransformer model
        
    def set_llm(self, llm):
        """Set LLM instance for embedding computation"""
        self.llm = llm
        logger.info("Set LLM instance for embedding computation")
        
    def load_from_files(self, entities_file, relations_file, triples_file, documents_file):
        """Load knowledge graph data from JSON files"""
        with open(entities_file, 'r', encoding='utf-8') as f:
            entities_data = json.load(f)
            for entity in entities_data:
                self.entities[entity['id']] = entity
        
        with open(relations_file, 'r', encoding='utf-8') as f:
            relations_data = json.load(f)
            for relation in relations_data:
                self.relations[relation['id']] = relation
        
        with open(triples_file, 'r', encoding='utf-8') as f:
            triples_data = json.load(f)
            for triple in triples_data:
                self.triples.append(triple)
        
        with open(documents_file, 'r', encoding='utf-8') as f:
            documents_data = json.load(f)
            for doc in documents_data:
                self.documents[doc['entity_id']] = doc['content']
        
        # Build graph
        self.build_graph()
        
        logger.info(f"Loaded knowledge graph with {len(self.entities)} entities, {len(self.relations)} relations, {len(self.triples)} triples, and {len(self.documents)} documents")
    
    def build_graph(self):
        """Build a NetworkX graph from triples"""
        self.graph = nx.DiGraph()
        
        # Add nodes
        for entity_id, entity in self.entities.items():
            self.graph.add_node(entity_id, label=entity['name'], type=entity.get('type', 'Entity'))
        
        # Add edges
        for triple in self.triples:
            head = triple['head']
            tail = triple['tail']
            relation = triple['relation']
            
            # Skip if head or tail entity doesn't exist
            if head not in self.entities or tail not in self.entities:
                continue
                
            relation_name = self.relations[relation]['name'] if relation in self.relations else relation
            self.graph.add_edge(head, tail, relation=relation, label=relation_name)
    
    def get_relation_candidates(self, entity_id: str) -> List[Dict]:
        """Get all relations connected to an entity"""
        if entity_id not in self.graph:
            return []
        
        candidates = []
        # Outgoing relations
        for _, neighbor, data in self.graph.out_edges(entity_id, data=True):
            candidates.append({
                'entity_id': entity_id,
                'entity_name': self.graph.nodes[entity_id]['label'],
                'relation': data['relation'],
                'relation_name': data['label'],
                'target_id': neighbor,
                'target_name': self.graph.nodes[neighbor]['label'],
                'head': True
            })
        
        # Incoming relations
        for neighbor, _, data in self.graph.in_edges(entity_id, data=True):
            candidates.append({
                'entity_id': entity_id,
                'entity_name': self.graph.nodes[entity_id]['label'],
                'relation': data['relation'],
                'relation_name': data['label'],
                'target_id': neighbor,
                'target_name': self.graph.nodes[neighbor]['label'],
                'head': False
            })
        
        return candidates
    
    def prune_relations(self, entity_id: str, query: str) -> List[Dict]:
        """Rank and prune relations based on relevance to query"""
        candidates = self.get_relation_candidates(entity_id)
        if not candidates:
            return []
        
        # Use LLM for relation ranking if available
        if self.llm:
            try:
                entity_name = self.graph.nodes[entity_id]['label']
                ranked_relations = self.llm.rank_relations(query, entity_name, candidates)
                
                # Sort by score and keep top relations with score > 0.2
                ranked_relations = [rel for rel in ranked_relations if rel['score'] > 0.2]
                ranked_relations = sorted(ranked_relations, key=lambda x: x['score'], reverse=True)
                return ranked_relations
            except Exception as e:
                logger.error(f"Error in LLM relation ranking: {e}")
        
        # Get model for fallback embedding calculation
        model = self.model
        if model is None:
            logger.info("No pre-loaded model provided, creating a new SentenceTransformer instance")
            model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Fallback to embedding-based ranking
        relation_names = [f"{cand['relation_name']} connects {cand['entity_name']} to {cand['target_name']}" for cand in candidates]
        
        if self.llm and self.llm.use_embedding_api:
            # Use LLM's embedding API
            scores = self.llm.compute_similarity(query, relation_names)
        else:
            # Use local sentence transformer
            query_emb = model.encode(query, convert_to_tensor=True)
            relation_embs = model.encode(relation_names, convert_to_tensor=True)
            scores = util.pytorch_cos_sim(query_emb, relation_embs)[0].cpu().numpy()
        
        # Combine scores with candidates
        for i, cand in enumerate(candidates):
            cand['score'] = float(scores[i])
        
        # Sort by score and keep top relations with score > 0.2
        candidates = [cand for cand in candidates if cand['score'] > 0.2]
        candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
        return candidates
    
    def get_entity_candidates(self, entity_id: str, relation: str, is_head: bool) -> List[str]:
        """Get entities connected via a specific relation"""
        candidates = []
        
        if is_head:
            # Entity is head, get tail entities
            for _, tail in self.graph.out_edges(entity_id):
                edge_data = self.graph.get_edge_data(entity_id, tail)
                if edge_data['relation'] == relation:
                    candidates.append(tail)
        else:
            # Entity is tail, get head entities
            for head, _ in self.graph.in_edges(entity_id):
                edge_data = self.graph.get_edge_data(head, entity_id)
                if edge_data['relation'] == relation:
                    candidates.append(head)
        
        return candidates
    
    def search_entity_documents(self, entity_ids: List[str], query: str) -> List[Dict]:
        """Search for relevant information in entity documents"""
        if not entity_ids:
            return []
            
        results = []
        
        # Get model for embedding calculation
        model = self.model
        if model is None:
            logger.info("No pre-loaded model provided, creating a new SentenceTransformer instance")
            model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Prepare for document search
        all_sentences = []
        entity_sentence_map = {}  # Maps sentence index to entity ID
        
        for entity_id in entity_ids:
            if entity_id not in self.documents:
                continue
                
            # Split document into sentences
            document = self.documents[entity_id]
            sentences = re.split(r'(?<=[.!?])\s+', document)
            
            # Store sentence index to entity mapping
            start_idx = len(all_sentences)
            all_sentences.extend(sentences)
            end_idx = len(all_sentences)
            
            for i in range(start_idx, end_idx):
                entity_sentence_map[i] = entity_id
        
        if not all_sentences:
            return results
            
        # Calculate relevance scores
        if self.llm and self.llm.use_embedding_api:
            # Use LLM's embedding API
            scores = self.llm.compute_similarity(query, all_sentences)
        else:
            # Use local sentence transformer
            try:
                query_emb = model.encode(query, convert_to_tensor=True)
                sentences_emb = model.encode(all_sentences, convert_to_tensor=True)
                scores = util.pytorch_cos_sim(query_emb, sentences_emb)[0].cpu().numpy()
            except Exception as e:
                logger.error(f"Error computing sentence embeddings: {e}")
                # Fallback to simple keyword matching
                scores = []
                query_words = set(query.lower().split())
                for sentence in all_sentences:
                    sentence_words = set(sentence.lower().split())
                    overlap = len(query_words.intersection(sentence_words))
                    score = overlap / max(1, len(query_words))
                    scores.append(score)
        
        # Create results with scores
        for i, (sentence, score) in enumerate(zip(all_sentences, scores)):
            entity_id = entity_sentence_map[i]
            entity_name = self.graph.nodes[entity_id]['label']
            
            results.append({
                'entity_id': entity_id,
                'entity_name': entity_name,
                'text': sentence,
                'score': float(score)
            })
        
        # Sort by relevance
        results = sorted(results, key=lambda x: x['score'], reverse=True)
        return results[:20]  # Return top 20 sentences
    
    def visualize_graph(self, highlight_entities=None, max_nodes=50):
        """Visualize the knowledge graph using NetworkX and Matplotlib"""
        plt.figure(figsize=(12, 8))
        
        # Create a subgraph if highlight_entities is provided
        if highlight_entities:
            # Get nodes within 2 steps of highlighted nodes
            nodes_to_include = set(highlight_entities)
            for node in highlight_entities:
                if node in self.graph:
                    # Add outgoing neighbors
                    nodes_to_include.update(nx.descendants(self.graph, node))
                    # Add incoming neighbors
                    nodes_to_include.update(nx.ancestors(self.graph, node))
            
            # Limit to max_nodes
            if len(nodes_to_include) > max_nodes:
                # Prioritize highlighted entities and their direct neighbors
                direct_neighbors = set()
                for node in highlight_entities:
                    if node in self.graph:
                        direct_neighbors.update(self.graph.successors(node))
                        direct_neighbors.update(self.graph.predecessors(node))
                
                # Create a prioritized set of nodes
                prioritized_nodes = set(highlight_entities)
                prioritized_nodes.update(direct_neighbors)
                
                # Add remaining nodes until max_nodes
                remaining_nodes = nodes_to_include - prioritized_nodes
                remaining_nodes = list(remaining_nodes)[:max_nodes - len(prioritized_nodes)]
                
                nodes_to_include = prioritized_nodes.union(remaining_nodes)
            
            # Create subgraph
            G = self.graph.subgraph(nodes_to_include).copy()
        else:
            # If no highlight, just show a subset of nodes for visibility
            if len(self.graph.nodes) > max_nodes:
                # Get nodes with highest degree centrality
                centrality = nx.degree_centrality(self.graph)
                top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
                nodes_to_include = [node for node, _ in top_nodes]
                G = self.graph.subgraph(nodes_to_include).copy()
            else:
                G = self.graph.copy()
        
        # Define node colors and sizes based on type and highlight status
        node_colors = []
        node_sizes = []
        for node in G.nodes():
            # Base size for all nodes
            base_size = 2000
            
            if highlight_entities and node in highlight_entities:
                node_colors.append('#FF4B4B')  # Bright red for highlighted nodes
                node_sizes.append(base_size * 1.5)  # 50% larger for highlighted nodes
            else:
                node_type = G.nodes[node].get('type', '')
                if node_type == 'Person':
                    node_colors.append('#64B5F6')  # Material Blue
                elif node_type == 'Organization':
                    node_colors.append('#81C784')  # Material Green
                elif node_type == 'Location':
                    node_colors.append('#FFB74D')  # Material Orange
                elif node_type == 'Work':
                    node_colors.append('#FFD54F')  # Material Amber
                else:
                    node_colors.append('#E0E0E0')  # Material Grey
                node_sizes.append(base_size)
        
        # Get node labels
        node_labels = {node: G.nodes[node]['label'] for node in G.nodes()}
        
        # Get edge labels
        edge_labels = {(u, v): G.edges[u, v]['label'] for u, v in G.edges()}
        
        # Use kamada_kawai_layout for better node distribution
        pos = nx.kamada_kawai_layout(G, scale=2.0)
        
        # Draw edges with curved arrows
        nx.draw_networkx_edges(
            G, pos,
            edge_color='#BDBDBD',
            width=1.0,
            alpha=0.6,
            arrowsize=20,
            arrowstyle='->',
            connectionstyle='arc3,rad=0.2'
        )
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9,
            edgecolors='white',
            linewidths=2
        )
        
        # Draw node labels with white background for better visibility
        for node, (x, y) in pos.items():
            plt.text(
                x, y,
                node_labels[node],
                fontsize=10,
                ha='center',
                va='center',
                bbox=dict(
                    facecolor='white',
                    edgecolor='none',
                    alpha=0.7,
                    pad=2.0
                )
            )
        
        # Draw edge labels with white background
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            font_size=8,
            bbox=dict(
                facecolor='white',
                edgecolor='none',
                alpha=0.7,
                pad=1.0
            ),
            label_pos=0.5,
            rotate=False
        )
        
        plt.axis('off')
        plt.tight_layout()
        return plt 
