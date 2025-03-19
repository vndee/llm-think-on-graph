import streamlit as st
import json
import os
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
import re
import requests
import time
import logging
from typing import List, Dict, Any, Optional, Tuple

# Import LLM class
from llm import LLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load custom CSS
def load_css():
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Thiết lập trang
st.set_page_config(
    page_title="Think-on-Graph 2.0 Demo",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
load_css()

# Add custom header
# st.markdown("""
#     <div style='text-align: center; padding: 2rem 0;'>
#         <h1>Think-on-Graph 2.0 Demo</h1>
#         <p style='font-size: 1.2rem; color: #666;'>
#             Kết hợp Knowledge Graph và LLM cho suy luận đa bước
#         </p>
#     </div>
# """, unsafe_allow_html=True)

# Cache model sentence transformer để tính cosine similarity
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Khởi tạo model
model = load_model()

class KnowledgeGraph:
    def __init__(self):
        self.entities = {}  # id -> entity
        self.relations = {}  # id -> relation
        self.triples = []   # list of (head, relation, tail)
        self.documents = {}  # entity_id -> document
        self.graph = nx.DiGraph()
        self.llm = None  # LLM instance for embedding computation
        
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
        plt.figure(figsize=(10, 8))
        
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
        
        # Define node colors based on type and highlight status
        node_colors = []
        for node in G.nodes():
            if highlight_entities and node in highlight_entities:
                node_colors.append('red')  # Highlighted nodes
            else:
                node_type = G.nodes[node].get('type', '')
                if node_type == 'Person':
                    node_colors.append('skyblue')
                elif node_type == 'Organization':
                    node_colors.append('lightgreen')
                elif node_type == 'Location':
                    node_colors.append('orange')
                elif node_type == 'Work':
                    node_colors.append('gold')
                else:
                    node_colors.append('lightgray')
        
        # Get node labels
        node_labels = {node: G.nodes[node]['label'] for node in G.nodes()}
        
        # Get edge labels
        edge_labels = {(u, v): G.edges[u, v]['label'] for u, v in G.edges()}
        
        # Draw the graph with a deterministic layout
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.8)
        nx.draw_networkx_edges(G, pos, width=1.0, arrowsize=15)
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
        
        # Draw edge labels with smaller font
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        plt.axis('off')
        return plt
    

# Use the updated ToG2Reasoner class
class ToG2Reasoner:
    def __init__(self, knowledge_graph: KnowledgeGraph, llm: LLM, max_depth: int = 3, max_width: int = 3):
        self.knowledge_graph = knowledge_graph
        self.llm = llm
        self.max_depth = max_depth
        self.max_width = max_width
        self.knowledge_graph.set_llm(llm)  # Set LLM instance for embedding computation
        logger.info(f"Initialized ToG2Reasoner with depth={max_depth}, width={max_width}")
        
    def identify_topic_entities(self, query: str) -> List[str]:
        """Identify topic entities from the query using LLM."""
        logger.info("Identifying topic entities from query")
        
        # Get all entity names and create a mapping from name to ID
        entity_names = []
        name_to_id = {}
        for entity_id, entity in self.knowledge_graph.entities.items():
            entity_names.append(entity['name'])
            name_to_id[entity['name']] = entity_id
        
        # Try LLM-based entity identification 
        if self.llm:
            try:
                entities = self.llm.identify_entities(query, entity_names)
                logger.info(f"LLM identified entities: {entities}")
                
                # Convert entity names to IDs
                valid_entity_ids = []
                for name in entities:
                    if name in name_to_id:
                        valid_entity_ids.append(name_to_id[name])
                
                if valid_entity_ids:
                    entity_names_found = [self.knowledge_graph.entities[eid]['name'] for eid in valid_entity_ids]
                    logger.info(f"Valid topic entities found: {entity_names_found} with IDs: {valid_entity_ids}")
                    return valid_entity_ids
                else:
                    logger.warning("No valid entities found from LLM results")
            except Exception as e:
                logger.error(f"Error in LLM entity identification: {e}")
        
        # Fallback to simple name matching
        matched_entity_ids = []
        for name, entity_id in name_to_id.items():
            if name.lower() in query.lower():
                matched_entity_ids.append(entity_id)
        
        if matched_entity_ids:
            entity_names_found = [self.knowledge_graph.entities[eid]['name'] for eid in matched_entity_ids]
            logger.info(f"Found entities by name matching: {entity_names_found} with IDs: {matched_entity_ids}")
            return matched_entity_ids
        else:
            logger.warning("No topic entities found by any method")
            return []
    
    def rank_relations(self, topic_entity_id: str, query: str):
        """Rank relations for a topic entity based on relevance to query."""
        logger.info(f"Ranking relations for entity {topic_entity_id}")
        
        # Get all relations for the entity
        relations = self.knowledge_graph.get_relation_candidates(topic_entity_id)
        
        # If no relations, return empty list
        if not relations:
            return []
        
        # Use LLM to rank relations if available
        if self.llm:
            try:
                entity_name = self.knowledge_graph.entities[topic_entity_id]['name']
                ranked_relations = self.llm.rank_relations(query, entity_name, relations)
                logger.info(f"LLM ranked {len(ranked_relations)} relations for {entity_name}")
                return ranked_relations
            except Exception as e:
                logger.error(f"Error in LLM relation ranking: {e}")
        
        # Fallback to embedding-based ranking
        return self.knowledge_graph.prune_relations(topic_entity_id, query)
    
    def search_entity_documents(self, entity_ids, query):
        """Search for relevant context in entity documents."""
        return self.knowledge_graph.search_entity_documents(entity_ids, query)
    
    def iterative_reasoning(self, query: str) -> Tuple[List[Dict], str]:
        """Main iterative reasoning method implementing ToG-2 approach."""
        total_start_time = time.time()
        reasoning_steps = []
        clues = None
        answer = None
        
        # Theo dõi các thực thể đã duyệt
        visited_entities = set()
        
        # Step 1: Initial Topic Entities
        step_start_time = time.time()
        topic_entities = self.identify_topic_entities(query)
        initial_topic_entities = topic_entities.copy()  # Save for visualization
        visited_entities.update(topic_entities)  # Thêm vào danh sách đã duyệt
        step_time = time.time() - step_start_time
        
        reasoning_steps.append({
            'step': 'Topic Entity Identification',
            'description': f'Identified {len(topic_entities)} initial topic entities',
            'entities': topic_entities,
            'visited_count': len(visited_entities),  # Thêm số lượng entities đã duyệt
            'time': f"{step_time:.2f}s"
        })
        
        if not topic_entities:
            return reasoning_steps, "Could not identify any relevant entities from the query."
        
        # Initial document retrieval for the topic entities
        initial_contexts = self.search_entity_documents(topic_entities, query)
        
        reasoning_steps.append({
            'step': 'Initial Context Retrieval',
            'description': f'Retrieved {len(initial_contexts)} initial context snippets',
            'documents': initial_contexts,
            'visited_count': len(visited_entities),  # Thêm số lượng entities đã duyệt
            'time': f"0.00s"  # Already counted in the previous step
        })
        
        # Check if initial information is sufficient
        if self.llm:
            try:
                # Simplified for the demo, could adapt with proper LLM prompt
                if len(initial_contexts) > 5 and sum(doc['score'] for doc in initial_contexts[:5]) > 4.0:
                    # Information might be sufficient
                    answer = self.generate_answer(query, initial_contexts, [])
                    reasoning_steps.append({
                        'step': 'Answer Generation',
                        'description': 'Generated answer from initial context',
                        'answer': answer,
                        'visited_count': len(visited_entities),  # Thêm số lượng entities đã duyệt
                        'time': f"0.00s"
                    })
                    return reasoning_steps, answer
            except Exception as e:
                logger.error(f"Error checking initial context sufficiency: {e}")
        
        # Iterative exploration
        all_entity_chains = []
        all_contexts = initial_contexts
        
        # Theo dõi số vòng lặp suy luận thực tế
        actual_iterations = 0
        
        for depth in range(self.max_depth):
            actual_iterations += 1
            logger.info(f"Starting iteration {depth+1}/{self.max_depth}")
            step_start_time = time.time()
            iteration_chains = []
            next_topic_entities = []
            
            # Step 2: Knowledge-guided Graph Search
            for entity_id in topic_entities:
                entity_name = self.knowledge_graph.entities[entity_id]['name']
                
                # Relation Discovery and Pruning
                ranked_relations = self.rank_relations(entity_id, query)
                
                # Keep only top relations based on width parameter
                top_relations = ranked_relations[:self.max_width]
                
                # Entity Discovery
                for relation in top_relations:
                    related_entity_ids = self.knowledge_graph.get_entity_candidates(
                        relation['entity_id'],
                        relation['relation'],
                        relation['head']
                    )
                    
                    for related_id in related_entity_ids:
                        if related_id in self.knowledge_graph.entities:
                            related_name = self.knowledge_graph.entities[related_id]['name']
                            chain = (entity_name, relation['relation_name'], related_name)
                            iteration_chains.append(chain)
                            
                            # Add to candidate entities for next iteration and track visited
                            if related_id not in next_topic_entities:
                                next_topic_entities.append(related_id)
                                visited_entities.add(related_id)  # Thêm vào danh sách đã duyệt
            
            # Add chains to the accumulated list
            all_entity_chains.extend(iteration_chains)
            
            # Step 3: Knowledge-guided Context Retrieval
            iteration_contexts = self.search_entity_documents(next_topic_entities, query)
            all_contexts.extend(iteration_contexts)
            
            # Context-based Entity Pruning - rank entities based on context relevance
            entity_scores = {}
            for doc in iteration_contexts:
                entity_id = doc['entity_id']
                if entity_id in entity_scores:
                    entity_scores[entity_id] += doc['score']
                else:
                    entity_scores[entity_id] = doc['score']
            
            # Sort by score and keep top entities for next iteration
            if entity_scores:
                sorted_entities = sorted(entity_scores.items(), key=lambda x: x[1], reverse=True)
                topic_entities = [entity_id for entity_id, _ in sorted_entities[:self.max_width]]
            else:
                # No relevant context found, use all discovered entities
                topic_entities = next_topic_entities[:self.max_width]
            
            step_time = time.time() - step_start_time
            
            reasoning_steps.append({
                'step': f'Iteration {depth+1} Exploration',
                'description': f'Explored {len(iteration_chains)} new entity chains and retrieved {len(iteration_contexts)} context snippets',
                'entity_chains': iteration_chains,
                'documents': iteration_contexts,
                'topic_entities': topic_entities,
                'visited_count': len(visited_entities),  # Thêm số lượng entities đã duyệt
                'iteration': actual_iterations,  # Thêm số vòng lặp hiện tại
                'time': f"{step_time:.2f}s"
            })
            
            # Step 4: Reasoning with Hybrid Knowledge
            # Check if we should continue or if we have enough information
            if self.llm and (depth == self.max_depth - 1 or len(all_contexts) > 20):
                # For the demo, we'll generate an answer at the end or if we have enough context
                answer = self.generate_answer(query, all_contexts, all_entity_chains)
                break
            
            # Nếu không tìm thấy thực thể mới, dừng vòng lặp
            if not topic_entities:
                break
        
        # Final answer generation if not done in the loop
        if not answer:
            answer = self.generate_answer(query, all_contexts, all_entity_chains)
        
        # Add answer generation step
        reasoning_steps.append({
            'step': 'Answer Generation',
            'description': 'Generated final answer based on all collected information',
            'answer': answer,
            'visited_count': len(visited_entities),  # Thêm số lượng entities đã duyệt
            'iteration': actual_iterations,  # Thêm số vòng lặp thực tế
            'time': f"0.00s"
        })
        
        # Add total processing time
        total_time = time.time() - total_start_time
        reasoning_steps.append({
            'step': 'Total Processing',
            'description': 'Total processing time',
            'visited_count': len(visited_entities),  # Thêm số lượng entities đã duyệt
            'iteration': actual_iterations,  # Thêm số vòng lặp thực tế
            'time': f"{total_time:.2f}s"
        })
        
        # Thêm metadata về quá trình
        reasoning_steps.append({
            'step': 'Process Metadata',
            'description': 'Statistical information about the reasoning process',
            'metadata': {
                'total_entities_visited': len(visited_entities),
                'total_iterations': actual_iterations,
                'total_chains': len(all_entity_chains),
                'total_contexts': len(all_contexts)
            }
        })
        
        logger.info(f"Completed reasoning process in {total_time:.2f}s")
        return reasoning_steps, answer
    
    def generate_answer(self, query, contexts, entity_chains):
        """Generate answer based on collected information."""
        if self.llm:
            try:
                return self.llm.generate_answer(query, contexts, entity_chains)
            except Exception as e:
                logger.error(f"Error in LLM answer generation: {e}")
        
        # Fallback if LLM is not available or fails
        if not contexts:
            return "Could not find sufficient information to answer the query."
        
        # Create a basic answer by combining top sentences
        answer = "Based on the information collected:\n\n"
        sorted_contexts = sorted(contexts, key=lambda x: x['score'], reverse=True)
        
        for i, ctx in enumerate(sorted_contexts[:5]):
            answer += f"{i+1}. {ctx['text']} (From {ctx['entity_name']})\n\n"
        
        # Add entity relationships that were explored
        if entity_chains:
            answer += "Entity connections explored:\n"
            for i, chain in enumerate(entity_chains[:5]):
                answer += f"- {chain[0]} → {chain[1]} → {chain[2]}\n"
        
        return answer

def load_sample_data(data_dir='data'):
    """Load sample data if no files are uploaded"""
    kg = KnowledgeGraph()
    kg.load_from_files(os.path.join(data_dir, 'entities.json'), os.path.join(data_dir, 'relations.json'), os.path.join(data_dir, 'triples.json'), os.path.join(data_dir, 'documents.json'))    
    return kg

def load_sample_queries(data_dir='data'):
    """Load sample queries from JSON file"""
    try:
        with open(os.path.join(data_dir, 'sample_queries.json'), 'r') as f:
            return json.load(f)['queries']
    except FileNotFoundError:
        return []

def main():
    # Sidebar configuration
    with st.sidebar:
        st.markdown("## Cấu hình")
        
        # Data configuration
        st.markdown("### Dữ liệu")
        use_sample_data = st.checkbox("Sử dụng dữ liệu mẫu", value=True)
        
        # Main content
        if use_sample_data:
            kg = load_sample_data()
            st.sidebar.success("✅ Đã tải dữ liệu mẫu")
        else:
            st.sidebar.markdown("### Tải lên dữ liệu của bạn")
            entities_file = st.sidebar.file_uploader("Tải lên entities.json", type=["json"])
            relations_file = st.sidebar.file_uploader("Tải lên relations.json", type=["json"])
            triples_file = st.sidebar.file_uploader("Tải lên triples.json", type=["json"])
            documents_file = st.sidebar.file_uploader("Tải lên documents.json", type=["json"])
            
            if entities_file and relations_file and triples_file and documents_file:
                # Save uploaded files
                os.makedirs('uploads', exist_ok=True)
                
                with open('uploads/entities.json', 'wb') as f:
                    f.write(entities_file.getbuffer())
                
                with open('uploads/relations.json', 'wb') as f:
                    f.write(relations_file.getbuffer())
                
                with open('uploads/triples.json', 'wb') as f:
                    f.write(triples_file.getbuffer())
                
                with open('uploads/documents.json', 'wb') as f:
                    f.write(documents_file.getbuffer())
                
                # Load data into KG
                kg = KnowledgeGraph()
                kg.load_from_files('uploads/entities.json', 'uploads/relations.json', 'uploads/triples.json', 'uploads/documents.json')
                st.sidebar.success("✅ Dữ liệu đã được tải lên thành công!")
            else:
                st.warning("Vui lòng tải lên đầy đủ các file dữ liệu cần thiết")
                return
        
        # LLM configuration
        st.markdown("### Cấu hình LLM")
        use_llm = st.checkbox("Sử dụng LLM cho suy luận", value=False)
        
        if use_llm:
            llm_provider = st.selectbox(
                "LLM Provider",
                ["openai", "ollama"],
                help="Chọn provider LLM để sử dụng"
            )
            
            model = st.selectbox(
                "Model",
                ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"] if llm_provider == "openai" else ["llama3", "mistral"],
                help="Chọn model để sử dụng"
            )
            
            use_embedding_api = st.checkbox(
                "Sử dụng Embedding API",
                value=True,
                help="Sử dụng OpenAI/Ollama embedding API để tăng tốc độ tính toán similarity"
            )
            
            api_key = st.text_input(
                "API Key",
                type="password",
                help="Nhập API key của bạn (chỉ cần thiết cho OpenAI)"
            )
            
            api_base = st.text_input(
                "API Base URL (tùy chọn)",
                help="URL cơ sở tùy chỉnh cho API (không bắt buộc)"
            )
            
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                help="Điều chỉnh độ ngẫu nhiên của câu trả lời"
            )
        
        # ToG-2 Parameters
        st.markdown("### Tham số ToG-2")
        max_depth = st.slider(
            "Độ sâu tìm kiếm tối đa",
            min_value=1,
            max_value=5,
            value=3,
            help="Số lượng vòng lặp tối đa trong quá trình tìm kiếm"
        )
        
        max_width = st.slider(
            "Độ rộng tìm kiếm tối đa",
            min_value=1,
            max_value=5,
            value=3,
            help="Số lượng thực thể được giữ lại trong mỗi vòng lặp"
        )
    
    # Initialize LLM instance
    llm_instance = None
    if use_llm:
        if llm_provider == "openai" and not api_key:
            st.error("⚠️ Vui lòng nhập API Key để sử dụng tính năng LLM")
            return
            
        try:
            llm_instance = LLM(
                provider=llm_provider,
                model=model,
                api_key=api_key,
                api_base=api_base if api_base else None,
                temperature=temperature,
                use_embedding_api=use_embedding_api
            )
        except ValueError as e:
            if "API key is required" in str(e):
                st.error("⚠️ Vui lòng nhập API Key để sử dụng tính năng LLM")
                return
            else:
                st.error(f"⚠️ Có lỗi xảy ra: {str(e)}")
                return
                    
    # Initialize reasoner with configured parameters
    reasoner = ToG2Reasoner(kg, llm=llm_instance, max_depth=max_depth, max_width=max_width)
    
    # Load sample queries
    sample_queries = load_sample_queries()
    
    # Add title and description
    st.markdown("""
        <div style='text-align: center; padding: 1rem 0;'>
            <h1>Think-on-Graph 2.0 Demo</h1>
            <p style='font-size: 1.2rem; color: #666;'>
                Kết hợp Knowledge Graph và LLM cho suy luận đa bước sâu và chính xác
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different sections
    tab1, tab2 = st.tabs(["Demo", "Giới thiệu"])
    
    with tab2:
        st.markdown("""
        ## Think-on-Graph 2.0
        
        Think-on-Graph 2.0 (ToG-2) là một cách tiếp cận mới kết hợp chặt chẽ giữa Knowledge Graph và tài liệu văn bản để nâng cao khả năng suy luận đa bước của Large Language Models (LLM).
        
        ### Điểm nổi bật của ToG-2:
        
        1. **Tích hợp chặt chẽ KG và văn bản**: Sử dụng KG để dẫn dắt việc tìm kiếm thông tin trong văn bản, đồng thời sử dụng văn bản để làm phong phú và tinh chỉnh việc tìm kiếm trên đồ thị.
        
        2. **Tìm kiếm đa bước sâu hơn**: Cho phép LLM thực hiện suy luận sâu hơn bằng cách lặp lại quá trình tìm kiếm và tích lũy thông tin.
        
        3. **Suy luận có cơ sở**: Tăng cường độ tin cậy của câu trả lời từ LLM bằng cách cung cấp thông tin được tìm thấy từ cả KG và văn bản.
        
        4. **Linh hoạt**: Có thể áp dụng cho nhiều loại LLM khác nhau mà không cần đào tạo lại.
        
        ### Cách thức hoạt động:
        
        1. **Khởi tạo**: Xác định các thực thể chủ đề từ câu hỏi.
        
        2. **Tìm kiếm đồ thị dựa trên tri thức**: Khám phá các mối quan hệ và thực thể liên quan trên đồ thị kiến thức.
        
        3. **Tìm kiếm văn bản theo hướng dẫn tri thức**: Truy xuất thông tin từ tài liệu dựa trên các thực thể đã khám phá.
        
        4. **Tinh chỉnh thực thể dựa trên ngữ cảnh**: Sử dụng thông tin trong văn bản để xác định thực thể nào là quan trọng nhất.
        
        5. **Lặp lại quá trình**: Tiếp tục khám phá sâu hơn cho đến khi thu thập đủ thông tin.
        
        6. **Đưa ra câu trả lời**: Sinh câu trả lời dựa trên tất cả thông tin thu thập được.
        """)
    
    with tab1:
        # Create columns for the query input section
        query_col1, query_col2 = st.columns([3, 1])
        
        with query_col2:
            if sample_queries:
                # Add a dropdown for sample queries
                selected_query = st.selectbox(
                    "Hoặc chọn câu hỏi mẫu",
                    options=[""] + [q["text"] for q in sample_queries],
                    format_func=lambda x: "Chọn câu hỏi mẫu..." if x == "" else x,
                    help="Chọn một trong những câu hỏi mẫu có sẵn"
                )
        
        with query_col1:
            # Text input for query
            query = st.text_input(
                "Nhập câu hỏi của bạn",
                value=selected_query,
                placeholder="Ví dụ: Nguyễn Nhật Ánh đã viết những tác phẩm nổi tiếng nào?",
                help="Nhập câu hỏi liên quan đến tri thức trong hệ thống"
            )
        
        # Process query
        if query:
            # Execute reasoning with real-time updates
            with st.spinner("Đang thực hiện suy luận..."):
                # Execute reasoning
                reasoning_steps, answer = reasoner.iterative_reasoning(query)
                
                # Sau khi có reasoning_steps, tính toán metrics
                entity_count = 0
                reasoning_steps_count = 0

                # Lấy metadata từ reasoning_steps nếu có
                for step in reasoning_steps:
                    if 'step' in step and step['step'] == 'Process Metadata' and 'metadata' in step:
                        metadata = step['metadata']
                        entity_count = metadata.get('total_entities_visited', 0)
                        break

                # Đếm tổng số bước suy luận thực tế
                for step in reasoning_steps:
                    # Chỉ đếm các bước thực sự là bước suy luận
                    if 'step' in step:
                        # Bỏ qua các bước metadata và tổng thời gian xử lý
                        if step['step'] not in ['Process Metadata', 'Total Processing']:
                            reasoning_steps_count += 1

                # Nếu không tìm thấy metadata về số thực thể
                if entity_count == 0:
                    for step in reasoning_steps:
                        if 'visited_count' in step:
                            entity_count = max(entity_count, step['visited_count'])

                # Hiển thị metrics trước khi tạo tabs
                col1, col2, col3, col4, col5 = st.columns(5)

                with col1:
                    st.markdown(f"""
                        <div class='metric-container'>
                            <div class='metric-value'>{max_width}</div>
                            <div class='metric-label'>Độ rộng tìm kiếm</div>
                        </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                        <div class='metric-container'>
                            <div class='metric-value'>{max_depth}</div>
                            <div class='metric-label'>Độ sâu tìm kiếm</div>
                        </div>
                    """, unsafe_allow_html=True)

                with col3:
                    st.markdown(f"""
                        <div class='metric-container'>
                            <div class='metric-value'>{entity_count}</div>
                            <div class='metric-label'>Thực thể đã duyệt</div>
                        </div>
                    """, unsafe_allow_html=True)

                with col4:
                    st.markdown(f"""
                        <div class='metric-container'>
                            <div class='metric-value'>{reasoning_steps_count}</div>
                            <div class='metric-label'>Bước suy luận</div>
                        </div>
                    """, unsafe_allow_html=True)

                with col5:
                    st.markdown(f"""
                        <div class='metric-container'>
                            <div class='metric-value'>ToG-2</div>
                            <div class='metric-label'>Phương pháp</div>
                        </div>
                    """, unsafe_allow_html=True)

                # Tạo tabs cho kết quả và quá trình suy luận
                result_tab, process_tab, viz_tab = st.tabs(["Kết quả", "Quá trình suy luận", "Trực quan hóa"])
                    
                with result_tab:
                    result_container = st.empty()
            
                with process_tab:
                    reasoning_container = st.container()
                    steps_placeholder = st.empty()
            
                with viz_tab:
                    viz_container = st.empty()
                
                # Function to update reasoning steps in UI
                def update_reasoning_steps(steps):
                    with steps_placeholder.container():
                        for i, step in enumerate(steps, 1):
                            if 'step' in step:
                                # Bỏ qua bước Process Metadata - chỉ dùng để lưu trữ dữ liệu
                                if step['step'] == 'Process Metadata':
                                    continue
                                    
                                # Tạo tiêu đề với thông tin bổ sung nếu có
                                step_title = f"Bước {i}: {step['step']}"
                                if 'visited_count' in step:
                                    step_title += f" ({step['visited_count']} thực thể đã duyệt)"
                                    
                                with st.expander(step_title, expanded=True):
                                    st.markdown(step['description'])
                                    
                                    # Display time if present
                                    if 'time' in step:
                                        st.markdown(f"**Thời gian thực thi:** {step['time']}")
                                    
                                    # Display entities if present
                                    if 'entities' in step:
                                        st.markdown("**Entities:**")
                                        if isinstance(step['entities'], list) and len(step['entities']) > 0:
                                            if isinstance(step['entities'][0], str):
                                                entity_names = [kg.entities[eid]['name'] for eid in step['entities'] if eid in kg.entities]
                                                st.write(", ".join(entity_names))
                                    
                                    # Display topic entities if present
                                    if 'topic_entities' in step:
                                        st.markdown("**Topic Entities:**")
                                        entity_names = [kg.entities[eid]['name'] for eid in step['topic_entities'] if eid in kg.entities]
                                        st.write(", ".join(entity_names))
                                    
                                    # Display entity chains if present
                                    if 'entity_chains' in step:
                                        st.markdown("**Entity chains explored:**")
                                        for chain in step['entity_chains'][:5]:  # Show top 5 chains
                                            st.write(f"- {chain[0]} → {chain[1]} → {chain[2]}")
                                    
                                    # Display documents if present
                                    if 'documents' in step:
                                        st.markdown("**Top relevant sentences:**")
                                        sorted_docs = sorted(step['documents'], key=lambda x: x['score'], reverse=True)
                                        for doc in sorted_docs[:5]:  # Show top 5 docs
                                            st.write(f"- {doc['text']} (From: {doc['entity_name']}, Score: {doc['score']:.2f})")
                                            
                                    # Display answer if present
                                    if 'answer' in step:
                                        st.markdown("**Answer:**")
                                        st.write(step['answer'])
                                        
                                    # Hiển thị metadata nếu có
                                    if 'metadata' in step:
                                        st.markdown("**Process Statistics:**")
                                        metadata = step['metadata']
                                        st.write(f"- Total entities visited: {metadata.get('total_entities_visited', 0)}")
                                        st.write(f"- Total iterations: {metadata.get('total_iterations', 0)}")
                                        st.write(f"- Total entity chains: {metadata.get('total_chains', 0)}")
                                        st.write(f"- Total context snippets: {metadata.get('total_contexts', 0)}")
                
                # Update reasoning steps
                update_reasoning_steps(reasoning_steps)
                
                # Get entities to highlight from reasoning steps
                highlight_entities = []
                entity_names = set()
                
                # Extract entity chains from reasoning steps
                all_chains = []
                for step in reasoning_steps:
                    if 'entity_chains' in step:
                        all_chains.extend(step['entity_chains'])
                
                # Get entity names from chains
                if all_chains:
                    for chain in all_chains:
                        entity_names.update([chain[0], chain[2]])
                
                # Map names back to IDs
                for entity_id, entity in kg.entities.items():
                    if entity['name'] in entity_names:
                        highlight_entities.append(entity_id)
                
                # Update visualization tab
                with viz_container:
                    st.markdown("### Đồ thị kiến thức khám phá")
                    st.pyplot(kg.visualize_graph(highlight_entities=highlight_entities))
            
            # Update results tab
            with result_container:
                st.markdown("### Kết quả")
                with st.container():
                    st.markdown(f"""
                        <div class='results-container'>
                            <h3>Câu hỏi:</h3>
                            <p>{query}</p>
                            <h3>Câu trả lời:</h3>
                            <p>{answer}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
if __name__ == "__main__":
    main()