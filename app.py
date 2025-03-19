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
st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1>Think-on-Graph 2.0 Demo</h1>
        <p style='font-size: 1.2rem; color: #666;'>
            Kết hợp Knowledge Graph và LLM cho suy luận đa bước
        </p>
    </div>
""", unsafe_allow_html=True)

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
        with open(entities_file, 'r') as f:
            entities_data = json.load(f)
            for entity in entities_data:
                self.entities[entity['id']] = entity
        
        with open(relations_file, 'r') as f:
            relations_data = json.load(f)
            for relation in relations_data:
                self.relations[relation['id']] = relation
        
        with open(triples_file, 'r') as f:
            self.triples = json.load(f)
        
        with open(documents_file, 'r') as f:
            documents_data = json.load(f)
            for doc in documents_data:
                self.documents[doc['entity_id']] = doc['content']
        
        # Build graph
        self.build_graph()
    
    def build_graph(self):
        """Build a NetworkX graph from triples"""
        self.graph = nx.DiGraph()
        
        # Add nodes
        for entity_id, entity in self.entities.items():
            self.graph.add_node(entity_id, label=entity['name'], type=entity['type'])
        
        # Add edges
        for triple in self.triples:
            head = triple['head']
            tail = triple['tail']
            relation = triple['relation']
            relation_name = self.relations[relation]['name'] if relation in self.relations else relation
            self.graph.add_edge(head, tail, relation=relation, label=relation_name)
    
    def identify_entities_from_query(self, query: str) -> List[str]:
        """Identify potential entities from the query by simple name matching"""
        entities_found = []
        for entity_id, entity in self.entities.items():
            if entity['name'].lower() in query.lower():
                entities_found.append(entity_id)
        return entities_found
    
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
                'head': True
            })
        
        # Incoming relations
        for neighbor, _, data in self.graph.in_edges(entity_id, data=True):
            candidates.append({
                'entity_id': entity_id,
                'entity_name': self.graph.nodes[entity_id]['label'],
                'relation': data['relation'],
                'relation_name': data['label'],
                'head': False
            })
        
        return candidates
    
    def prune_relations(self, entity_id: str, query: str) -> List[Dict]:
        """Rank and prune relations based on relevance to query"""
        candidates = self.get_relation_candidates(entity_id)
        if not candidates:
            return []
        
        # Compute relevance scores based on relation names
        relation_names = [f"{cand['relation_name']}" for cand in candidates]
        
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
        
        # Sort by score and keep top 3
        candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
        return candidates[:3]  # Keep top 3
    
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
        results = []
        
        # Prepare texts for embedding
        texts = []
        for entity_id in entity_ids:
            if entity_id not in self.documents:
                continue
                
            # Split document into sentences
            document = self.documents[entity_id]
            sentences = re.split(r'(?<=[.!?])\s+', document)
            texts.extend(sentences)
        
        if not texts:
            return results
            
        if self.llm and self.llm.use_embedding_api:
            # Use LLM's embedding API
            scores = self.llm.compute_similarity(query, texts)
        else:
            # Use local sentence transformer
            query_emb = model.encode(query, convert_to_tensor=True)
            sentences_emb = model.encode(texts, convert_to_tensor=True)
            scores = util.pytorch_cos_sim(query_emb, sentences_emb)[0].cpu().numpy()
        
        # Add top sentences to results
        for i, score in enumerate(scores):
            # Find which entity this sentence belongs to
            current_pos = 0
            entity_id = None
            for eid in entity_ids:
                if eid not in self.documents:
                    continue
                doc_sentences = re.split(r'(?<=[.!?])\s+', self.documents[eid])
                if current_pos <= i < current_pos + len(doc_sentences):
                    entity_id = eid
                    break
                current_pos += len(doc_sentences)
            
            if entity_id:
                results.append({
                    'entity_id': entity_id,
                    'entity_name': self.graph.nodes[entity_id]['label'],
                    'text': texts[i],
                    'score': float(score)
                })
        
        # Sort by relevance
        results = sorted(results, key=lambda x: x['score'], reverse=True)
        return results[:10]  # Return top 10 sentences
    
    def visualize_graph(self, highlight_entities=None):
        """Visualize the knowledge graph using NetworkX and Matplotlib"""
        plt.figure(figsize=(12, 8))
        
        # Create a copy of the graph for visualization
        G = self.graph.copy()
        
        # Define node colors based on type
        node_colors = []
        for node in G.nodes():
            if highlight_entities and node in highlight_entities:
                node_colors.append('red')  # Highlighted nodes
            else:
                node_type = G.nodes[node].get('type', '')
                if node_type == 'Person':
                    node_colors.append('skyblue')
                elif node_type == 'Field':
                    node_colors.append('lightgreen')
                elif node_type == 'Country':
                    node_colors.append('orange')
                elif node_type == 'Award':
                    node_colors.append('gold')
                else:
                    node_colors.append('lightgray')
        
        # Get node labels
        node_labels = {node: G.nodes[node]['label'] for node in G.nodes()}
        
        # Get edge labels
        edge_labels = {(u, v): G.edges[u, v]['label'] for u, v in G.edges()}
        
        # Draw the graph
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
    def __init__(self, knowledge_graph: KnowledgeGraph, llm: LLM, depth: int = 2, width: int = 3):
        self.knowledge_graph = knowledge_graph
        self.llm = llm
        self.depth = depth
        self.width = width
        self.knowledge_graph.set_llm(llm)  # Set LLM instance for embedding computation
        logger.info(f"Initialized ToG2Reasoner with depth={depth}, width={width}")
        
    def identify_topic_entities(self, query: str) -> List[str]:
        """Identify topic entities from query"""
        if self.llm:
            # Use LLM for entity identification
            entity_names = [entity['name'] for entity_id, entity in self.knowledge_graph.entities.items()]
            identified_names = self.llm.identify_entities(query, entity_names)
            logger.info("LLM identified entities: %s", identified_names)
            
            # Map back to entity ids
            topic_entities = []
            for entity_id, entity in self.knowledge_graph.entities.items():
                if entity['name'] in identified_names:
                    topic_entities.append(entity_id)
        else:
            # Fallback to simple name matching
            topic_entities = self.knowledge_graph.identify_entities_from_query(query)
            logger.info("Simple matching identified entities: %s", topic_entities)
            
        return topic_entities
        
    def explore_relations(self, topic_entities: List[str], query: str) -> List[Tuple[str, str, str]]:
        """Explore relations and discover connected entities"""
        entity_chains = []
        current_entities = topic_entities
        
        for depth in range(self.depth):
            if not current_entities:
                break
                
            next_entities = []
            for entity_id in current_entities:
                # Get and prune relations
                relations = self.knowledge_graph.prune_relations(entity_id, query)
                
                for relation in relations:
                    # Get connected entities
                    candidates = self.knowledge_graph.get_entity_candidates(
                        relation['entity_id'],
                        relation['relation'],
                        relation['head']
                    )
                    
                    # Add to chains
                    for candidate_id in candidates:
                        chain = (
                            self.knowledge_graph.entities[relation['entity_id']]['name'],
                            relation['relation_name'],
                            self.knowledge_graph.entities[candidate_id]['name']
                        )
                        entity_chains.append(chain)
                        next_entities.append(candidate_id)
            
            # Keep top entities for next iteration
            if next_entities:
                # Score entities based on their relations
                entity_scores = {}
                for entity_id in next_entities:
                    entity_scores[entity_id] = len(self.knowledge_graph.get_relation_candidates(entity_id))
                
                # Sort by score and keep top self.width
                current_entities = sorted(entity_scores.items(), key=lambda x: x[1], reverse=True)[:self.width]
                current_entities = [entity_id for entity_id, _ in current_entities]
        
        return entity_chains
        
    def rank_entities(self, entity_chains: List[Tuple[str, str, str]], doc_results: List[Dict]) -> List[Dict]:
        """Rank entities based on relevance"""
        entity_scores = {}
        
        # Score entities based on document relevance
        for doc in doc_results:
            entity_id = doc['entity_id']
            if entity_id in entity_scores:
                entity_scores[entity_id] += doc['score']
            else:
                entity_scores[entity_id] = doc['score']
        
        # Add scores from entity chains
        for chain in entity_chains:
            # Get entity IDs from names
            entity_ids = []
            for name in [chain[0], chain[2]]:
                for entity_id, entity in self.knowledge_graph.entities.items():
                    if entity['name'] == name:
                        entity_ids.append(entity_id)
                        break
            
            # Add chain score to entities
            for entity_id in entity_ids:
                if entity_id in entity_scores:
                    entity_scores[entity_id] += 1
                else:
                    entity_scores[entity_id] = 1
        
        # Convert to list of dicts with entity info
        ranked_entities = []
        for entity_id, score in sorted(entity_scores.items(), key=lambda x: x[1], reverse=True):
            entity = self.knowledge_graph.entities[entity_id]
            ranked_entities.append({
                'id': entity_id,
                'name': entity['name'],
                'type': entity['type'],
                'score': score
            })
        
        return ranked_entities
        
    def reason(self, query: str) -> Tuple[List[Dict], List[str], str]:
        """Main reasoning method"""
        total_start_time = time.time()
        reasoning_steps = []
        entity_chains = None
        answer = None
        
        # Step 1: Identify topic entities from query
        step_start_time = time.time()
        logger.info("Step 1: Identifying topic entities from query")
        topic_entities = self.identify_topic_entities(query)
        step_time = time.time() - step_start_time
        reasoning_steps.append({
            'step': 'Entity Identification',
            'description': f'Identified {len(topic_entities)} topic entities',
            'entities': topic_entities,
            'time': f"{step_time:.2f}s"
        })
        
        if not topic_entities:
            logger.warning("No topic entities found")
            return reasoning_steps, None, "Could not identify any relevant entities from the query."
            
        # Step 2: Explore relations and discover connected entities
        step_start_time = time.time()
        logger.info("Step 2: Exploring relations and discovering connected entities")
        entity_chains = self.explore_relations(topic_entities, query)
        step_time = time.time() - step_start_time
        reasoning_steps.append({
            'step': 'Entity Discovery',
            'description': f'Discovered {len(entity_chains)} entity chains',
            'entities': entity_chains,
            'time': f"{step_time:.2f}s"
        })
        
        if not entity_chains:
            logger.warning("No entity chains found")
            return reasoning_steps, None, "Could not find any relevant information to answer the query."
            
        # Step 3: Search documents for relevant information
        step_start_time = time.time()
        logger.info("Step 3: Searching documents for relevant information")
        doc_results = self.knowledge_graph.search_entity_documents(
            [entity['id'] for chain in entity_chains for entity in chain],
            query
        )
        step_time = time.time() - step_start_time
        reasoning_steps.append({
            'step': 'Document Search',
            'description': f'Found {len(doc_results)} relevant document snippets',
            'documents': doc_results,
            'time': f"{step_time:.2f}s"
        })
        
        # Step 4: Rank entities based on relevance
        step_start_time = time.time()
        logger.info("Step 4: Ranking entities based on relevance")
        ranked_entities = self.rank_entities(entity_chains, doc_results)
        step_time = time.time() - step_start_time
        reasoning_steps.append({
            'step': 'Entity Ranking',
            'description': f'Ranked {len(ranked_entities)} entities by relevance',
            'entities': ranked_entities,
            'time': f"{step_time:.2f}s"
        })
        
        # Step 5: Generate final answer
        step_start_time = time.time()
        logger.info("Step 5: Generating final answer")
        answer = self.generate_answer(query, ranked_entities, doc_results)
        step_time = time.time() - step_start_time
        reasoning_steps.append({
            'step': 'Answer Generation',
            'description': 'Generated final answer',
            'answer': answer,
            'time': f"{step_time:.2f}s"
        })
        
        # Add total processing time
        total_time = time.time() - total_start_time
        reasoning_steps.append({
            'step': 'Total Processing',
            'description': 'Total processing time',
            'time': f"{total_time:.2f}s"
        })
        
        logger.info(f"Completed reasoning process in {total_time:.2f}s")
        return reasoning_steps, entity_chains, answer
    
    def generate_answer(self, query, sentences, entity_chains):
        """Generate final answer based on collected information (fallback method)"""
        # In a real implementation, this would use an LLM
        # For this demo, we'll just return the top sentences
        
        if not sentences:
            return "Could not find sufficient information to answer the query."
        
        # Sort sentences by relevance
        sorted_sentences = sorted(sentences, key=lambda x: x['score'], reverse=True)
        
        # Create a basic answer by combining top 3 sentences
        answer = "Based on the information collected:\n\n"
        for i, sent in enumerate(sorted_sentences[:3]):
            answer += f"{i+1}. {sent['text']} (From {sent['entity_name']})\n\n"
        
        # Add entity paths that were explored
        if entity_chains:
            answer += "Entity connections explored:\n"
            for i, chain in enumerate(entity_chains[:5]):  # Show up to 5 chains
                answer += f"- {chain[0]} -> {chain[1]} -> {chain[2]}\n"
        
        return answer

def load_sample_data():
    """Load sample data if no files are uploaded"""
    kg = KnowledgeGraph()
    
    # Create temp files with sample data
    os.makedirs('temp', exist_ok=True)
    
    entities = [
        {"id": "E1", "name": "Albert Einstein", "type": "Person"},
        {"id": "E2", "name": "Physics", "type": "Field"},
        {"id": "E3", "name": "Germany", "type": "Country"},
        {"id": "E4", "name": "Nobel Prize in Physics", "type": "Award"},
        {"id": "E5", "name": "Theory of Relativity", "type": "Scientific Theory"},
        {"id": "E6", "name": "Marie Curie", "type": "Person"},
        {"id": "E7", "name": "Poland", "type": "Country"},
        {"id": "E8", "name": "Chemistry", "type": "Field"}
    ]
    
    relations = [
        {"id": "R1", "name": "field_of_work"},
        {"id": "R2", "name": "born_in"},
        {"id": "R3", "name": "award_received"},
        {"id": "R4", "name": "developed"}
    ]
    
    triples = [
        {"head": "E1", "relation": "R1", "tail": "E2"},
        {"head": "E1", "relation": "R2", "tail": "E3"},
        {"head": "E1", "relation": "R3", "tail": "E4"},
        {"head": "E1", "relation": "R4", "tail": "E5"},
        {"head": "E6", "relation": "R1", "tail": "E2"},
        {"head": "E6", "relation": "R1", "tail": "E8"},
        {"head": "E6", "relation": "R2", "tail": "E7"},
        {"head": "E6", "relation": "R3", "tail": "E4"}
    ]
    
    documents = [
        {"entity_id": "E1", "content": "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity. He received the Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect."},
        {"entity_id": "E2", "content": "Physics is the natural science that studies matter, its motion and behavior through space and time, and the related entities of energy and force."},
        {"entity_id": "E3", "content": "Germany is a country in Central Europe. It is the second most populous country in Europe after Russia and the most populous member state of the European Union."},
        {"entity_id": "E4", "content": "The Nobel Prize in Physics is a yearly award given by the Royal Swedish Academy of Sciences for those who have made the most outstanding contributions for mankind in the field of physics."},
        {"entity_id": "E5", "content": "The theory of relativity usually encompasses two interrelated theories by Albert Einstein: special relativity and general relativity, proposed and published in 1905 and 1915, respectively."},
        {"entity_id": "E6", "content": "Marie Curie was a Polish and naturalized-French physicist and chemist who conducted pioneering research on radioactivity. She was the first woman to win a Nobel Prize and remains the only person to win Nobel Prizes in two scientific fields."},
        {"entity_id": "E7", "content": "Poland is a country in Central Europe. It is divided into 16 administrative provinces called voivodeships."},
        {"entity_id": "E8", "content": "Chemistry is the scientific study of the properties and behavior of matter. It is a natural science that covers the elements that make up matter to the compounds made of atoms, molecules and ions."}
    ]
    
    # Save sample data
    with open('temp/entities.json', 'w') as f:
        json.dump(entities, f)
    
    with open('temp/relations.json', 'w') as f:
        json.dump(relations, f)
    
    with open('temp/triples.json', 'w') as f:
        json.dump(triples, f)
    
    with open('temp/documents.json', 'w') as f:
        json.dump(documents, f)
    
    # Load data into KG
    kg.load_from_files('temp/entities.json', 'temp/relations.json', 'temp/triples.json', 'temp/documents.json')
    
    return kg

def load_sample_queries():
    """Load sample queries from JSON file"""
    try:
        with open('sample_queries.json', 'r') as f:
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
                    
    # Initialize reasoner with default depth and width
    reasoner = ToG2Reasoner(kg, llm=llm_instance, depth=2, width=3)
    
    # Load sample queries
    sample_queries = load_sample_queries()
    
    # Create columns for the query input section
    query_col1, query_col2 = st.columns([2, 1])
    
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
            "",
            value=selected_query,
            placeholder="Nhập câu hỏi của bạn...",
            help="Ví dụ: Which country was Albert Einstein born in?"
        )
    
    # Process query
    if query:
        # Create columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
                <div class='metric-container'>
                    <div class='metric-value'>1</div>
                    <div class='metric-label'>Số thực thể ban đầu</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class='metric-container'>
                    <div class='metric-value'>3</div>
                    <div class='metric-label'>Độ sâu tìm kiếm</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
                <div class='metric-container'>
                    <div class='metric-value'>3</div>
                    <div class='metric-label'>Số thực thể mỗi bước</div>
                </div>
            """, unsafe_allow_html=True)
        
        # Create tabs before reasoning
        tab1, tab2, tab3 = st.tabs(["Kết quả", "Quá trình suy luận", "Trực quan hóa đường dẫn"])
        
        # Initialize empty containers for each tab
        with tab1:
            results_container = st.empty()
        
        with tab2:
            reasoning_container = st.container()
        
        with tab3:
            visualization_container = st.empty()
        
        # Execute reasoning with real-time updates
        with st.spinner("Đang thực hiện suy luận..."):
            # Switch to reasoning tab while processing
            st.query_params["active_tab"] = "Quá trình suy luận"
            
            # Execute reasoning
            reasoning_steps = []
            entity_chains = None
            answer = None
            
            # Create a placeholder for steps in the reasoning tab
            with reasoning_container:
                steps_placeholder = st.empty()
                
                def update_reasoning_steps(steps):
                    with steps_placeholder.container():
                        for i, step in enumerate(steps, 1):
                            with st.expander(f"Bước {i}: {step['step']}", expanded=True):
                                st.markdown(step['description'])
                                
                                # Display time if present
                                if 'time' in step:
                                    st.markdown(f"**Thời gian thực thi:** {step['time']}")
                                
                                # Display entities if present
                                if 'entities' in step:
                                    st.markdown("**Entities:**")
                                    entity_names = [kg.entities[eid]['name'] for eid in step['entities'] if eid in kg.entities]
                                    st.write(", ".join(entity_names))
                                
                                # Display relations if present
                                if 'relations' in step:
                                    st.markdown("**Relations:**")
                                    for rel in step['relations']:
                                        st.write(f"- {rel['entity_name']} → {rel['relation_name']} (Score: {rel['score']:.2f})")
                                
                                # Display top sentences if present
                                if 'top_sentences' in step:
                                    st.markdown("**Top relevant sentences:**")
                                    for sent in step['top_sentences']:
                                        st.write(f"- {sent['text']} (From: {sent['entity_name']}, Score: {sent['score']:.2f})")
                                
                                # Display selected entities if present
                                if 'selected_entities' in step:
                                    st.markdown("**Selected entities:**")
                                    st.write(", ".join(step['selected_entities']))
                                
                                # Display entity chains if present
                                if 'entity_chains' in step:
                                    st.markdown("**Entity chains explored:**")
                                    for chain in step['entity_chains'][:5]:  # Show top 5 chains
                                        st.write(f"- {chain[0]} → {chain[1]} → {chain[2]}")
                
                # Execute reasoning with step updates
                reasoning_steps, entity_chains, answer = reasoner.reason(query)
                update_reasoning_steps(reasoning_steps)
            
            # Switch back to results tab when done
            st.query_params["active_tab"] = "Kết quả"
        
        # Update results tab
        with results_container:
            st.markdown("### Kết quả")
            with st.container():
                st.markdown("""
                    <div class='results-container'>
                        <h3>Phương pháp suy luận: {}</h3>
                        <h4>Câu hỏi:</h4>
                        <p>{}</p>
                        <h4>Câu trả lời:</h4>
                        <p>{}</p>
                    </div>
                """.format(
                    "Embedding-based" if not use_llm else "LLM-enhanced",
                    query,
                    answer
                ), unsafe_allow_html=True)
        
        # Update visualization tab
        with visualization_container:
            st.markdown("### Đồ thị kiến thức")
            with st.container():
                # Get entities to highlight from the final reasoning step
                highlight_entities = []
                if entity_chains:
                    entity_names = set()
                    for chain in entity_chains:
                        entity_names.update([chain[0], chain[2]])
                    
                    # Map names back to IDs
                    for entity_id, entity in kg.entities.items():
                        if entity['name'] in entity_names:
                            highlight_entities.append(entity_id)
                
                st.pyplot(kg.visualize_graph(highlight_entities=highlight_entities))

if __name__ == "__main__":
    main()