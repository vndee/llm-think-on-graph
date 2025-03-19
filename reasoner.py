import time
import logging
from typing import List, Dict, Any, Optional, Tuple
import re

# Import from our modules
from knowledge_graph import KnowledgeGraph
from llm import LLM

logger = logging.getLogger(__name__)

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