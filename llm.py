import json
import requests
import os
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
from openai import OpenAI
from const import (
    ENTITY_RECOGNITION_SYSTEM_PROMPT,
    ENTITY_RECOGNITION_PROMPT,
    RELATION_RANKING_SYSTEM_PROMPT,
    RELATION_RANKING_PROMPT,
    ANSWER_GENERATION_SYSTEM_PROMPT,
    ANSWER_GENERATION_PROMPT,
    CONTEXT_ANALYSIS_SYSTEM_PROMPT,
    CONTEXT_ANALYSIS_PROMPT,
)

logger = logging.getLogger(__name__)

class LLM:
    """Class to handle interactions with Large Language Models using either OpenAI compatible API or Ollama."""
    
    def __init__(
        self, 
        provider: str = "openai", 
        model: str = "gpt-3.5-turbo", 
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        use_embedding_api: bool = True
    ):
        """
        Initialize LLM client.
        
        Args:
            provider: Either "openai" or "ollama"
            model: Model name (e.g., "gpt-3.5-turbo" for OpenAI or "llama3" for Ollama)
            api_key: API key for OpenAI (not needed for Ollama)
            api_base: Custom API base URL (defaults to OpenAI's or Ollama's local URL)
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum output tokens
            use_embedding_api: Whether to use OpenAI/Ollama embedding API instead of local embedding
        """
        self.provider = provider.lower()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_embedding_api = use_embedding_api
        
        # Set up API endpoints
        if self.provider == "openai":
            self.api_key = api_key
            self.api_base = api_base or "https://api.openai.com/v1"
            if not self.api_key:
                raise ValueError("API key is required for OpenAI")
            # Initialize OpenAI client
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base
            )
        elif self.provider == "ollama":
            self.api_key = None
            self.api_base = api_base or "http://localhost:11434"
            self.client = None
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
            
        logger.info("Initialized LLM with provider=%s, model=%s, use_embedding_api=%s", 
                   self.provider, self.model, self.use_embedding_api)

    def get_embeddings(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Get embeddings for text(s)"""
        if isinstance(texts, str):
            texts = [texts]
            
        if self.provider == "openai":
            try:
                response = self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=texts
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                logger.error(f"Error getting embeddings from OpenAI: {e}")
                raise
        elif self.provider == "ollama":
            # Ollama embeddings implementation
            url = f"{self.api_base}/api/embeddings"
            embeddings = []
            
            for text in texts:
                payload = {
                    "model": self.model,
                    "prompt": text
                }
                
                try:
                    response = requests.post(url, json=payload)
                    response.raise_for_status()
                    embeddings.append(response.json()["embedding"])
                except Exception as e:
                    logger.error(f"Error getting embedding from Ollama: {e}")
                    raise
                    
            return embeddings
        else:
            raise ValueError(f"Unsupported provider for embeddings: {self.provider}")

    def compute_similarity(self, query: str, texts: List[str]) -> List[float]:
        """Compute similarity scores between query and texts"""
        if not texts:
            return []
            
        # Get embeddings
        query_emb = self.get_embeddings(query)[0]
        text_embs = self.get_embeddings(texts)
        
        # Compute cosine similarity
        import numpy as np
        query_emb = np.array(query_emb)
        text_embs = np.array(text_embs)
        
        # Normalize vectors for cosine similarity
        query_norm = np.linalg.norm(query_emb)
        text_norms = np.linalg.norm(text_embs, axis=1)
        
        # Avoid division by zero
        similarities = np.zeros(len(texts))
        valid_indices = np.where(text_norms > 0)[0]
        
        if query_norm > 0 and len(valid_indices) > 0:
            similarities[valid_indices] = np.dot(text_embs[valid_indices], query_emb) / (
                text_norms[valid_indices] * query_norm
            )
        
        return similarities.tolist()

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text with the LLM"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        messages.append({"role": "user", "content": prompt})
        
        if self.provider == "openai":
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"Error calling OpenAI API: {e}")
                return f"Error: {str(e)}"
        elif self.provider == "ollama":
            url = f"{self.api_base}/api/generate"
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "temperature": self.temperature,
            }
            
            if system_prompt:
                payload["system"] = system_prompt
                
            try:
                response = requests.post(url, json=payload)
                response.raise_for_status()
                return response.json().get("response", "")
            except Exception as e:
                logger.error(f"Error calling Ollama API: {e}")
                return f"Error: {str(e)}"
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def identify_entities(self, query: str, entity_names: List[str]) -> List[str]:
        """Identify entities in the query using LLM"""
        # Format entity_names for the prompt
        if len(entity_names) > 100:
            # If too many entities, select a subset that might be relevant
            import re
            # Extract words from the query (non-stopwords)
            query_words = set(re.findall(r'\b\w+\b', query.lower()))
            stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'about', 'by'}
            query_words = query_words - stopwords
            
            # Filter entities that might be relevant to the query
            filtered_entities = []
            for entity in entity_names:
                entity_lower = entity.lower()
                # Check if any query word appears in the entity name
                if any(word in entity_lower for word in query_words):
                    filtered_entities.append(entity)
                    
            # If we still have too many, take the first 100
            if len(filtered_entities) > 100:
                entity_names_str = ', '.join(filtered_entities[:100])
            else:
                entity_names_str = ', '.join(filtered_entities)
        else:
            entity_names_str = ', '.join(entity_names)
        
        prompt = f"""
        Query: {query}

        Available entities: {entity_names_str}

        List only the entities that are directly mentioned or clearly referenced in the query. Return them as a comma-separated list.
        """
        
        system_prompt = """
        You are an entity linker that identifies entities mentioned in queries. Extract only the entities that exist in the provided list of entity names.
        """
        
        result = self.generate(prompt, system_prompt)
        
        # Parse the result - expecting a comma-separated list of entity names
        entities = [name.strip() for name in result.split(',') if name.strip()]
        return entities
    
    def rank_relations(self, query: str, entity_name: str, relations: List[Dict]) -> List[Dict]:
        """Rank relations by their relevance to the query"""
        if not relations:
            return []
            
        relation_info = '\n'.join([f"- {rel['relation_name']} (connects {entity_name} to other entities)" for rel in relations])
        
        prompt = f"""
        Query: {query}

        Entity: {entity_name}

        Available relations:
        {relation_info}

        Rank the relations above by how relevant they are for answering the query about {entity_name}.
        For each relation, provide a score from 0 to 10, where:
        - 0 means completely irrelevant
        - 10 means highly relevant to answering the query

        Return a JSON array in this format:
        [
          {{"relation_name": "relation1", "score": score1}},
          {{"relation_name": "relation2", "score": score2}},
          ...
        ]
        """
        
        system_prompt = """
        You are a knowledge graph relation ranker that helps identify which relations are most relevant to answering a query.
        """
        
        result = self.generate(prompt, system_prompt)
        
        # Extract JSON from the response
        try:
            # Try to find JSON by looking for array notation
            import re
            import json
            
            json_match = re.search(r'\[.*\]', result, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                ranked_relations = json.loads(json_str)
                
                # Match the ranked relations back to the original relations data
                enriched_relations = []
                for ranked in ranked_relations:
                    for original in relations:
                        if ranked['relation_name'] == original['relation_name']:
                            # Normalize score to 0-1 range
                            enriched = {**original, 'score': ranked['score'] / 10.0}
                            enriched_relations.append(enriched)
                            break
                            
                # Sort by score in descending order
                enriched_relations.sort(key=lambda x: x['score'], reverse=True)
                return enriched_relations
            else:
                logger.error("Could not find JSON array in response")
                # Fallback
                for i, rel in enumerate(relations):
                    rel['score'] = (len(relations) - i) / len(relations)
                return relations
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Error parsing relation ranking: {e}")
            # Fallback - just assign scores based on position in the list
            for i, rel in enumerate(relations):
                rel['score'] = (len(relations) - i) / len(relations)
            return relations
    
    def generate_answer(self, query: str, contexts: List[Dict], entity_chains: List[Tuple]) -> str:
        """Generate answer based on contexts and entity chains"""
        # Format the contexts for the prompt
        formatted_contexts = ""
        if contexts:
            # Sort by relevance
            sorted_contexts = sorted(contexts, key=lambda x: x.get('score', 0), reverse=True)
            for i, ctx in enumerate(sorted_contexts[:10]):  # Use top 10 contexts
                formatted_contexts += f"{i+1}. \"{ctx['text']}\" (Source: {ctx.get('entity_name', 'Unknown')})\n"
        
        # Format the entity chains
        formatted_chains = ""
        if entity_chains:
            for i, chain in enumerate(entity_chains[:5]):  # Use top 5 chains
                formatted_chains += f"{i+1}. {chain[0]} → {chain[1]} → {chain[2]}\n"
        
        prompt = f"""
        Answer the following query based ONLY on the information provided in the context snippets and entity chains.

        Query: {query}

        Relevant context snippets:
        {formatted_contexts}

        Entity connections:
        {formatted_chains}

        Provide a comprehensive, accurate answer using only the information above. If the information is insufficient to fully answer the query, clearly state what remains unknown.
        """
        
        system_prompt = """
        You are a helpful assistant that provides accurate, comprehensive answers based on the provided information.
        """
        
        return self.generate(prompt, system_prompt)