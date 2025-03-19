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
        """
        Get embeddings for a text or list of texts using OpenAI or Ollama API.
        
        Args:
            texts: Single text or list of texts to get embeddings for
            
        Returns:
            List of embeddings (each embedding is a list of floats)
        """
        if isinstance(texts, str):
            texts = [texts]
            
        if self.provider == "openai":
            url = f"{self.api_base}/embeddings"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "text-embedding-ada-002",
                "input": texts
            }
            
            try:
                response = requests.post(url, headers=headers, json=payload)
                response.raise_for_status()
                return [item["embedding"] for item in response.json()["data"]]
            except Exception as e:
                logger.error(f"Error getting embeddings from OpenAI: {e}")
                raise
                
        elif self.provider == "ollama":
            url = f"{self.api_base}/api/embeddings"
            payload = {
                "model": self.model,
                "prompt": texts[0] if len(texts) == 1 else texts
            }
            
            try:
                response = requests.post(url, json=payload)
                response.raise_for_status()
                result = response.json()
                if isinstance(result, list):
                    return [item["embedding"] for item in result]
                else:
                    return [result["embedding"]]
            except Exception as e:
                logger.error(f"Error getting embeddings from Ollama: {e}")
                raise
        else:
            raise ValueError(f"Unsupported provider for embeddings: {self.provider}")

    def compute_similarity(self, query: str, texts: List[str]) -> List[float]:
        """
        Compute similarity scores between a query and a list of texts using embeddings.
        
        Args:
            query: Query text
            texts: List of texts to compare against
            
        Returns:
            List of similarity scores (0-1)
        """
        # Get embeddings for query and texts
        query_emb = self.get_embeddings(query)[0]
        text_embs = self.get_embeddings(texts)
        
        # Convert to numpy arrays for efficient computation
        import numpy as np
        query_emb = np.array(query_emb)
        text_embs = np.array(text_embs)
        
        # Compute cosine similarity
        similarities = np.dot(text_embs, query_emb) / (
            np.linalg.norm(text_embs, axis=1) * np.linalg.norm(query_emb)
        )
        
        return similarities.tolist()

    def _call_openai(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Call OpenAI API with the given prompt."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        messages.append({"role": "user", "content": prompt})
        
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
    
    def _call_ollama(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Call Ollama API with the given prompt."""
        url = f"{self.api_base}/generate"
        
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
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text from the LLM with the given prompt."""
        if self.provider == "openai":
            return self._call_openai(prompt, system_prompt)
        elif self.provider == "ollama":
            return self._call_ollama(prompt, system_prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def identify_entities(self, query: str, entity_names: List[str]) -> List[str]:
        """
        Identify potential entities from the query using LLM.
        Returns the names of entities identified in the query.
        """
        prompt = ENTITY_RECOGNITION_PROMPT.format(
            query=query,
            entity_names=', '.join(entity_names)
        )
        
        result = self.generate(prompt, ENTITY_RECOGNITION_SYSTEM_PROMPT)
        
        # Parse the result - expecting a comma-separated list of entity names
        entities = [name.strip() for name in result.split(',') if name.strip()]
        return entities
    
    def rank_relations(self, query: str, entity_name: str, relations: List[Dict]) -> List[Dict]:
        """
        Rank a list of potential relations by their relevance to the query.
        Returns the ranked relations with scores.
        """
        relation_info = '\n'.join([f"- {rel['relation_name']} (connects {entity_name} to other entities)" for rel in relations])
        
        prompt = RELATION_RANKING_PROMPT.format(
            query=query,
            entity_name=entity_name,
            relation_info=relation_info
        )
        
        result = self.generate(prompt, RELATION_RANKING_SYSTEM_PROMPT)
        
        # Extract JSON from the response
        try:
            # Try to find JSON by looking for array notation
            json_str = result[result.find('['):result.rfind(']')+1]
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
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Error parsing relation ranking: {e}")
            # Fallback - just assign scores based on position in the list
            for i, rel in enumerate(relations):
                rel['score'] = (len(relations) - i) / len(relations)
            return relations
    
    def generate_answer(self, query: str, sentences: List[Dict], entity_chains: List[Tuple]) -> str:
        """
        Generate a comprehensive answer based on the query, relevant sentences, and entity chains.
        """
        # Format the sentences for the prompt
        formatted_sentences = ""
        if sentences:
            # Sort by relevance
            sorted_sentences = sorted(sentences, key=lambda x: x['score'], reverse=True)
            for i, sent in enumerate(sorted_sentences[:10]):  # Use top 10 sentences
                formatted_sentences += f"{i+1}. \"{sent['text']}\" (Source: {sent['entity_name']}, Relevance: {sent['score']:.2f})\n"
        
        # Format the entity chains
        formatted_chains = ""
        if entity_chains:
            for i, chain in enumerate(entity_chains[:5]):  # Use top 5 chains
                formatted_chains += f"{i+1}. {chain[0]} → {chain[1]} → {chain[2]}\n"
        
        prompt = ANSWER_GENERATION_PROMPT.format(
            query=query,
            formatted_sentences=formatted_sentences,
            formatted_chains=formatted_chains
        )
        
        return self.generate(prompt, ANSWER_GENERATION_SYSTEM_PROMPT)
    
    def analyze_entity_context(self, query: str, entity_name: str, context: str) -> float:
        """
        Analyze how relevant an entity's context is to the query.
        Returns a relevance score between 0 and 1.
        """
        prompt = CONTEXT_ANALYSIS_PROMPT.format(
            query=query,
            entity_name=entity_name,
            context=context
        )
        
        result = self.generate(prompt, CONTEXT_ANALYSIS_SYSTEM_PROMPT)
        
        # Extract the numeric score
        try:
            # Find any number in the response
            import re
            match = re.search(r'\b(\d+(\.\d+)?)\b', result)
            if match:
                score = float(match.group(1))
                # Normalize to 0-1 range
                return min(score / 10.0, 1.0)
            else:
                return 0.5  # Default mid-value if no score found
        except ValueError:
            return 0.5  # Default mid-value