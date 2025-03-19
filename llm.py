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
    SIMPLE_ANSWER_PROMPT,
    SIMPLE_ANSWER_SYSTEM_PROMPT,
    FALLBACK_ANSWER_TEMPLATE,
    FALLBACK_ANSWER_NO_INFO,
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

        try:
            # Use the improved prompt from const.py
            prompt = ENTITY_RECOGNITION_PROMPT.format(
                query=query,
                entity_names=entity_names_str
            )
            
            # Log the prompt for debugging
            logger.debug(f"Entity recognition prompt: {prompt}")
            
            result = self.generate(prompt, ENTITY_RECOGNITION_SYSTEM_PROMPT)
            
            # Log the result for debugging
            logger.debug(f"Entity recognition result: {result}")
            
            # Try to extract JSON from the response
            import re
            import json
            
            # First, try direct JSON parsing
            try:
                entities_data = json.loads(result)
                logger.info("Successfully parsed entity JSON response")
                # Extract entities with high confidence (> 0.7)
                return [item['entity'] for item in entities_data if item['confidence'] > 0.7]
            except json.JSONDecodeError:
                # Try to extract JSON array pattern from the text
                json_match = re.search(r'\[\s*\{.*\}\s*\]', result, re.DOTALL)
                if json_match:
                    try:
                        json_str = json_match.group(0)
                        entities_data = json.loads(json_str)
                        logger.info("Successfully extracted and parsed entity JSON from response")
                        return [item['entity'] for item in entities_data if item['confidence'] > 0.7]
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Failed to parse extracted JSON: {e}")
                
                # If JSON parsing fails, fallback to simple text extraction
                # Check for entity names directly in the text
                found_entities = []
                for name in entity_names:
                    if name.lower() in result.lower():
                        found_entities.append(name)
                
                if found_entities:
                    logger.info(f"Extracted entities by name matching in response: {found_entities}")
                    return found_entities
                
                # Last resort: look for quotes which might contain entity names
                quoted_entities = re.findall(r'"([^"]*)"', result)
                valid_entities = [e for e in quoted_entities if e in entity_names]
                
                if valid_entities:
                    logger.info(f"Extracted entities from quotes in response: {valid_entities}")
                    return valid_entities
                
                # If all else fails, fall back to comma-separated parsing
                logger.warning("All JSON parsing methods failed, falling back to comma-separated parsing")
                entities = [name.strip() for name in result.split(',') if name.strip() and name.strip() in entity_names]
                return entities
                
        except Exception as e:
            logger.error(f"Error in entity identification: {e}")
            # Emergency fallback - simple text matching
            return [name for name in entity_names if name.lower() in query.lower()]
    
    def rank_relations(self, query: str, entity_name: str, relations: List[Dict]) -> List[Dict]:
        """Rank relations by their relevance to the query"""
        if not relations:
            return []
            
        relation_info = '\n'.join([f"- {rel['relation_name']} (connects {entity_name} to other entities)" for rel in relations])
        
        try:
            # Use the improved prompt from const.py
            prompt = RELATION_RANKING_PROMPT.format(
                query=query,
                entity_name=entity_name,
                relation_info=relation_info
            )
            
            # Log the prompt for debugging
            logger.debug(f"Relation ranking prompt: {prompt}")
            
            result = self.generate(prompt, RELATION_RANKING_SYSTEM_PROMPT)
            
            # Log the result for debugging
            logger.debug(f"Relation ranking result: {result}")
            
            # Try to extract JSON from the response
            import re
            import json
            
            # First, try direct JSON parsing
            try:
                ranked_relations = json.loads(result)
                logger.info("Successfully parsed relation ranking JSON response")
            except json.JSONDecodeError:
                # Try to extract JSON array pattern from the text
                json_match = re.search(r'\[\s*\{.*\}\s*\]', result, re.DOTALL)
                if json_match:
                    try:
                        json_str = json_match.group(0)
                        ranked_relations = json.loads(json_str)
                        logger.info("Successfully extracted and parsed relation JSON from response")
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse extracted relation JSON: {e}")
                        # Fallback to simpler format that might be present
                        simple_match = re.findall(r'"?relation_name"?\s*:\s*"([^"]+)".*?"?score"?\s*:\s*(\d+(?:\.\d+)?)', result)
                        if simple_match:
                            ranked_relations = [{"relation_name": name, "score": float(score)} for name, score in simple_match]
                            logger.info("Extracted relation rankings using regex patterns")
                        else:
                            raise ValueError("Could not extract any valid relation rankings")
                else:
                    raise ValueError("No JSON array pattern found in response")
            
            # Match the ranked relations back to the original relations data
            enriched_relations = []
            for ranked in ranked_relations:
                for original in relations:
                    if ranked['relation_name'] == original['relation_name']:
                        # Check if we have the new format with multiple scores
                        if 'query_relevance' in ranked:
                            # Calculate weighted average score
                            total_score = (
                                ranked.get('query_relevance', 0) + 
                                ranked.get('reasoning_potential', 0) + 
                                ranked.get('information_completeness', 0)
                            ) / 30.0  # Normalize to 0-1 range
                        else:
                            # Old format with single score
                            total_score = float(ranked.get('score', 0)) / 10.0
                            
                        enriched = {
                            **original,
                            'score': total_score,
                            'reasoning_path': ranked.get('reasoning_path', '')
                        }
                        enriched_relations.append(enriched)
                        break
                    
            # Sort by score in descending order
            enriched_relations.sort(key=lambda x: x['score'], reverse=True)
            return enriched_relations
                
        except Exception as e:
            logger.error(f"Error in relation ranking: {e}")
            # Fallback - assign scores based on position
            for i, rel in enumerate(relations):
                rel['score'] = (len(relations) - i) / len(relations)
            return relations
    
    def generate_answer(self, query: str, contexts: List[Dict], entity_chains: List[Tuple]) -> str:
        """Generate answer based on contexts and entity chains"""
        # Format the contexts for the prompt
        formatted_sentences = ""
        if contexts:
            # Sort by relevance
            sorted_contexts = sorted(contexts, key=lambda x: x.get('score', 0), reverse=True)
            for i, ctx in enumerate(sorted_contexts[:10]):  # Use top 10 contexts
                formatted_sentences += f"{i+1}. \"{ctx['text']}\" (Source: {ctx.get('entity_name', 'Unknown')})\n"
        
        # Format the entity chains
        formatted_chains = ""
        if entity_chains:
            for i, chain in enumerate(entity_chains[:5]):  # Use top 5 chains
                formatted_chains += f"{i+1}. {chain[0]} → {chain[1]} → {chain[2]}\n"
        
        try:
            # Use the simplified prompt from const.py
            prompt = SIMPLE_ANSWER_PROMPT.format(
                query=query,
                formatted_sentences=formatted_sentences,
                formatted_chains=formatted_chains
            )
            
            # Log the full prompt and result for debugging
            logger.debug(f"Answer generation prompt: {prompt}")
            
            result = self.generate(prompt, SIMPLE_ANSWER_SYSTEM_PROMPT)
            
            logger.debug(f"Answer generation result: {result}")
            
            # Simply return the result as is - it's already in plain text format
            return result
                
        except Exception as e:
            logger.error(f"Error in answer generation: {str(e)}")
            # Log the error details for debugging
            import traceback
            logger.debug(f"Error details: {traceback.format_exc()}")
            
            # Emergency fallback - return a simple answer from the top contexts
            if contexts:
                top_contexts = sorted(contexts, key=lambda x: x.get('score', 0), reverse=True)[:3]
                contexts_text = ""
                for i, ctx in enumerate(top_contexts):
                    contexts_text += f"{i+1}. {ctx['text']} (từ {ctx['entity_name']})\n"
                return FALLBACK_ANSWER_TEMPLATE.format(contexts=contexts_text)
            else:
                return FALLBACK_ANSWER_NO_INFO