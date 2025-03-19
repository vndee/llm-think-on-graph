"""Constants for LLM prompts."""

ENTITY_RECOGNITION_SYSTEM_PROMPT = "You are an expert entity recognition system. Extract entities from the given query."

ENTITY_RECOGNITION_PROMPT = """Given the following query and a list of known entities, identify which entities (if any) are
mentioned or strongly implied in the query. Return only the names of the entities, separated by commas.

Query: {query}

Known entities:
{entity_names}

Entities mentioned or implied in the query:"""

RELATION_RANKING_SYSTEM_PROMPT = "You are an AI assistant that helps rank relations by relevance to a query."

RELATION_RANKING_PROMPT = """Given a query and a set of relations connected to entity "{entity_name}", rank the relations 
by their likely relevance to answering the query. Return the results as a JSON array where each item has 
"relation_name" and "score" (0-10, with 10 being most relevant).

Query: {query}

Relations:
{relation_info}

JSON ranking:"""

ANSWER_GENERATION_SYSTEM_PROMPT = """You are an AI assistant specialized in generating comprehensive answers based on retrieved information.
Focus on being accurate, concise, and directly answering the question using only the provided information."""

ANSWER_GENERATION_PROMPT = """Please answer the following question based strictly on the provided information.
Do not introduce facts not present in the provided sentences or entity relationships.

Question: {query}

Relevant Information:
{formatted_sentences}

Entity Relationships:
{formatted_chains}

Provide a comprehensive, factual answer to the question based solely on the information above:"""

CONTEXT_ANALYSIS_SYSTEM_PROMPT = "You are an expert at evaluating text relevance to questions."

CONTEXT_ANALYSIS_PROMPT = """On a scale of 0 to 10, rate how relevant the following text about "{entity_name}" is to 
answering the query. Return only the numeric score, nothing else.

Query: {query}

Text about {entity_name}:
{context}

Relevance score (0-10):"""
