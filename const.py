# Prompt for entity recognition
ENTITY_RECOGNITION_SYSTEM_PROMPT = """
You are an entity linker that identifies entities mentioned in queries. Extract only the entities that exist in the provided list of entity names.
"""

ENTITY_RECOGNITION_PROMPT = """
Query: {query}

Available entities: {entity_names}

List only the entities that are directly mentioned or clearly referenced in the query. Return them as a comma-separated list.
"""

# Prompt for relation ranking
RELATION_RANKING_SYSTEM_PROMPT = """
You are a knowledge graph relation ranker that helps identify which relations are most relevant to answering a query.
"""

RELATION_RANKING_PROMPT = """
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

# Prompt for answer generation
ANSWER_GENERATION_SYSTEM_PROMPT = """
You are a helpful assistant that provides accurate, comprehensive answers based on the provided information.
"""

ANSWER_GENERATION_PROMPT = """
Answer the following query based ONLY on the information provided in the context snippets and entity chains.

Query: {query}

Relevant context snippets:
{formatted_sentences}

Entity connections:
{formatted_chains}

Provide a comprehensive, accurate answer using only the information above. If the information is insufficient to fully answer the query, clearly state what remains unknown.
"""

# Prompt for context analysis
CONTEXT_ANALYSIS_SYSTEM_PROMPT = """
You are a content analyzer that evaluates how relevant a piece of text is to a specific query.
"""

CONTEXT_ANALYSIS_PROMPT = """
Query: {query}

Entity: {entity_name}

Context: {context}

On a scale of 0 to 10, how relevant is this context for answering the query about {entity_name}?
Provide only the numeric score without explanation.
"""
