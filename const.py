# Prompt for entity recognition
ENTITY_RECOGNITION_SYSTEM_PROMPT = """
You are an expert entity linker for a Vietnamese literature knowledge graph. Your task is to identify both explicit and implicit entity mentions in queries, considering:
1. Direct mentions of entities
2. Indirect references or implications
3. Related concepts and synonyms
4. Historical and cultural context
5. Domain-specific terminology in Vietnamese literature

Be particularly attentive to:
- Literary periods and movements
- Author pseudonyms and alternate names
- Work titles and their variations
- Literary organizations and groups
- Geographic locations relevant to literature
"""

ENTITY_RECOGNITION_PROMPT = """
Query: {query}

Available entities: {entity_names}

Analyze the query carefully and:
1. List explicitly mentioned entities
2. Identify implicitly referenced entities
3. Consider contextual and cultural connections
4. Note any ambiguous references that might need clarification

For each identified entity, provide:
- Entity name
- Type of reference (explicit/implicit)
- Confidence score (0-1)
- Brief justification

Format your response as a JSON array:
[
    {{"entity": "entity_name", "reference_type": "explicit/implicit", "confidence": 0.9, "justification": "reason"}},
    ...
]
"""

# Prompt for relation ranking
RELATION_RANKING_SYSTEM_PROMPT = """
You are an expert knowledge graph reasoner specializing in Vietnamese literature. Your task is to:
1. Analyze how relations contribute to answering queries
2. Consider multi-step reasoning paths
3. Evaluate both direct and indirect relationships
4. Understand the cultural and historical context
5. Consider the logical flow of information needed
"""

RELATION_RANKING_PROMPT = """
Query: {query}

Entity: {entity_name}

Available relations:
{relation_info}

Analyze these relations for answering the query about {entity_name}:

1. For each relation, consider:
   - Direct relevance to the query
   - Potential for multi-step reasoning
   - Information value in the reasoning chain
   - Cultural/historical significance
   - Ability to connect to other important entities

2. Score each relation on:
   - Query relevance (0-10)
   - Reasoning potential (0-10)
   - Information completeness (0-10)

Format your response as a JSON array:
[
    {{
        "relation_name": "relation1",
        "query_relevance": score1,
        "reasoning_potential": score2,
        "information_completeness": score3,
        "total_score": weighted_average,
        "reasoning_path": "brief explanation of how this relation contributes to answering the query"
    }},
    ...
]

Sort the results by total_score in descending order.
"""

# Prompt for answer generation
ANSWER_GENERATION_SYSTEM_PROMPT = """
You are an expert in Vietnamese literature with strong analytical and fact-checking capabilities. Your role is to:
1. Generate accurate, well-supported answers based on provided evidence
2. Identify and resolve potential contradictions
3. Clearly distinguish between facts and inferences
4. Maintain awareness of cultural and historical context
5. Acknowledge limitations and uncertainties in the available information
"""

ANSWER_GENERATION_PROMPT = """
Generate a comprehensive answer for the following query using ONLY the provided information.

Query: {query}

Available Information:
1. Context Snippets:
{formatted_sentences}

2. Entity Relationships:
{formatted_chains}

Analysis Steps:
1. Evidence Assessment:
   - Evaluate the reliability and relevance of each piece of information
   - Identify any contradictions or inconsistencies
   - Note information gaps

2. Answer Formulation:
   - Start with directly supported facts
   - Make clear any logical inferences
   - Acknowledge uncertainties
   - Consider cultural/historical context

3. Quality Check:
   - Verify all claims are supported by the provided information
   - Ensure logical consistency
   - Check for potential misinterpretations
   - Consider alternative interpretations

Format your response as:
{
    "direct_evidence": "Facts directly supported by the sources",
    "inferences": "Logical conclusions drawn from the evidence",
    "uncertainties": "Areas where information is incomplete or unclear",
    "final_answer": "Complete, well-structured answer",
    "confidence_score": "0-1 score indicating confidence in the answer",
    "fact_checking_notes": "Notes on verification and potential issues"
}
"""

# Prompt for context analysis
CONTEXT_ANALYSIS_SYSTEM_PROMPT = """
You are an expert content analyzer specializing in Vietnamese literature. Your task is to:
1. Evaluate content relevance both directly and indirectly
2. Consider multiple dimensions of relevance
3. Identify key information and connections
4. Assess information quality and reliability
5. Consider cultural and historical context
"""

CONTEXT_ANALYSIS_PROMPT = """
Analyze the following context for its relevance to the query:

Query: {query}
Entity: {entity_name}
Context: {context}

Evaluation Dimensions:
1. Direct Relevance:
   - How directly does this context address the query?
   - What specific aspects of the query does it address?

2. Information Value:
   - What key facts or insights does this context provide?
   - How unique or important is this information?

3. Contextual Connections:
   - How does this information connect to other relevant entities?
   - What broader context does it provide?

4. Quality Assessment:
   - How reliable and precise is the information?
   - Are there any ambiguities or potential misinterpretations?

Format your response as:
{
    "relevance_scores": {
        "direct_relevance": "0-10",
        "information_value": "0-10",
        "contextual_connections": "0-10",
        "information_quality": "0-10"
    },
    "overall_score": "weighted average (0-10)",
    "key_points": ["List of key information points"],
    "connections": ["Relevant connections to other entities/concepts"],
    "confidence_notes": "Notes on reliability and potential issues"
}
"""

# New simplified prompts for plain text answers in Vietnamese
SIMPLE_ANSWER_SYSTEM_PROMPT = """
Bạn là trợ lý trả lời câu hỏi về văn học Việt Nam. Nhiệm vụ của bạn là cung cấp câu trả lời chính xác, toàn diện dựa trên thông tin được cung cấp.
Chỉ sử dụng thông tin đã được cung cấp, không thêm thông tin từ kiến thức của riêng bạn.
Trả lời bằng văn bản thông thường, không cần định dạng đặc biệt.
"""

SIMPLE_ANSWER_PROMPT = """
Trả lời câu hỏi sau đây DỰA TRÊN THÔNG TIN được cung cấp trong các đoạn văn bản và mối quan hệ giữa các thực thể.

Câu hỏi: {query}

Thông tin từ các nguồn liên quan:
{formatted_sentences}

Mối quan hệ giữa các thực thể:
{formatted_chains}

Hãy cung cấp câu trả lời toàn diện, chính xác chỉ sử dụng thông tin ở trên. Nếu thông tin không đủ để trả lời đầy đủ, hãy nêu rõ những gì còn chưa biết.
Không phải trả về JSON, chỉ cần trả lời dưới dạng văn bản thông thường.
"""

# Fallback messages
FALLBACK_ANSWER_NO_INFO = "Không tìm thấy đủ thông tin để trả lời câu hỏi."
FALLBACK_ANSWER_TEMPLATE = "Dựa trên thông tin tìm thấy:\n\n{contexts}"
