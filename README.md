# Think-on-Graph 2.0 (ToG-2)

A demonstration of the ToG-2 approach for multi-step reasoning using Knowledge Graphs and Large Language Models.

## Overview

Think-on-Graph 2.0 (ToG-2) is an approach that tightly integrates Knowledge Graphs with textual documents to enhance the multi-step reasoning capabilities of Large Language Models (LLMs).

### Key Features

1. **Tight integration of KG and text**: Uses KG to guide information search in text, while using text to enrich and refine graph search.
2. **Deep multi-step search**: Allows LLMs to perform deeper reasoning by iteratively searching and accumulating information.
3. **Grounded reasoning**: Enhances the reliability of LLM answers by providing information from both KG and text.
4. **Flexibility**: Can be applied to various LLMs without retraining.

## Project Structure

The project is organized into several modules:

- `app.py`: Main Streamlit application for the web interface
- `knowledge_graph.py`: Knowledge Graph module for storing and navigating the knowledge graph
- `reasoner.py`: The ToG-2 reasoner that implements the reasoning algorithm
- `llm.py`: LLM interface for various language model providers
- `const.py`: Constants and prompt templates
- `style.css`: CSS styling for the Streamlit interface
- `data/`: Contains sample data for the application

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/llm-thinking-on-graph.git
cd llm-thinking-on-graph
```

2. Install UV (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Create a virtual environment and install dependencies:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install .
```

## Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

This will start the web interface where you can:
1. Select data sources (sample data or upload your own)
2. Configure LLM parameters 
3. Set ToG-2 search parameters
4. Ask questions and see the reasoning process

## Data Format

The system expects the following JSON files:
- `entities.json`: List of entities with ID, name, and type
- `relations.json`: List of relation types with ID and name
- `triples.json`: Graph triples (head, relation, tail)
- `documents.json`: Textual documents associated with entities

## LLM Support

The system supports:
- OpenAI API (GPT-3.5-Turbo, GPT-4o-mini, GPT-4o)
- Ollama (llama3, mistral)

## Customizing

You can extend this project by:
1. Adding new LLM providers in `llm.py`
2. Extending reasoning strategies in `reasoner.py`
3. Adapting the knowledge graph structure in `knowledge_graph.py`

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project demonstrates the Think-on-Graph approach for multi-step reasoning.
