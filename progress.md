# LangGraph Research Project

## Project Overview
This is an automated literature review generation tool built with LangGraph. The system uses a multi-agent architecture to research academic papers and generate comprehensive literature surveys on specified topics.

**Key Features:**
- Automated research planning and execution
- Unified LLM provider via OpenRouter (access to 200+ models)
- Document retrieval and processing with RAG
- Literature survey generation
- Plan caching for efficiency
- Interactive workflow with human-in-the-loop capability

## Architecture

### Main Workflow (`agents/graph.py`)
The main graph orchestrates two sequential subgraphs:
1. **Planning Agent** → **Refinement Agent**

### Core Components

#### Planning Agent (`agents/planning_agent/`)
- **Purpose**: Generates research plans and search queries
- **Key Files**:
  - `graph.py` - Planning workflow graph
  - `nodes/reserch_plan_nodes.py` - Planning logic nodes
  - `tools.py` - ArXiv search and document tools
  - `prompts.py` - LLM prompts for planning

#### Refinement Agent (`agents/refinement_agent/`)
- **Purpose**: Processes documents and creates literature surveys
- **Key Files**:
  - `graph.py` - Refinement workflow graph
  - `nodes/writing.py` - RAG-based content generation
  - `nodes/review_content.py` - Content quality review
  - `nodes/review_grounding.py` - Citation verification
  - `nodes/feedback_processing.py` - Iterative refinement

#### Shared Components (`agents/shared/`)
- **State Management**: `state/main_state.py` - Central AgentState class
- **Configuration**: `main_config.py` - Unified OpenRouter configuration
- **LLM Utilities**: `utils/llm_utils.py` - Model initialization helpers
- **Common Nodes**: `nodes/general_nodes.py`

### State Structure (`AgentState`)
```python
{
    "topic": str,                    # Research topic
    "paper_recency": str,           # Time filter for papers
    "caching_options": dict,        # Plan/section caching
    "messages": list,               # Conversation history
    "search_queries": list,         # Generated search queries
    "plan": Plan,                   # Research plan structure
    "refinement_progress": dict,    # Processing status
    "literature_survey": list,      # Final sections
    "completed": bool              # Workflow completion
}
```

## Configuration

### Environment Variables (`.env`)
All configuration is managed via environment variables:

```bash
# API Keys
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# Model Configuration (OpenRouter format: provider/model-name)
ORCHESTRATOR_MODEL=openai/gpt-4o          # For planning and review
TEXT_MODEL=openai/gpt-4-turbo            # For content generation
EMBEDDING_MODEL=qwen/qwen3-embedding-8b  # For RAG (MTEB #1)
```

### Model Roles
- **Orchestrator Model**: Used for complex reasoning tasks (planning, review, grounding checks)
- **Text Model**: Used for text generation (writing subsections, summaries)
- **Embedding Model**: Used for RAG vector search (document retrieval)

### Supported Models
Via OpenRouter, you can use any model from:
- OpenAI (GPT-4, GPT-3.5, etc.)
- Anthropic (Claude 3.5 Sonnet, Opus, etc.)
- Google (Gemini Pro, etc.)
- Meta (Llama 3.1, 3.2, etc.)
- Mistral, Cohere, and 200+ others

See https://openrouter.ai/models for full list.

## Key Files

### Entry Points
- **`main.py`** - Main execution script with configuration
- **`agents/graph.py`** - Main workflow definition

### Configuration
- **`pyproject.toml`** - Poetry dependencies and project metadata
- **`.env`** - Environment variables (API keys, model selection)
- **`agents/shared/main_config.py`** - Configuration dataclass
- **`agents/shared/utils/llm_utils.py`** - LLM initialization utilities

### Data & Caching
- **`cache/plans/`** - Cached research plans (JSON files)
- **`graph_diagrams/`** - Auto-generated workflow visualizations

## Development

### Dependencies
Managed via Poetry. Key dependencies:
- `langgraph >= 1.0.3` - Workflow orchestration
- `langchain-core >= 1.1.0` - LangChain core functionality
- `langchain-openai >= 1.0.3` - OpenAI-compatible LLM integration (used with OpenRouter)
- `langchain-community >= 0.4.1` - Community integrations
- `arxiv >= 2.2.0` - Academic paper search
- `pymupdf >= 1.26.3` - PDF processing

### Running the Application
```bash
# Install dependencies
poetry install

# Run the main workflow
poetry run python main.py
```

### Configuration
Edit `main.py` to configure:
- `TOPIC` - Research topic
- `PAPER_RECENCY` - Time filter (e.g., "after 2023")
- `cached_plan_id` - Reuse existing plans

Or modify `.env` to change models:
- `ORCHESTRATOR_MODEL` - Planning/review model
- `TEXT_MODEL` - Content generation model
- `EMBEDDING_MODEL` - RAG embedding model

### Workflow Execution
The system runs asynchronously and supports:
- Plan caching for efficiency
- Human interrupts for input
- Graph visualization generation
- Multi-step processing with state persistence

### Graph Visualization
Workflow diagrams are automatically generated:
- `graph_diagrams/main_graph.png` - Overall workflow
- `graph_diagrams/planning_graph.png` - Planning subgraph

## Common Tasks

### Changing Models
Simply update the `.env` file with your desired model:
```bash
ORCHESTRATOR_MODEL=anthropic/claude-3.5-sonnet
TEXT_MODEL=openai/gpt-4-turbo
EMBEDDING_MODEL=qwen/qwen3-embedding-8b
```

No code changes required! All models are accessed through OpenRouter's unified API.

### Modifying Research Logic
- **Planning**: Edit nodes in `agents/planning_agent/nodes/`
- **Refinement**: Edit nodes in `agents/refinement_agent/nodes/`

### Extending State
1. Update `agents/shared/state/main_state.py`
2. Update component files (`planning_components.py`, `refinement_components.py`)

### Cache Management
- Plans cached in `cache/plans/` with UUID filenames
- Clear cache by deleting JSON files
- Use `cached_plan_id` in main.py to reuse plans

### Adding New Model Providers
The system uses OpenRouter's unified API, so no code changes are needed to access new models. Simply use the model ID from https://openrouter.ai/models in your `.env` file.

## Troubleshooting

### Common Issues
- **Missing API keys**: Check `.env` file has `OPENROUTER_API_KEY` set
- **Import errors**: Run `poetry install` to ensure dependencies
- **Graph execution errors**: Check recursion limit in `RunnableConfig`
- **Model not found**: Verify model ID at https://openrouter.ai/models

### Debug Mode
Monitor execution via message history in final state:
```python
final_state["messages"]  # View conversation flow
final_state["plan"]      # Check generated plan
```

## Recent Updates

### January 2025 - OpenRouter Migration
- **Simplified Configuration**: Removed `model_config.json`, all config in `.env`
- **Single Provider**: Migrated from multi-provider (OpenAI, Anthropic, Google) to OpenRouter
- **Unified API**: Access 200+ models through single API endpoint
- **Updated Dependencies**: Upgraded to LangChain 1.x and LangGraph 1.x
- **Better Embeddings**: Using Qwen3-Embedding-8B (#1 on MTEB leaderboard)
- **Code Reduction**: ~40% less configuration code
- **Removed Dependencies**: Eliminated `langchain-anthropic` and `langchain-google-genai`

### Architecture Improvements
- Simplified `agents/shared/main_config.py` (removed multi-provider logic)
- Streamlined `agents/shared/utils/llm_utils.py` (single provider pattern)
- Added `get_embedding_model()` utility for consistent model initialization
- Environment-based configuration for easier deployment

## Performance Notes

### Embedding Model Choice
Currently using **Qwen3-Embedding-8B** for RAG:
- Ranked #1 on MTEB multilingual leaderboard (score: 70.58)
- 32k token context window
- Very cost-effective: $0.01 per million tokens
- Superior to OpenAI embeddings for most tasks

### Model Recommendations
- **Fast iterations**: Use `openai/gpt-4o-mini` or `anthropic/claude-3-haiku`
- **Best quality**: Use `openai/gpt-4` or `anthropic/claude-3.5-sonnet`
- **Cost-effective**: Use `meta-llama/llama-3.1-70b` or `google/gemini-pro`
- **Specialized**: Check https://openrouter.ai/models for task-specific models
