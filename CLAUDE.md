# LangGraph Research Project

## Project Overview
This is an automated literature review generation tool built with LangGraph. The system uses a multi-agent architecture to research academic papers and generate comprehensive literature surveys on specified topics.

**Key Features:**
- Automated research planning and execution
- Multi-LLM support (OpenAI, Anthropic, Google)
- Document retrieval and processing
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
  - `nodes/rag_nodes.py` - RAG processing nodes
  - `nodes/refinement_nodes.py` - Survey generation nodes

#### Shared Components (`agents/shared/`)
- **State Management**: `state/main_state.py` - Central AgentState class
- **Common Nodes**: `nodes/general_nodes.py`
- **Utilities**: `utils/llm_utils.py`

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

## Key Files

### Entry Points
- **`main.py`** - Main execution script with configuration
- **`agents/graph.py`** - Main workflow definition

### Configuration
- **`pyproject.toml`** - Poetry dependencies and project metadata
- **`.env`** - Environment variables (API keys, etc.)

### Data & Caching
- **`cache/plans/`** - Cached research plans (JSON files)
- **`graph_diagrams/`** - Auto-generated workflow visualizations

## Development

### Dependencies
Managed via Poetry. Key dependencies:
- `langgraph` - Workflow orchestration
- `langchain-openai`, `langchain-anthropic`, `langchain-google-genai` - LLM providers
- `arxiv` - Academic paper search
- `pymupdf` - PDF processing

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

### Adding New LLM Providers
1. Add dependency to `pyproject.toml`
2. Update `agents/shared/utils/llm_utils.py`
3. Configure environment variables

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

## Troubleshooting

### Common Issues
- **Missing API keys**: Check `.env` file configuration
- **Import errors**: Run `poetry install` to ensure dependencies
- **Graph execution errors**: Check recursion limit in `RunnableConfig`

### Debug Mode
Monitor execution via message history in final state:
```python
final_state["messages"]  # View conversation flow
final_state["plan"]      # Check generated plan
```