- Simplify grounding review to match the structure of content review - so for each issue we get single detailed point (include arxiv id, citation, rexplanation, reccomendation, etc)

- Consider adding two message threads - quality and grounding review that are fed to the model as a whole

- Add image figure parsing

- Try loading whole papers into the context when writing a subsection instead of the rag (it should fit)

- First to grounding refinement, then do content refinement

- store the citations etc as json, fix the caching / storing, make a more general solution

---

## Desktop App Concept

### Vision
Build a local desktop application that provides a GUI interface for the LangGraph research system with:
- Chat interface for natural interaction
- LaTeX preview for rendered literature reviews
- Session storage and history
- Export capabilities (LaTeX/PDF)

### Technology Stack
**Core Framework:**
- PyQt6 - Pure Python GUI framework for native desktop experience
- PyQt6-WebEngine - For rendering LaTeX via MathJax

**LaTeX & Export:**
- MathJax - Client-side LaTeX rendering in preview panel
- Jinja2 - Template engine for LaTeX document generation
- PyLaTeX or pypandoc - LaTeX/PDF export

**Storage:**
- SQLite - Session metadata, search/filter capabilities
- File-based - Session content (JSON/Markdown in cache/)
- Hybrid approach: metadata in DB, content in files

### Key Features

**1. Chat Interface**
- Message bubbles (user/assistant)
- Support markdown + LaTeX in messages
- Progress indicators for long-running research tasks
- Async execution with real-time updates

**2. LaTeX Preview Panel**
- QWebEngineView with MathJax for beautiful equation rendering
- Live updates as research progresses
- Section navigation (jump to intro, methodology, etc.)
- Support for inline ($...$) and block ($$...$$) equations

**3. Session Management**
- Session history sidebar with search/filter
- Save/load research sessions
- Metadata tracking (topic, date, status, paper count)
- Extend existing cache/plans/ structure

**4. Export Capabilities**
- Export to LaTeX (.tex) with academic formatting
- Export to PDF via pdflatex or pandoc
- Include bibliography from researched papers
- Customizable templates

### Architecture Approach

**Dual Operation Modes:**

1. **Direct Integration (Default):**
   - Import and run existing LangGraph graph directly in app
   - Async execution with progress callbacks
   - Stream results to chat interface
   - All-in-one Python application

2. **MCP Mode (Optional):**
   - Create `mcp_server.py` exposing research tools
   - Desktop app acts as MCP client
   - Modular, can connect to other MCP servers
   - More complex but extensible

### Proposed File Structure
```
langgraph-research/
├── desktop_app/
│   ├── main.py                    # GUI entry point
│   ├── ui/
│   │   ├── main_window.py        # Main window layout
│   │   ├── chat_widget.py        # Chat interface
│   │   ├── preview_widget.py     # LaTeX preview panel
│   │   └── session_browser.py    # Session history sidebar
│   ├── backend/
│   │   ├── graph_wrapper.py      # LangGraph integration layer
│   │   ├── mcp_client.py         # Optional MCP client
│   │   └── session_manager.py    # Session persistence
│   ├── storage/
│   │   ├── database.py           # SQLite manager
│   │   └── schema.sql            # DB schema
│   ├── exporters/
│   │   └── latex_exporter.py     # LaTeX/PDF export
│   └── templates/
│       ├── preview.html          # MathJax template
│       └── paper.tex.j2          # LaTeX paper template
├── mcp_server.py                  # Optional MCP server
├── agents/                        # Existing LangGraph code (unchanged)
└── main.py                        # Existing CLI (unchanged)
```

### Implementation Notes

**Integration Strategy:**
- Minimal changes to existing LangGraph agents
- Create wrapper class to bridge graph execution with GUI signals
- Reuse all existing nodes, prompts, tools
- Keep CLI `main.py` functional alongside desktop app

**Database Schema:**
```sql
sessions (
    id TEXT PRIMARY KEY,
    topic TEXT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    status TEXT,  -- 'planning', 'researching', 'completed'
    plan_id TEXT,
    metadata JSON
)

messages (
    id INTEGER PRIMARY KEY,
    session_id TEXT,
    role TEXT,  -- 'user', 'assistant', 'system'
    content TEXT,
    timestamp TIMESTAMP,
    FOREIGN KEY(session_id) REFERENCES sessions(id)
)
```

**UI Layout:**
```
┌─────────────────────────────────────────────────────────────┐
│ [New] [Export] [Settings]                          [Theme]  │
├────────────┬──────────────────────────┬────────────────────┤
│            │                          │                    │
│  Sessions  │      Chat Messages       │   LaTeX Preview    │
│            │                          │                    │
│  • ML 2024 │  User: Research topic... │  # Introduction    │
│  • NLP     │  Assistant: Starting...  │                    │
│  • CV      │  ...                     │  Recent advances...│
│            │                          │                    │
│  [Search]  │  ┌────────────────────┐  │  $$\alpha = ...$$  │
│            │  │ Type message...    │  │                    │
│            │  └────────────────────┘  │                    │
└────────────┴──────────────────────────┴────────────────────┘
```

### MCP Server Approach

If implementing MCP mode, expose tools like:

```python
@mcp.tool()
async def research_topic(topic: str, paper_recency: str = "after 2023") -> dict:
    """Generate a literature review on a given topic"""

@mcp.tool()
async def search_papers(query: str, max_results: int = 10) -> list:
    """Search ArXiv for academic papers"""

@mcp.resource("cache://plans/{plan_id}")
def get_cached_plan(plan_id: str) -> str:
    """Access cached research plans"""
```

### Dependencies to Add
```toml
[tool.poetry.dependencies]
PyQt6 = "^6.6.0"
PyQt6-WebEngine = "^6.6.0"
jinja2 = "^3.1.0"
pylatex = "^1.4.0"  # or pypandoc
mcp = {extras = ["cli"], version = "^1.0.0"}  # Optional, for MCP mode
```

### Future Enhancements
- Multi-session comparison view
- Citation manager integration
- Collaborative features (shared sessions)
- Plugin system for custom exporters
- Voice input for chat
- GPU acceleration for preview rendering