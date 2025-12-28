# LaTeX Export & Tkinter GUI Implementation Plan

## Overview

Two major features to implement:
1. **LaTeX Export** - Auto-generate `.tex` + `.bib` files in ZIP after review completion
2. **Tkinter GUI** - Full-featured GUI for managing literature reviews

---

## Feature 1: LaTeX Export

### Objective
Convert the completed literature survey into a LaTeX document with proper BibTeX citations, packaged as a ZIP file for easy compilation.

### New Files
- `export/__init__.py`
- `export/latex_exporter.py` - Main export logic

### Changes to Existing Files
- `main.py` - Add export call after review completion

---

### Implementation Details

#### 1. LatexExporter Class (`export/latex_exporter.py`)

```python
import re
import zipfile
from pathlib import Path
from typing import List, Dict
from datetime import datetime

from agents.shared.state.refinement_components import Section, Subsection, PaperWithSegements
from agents.shared.state.planning_components import Plan


class LatexExporter:
    """Export literature survey to LaTeX format with BibTeX citations."""

    def __init__(
        self,
        literature_survey: List[Section],
        plan: Plan,
        topic: str,
        review_id: str
    ):
        self.literature_survey = literature_survey
        self.plan = plan
        self.topic = topic
        self.review_id = review_id
        self.papers_map: Dict[str, PaperWithSegements] = {}  # arxiv_id -> paper

    def export_to_zip(self, output_dir: str = "exports") -> str:
        """
        Creates ZIP with main.tex + references.bib
        Returns path to the created ZIP file.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Collect all papers
        self._collect_papers()

        # Generate content
        latex_content = self._generate_latex()
        bibtex_content = self._generate_bibtex()

        # Create ZIP
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"review_{self.review_id[:8]}_{timestamp}.zip"
        zip_path = output_path / zip_filename

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("main.tex", latex_content)
            zf.writestr("references.bib", bibtex_content)

        return str(zip_path)

    def _collect_papers(self) -> None:
        """Collect all unique papers from subsections."""
        for section in self.literature_survey:
            for subsection in section.subsections:
                for paper in subsection.papers:
                    if paper.arxiv_id not in self.papers_map:
                        self.papers_map[paper.arxiv_id] = paper

    def _generate_latex(self) -> str:
        """Generate the main.tex content."""
        # Document preamble
        latex = r"""\documentclass[12pt,a4paper]{article}

% Packages
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{hyperref}
\usepackage[backend=biber,style=numeric,sorting=none]{biblatex}
\usepackage{geometry}
\usepackage{setspace}

% Bibliography
\addbibresource{references.bib}

% Page setup
\geometry{margin=1in}
\onehalfspacing

% Document info
"""
        # Add title
        escaped_topic = self._escape_latex(self.topic)
        latex += f"\\title{{{escaped_topic}}}\n"
        latex += "\\author{Generated Literature Review}\n"
        latex += f"\\date{{\\today}}\n\n"

        # Begin document
        latex += r"""\begin{document}

\maketitle
\tableofcontents
\newpage

"""

        # Add sections
        for section in self.literature_survey:
            latex += self._format_section(section)

        # Bibliography and end
        latex += r"""
\newpage
\printbibliography

\end{document}
"""
        return latex

    def _format_section(self, section: Section) -> str:
        """Format a single section with subsections."""
        content = ""

        # Section title
        escaped_title = self._escape_latex(section.section_title)
        content += f"\\section{{{escaped_title}}}\n\n"

        # Section introduction (the 2-sentence description)
        if section.section_introduction:
            escaped_intro = self._escape_latex(section.section_introduction)
            content += f"{escaped_intro}\n\n"

        # Subsections
        for subsection in section.subsections:
            content += self._format_subsection(subsection)

        return content

    def _format_subsection(self, subsection: Subsection) -> str:
        """Format a single subsection."""
        content = ""

        # Subsection title (key point text)
        escaped_title = self._escape_latex(subsection.key_point_text)
        content += f"\\subsection{{{escaped_title}}}\n\n"

        # Content with converted citations
        if subsection.content:
            converted_content = self._convert_citations(subsection.content)
            escaped_content = self._escape_latex_preserve_commands(converted_content)
            content += f"{escaped_content}\n\n"

        return content

    def _convert_citations(self, text: str) -> str:
        """
        Convert [Author_YEAR(ArxivID)] citations to \cite{key} format.

        Examples:
        - [Smith_2023(2301.12345)] -> \cite{arxiv_2301_12345}
        - [Smith_2023(2301.12345); Jones_2022(2201.54321)] -> \cite{arxiv_2301_12345,arxiv_2201_54321}
        """
        def replace_citation(match):
            citation_block = match.group(1)
            # Split by semicolon for multiple citations
            parts = citation_block.split(';')
            cite_keys = []

            for part in parts:
                # Extract arxiv_id from parentheses
                arxiv_match = re.search(r'\(([0-9v.]+)\)', part.strip())
                if arxiv_match:
                    arxiv_id = arxiv_match.group(1)
                    # Convert to valid BibTeX key (replace . with _)
                    cite_key = f"arxiv_{arxiv_id.replace('.', '_')}"
                    cite_keys.append(cite_key)

            if cite_keys:
                return f"\\cite{{{','.join(cite_keys)}}}"
            return match.group(0)  # Return original if no match

        # Pattern: [anything(arxiv_id)] or [a(id); b(id); ...]
        pattern = r'\[([^\]]+\([0-9v.]+\)[^\]]*)\]'
        return re.sub(pattern, replace_citation, text)

    def _generate_bibtex(self) -> str:
        """Generate references.bib content."""
        entries = []

        for arxiv_id, paper in self.papers_map.items():
            entry = self._format_bibtex_entry(paper)
            entries.append(entry)

        return "\n\n".join(entries)

    def _format_bibtex_entry(self, paper: PaperWithSegements) -> str:
        """Format a single BibTeX entry."""
        # Create key from arxiv_id
        cite_key = f"arxiv_{paper.arxiv_id.replace('.', '_')}"

        # Format authors (join with " and ")
        authors = " and ".join(paper.authors) if paper.authors else "Unknown"

        # Extract year from arxiv_id (first 2 digits indicate year)
        try:
            year_prefix = paper.arxiv_id[:2]
            year = f"20{year_prefix}" if int(year_prefix) < 50 else f"19{year_prefix}"
        except:
            year = "2023"

        # Escape special characters in title
        title = self._escape_bibtex(paper.title)

        entry = f"""@article{{{cite_key},
    title = {{{title}}},
    author = {{{authors}}},
    year = {{{year}}},
    eprint = {{{paper.arxiv_id}}},
    archivePrefix = {{arXiv}},
    primaryClass = {{cs}},
    url = {{{paper.arxiv_url}}}
}}"""
        return entry

    def _escape_latex(self, text: str) -> str:
        """Escape special LaTeX characters."""
        if not text:
            return ""

        # Characters that need escaping in LaTeX
        replacements = [
            ('\\', r'\textbackslash{}'),
            ('&', r'\&'),
            ('%', r'\%'),
            ('$', r'\$'),
            ('#', r'\#'),
            ('_', r'\_'),
            ('{', r'\{'),
            ('}', r'\}'),
            ('~', r'\textasciitilde{}'),
            ('^', r'\textasciicircum{}'),
        ]

        for old, new in replacements:
            text = text.replace(old, new)

        return text

    def _escape_latex_preserve_commands(self, text: str) -> str:
        """Escape LaTeX but preserve \cite commands."""
        if not text:
            return ""

        # Temporarily replace \cite commands
        cite_pattern = r'(\\cite\{[^}]+\})'
        cites = re.findall(cite_pattern, text)

        for i, cite in enumerate(cites):
            text = text.replace(cite, f"__CITE_PLACEHOLDER_{i}__")

        # Escape the rest
        text = self._escape_latex(text)

        # Restore \cite commands
        for i, cite in enumerate(cites):
            text = text.replace(f"__CITE_PLACEHOLDER_{i}__", cite)

        return text

    def _escape_bibtex(self, text: str) -> str:
        """Escape special characters for BibTeX."""
        if not text:
            return ""

        # BibTeX special characters
        text = text.replace('&', r'\&')
        text = text.replace('%', r'\%')
        text = text.replace('$', r'\$')
        text = text.replace('#', r'\#')
        text = text.replace('_', r'\_')

        return text
```

#### 2. Integration in `main.py`

Add after successful completion (around line 110):

```python
from export.latex_exporter import LatexExporter

# ... existing code ...

if final_state.completed:
    db.update_review_status(review.id, 'completed')
    db.update_review_metrics(
        review.id,
        total_sections=len(final_state.literature_survey),
        total_papers_used=len(db.get_papers_for_review(review.id))
    )

    # Export to LaTeX
    try:
        exporter = LatexExporter(
            final_state.literature_survey,
            final_state.plan,
            topic,
            review.id
        )
        zip_path = exporter.export_to_zip("exports")
        console.print(Panel(
            f"LaTeX export saved to: {zip_path}",
            title="[bold green]Export Complete[/bold green]",
            expand=False
        ))
    except Exception as e:
        logger.error(f"LaTeX export failed: {e}")
        console.print(Panel(f"Export failed: {e}", style="bold yellow"))

    console.print(Panel(
        f"Review ID: {review.id}",
        title="[bold green]Review Completed and Saved[/bold green]",
        expand=False
    ))
```

---

## Feature 2: Tkinter GUI

### Objective
Create a desktop GUI for managing literature reviews with the ability to:
- View all past reviews with filtering
- Create and run new reviews
- Preview review content and results
- Export reviews to LaTeX

### New Files Structure
```
gui/
├── __init__.py
├── app.py              # Main application class
├── styles.py           # Shared styles and colors
└── views/
    ├── __init__.py
    ├── home.py         # Dashboard/home view
    ├── new_review.py   # Create new review form
    ├── review_detail.py # View review details
    └── review_runner.py # Run review with progress
gui.py                  # Entry point
```

---

### Implementation Details

#### 1. Main Application (`gui/app.py`)

```python
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import os
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from data.database.crud import ReviewDB

from gui.views.home import HomeView
from gui.views.new_review import NewReviewView
from gui.views.review_detail import ReviewDetailView
from gui.views.review_runner import ReviewRunnerView


class LitReviewApp(tk.Tk):
    """Main Literature Review Manager Application."""

    def __init__(self):
        super().__init__()

        # Load environment
        load_dotenv(Path(__file__).parent.parent / ".env", override=False)

        # Window setup
        self.title("Literature Review Manager")
        self.geometry("1200x800")
        self.minsize(800, 600)

        # Database connection
        self.db = ReviewDB()

        # Configure styles
        self._setup_styles()

        # Create main layout
        self._create_layout()

        # Show home view by default
        self.show_home()

    def _setup_styles(self):
        """Configure ttk styles."""
        style = ttk.Style()
        style.theme_use('clam')

        # Custom styles
        style.configure('Sidebar.TFrame', background='#2c3e50')
        style.configure('Sidebar.TButton',
                       padding=10,
                       font=('Helvetica', 11))
        style.configure('Header.TLabel',
                       font=('Helvetica', 18, 'bold'))
        style.configure('Subheader.TLabel',
                       font=('Helvetica', 12))
        style.configure('Status.TLabel',
                       font=('Helvetica', 10))

    def _create_layout(self):
        """Create the main application layout."""
        # Sidebar (left)
        self.sidebar = ttk.Frame(self, style='Sidebar.TFrame', width=200)
        self.sidebar.pack(side='left', fill='y')
        self.sidebar.pack_propagate(False)

        # Sidebar title
        title_label = ttk.Label(
            self.sidebar,
            text="Lit Review",
            font=('Helvetica', 16, 'bold'),
            foreground='white',
            background='#2c3e50'
        )
        title_label.pack(pady=20)

        # Navigation buttons
        self.nav_buttons = {}

        home_btn = ttk.Button(
            self.sidebar,
            text="Home",
            command=self.show_home,
            style='Sidebar.TButton'
        )
        home_btn.pack(fill='x', padx=10, pady=5)
        self.nav_buttons['home'] = home_btn

        new_btn = ttk.Button(
            self.sidebar,
            text="New Review",
            command=self.show_new_review,
            style='Sidebar.TButton'
        )
        new_btn.pack(fill='x', padx=10, pady=5)
        self.nav_buttons['new'] = new_btn

        # Content area (right)
        self.content = ttk.Frame(self)
        self.content.pack(side='right', fill='both', expand=True)

        # Current view reference
        self.current_view = None

    def _clear_content(self):
        """Clear the content area."""
        if self.current_view:
            self.current_view.destroy()
            self.current_view = None

    def show_home(self):
        """Display the home/dashboard view."""
        self._clear_content()
        self.current_view = HomeView(self.content, self)
        self.current_view.pack(fill='both', expand=True)

    def show_new_review(self):
        """Display the new review form."""
        self._clear_content()
        self.current_view = NewReviewView(self.content, self)
        self.current_view.pack(fill='both', expand=True)

    def show_review_detail(self, review_id: str):
        """Display details for a specific review."""
        self._clear_content()
        self.current_view = ReviewDetailView(self.content, self, review_id)
        self.current_view.pack(fill='both', expand=True)

    def run_review(self, review_id: str):
        """Run a review workflow."""
        self._clear_content()
        self.current_view = ReviewRunnerView(self.content, self, review_id)
        self.current_view.pack(fill='both', expand=True)
```

#### 2. Home View (`gui/views/home.py`)

```python
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime


class HomeView(ttk.Frame):
    """Dashboard view showing all reviews."""

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.db = app.db

        self._create_widgets()
        self._load_reviews()

    def _create_widgets(self):
        """Create the home view widgets."""
        # Header
        header_frame = ttk.Frame(self)
        header_frame.pack(fill='x', padx=20, pady=20)

        ttk.Label(
            header_frame,
            text="Literature Reviews",
            style='Header.TLabel'
        ).pack(side='left')

        # Action buttons
        btn_frame = ttk.Frame(header_frame)
        btn_frame.pack(side='right')

        ttk.Button(
            btn_frame,
            text="New Review",
            command=self.app.show_new_review
        ).pack(side='left', padx=5)

        ttk.Button(
            btn_frame,
            text="Refresh",
            command=self._load_reviews
        ).pack(side='left', padx=5)

        # Filter
        filter_frame = ttk.Frame(self)
        filter_frame.pack(fill='x', padx=20, pady=(0, 10))

        ttk.Label(filter_frame, text="Filter:").pack(side='left')

        self.filter_var = tk.StringVar(value="All")
        filter_combo = ttk.Combobox(
            filter_frame,
            textvariable=self.filter_var,
            values=["All", "planning", "writing", "completed", "failed"],
            state='readonly',
            width=15
        )
        filter_combo.pack(side='left', padx=10)
        filter_combo.bind('<<ComboboxSelected>>', lambda e: self._load_reviews())

        # Stats panel
        self.stats_frame = ttk.Frame(self)
        self.stats_frame.pack(fill='x', padx=20, pady=(0, 10))

        # Reviews table
        table_frame = ttk.Frame(self)
        table_frame.pack(fill='both', expand=True, padx=20, pady=10)

        # Columns
        columns = ('topic', 'status', 'created', 'sections', 'papers')
        self.tree = ttk.Treeview(table_frame, columns=columns, show='headings')

        self.tree.heading('topic', text='Topic')
        self.tree.heading('status', text='Status')
        self.tree.heading('created', text='Created')
        self.tree.heading('sections', text='Sections')
        self.tree.heading('papers', text='Papers')

        self.tree.column('topic', width=400)
        self.tree.column('status', width=100)
        self.tree.column('created', width=150)
        self.tree.column('sections', width=80)
        self.tree.column('papers', width=80)

        # Scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # Double-click to view details
        self.tree.bind('<Double-1>', self._on_double_click)

        # Store review IDs mapping
        self.review_ids = {}

    def _load_reviews(self):
        """Load reviews from database."""
        # Clear existing
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.review_ids.clear()

        # Get filter
        status_filter = self.filter_var.get()
        if status_filter == "All":
            status_filter = None

        # Load from DB
        reviews = self.db.list_reviews(status=status_filter, limit=100)

        # Update stats
        self._update_stats(reviews if status_filter is None else self.db.list_reviews(limit=1000))

        # Populate table
        for review in reviews:
            created = review.created_at.strftime("%Y-%m-%d %H:%M") if review.created_at else "N/A"
            item_id = self.tree.insert('', 'end', values=(
                review.topic[:80] + "..." if len(review.topic) > 80 else review.topic,
                review.status,
                created,
                review.total_sections or 0,
                review.total_papers_used or 0
            ))
            self.review_ids[item_id] = review.id

    def _update_stats(self, all_reviews):
        """Update stats panel."""
        # Clear existing stats
        for widget in self.stats_frame.winfo_children():
            widget.destroy()

        total = len(all_reviews)
        completed = sum(1 for r in all_reviews if r.status == 'completed')
        in_progress = sum(1 for r in all_reviews if r.status in ('planning', 'writing'))
        failed = sum(1 for r in all_reviews if r.status == 'failed')

        stats_text = f"Total: {total} | Completed: {completed} | In Progress: {in_progress} | Failed: {failed}"
        ttk.Label(self.stats_frame, text=stats_text, style='Status.TLabel').pack(side='left')

    def _on_double_click(self, event):
        """Handle double-click on review."""
        selection = self.tree.selection()
        if selection:
            item_id = selection[0]
            review_id = self.review_ids.get(item_id)
            if review_id:
                self.app.show_review_detail(review_id)
```

#### 3. New Review Form (`gui/views/new_review.py`)

```python
import tkinter as tk
from tkinter import ttk, messagebox
import os


class NewReviewView(ttk.Frame):
    """Form for creating a new review."""

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.db = app.db

        self._create_widgets()

    def _create_widgets(self):
        """Create form widgets."""
        # Header
        header_frame = ttk.Frame(self)
        header_frame.pack(fill='x', padx=20, pady=20)

        ttk.Label(
            header_frame,
            text="Create New Review",
            style='Header.TLabel'
        ).pack(side='left')

        # Form
        form_frame = ttk.Frame(self)
        form_frame.pack(fill='both', expand=True, padx=40, pady=20)

        # Topic
        ttk.Label(form_frame, text="Research Topic:").grid(row=0, column=0, sticky='w', pady=10)
        self.topic_entry = ttk.Entry(form_frame, width=60)
        self.topic_entry.grid(row=0, column=1, sticky='ew', pady=10, padx=10)

        # Paper recency
        ttk.Label(form_frame, text="Paper Recency:").grid(row=1, column=0, sticky='w', pady=10)
        self.recency_entry = ttk.Entry(form_frame, width=60)
        self.recency_entry.insert(0, "after 2023")
        self.recency_entry.grid(row=1, column=1, sticky='ew', pady=10, padx=10)

        # Model selection
        models = [
            "openai/gpt-4o",
            "openai/gpt-4-turbo",
            "anthropic/claude-3-opus",
            "anthropic/claude-3-sonnet"
        ]

        # Orchestrator model
        ttk.Label(form_frame, text="Orchestrator Model:").grid(row=2, column=0, sticky='w', pady=10)
        self.orchestrator_var = tk.StringVar(value=os.getenv("ORCHESTRATOR_MODEL", "openai/gpt-4o"))
        orchestrator_combo = ttk.Combobox(form_frame, textvariable=self.orchestrator_var, values=models, width=57)
        orchestrator_combo.grid(row=2, column=1, sticky='ew', pady=10, padx=10)

        # Text model
        ttk.Label(form_frame, text="Text Model:").grid(row=3, column=0, sticky='w', pady=10)
        self.text_model_var = tk.StringVar(value=os.getenv("TEXT_MODEL", "openai/gpt-4-turbo"))
        text_combo = ttk.Combobox(form_frame, textvariable=self.text_model_var, values=models, width=57)
        text_combo.grid(row=3, column=1, sticky='ew', pady=10, padx=10)

        # Embedding model
        embedding_models = [
            "qwen/qwen3-embedding-8b",
            "openai/text-embedding-3-small",
            "openai/text-embedding-3-large"
        ]
        ttk.Label(form_frame, text="Embedding Model:").grid(row=4, column=0, sticky='w', pady=10)
        self.embedding_var = tk.StringVar(value=os.getenv("EMBEDDING_MODEL", "qwen/qwen3-embedding-8b"))
        embed_combo = ttk.Combobox(form_frame, textvariable=self.embedding_var, values=embedding_models, width=57)
        embed_combo.grid(row=4, column=1, sticky='ew', pady=10, padx=10)

        form_frame.columnconfigure(1, weight=1)

        # Buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill='x', padx=40, pady=20)

        ttk.Button(
            btn_frame,
            text="Start Review",
            command=self._start_review
        ).pack(side='left', padx=5)

        ttk.Button(
            btn_frame,
            text="Cancel",
            command=self.app.show_home
        ).pack(side='left', padx=5)

    def _start_review(self):
        """Validate and start the review."""
        topic = self.topic_entry.get().strip()
        recency = self.recency_entry.get().strip()

        if not topic:
            messagebox.showerror("Error", "Please enter a research topic.")
            return

        if not recency:
            messagebox.showerror("Error", "Please enter paper recency criteria.")
            return

        # Create review in database
        try:
            review = self.db.create_review(
                topic=topic,
                paper_recency=recency,
                orchestrator_model=self.orchestrator_var.get(),
                text_model=self.text_model_var.get(),
                embedding_model=self.embedding_var.get()
            )

            # Navigate to review runner
            self.app.run_review(review.id)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to create review: {e}")
```

#### 4. Review Detail View (`gui/views/review_detail.py`)

```python
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json

from export.latex_exporter import LatexExporter
from agents.shared.state.planning_components import Plan


class ReviewDetailView(ttk.Frame):
    """View for displaying review details."""

    def __init__(self, parent, app, review_id: str):
        super().__init__(parent)
        self.app = app
        self.db = app.db
        self.review_id = review_id

        self.review = self.db.get_review(review_id)
        if not self.review:
            messagebox.showerror("Error", "Review not found")
            app.show_home()
            return

        self._create_widgets()

    def _create_widgets(self):
        """Create detail view widgets."""
        # Header
        header_frame = ttk.Frame(self)
        header_frame.pack(fill='x', padx=20, pady=20)

        # Back button
        ttk.Button(
            header_frame,
            text="< Back",
            command=self.app.show_home
        ).pack(side='left')

        # Title
        title_frame = ttk.Frame(header_frame)
        title_frame.pack(side='left', padx=20)

        ttk.Label(
            title_frame,
            text=self.review.topic[:60] + "..." if len(self.review.topic) > 60 else self.review.topic,
            style='Header.TLabel'
        ).pack(anchor='w')

        # Status badge
        status_colors = {
            'planning': 'orange',
            'writing': 'blue',
            'completed': 'green',
            'failed': 'red'
        }
        status_label = ttk.Label(
            title_frame,
            text=self.review.status.upper(),
            foreground=status_colors.get(self.review.status, 'gray')
        )
        status_label.pack(anchor='w')

        # Action buttons
        action_frame = ttk.Frame(header_frame)
        action_frame.pack(side='right')

        if self.review.status == 'completed':
            ttk.Button(
                action_frame,
                text="Export LaTeX",
                command=self._export_latex
            ).pack(side='left', padx=5)

        ttk.Button(
            action_frame,
            text="Delete",
            command=self._delete_review
        ).pack(side='left', padx=5)

        # Notebook for tabs
        notebook = ttk.Notebook(self)
        notebook.pack(fill='both', expand=True, padx=20, pady=10)

        # Overview tab
        overview_frame = ttk.Frame(notebook)
        notebook.add(overview_frame, text="Overview")
        self._create_overview_tab(overview_frame)

        # Plan tab
        plan_frame = ttk.Frame(notebook)
        notebook.add(plan_frame, text="Plan")
        self._create_plan_tab(plan_frame)

        # Content tab
        content_frame = ttk.Frame(notebook)
        notebook.add(content_frame, text="Content")
        self._create_content_tab(content_frame)

        # Papers tab
        papers_frame = ttk.Frame(notebook)
        notebook.add(papers_frame, text="Papers")
        self._create_papers_tab(papers_frame)

    def _create_overview_tab(self, parent):
        """Create overview tab content."""
        info_frame = ttk.Frame(parent)
        info_frame.pack(fill='x', padx=20, pady=20)

        info = [
            ("Topic:", self.review.topic),
            ("Paper Recency:", self.review.paper_recency),
            ("Status:", self.review.status),
            ("Created:", self.review.created_at.strftime("%Y-%m-%d %H:%M") if self.review.created_at else "N/A"),
            ("Completed:", self.review.completed_at.strftime("%Y-%m-%d %H:%M") if self.review.completed_at else "N/A"),
            ("Sections:", str(self.review.total_sections or 0)),
            ("Papers Used:", str(self.review.total_papers_used or 0)),
            ("Orchestrator Model:", self.review.orchestrator_model or "N/A"),
            ("Text Model:", self.review.text_model or "N/A"),
            ("Embedding Model:", self.review.embedding_model or "N/A"),
        ]

        for i, (label, value) in enumerate(info):
            ttk.Label(info_frame, text=label, font=('Helvetica', 10, 'bold')).grid(
                row=i, column=0, sticky='w', pady=5
            )
            ttk.Label(info_frame, text=value, wraplength=600).grid(
                row=i, column=1, sticky='w', pady=5, padx=10
            )

    def _create_plan_tab(self, parent):
        """Create plan tab content."""
        plan_obj = self.db.get_plan(self.review_id)

        if not plan_obj:
            ttk.Label(parent, text="No plan available yet.").pack(pady=20)
            return

        # Scrollable text
        text_frame = ttk.Frame(parent)
        text_frame.pack(fill='both', expand=True, padx=10, pady=10)

        text = tk.Text(text_frame, wrap='word', font=('Courier', 10))
        scrollbar = ttk.Scrollbar(text_frame, command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)

        text.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # Format plan content
        content = f"Reasoning:\n{plan_obj.reasoning}\n\n"
        content += "=" * 60 + "\n\n"

        for section in plan_obj.plan:
            content += f"Section {section.number}: {section.title}\n"
            content += f"  {section.outline}\n\n"
            for kp in section.key_points:
                content += f"  - {kp.text}\n"
                for paper in kp.papers:
                    content += f"      [{paper.title} ({paper.year})]\n"
            content += "\n"

        text.insert('1.0', content)
        text.configure(state='disabled')

    def _create_content_tab(self, parent):
        """Create content tab with literature survey."""
        # This would need to load from database or state
        # For now, show placeholder
        ttk.Label(
            parent,
            text="Content preview requires loading survey data from storage.\nImplementation pending."
        ).pack(pady=20)

    def _create_papers_tab(self, parent):
        """Create papers tab."""
        papers = self.db.get_papers_for_review(self.review_id)

        if not papers:
            ttk.Label(parent, text="No papers associated with this review.").pack(pady=20)
            return

        # Treeview for papers
        columns = ('title', 'authors', 'year', 'arxiv_id')
        tree = ttk.Treeview(parent, columns=columns, show='headings')

        tree.heading('title', text='Title')
        tree.heading('authors', text='Authors')
        tree.heading('year', text='Year')
        tree.heading('arxiv_id', text='ArXiv ID')

        tree.column('title', width=400)
        tree.column('authors', width=200)
        tree.column('year', width=60)
        tree.column('arxiv_id', width=100)

        scrollbar = ttk.Scrollbar(parent, orient='vertical', command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)

        tree.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        scrollbar.pack(side='right', fill='y', pady=10)

        for paper in papers:
            authors = json.loads(paper.authors_json) if paper.authors_json else []
            authors_str = ", ".join(authors[:3])
            if len(authors) > 3:
                authors_str += " et al."

            tree.insert('', 'end', values=(
                paper.title[:60] + "..." if len(paper.title) > 60 else paper.title,
                authors_str,
                paper.year or "N/A",
                paper.arxiv_id
            ))

    def _export_latex(self):
        """Export review to LaTeX."""
        # Would need to load literature_survey from storage
        messagebox.showinfo(
            "Export",
            "LaTeX export from GUI requires loading survey data.\nUse CLI for now."
        )

    def _delete_review(self):
        """Delete the review."""
        if messagebox.askyesno("Confirm Delete", "Are you sure you want to delete this review?"):
            try:
                # Would need to add delete method to CRUD
                messagebox.showinfo("Info", "Delete not yet implemented in database CRUD.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete: {e}")
```

#### 5. Review Runner (`gui/views/review_runner.py`)

```python
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import asyncio
import threading
from queue import Queue

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from agents.shared.state.main_state import AgentState
from agents.graph import graph


class ReviewRunnerView(ttk.Frame):
    """View for running a review workflow."""

    def __init__(self, parent, app, review_id: str):
        super().__init__(parent)
        self.app = app
        self.db = app.db
        self.review_id = review_id

        self.review = self.db.get_review(review_id)
        if not self.review:
            messagebox.showerror("Error", "Review not found")
            app.show_home()
            return

        self.message_queue = Queue()
        self.is_running = False

        self._create_widgets()
        self._start_workflow()

    def _create_widgets(self):
        """Create runner widgets."""
        # Header
        header_frame = ttk.Frame(self)
        header_frame.pack(fill='x', padx=20, pady=20)

        ttk.Label(
            header_frame,
            text=f"Running: {self.review.topic[:50]}...",
            style='Header.TLabel'
        ).pack(side='left')

        self.cancel_btn = ttk.Button(
            header_frame,
            text="Cancel",
            command=self._cancel
        )
        self.cancel_btn.pack(side='right')

        # Progress
        progress_frame = ttk.Frame(self)
        progress_frame.pack(fill='x', padx=20, pady=10)

        self.progress = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.progress.pack(fill='x')

        # Status
        self.status_var = tk.StringVar(value="Initializing...")
        ttk.Label(progress_frame, textvariable=self.status_var).pack(pady=10)

        # Log output
        log_frame = ttk.Frame(self)
        log_frame.pack(fill='both', expand=True, padx=20, pady=10)

        self.log_text = tk.Text(log_frame, wrap='word', height=20)
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)

        self.log_text.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

    def _start_workflow(self):
        """Start the workflow in a background thread."""
        self.is_running = True
        self.progress.start()

        # Start workflow thread
        thread = threading.Thread(target=self._run_workflow_thread)
        thread.daemon = True
        thread.start()

        # Start message polling
        self._poll_messages()

    def _run_workflow_thread(self):
        """Run workflow in background thread."""
        try:
            asyncio.run(self._run_workflow_async())
        except Exception as e:
            self.message_queue.put(('error', str(e)))

    async def _run_workflow_async(self):
        """Async workflow execution."""
        init_state = AgentState(
            review_id=self.review_id,
            topic=self.review.topic,
            paper_recency=self.review.paper_recency,
            completed=False,
            messages=[],
            search_queries=[],
            plan=None
        )

        config = RunnableConfig(
            recursion_limit=200,
            configurable={"thread_id": self.review_id}
        )

        current_input = init_state

        while self.is_running:
            self.message_queue.put(('status', "Processing..."))

            result = await graph.ainvoke(current_input, config)

            if "__interrupt__" in result:
                # Human-in-the-loop
                query = result["__interrupt__"][0].value["query"]
                self.message_queue.put(('interrupt', query))

                # Wait for response
                while self.is_running:
                    if hasattr(self, 'interrupt_response'):
                        answer = self.interrupt_response
                        delattr(self, 'interrupt_response')
                        current_input = Command(resume={"data": answer})
                        break
                    await asyncio.sleep(0.1)
                continue

            # Workflow complete
            self.message_queue.put(('complete', result))
            break

    def _poll_messages(self):
        """Poll message queue for updates."""
        while not self.message_queue.empty():
            msg_type, data = self.message_queue.get()

            if msg_type == 'status':
                self.status_var.set(data)
                self._log(data)

            elif msg_type == 'error':
                self.progress.stop()
                self.status_var.set(f"Error: {data}")
                self._log(f"ERROR: {data}")
                messagebox.showerror("Error", data)
                self.is_running = False

            elif msg_type == 'interrupt':
                # Show dialog for human input
                response = simpledialog.askstring("Input Required", data)
                if response:
                    self.interrupt_response = response
                else:
                    self.is_running = False

            elif msg_type == 'complete':
                self.progress.stop()
                self.status_var.set("Complete!")
                self._log("Workflow completed successfully!")
                self.is_running = False

                # Update DB and show completion
                self.db.update_review_status(self.review_id, 'completed')
                messagebox.showinfo("Success", "Review completed!")
                self.app.show_review_detail(self.review_id)

        if self.is_running:
            self.after(100, self._poll_messages)

    def _log(self, message: str):
        """Add message to log."""
        self.log_text.insert('end', f"{message}\n")
        self.log_text.see('end')

    def _cancel(self):
        """Cancel the workflow."""
        if messagebox.askyesno("Confirm", "Cancel the running workflow?"):
            self.is_running = False
            self.progress.stop()
            self.status_var.set("Cancelled")
            self._log("Workflow cancelled by user.")
            self.db.update_review_status(self.review_id, 'failed')
            self.app.show_home()
```

#### 6. Entry Point (`gui.py`)

```python
#!/usr/bin/env python3
"""
Literature Review Manager - GUI Entry Point

Run with: python gui.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from gui.app import LitReviewApp


def main():
    app = LitReviewApp()
    app.mainloop()


if __name__ == "__main__":
    main()
```

---

## Implementation Order

### Phase 1: LaTeX Export (Foundation)
1. Create `export/__init__.py` and `export/latex_exporter.py`
2. Test citation regex conversion
3. Test BibTeX generation
4. Integrate into `main.py`
5. Test end-to-end export

### Phase 2: GUI Core
1. Create `gui/__init__.py` and `gui/app.py`
2. Create `gui/views/__init__.py`
3. Implement `gui/views/home.py`
4. Create `gui.py` entry point
5. Test basic navigation

### Phase 3: GUI Features
1. Implement `gui/views/new_review.py`
2. Implement `gui/views/review_detail.py`
3. Implement `gui/views/review_runner.py`
4. Add export integration to detail view
5. Test full workflow

---

## Dependencies

No new dependencies required - uses Python standard library:
- `tkinter` (standard library)
- `zipfile` (standard library)
- `re` (standard library)

---

## Testing Checklist

### LaTeX Export
- [ ] Citation conversion: `[Author_2023(2301.12345)]` → `\cite{arxiv_2301_12345}`
- [ ] Multiple citations: `[A(id1); B(id2)]` → `\cite{arxiv_id1,arxiv_id2}`
- [ ] BibTeX entry generation with all fields
- [ ] ZIP file creation with main.tex and references.bib
- [ ] LaTeX compiles without errors

### GUI
- [ ] App launches without errors
- [ ] Home view shows reviews from database
- [ ] Filter dropdown works
- [ ] Double-click opens review detail
- [ ] New review form validates input
- [ ] Review runner shows progress
- [ ] Human-in-the-loop dialogs work
- [ ] Export button generates ZIP
