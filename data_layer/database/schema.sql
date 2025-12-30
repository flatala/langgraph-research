-- LangGraph Research Database Schema
-- Tracks literature reviews, plans, papers, and vector embeddings

-- Reviews table: Top-level tracking of each literature review run
CREATE TABLE IF NOT EXISTS reviews (
    id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    paper_recency TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    status TEXT NOT NULL DEFAULT 'planning',  -- 'planning', 'writing', 'completed', 'failed'

    -- Metrics
    total_sections INTEGER DEFAULT 0,
    total_papers_used INTEGER DEFAULT 0,

    -- Model configuration used
    orchestrator_model TEXT,
    text_model TEXT,
    embedding_model TEXT
);

-- Plans table: Store research plans as JSON
CREATE TABLE IF NOT EXISTS plans (
    id TEXT PRIMARY KEY,
    review_id TEXT NOT NULL,
    plan_json TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (review_id) REFERENCES reviews(id) ON DELETE CASCADE
);

-- Sections table: Track each section in a review
CREATE TABLE IF NOT EXISTS sections (
    id TEXT PRIMARY KEY,
    review_id TEXT NOT NULL,
    section_index INTEGER NOT NULL,
    title TEXT NOT NULL,
    outline TEXT,
    markdown_content TEXT,  -- Final section markdown

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,

    FOREIGN KEY (review_id) REFERENCES reviews(id) ON DELETE CASCADE,
    UNIQUE(review_id, section_index)
);

-- Subsections table: Track each subsection (smallest writing unit)
CREATE TABLE IF NOT EXISTS subsections (
    id TEXT PRIMARY KEY,
    section_id TEXT NOT NULL,
    subsection_index INTEGER NOT NULL,
    title TEXT NOT NULL,
    content TEXT,  -- Final content after all revisions
    key_point TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,

    FOREIGN KEY (section_id) REFERENCES sections(id) ON DELETE CASCADE,
    UNIQUE(section_id, subsection_index)
);

-- Papers table: All papers encountered (deduplicated)
CREATE TABLE IF NOT EXISTS papers (
    id TEXT PRIMARY KEY,  -- arxiv_id used as primary key
    arxiv_id TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    authors_json TEXT,  -- JSON array of author names
    url TEXT,
    year INTEGER,
    summary TEXT,

    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    times_used INTEGER DEFAULT 0
);

-- Review Papers junction: Which papers used in which review/section
CREATE TABLE IF NOT EXISTS review_papers (
    review_id TEXT NOT NULL,
    paper_id TEXT NOT NULL,
    section_index INTEGER NOT NULL,
    subsection_index INTEGER NOT NULL,
    citation TEXT,  -- How it was cited in the text
    relevance_score FLOAT,  -- Similarity score from vector search

    PRIMARY KEY (review_id, paper_id, section_index, subsection_index),
    FOREIGN KEY (review_id) REFERENCES reviews(id) ON DELETE CASCADE,
    FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE
);

-- Vector Collections: Track ChromaDB collections metadata
CREATE TABLE IF NOT EXISTS vector_collections (
    id TEXT PRIMARY KEY,
    review_id TEXT NOT NULL,
    paper_id TEXT NOT NULL,
    collection_name TEXT NOT NULL,  -- e.g., "review_{uuid}_paper_{arxiv_id}"
    embedding_model TEXT NOT NULL,
    chunk_size INTEGER NOT NULL,
    chunk_overlap INTEGER NOT NULL,
    total_chunks INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (review_id) REFERENCES reviews(id) ON DELETE CASCADE,
    FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE,
    UNIQUE(review_id, paper_id)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_reviews_status ON reviews(status);
CREATE INDEX IF NOT EXISTS idx_reviews_topic ON reviews(topic);
CREATE INDEX IF NOT EXISTS idx_reviews_created ON reviews(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_sections_review ON sections(review_id, section_index);
CREATE INDEX IF NOT EXISTS idx_subsections_section ON subsections(section_id, subsection_index);
CREATE INDEX IF NOT EXISTS idx_papers_arxiv ON papers(arxiv_id);
CREATE INDEX IF NOT EXISTS idx_papers_times_used ON papers(times_used DESC);
CREATE INDEX IF NOT EXISTS idx_review_papers_review ON review_papers(review_id);
CREATE INDEX IF NOT EXISTS idx_vector_collections_review ON vector_collections(review_id, paper_id);
