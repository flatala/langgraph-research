"""SQLAlchemy ORM models for literature review database."""
from sqlalchemy import Column, String, Integer, Text, Float, DateTime, ForeignKey, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime
import json
from typing import List, Dict, Any

Base = declarative_base()


class Review(Base):
    """Top-level literature review tracking."""
    __tablename__ = 'reviews'

    id = Column(String, primary_key=True)
    topic = Column(String, nullable=False)
    paper_recency = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    status = Column(String, default='planning')

    total_sections = Column(Integer, default=0)
    total_papers_used = Column(Integer, default=0)

    orchestrator_model = Column(String)
    text_model = Column(String)
    embedding_model = Column(String)

    # Relationships
    plans = relationship("Plan", back_populates="review", cascade="all, delete-orphan")
    sections = relationship("Section", back_populates="review", cascade="all, delete-orphan")
    papers = relationship("ReviewPaper", back_populates="review", cascade="all, delete-orphan")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "topic": self.topic,
            "paper_recency": self.paper_recency,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_sections": self.total_sections,
            "total_papers_used": self.total_papers_used,
            "orchestrator_model": self.orchestrator_model,
            "text_model": self.text_model,
            "embedding_model": self.embedding_model,
        }


class Plan(Base):
    """Research plan stored as JSON."""
    __tablename__ = 'plans'

    id = Column(String, primary_key=True)
    review_id = Column(String, ForeignKey('reviews.id'), nullable=False)
    plan_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    review = relationship("Review", back_populates="plans")

    def get_plan_object(self):
        """Deserialize plan JSON to Plan Pydantic model."""
        from agentic_workflow.shared.state.planning_components import Plan as PlanModel
        return PlanModel.model_validate_json(self.plan_json)

    def set_plan_object(self, plan_obj):
        """Serialize Plan Pydantic model to JSON."""
        self.plan_json = plan_obj.model_dump_json()


class Section(Base):
    """Section within a literature review."""
    __tablename__ = 'sections'

    id = Column(String, primary_key=True)
    review_id = Column(String, ForeignKey('reviews.id'), nullable=False)
    section_index = Column(Integer, nullable=False)
    title = Column(String, nullable=False)
    outline = Column(Text)
    markdown_content = Column(Text)

    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)

    review = relationship("Review", back_populates="sections")
    subsections = relationship("Subsection", back_populates="section", cascade="all, delete-orphan")


class Subsection(Base):
    """Subsection within a section (smallest writing unit)."""
    __tablename__ = 'subsections'

    id = Column(String, primary_key=True)
    section_id = Column(String, ForeignKey('sections.id'), nullable=False)
    subsection_index = Column(Integer, nullable=False)
    title = Column(String, nullable=False)
    content = Column(Text)  # Final content after all revisions
    key_point = Column(Text)

    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)

    section = relationship("Section", back_populates="subsections")


class Paper(Base):
    """Academic paper metadata."""
    __tablename__ = 'papers'

    id = Column(String, primary_key=True)  # arxiv_id
    arxiv_id = Column(String, unique=True, nullable=False)
    title = Column(String, nullable=False)
    authors_json = Column(Text)
    url = Column(String)
    year = Column(Integer)
    summary = Column(Text)

    first_seen = Column(DateTime, default=datetime.utcnow)
    times_used = Column(Integer, default=0)

    reviews = relationship("ReviewPaper", back_populates="paper", cascade="all, delete-orphan")
    vector_collections = relationship("VectorCollection", back_populates="paper", cascade="all, delete-orphan")

    @property
    def authors(self) -> List[str]:
        """Get authors as list."""
        return json.loads(self.authors_json) if self.authors_json else []

    @authors.setter
    def authors(self, value: List[str]):
        """Set authors from list."""
        self.authors_json = json.dumps(value)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "authors": self.authors,
            "url": self.url,
            "year": self.year,
            "summary": self.summary,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "times_used": self.times_used,
        }


class ReviewPaper(Base):
    """Junction table: Papers used in reviews."""
    __tablename__ = 'review_papers'

    review_id = Column(String, ForeignKey('reviews.id'), primary_key=True)
    paper_id = Column(String, ForeignKey('papers.id'), primary_key=True)
    section_index = Column(Integer, primary_key=True)
    subsection_index = Column(Integer, primary_key=True)
    citation = Column(Text)
    relevance_score = Column(Float)

    review = relationship("Review", back_populates="papers")
    paper = relationship("Paper", back_populates="reviews")


class VectorCollection(Base):
    """Metadata about ChromaDB vector collections."""
    __tablename__ = 'vector_collections'

    id = Column(String, primary_key=True)
    review_id = Column(String, ForeignKey('reviews.id'), nullable=False)
    paper_id = Column(String, ForeignKey('papers.id'), nullable=False)
    collection_name = Column(String, nullable=False)
    embedding_model = Column(String, nullable=False)
    chunk_size = Column(Integer, nullable=False)
    chunk_overlap = Column(Integer, nullable=False)
    total_chunks = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    review = relationship("Review")
    paper = relationship("Paper", back_populates="vector_collections")
