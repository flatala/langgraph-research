"""Database CRUD operations for literature reviews."""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from typing import Optional, List
import uuid
from datetime import datetime
import json
import logging

from .models import Base, Review, Plan, Section, Subsection, Paper, ReviewPaper, VectorCollection

logger = logging.getLogger(__name__)

class ReviewDB:
    """Database operations for literature reviews."""

    def __init__(self, db_path: str = "data/research.db"):
        """Initialize database connection."""
        self.engine = create_engine(f'sqlite:///{db_path}', echo=False)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()

    # Review operations
    def create_review(self, topic: str, paper_recency: str,
                     orchestrator_model: str, text_model: str,
                     embedding_model: str) -> Review:
        """Create a new review."""
        session = self.get_session()
        try:
            review = Review(
                id=str(uuid.uuid4()),
                topic=topic,
                paper_recency=paper_recency,
                status='planning',
                orchestrator_model=orchestrator_model,
                text_model=text_model,
                embedding_model=embedding_model
            )
            session.add(review)
            session.commit()
            session.refresh(review)
            logger.info(f"Created review: {review.id}")
            return review
        finally:
            session.close()

    def get_review(self, review_id: str) -> Optional[Review]:
        """Get review by ID."""
        session = self.get_session()
        try:
            return session.query(Review).filter(Review.id == review_id).first()
        finally:
            session.close()

    def update_review_status(self, review_id: str, status: str):
        """Update review status."""
        session = self.get_session()
        try:
            review = session.query(Review).filter(Review.id == review_id).first()
            if review:
                review.status = status
                if status == 'completed':
                    review.completed_at = datetime.utcnow()
                session.commit()
                logger.info(f"Updated review {review_id} status to: {status}")
        finally:
            session.close()

    def update_review_metrics(self, review_id: str, total_sections: int = None,
                            total_papers_used: int = None):
        """Update review metrics."""
        session = self.get_session()
        try:
            review = session.query(Review).filter(Review.id == review_id).first()
            if review:
                if total_sections is not None:
                    review.total_sections = total_sections
                if total_papers_used is not None:
                    review.total_papers_used = total_papers_used
                session.commit()
        finally:
            session.close()

    def list_reviews(self, status: Optional[str] = None, limit: int = 100) -> List[Review]:
        """List all reviews, optionally filtered by status."""
        session = self.get_session()
        try:
            query = session.query(Review)
            if status:
                query = query.filter(Review.status == status)
            return query.order_by(Review.created_at.desc()).limit(limit).all()
        finally:
            session.close()

    # Plan operations
    def save_plan(self, review_id: str, plan_obj) -> Plan:
        """Save a research plan."""
        session = self.get_session()
        try:
            plan = Plan(
                id=str(uuid.uuid4()),
                review_id=review_id,
                plan_json=plan_obj.model_dump_json()
            )
            session.add(plan)
            session.commit()
            session.refresh(plan)
            logger.info(f"Saved plan for review {review_id}")
            return plan
        finally:
            session.close()

    def get_plan(self, review_id: str):
        """Get plan for a review (returns Pydantic Plan object)."""
        session = self.get_session()
        try:
            plan = session.query(Plan).filter(Plan.review_id == review_id).first()
            return plan.get_plan_object() if plan else None
        finally:
            session.close()

    # Paper operations
    def get_or_create_paper(self, arxiv_id: str, title: str, authors: List[str],
                           url: str, year: int, summary: str = "") -> Paper:
        """Get existing paper or create new one."""
        session = self.get_session()
        try:
            paper = session.query(Paper).filter(Paper.arxiv_id == arxiv_id).first()

            if paper:
                paper.times_used += 1
                session.commit()
                session.refresh(paper)
                return paper

            paper = Paper(
                id=arxiv_id,
                arxiv_id=arxiv_id,
                title=title,
                authors_json=json.dumps(authors),
                url=url,
                year=year,
                summary=summary,
                times_used=1
            )
            session.add(paper)
            session.commit()
            session.refresh(paper)
            logger.info(f"Added paper to database: {title}")
            return paper
        finally:
            session.close()

    def get_paper(self, paper_id: str) -> Optional[Paper]:
        """Get paper by ID."""
        session = self.get_session()
        try:
            return session.query(Paper).filter(Paper.id == paper_id).first()
        finally:
            session.close()

    def link_paper_to_review(self, review_id: str, paper_id: str,
                            section_index: int, subsection_index: int,
                            citation: str = "", relevance_score: float = 0.0):
        """Link a paper to a specific subsection in a review."""
        session = self.get_session()
        try:
            # Check if already exists
            existing = session.query(ReviewPaper).filter(
                ReviewPaper.review_id == review_id,
                ReviewPaper.paper_id == paper_id,
                ReviewPaper.section_index == section_index,
                ReviewPaper.subsection_index == subsection_index
            ).first()

            if existing:
                return  # Already linked

            review_paper = ReviewPaper(
                review_id=review_id,
                paper_id=paper_id,
                section_index=section_index,
                subsection_index=subsection_index,
                citation=citation,
                relevance_score=relevance_score
            )
            session.add(review_paper)
            session.commit()
        finally:
            session.close()

    def get_papers_for_review(self, review_id: str) -> List[Paper]:
        """Get all papers used in a review."""
        session = self.get_session()
        try:
            return (session.query(Paper)
                   .join(ReviewPaper)
                   .filter(ReviewPaper.review_id == review_id)
                   .distinct()
                   .all())
        finally:
            session.close()

    # Section operations
    def create_section(self, review_id: str, section_index: int, title: str,
                      outline: str = "") -> Section:
        """Create a new section."""
        session = self.get_session()
        try:
            section = Section(
                id=str(uuid.uuid4()),
                review_id=review_id,
                section_index=section_index,
                title=title,
                outline=outline
            )
            session.add(section)
            session.commit()
            session.refresh(section)
            return section
        finally:
            session.close()

    def update_section_content(self, section_id: str, markdown_content: str):
        """Update section markdown content."""
        session = self.get_session()
        try:
            section = session.query(Section).filter(Section.id == section_id).first()
            if section:
                section.markdown_content = markdown_content
                section.completed_at = datetime.utcnow()
                session.commit()
        finally:
            session.close()

    # Subsection operations
    def create_subsection(self, section_id: str, subsection_index: int,
                         title: str, key_point: str = "") -> Subsection:
        """Create a new subsection."""
        session = self.get_session()
        try:
            subsection = Subsection(
                id=str(uuid.uuid4()),
                section_id=section_id,
                subsection_index=subsection_index,
                title=title,
                key_point=key_point
            )
            session.add(subsection)
            session.commit()
            session.refresh(subsection)
            return subsection
        finally:
            session.close()

    def update_subsection_content(self, subsection_id: str, content: str):
        """Update subsection final content."""
        session = self.get_session()
        try:
            subsection = session.query(Subsection).filter(Subsection.id == subsection_id).first()
            if subsection:
                subsection.content = content
                subsection.completed_at = datetime.utcnow()
                session.commit()
        finally:
            session.close()

    # Vector collection operations
    def register_vector_collection(self, review_id: str, paper_id: str,
                                   collection_name: str, embedding_model: str,
                                   chunk_size: int, chunk_overlap: int,
                                   total_chunks: int) -> VectorCollection:
        """Register a vector collection for a paper."""
        session = self.get_session()
        try:
            vc = VectorCollection(
                id=str(uuid.uuid4()),
                review_id=review_id,
                paper_id=paper_id,
                collection_name=collection_name,
                embedding_model=embedding_model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                total_chunks=total_chunks
            )
            session.add(vc)
            session.commit()
            session.refresh(vc)
            logger.info(f"Registered vector collection for paper {paper_id}")
            return vc
        finally:
            session.close()

    def get_vector_collection(self, review_id: str, paper_id: str) -> Optional[VectorCollection]:
        """Check if vector collection exists for a paper in a review."""
        session = self.get_session()
        try:
            return (session.query(VectorCollection)
                   .filter(VectorCollection.review_id == review_id,
                          VectorCollection.paper_id == paper_id)
                   .first())
        finally:
            session.close()

    # Query helpers
    def get_most_used_papers(self, limit: int = 10) -> List[Paper]:
        """Get most frequently used papers across all reviews."""
        session = self.get_session()
        try:
            return (session.query(Paper)
                   .filter(Paper.times_used > 0)
                   .order_by(Paper.times_used.desc())
                   .limit(limit)
                   .all())
        finally:
            session.close()
