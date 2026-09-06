"""
database.py — SQLAlchemy models + DB session setup.
Supports SQLite (default) and PostgreSQL via DATABASE_URL env var.
"""

import os
import json
from typing import Optional
from datetime import datetime, timezone
from sqlalchemy import (
    create_engine, Column, String, Integer, Text,
    DateTime
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session, Mapped, mapped_column

# ---------------------------------------------------------------------------
# Engine setup
# ---------------------------------------------------------------------------

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/advisory.db")

# SQLite needs check_same_thread=False for use with FastAPI's thread pool
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class UserProfile(Base):
    """
    Structured business profile for each user.
    Updated every 5 chat turns by the background entity-extractor.
    """
    __tablename__ = "user_profiles"

    user_id: Mapped[str] = mapped_column(String(128), primary_key=True, index=True)
    name: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    business_type: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    location: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    turnover: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    employee_count: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    schemes_enrolled: Mapped[Optional[str]] = mapped_column(Text, nullable=True)   # comma-separated
    goals: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    challenges: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    extra_facts_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)   # JSON blob
    turn_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        extra = {}
        if self.extra_facts_json:
            try:
                extra = json.loads(str(self.extra_facts_json))
            except Exception:
                pass
        return {
            "user_id":          self.user_id,
            "name":             self.name,
            "business_type":    self.business_type,
            "location":         self.location,
            "turnover":         self.turnover,
            "employee_count":   self.employee_count,
            "schemes_enrolled": self.schemes_enrolled,
            "goals":            self.goals,
            "challenges":       self.challenges,
            "extra_facts":      extra,
            "turn_count":       self.turn_count,
            "updated_at":       self.updated_at.isoformat() if self.updated_at else None,
        }

    def to_profile_text(self) -> str:
        """Formats the profile as a human-readable string for Gemini system prompts."""
        extra = {}
        if self.extra_facts_json:
            try:
                extra = json.loads(str(self.extra_facts_json))
            except Exception:
                pass

        lines = [
            f"User ID: {self.user_id}",
            f"Name: {self.name or 'Unknown'}",
            f"Business Type: {self.business_type or 'Not specified'}",
            f"Location: {self.location or 'Not specified'}",
            f"Annual Turnover: {self.turnover or 'Not specified'}",
            f"Number of Employees: {self.employee_count or 'Not specified'}",
            f"Government Schemes Enrolled: {self.schemes_enrolled or 'None known'}",
            f"Business Goals: {self.goals or 'Not specified'}",
            f"Key Challenges: {self.challenges or 'Not specified'}",
        ]
        if extra:
            lines.append("Additional Facts:")
            for k, v in extra.items():
                lines.append(f"  - {k}: {v}")
        return "\n".join(lines)


class Message(Base):
    """
    Stores every chat turn for history retrieval and profile updates.
    """
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)   # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[Optional[str]] = mapped_column(String(32), nullable=True, default="en")
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "id":         self.id,
            "user_id":    self.user_id,
            "role":       self.role,
            "content":    self.content,
            "language":   self.language,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def init_db():
    """Create all tables (idempotent)."""
    os.makedirs("./data", exist_ok=True)
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency: yields a DB session and ensures it's closed."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_or_create_profile(db: Session, user_id: str) -> UserProfile:
    """Fetch existing profile or create a blank one."""
    profile = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
    if not profile:
        profile = UserProfile(user_id=user_id)
        db.add(profile)
        db.commit()
        db.refresh(profile)
    return profile


def save_message(db: Session, user_id: str, role: str, content: str, language: str = "en") -> Message:
    """Persist a chat message and return the saved object."""
    msg = Message(user_id=user_id, role=role, content=content, language=language)
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg


def get_recent_messages(db: Session, user_id: str, limit: int = 6) -> list[Message]:
    """Return the N most recent messages for a user, ordered oldest-first."""
    rows = (
        db.query(Message)
        .filter(Message.user_id == user_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
        .all()
    )
    return list(reversed(rows))


def get_all_messages(db: Session, user_id: str) -> list[Message]:
    """Return ALL messages for a user, oldest-first."""
    return (
        db.query(Message)
        .filter(Message.user_id == user_id)
        .order_by(Message.created_at.asc())
        .all()
    )


def get_recent_messages_for_extraction(db: Session, user_id: str, limit: int = 10) -> list[Message]:
    """Return last N messages for profile entity extraction."""
    rows = (
        db.query(Message)
        .filter(Message.user_id == user_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
        .all()
    )
    return list(reversed(rows))
