import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def new_id():
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# 法律案例向量文档（pgvector + pg_trgm）
# ---------------------------------------------------------------------------

class LegalDocument(Base):
    __tablename__ = "legal_documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    accusations: Mapped[str] = mapped_column(Text, nullable=False, default="")
    articles: Mapped[str] = mapped_column(Text, nullable=False, default="")
    punishment: Mapped[int] = mapped_column(nullable=False, default=0)
    source_case_id: Mapped[str] = mapped_column(String(80), nullable=False, default="")
    source_chunk_index: Mapped[int] = mapped_column(nullable=False, default=0)
    embedding: Mapped[list[float]] = mapped_column(Vector(512), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.now())


# ---------------------------------------------------------------------------
# 业务模型（案卷、会话、消息、记忆）
# ---------------------------------------------------------------------------

class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    display_name: Mapped[str] = mapped_column(String(120), nullable=False, default="默认用户")
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.now())

    cases: Mapped[list["Case"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Case(Base):
    __tablename__ = "cases"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title: Mapped[str] = mapped_column(String(240), nullable=False)
    case_no: Mapped[str] = mapped_column(String(120), nullable=False, default="")
    case_type: Mapped[str] = mapped_column(String(120), nullable=False, default="法律咨询")
    status: Mapped[str] = mapped_column(String(60), nullable=False, default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())

    user: Mapped["User"] = relationship(back_populates="cases")
    conversations: Mapped[list["Conversation"]] = relationship(back_populates="case", cascade="all, delete-orphan")
    memory: Mapped["CaseMemory"] = relationship(back_populates="case", cascade="all, delete-orphan", uselist=False)


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.id", ondelete="CASCADE"), nullable=False)
    title: Mapped[str] = mapped_column(String(240), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())

    case: Mapped["Case"] = relationship(back_populates="conversations")
    messages: Mapped[list["Message"]] = relationship(back_populates="conversation", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    role: Mapped[str] = mapped_column(String(24), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(String(120), nullable=False, default="")
    references_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.now())

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")


class CaseMemory(Base):
    __tablename__ = "case_memories"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, unique=True)
    facts_summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    user_goal: Mapped[str] = mapped_column(Text, nullable=False, default="")
    dispute_focus: Mapped[str] = mapped_column(Text, nullable=False, default="")
    confirmed_points: Mapped[str] = mapped_column(Text, nullable=False, default="")
    missing_evidence: Mapped[str] = mapped_column(Text, nullable=False, default="")
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())

    case: Mapped["Case"] = relationship(back_populates="memory")


class UserPreference(Base):
    __tablename__ = "user_preferences"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    preference_key: Mapped[str] = mapped_column(String(120), nullable=False)
    preference_value: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())
