import json
from datetime import datetime, timedelta

from sqlalchemy import select

from agent import LegalAgent
from db import create_session_factory, session_scope
from memory import MemoryService
from models import Case, Conversation, Message, User
from settings import get_settings


class ConversationService:
    max_context_chars = 1200

    def __init__(self, session_factory=None, agent=None, memory_service=None, history_limit=None):
        self.session_factory = session_factory or create_session_factory()
        self.agent = agent or LegalAgent()
        self.memory_service = memory_service or MemoryService()
        self.history_limit = history_limit or get_settings().conversation_history_limit

    def chat(self, question, conversation_id=None, case_id=None):
        with session_scope(self.session_factory) as session:
            case, conversation = self._resolve_case_and_conversation(session, question, conversation_id, case_id)
            history = self._recent_history(session, conversation.id)
            memory = self.memory_service.get_or_create(session, case.id)
            result = self.agent.chat(question, history=history, memory=self.memory_service.to_dict(memory))

            user_message_time = datetime.now()
            assistant_message_time = user_message_time + timedelta(microseconds=1)
            session.add(Message(conversation_id=conversation.id, role="user", content=question, created_at=user_message_time))
            session.add(
                Message(
                    conversation_id=conversation.id,
                    role="assistant",
                    content=result["answer"],
                    model=result.get("model", ""),
                    references_json=json.dumps(result.get("references", []), ensure_ascii=False),
                    created_at=assistant_message_time,
                )
            )
            updated_memory = self.memory_service.update_from_turn(session, case.id, question, result["answer"])

            return {
                **result,
                "case_id": case.id,
                "conversation_id": conversation.id,
                "memory": self.memory_service.to_dict(updated_memory),
            }

    def create_case(self, title, case_no="", case_type="法律咨询"):
        with session_scope(self.session_factory) as session:
            user = self._get_default_user(session)
            case = Case(user_id=user.id, title=title, case_no=case_no, case_type=case_type)
            session.add(case)
            session.flush()
            self.memory_service.get_or_create(session, case.id)
            return self._case_to_dict(case)

    def list_cases(self):
        with session_scope(self.session_factory) as session:
            cases = session.scalars(select(Case).order_by(Case.updated_at.desc())).all()
            return [self._case_to_dict(case) for case in cases]

    def list_conversations(self, case_id=None):
        with session_scope(self.session_factory) as session:
            stmt = select(Conversation).order_by(Conversation.updated_at.desc())
            if case_id:
                stmt = stmt.where(Conversation.case_id == case_id)
            return [self._conversation_to_dict(item) for item in session.scalars(stmt).all()]

    def get_conversation(self, conversation_id):
        with session_scope(self.session_factory) as session:
            conversation = session.get(Conversation, conversation_id)
            if conversation is None:
                raise KeyError(f"Conversation not found: {conversation_id}")
            messages = session.scalars(
                select(Message).where(Message.conversation_id == conversation_id).order_by(Message.created_at.asc())
            ).all()
            return {
                **self._conversation_to_dict(conversation),
                "messages": [self._message_to_dict(message) for message in messages],
            }

    def get_case_memory(self, case_id):
        with session_scope(self.session_factory) as session:
            memory = self.memory_service.get_or_create(session, case_id)
            return self.memory_service.to_dict(memory)

    def delete_case(self, case_id):
        with session_scope(self.session_factory) as session:
            case = session.get(Case, case_id)
            if case is None:
                return {"deleted": False}
            session.delete(case)
            return {"deleted": True}

    def _resolve_case_and_conversation(self, session, question, conversation_id, case_id):
        if conversation_id:
            conversation = session.get(Conversation, conversation_id)
            if conversation is None:
                raise KeyError(f"Conversation not found: {conversation_id}")
            return conversation.case, conversation

        if case_id:
            case = session.get(Case, case_id)
            if case is None:
                raise KeyError(f"Case not found: {case_id}")
        else:
            user = self._get_default_user(session)
            case = Case(user_id=user.id, title=self._title_from_question(question), case_type="法律咨询")
            session.add(case)
            session.flush()
            self.memory_service.get_or_create(session, case.id)

        conversation = Conversation(case_id=case.id, title=self._title_from_question(question))
        session.add(conversation)
        session.flush()
        return case, conversation

    def _recent_history(self, session, conversation_id):
        messages = session.scalars(
            select(Message).where(Message.conversation_id == conversation_id).order_by(Message.created_at.desc()).limit(self.history_limit * 2)
        ).all()
        messages = self._drop_leading_assistant(list(reversed(messages)))
        return [
            {"role": message.role, "content": self._trim_context(message.content)}
            for message in messages
        ]

    def _trim_context(self, content):
        content = content or ""
        if len(content) <= self.max_context_chars:
            return content
        return f"{content[:self.max_context_chars]}..."

    def _drop_leading_assistant(self, messages):
        while messages and messages[0].role != "user":
            messages.pop(0)
        return messages

    def _get_default_user(self, session):
        user = session.scalars(select(User).limit(1)).first()
        if user is None:
            user = User(display_name="默认用户")
            session.add(user)
            session.flush()
        return user

    def _title_from_question(self, question):
        return (question.strip()[:32] or "新法律咨询")

    def _case_to_dict(self, case):
        return {
            "id": case.id,
            "title": case.title,
            "case_no": case.case_no,
            "case_type": case.case_type,
            "status": case.status,
        }

    def _conversation_to_dict(self, conversation):
        return {
            "id": conversation.id,
            "case_id": conversation.case_id,
            "title": conversation.title,
        }

    def _message_to_dict(self, message):
        return {
            "id": message.id,
            "role": message.role,
            "content": message.content,
            "model": message.model,
            "references": json.loads(message.references_json or "[]"),
        }
