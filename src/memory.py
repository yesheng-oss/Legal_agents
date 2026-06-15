import json
import logging
from typing import Optional

from llm import create_llm_provider
from models import CaseMemory

logger = logging.getLogger("legal_agent.memory")

MEMORY_FIELDS = ("facts_summary", "user_goal", "dispute_focus", "confirmed_points", "missing_evidence")

MEMORY_EXTRACTION_PROMPT = """你是一名法律案件记忆提取助手。请根据当前对话和已有记忆，提取或更新以下5个字段。
要求：
1. 只输出合法JSON，不要解释
2. 每个字段值应为简洁中文，控制在120字以内
3. 如果某字段信息不足，保留空字符串""
4. 如果已有记忆中的信息仍然有效，应保留并整合

已有记忆：
{current_memory}

用户新问题：{question}

助手回答：{answer}

请输出JSON格式：
{{
  "facts_summary": "案件事实摘要",
  "user_goal": "用户目标",
  "dispute_focus": "争议焦点",
  "confirmed_points": "已确认结论",
  "missing_evidence": "待补充证据"
}}
"""


class LLMMemoryExtractor:
    def __init__(self, llm_provider=None):
        self._llm = llm_provider

    @property
    def llm(self):
        if self._llm is None:
            self._llm = create_llm_provider()
        return self._llm

    def extract(self, question: str, answer: str, current_memory: dict) -> dict:
        prompt = MEMORY_EXTRACTION_PROMPT.format(
            current_memory=json.dumps(current_memory, ensure_ascii=False, indent=2),
            question=question,
            answer=answer or "",
        )
        try:
            response = self.llm.generate(prompt)
            data = self._parse_json(response)
            if data:
                return {field: str(data.get(field, "")).strip()[:200] for field in MEMORY_FIELDS}
        except Exception as exc:
            logger.warning("LLM memory extraction failed: %s", exc)
        return {}

    @staticmethod
    def _parse_json(text: str) -> Optional[dict]:
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start : end + 1])
            raise


class MemoryService:
    def __init__(self, llm_extractor: Optional[LLMMemoryExtractor] = None):
        self.extractor = llm_extractor or LLMMemoryExtractor()

    def get_or_create(self, session, case_id):
        memory = session.query(CaseMemory).filter(CaseMemory.case_id == case_id).one_or_none()
        if memory is None:
            memory = CaseMemory(case_id=case_id)
            session.add(memory)
            session.flush()
        return memory

    def to_dict(self, memory):
        return {
            "case_id": memory.case_id,
            "facts_summary": memory.facts_summary,
            "user_goal": memory.user_goal,
            "dispute_focus": memory.dispute_focus,
            "confirmed_points": memory.confirmed_points,
            "missing_evidence": memory.missing_evidence,
        }

    def update_from_turn(self, session, case_id, question, answer):
        memory = self.get_or_create(session, case_id)
        current = self.to_dict(memory)

        extracted = self.extractor.extract(question, answer or "", current)
        if extracted:
            memory.facts_summary = extracted["facts_summary"] or memory.facts_summary
            memory.user_goal = extracted["user_goal"] or memory.user_goal
            memory.dispute_focus = extracted["dispute_focus"] or memory.dispute_focus
            memory.confirmed_points = extracted["confirmed_points"] or memory.confirmed_points
            memory.missing_evidence = extracted["missing_evidence"] or memory.missing_evidence
        else:
            self._heuristic_update(memory, question, answer)

        session.flush()
        return memory

    def _heuristic_update(self, memory, question, answer):
        if not memory.facts_summary:
            memory.facts_summary = f"用户咨询：{question[:160]}"
        else:
            memory.facts_summary = self._append_unique(memory.facts_summary, question[:120])

        if not memory.user_goal:
            memory.user_goal = "获得可引用案例和法律风险判断"

        if not memory.dispute_focus:
            memory.dispute_focus = self._infer_focus(question)

        if answer:
            memory.confirmed_points = self._append_unique(memory.confirmed_points, answer[:160])

        if not memory.missing_evidence:
            memory.missing_evidence = "需补充合同、付款凭证、沟通记录、案发金额或损失证明等关键材料。"

    def _infer_focus(self, question):
        if "合同" in question:
            return "合同履行、解除权、违约责任和证据补强"
        if "盗窃" in question:
            return "盗窃行为认定、涉案金额、量刑情节和证据链"
        if "诈骗" in question:
            return "非法占有目的、欺骗行为、损失金额和量刑情节"
        return "案件事实认定、法律适用和风险等级"

    def _append_unique(self, current, addition):
        if not addition or addition in current:
            return current
        if not current:
            return addition
        return f"{current}\n{addition}"
