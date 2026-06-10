from models import CaseMemory


MEMORY_FIELDS = ("facts_summary", "user_goal", "dispute_focus", "confirmed_points", "missing_evidence")


class MemoryService:
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

        session.flush()
        return memory

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
