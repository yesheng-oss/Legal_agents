import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag import LegalRAG


def test_format_history_uses_chinese_role_labels_and_limits_long_items():
    rag = LegalRAG.__new__(LegalRAG)

    text = rag._format_history(
        [
            {"role": "user", "content": "合同解除风险" * 120},
            {"role": "assistant", "content": "需要结合催告记录"},
        ]
    )

    assert "用户：" in text
    assert "助手：" in text
    assert "user:" not in text
    assert "assistant:" not in text
    assert "..." in text
    assert len(text) < 900
