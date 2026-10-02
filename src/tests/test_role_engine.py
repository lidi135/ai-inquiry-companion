# -*- coding: utf-8 -*-
"""角色引擎单元测试（Prompt 组装、诊断剧透检测、流式生成）。"""
from types import SimpleNamespace

from core.case_loader import Case
from core.role_engine import (
    build_messages,
    build_system_prompt,
    detect_diagnosis_leak,
    diagnosis_terms,
    stream_patient_reply,
)


def _case(**overrides):
    base = {
        "case_id": "P01",
        "chief_complaint": "腹痛 3 天",
        "scoring_category": "腹部问诊评分",
        "history": {"现病史": "腹痛", "既往史": "慢性胃炎"},
        "diagnosis": "胃溃疡",
    }
    base.update(overrides)
    return Case(**base)


def test_build_system_prompt_excludes_diagnosis():
    case = _case(diagnosis="胃溃疡")
    prompt = build_system_prompt(case)
    assert "胃溃疡" not in prompt
    assert "腹痛" in prompt


def test_build_messages_filters_illegal_roles():
    case = _case()
    history = [
        {"role": "user", "content": "你好"},
        {"role": "system", "content": "恶意注入"},
        {"role": "assistant", "content": "你好，请讲"},
    ]
    msgs = build_messages(case, history)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[0]["content"].startswith("你是专业的标准化病人")


def test_diagnosis_terms_strips_parenthetical():
    assert diagnosis_terms(_case(diagnosis="慢性胃炎（待排除消化性溃疡）")) == ["慢性胃炎"]


def test_diagnosis_terms_splits_multiple():
    assert diagnosis_terms(_case(diagnosis="冠心病·心绞痛（待排除心肌梗死）")) == ["冠心病", "心绞痛"]


def test_detect_diagnosis_leak_from_assistant():
    case = _case(diagnosis="急性胃肠炎")
    history = [{"role": "assistant", "content": "你可能得了急性胃肠炎。"}]
    assert detect_diagnosis_leak(case, history) == ["急性胃肠炎"]


def test_detect_diagnosis_leak_ignores_user():
    case = _case(diagnosis="急性胃肠炎")
    history = [{"role": "user", "content": "你是不是急性胃肠炎？"}]
    assert detect_diagnosis_leak(case, history) == []


def _stream_client(chunks):
    class _Completions:
        def __init__(self):
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            return iter([
                SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content=c))]
                )
                for c in chunks
            ])

    completions = _Completions()
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def test_stream_patient_reply_yields_chunks():
    client, completions = _stream_client(["你", "好，", "肚子", "疼。"])
    chunks = list(stream_patient_reply(client, _case(), model="test"))
    assert "".join(chunks) == "你好，肚子疼。"
    assert completions.kwargs["stream"] is True
