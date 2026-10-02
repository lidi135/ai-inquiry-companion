"""case_generator 模块单测：覆盖解析、去重、缓存命中、非法字段。"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from core.case_generator import (
    build_generation_prompt,
    generate_case,
    parse_generated_case,
)
from core.case_loader import CaseLoadError

# 完整有效生成结果（用于解析测试 + 假模型 mock）
_VALID_PAYLOAD = {
    "case_id": "P99",
    "主诉": "反复胸闷 1 周",
    "诊断": "稳定型心绞痛",
    "科室": "心血管内科",
    "难度": "中",
    "病史": {
        "现病史": "近 1 周活动后出现胸骨后压榨样疼痛，休息 3-5 分钟可缓解",
        "既往史": "高血压病史 5 年",
        "个人史": "吸烟 20 年",
        "婚育史": "已婚已育",
        "家族史": "父亲冠心病史",
    },
    "评分维度": [
        "询问主诉", "询问现病史", "询问既往史", "询问个人史",
        "询问家族史", "询问体格检查", "询问辅助检查", "沟通技巧",
    ],
}


def test_parse_generated_case_ok():
    case = parse_generated_case(_VALID_PAYLOAD)
    assert case.case_id == "P99"
    assert case.department == "心血管内科"
    assert case.difficulty == "中"
    assert case.diagnosis == "稳定型心绞痛"
    assert "现病史" in case.history
    assert len(case.scoring_points) >= 5


def test_parse_generated_case_missing_field():
    bad = dict(_VALID_PAYLOAD)
    del bad["诊断"]
    with pytest.raises(CaseLoadError, match="诊断"):
        parse_generated_case(bad)


def test_parse_generated_case_invalid_difficulty():
    bad = dict(_VALID_PAYLOAD)
    bad["难度"] = "高"
    with pytest.raises(CaseLoadError, match="难度"):
        parse_generated_case(bad)


def test_build_generation_prompt_contains_schema_hint():
    p = build_generation_prompt("消化内科", "易")
    assert "消化内科" in p
    assert "易" in p
    assert "case_id" in p
    assert "评分维度" in p


def test_generate_case_calls_model_and_parses(tmp_path, monkeypatch):
    """模拟一次完整的「调 API → 解析 → 入缓存」链路。"""
    # 用临时 RECORDS_DIR 隔离真实缓存
    monkeypatch.setattr("core.case_generator._CACHE_FILE",
                        tmp_path / "cache.json")

    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = json.dumps(
        _VALID_PAYLOAD, ensure_ascii=False
    )
    fake_client = MagicMock()

    with patch("core.case_generator.chat_completion",
               return_value=mock_resp), \
         patch("core.case_generator.create_client",
               return_value=fake_client), \
         patch("core.case_generator.load_all_cases", return_value={}):
        case = generate_case("心血管内科", "中", use_cache=True)

    assert case.case_id == "P99"
    assert case.department == "心血管内科"
    # 缓存应已写入
    cache_file = tmp_path / "cache.json"
    assert cache_file.exists()
    cached = json.loads(cache_file.read_text(encoding="utf-8"))
    assert any("稳定型心绞痛" in str(v) for v in cached.values())
    # 缓存 key 必须带「科室|难度|」前缀，才能被后续命中逻辑（startswith）读到
    assert all(k.startswith("心血管内科|中|") for k in cached)


def test_generate_case_cache_hit_skips_api(tmp_path, monkeypatch):
    """第二次相同 (科室, 难度) 调用应走缓存，不触达模型。"""
    monkeypatch.setattr("core.case_generator._CACHE_FILE",
                        tmp_path / "cache.json")
    # 先写入一份有效缓存
    key_payload = json.dumps(_VALID_PAYLOAD, ensure_ascii=False)
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "cache.json").write_text(
        json.dumps({"心血管内科|中|": key_payload}, ensure_ascii=False),
        encoding="utf-8",
    )

    def _must_not_call(*args, **kwargs):
        raise AssertionError("chat_completion 不应被调用")

    with patch("core.case_generator.chat_completion",
               side_effect=_must_not_call), \
         patch("core.case_generator.create_client"), \
         patch("core.case_generator.load_all_cases", return_value={}):
        case = generate_case("心血管内科", "中", use_cache=True)

    assert case.case_id == "P99"


def test_generate_case_retries_on_conflict_and_succeeds(tmp_path, monkeypatch):
    """第一次主诉冲突时自动重试，第二次换 id 与主诉后成功。"""
    monkeypatch.setattr("core.case_generator._CACHE_FILE",
                        tmp_path / "cache.json")

    # 第 1 次：与 P77 主诉冲突；第 2 次：换主诉 + id，全部不冲突
    first = dict(_VALID_PAYLOAD)
    second = dict(_VALID_PAYLOAD)
    second["case_id"] = "P99"
    second["主诉"] = "活动后气短 5 天"

    state = {"n": 0}

    def _resp_factory(*args, **kwargs):
        r = MagicMock()
        r.choices = [MagicMock()]
        r.choices[0].message.content = json.dumps(
            second if state["n"] else first, ensure_ascii=False
        )
        state["n"] += 1
        return r

    conflicting = {
        "P77": _make_case(case_id="P77", chief_complaint="反复胸闷 1 周"),
    }

    with patch("core.case_generator.chat_completion",
               side_effect=_resp_factory), \
         patch("core.case_generator.create_client",
               return_value=MagicMock()), \
         patch("core.case_generator.load_all_cases",
               return_value=conflicting):
        case = generate_case("心血管内科", "中", use_cache=False)

    assert case.case_id == "P99"
    assert case.chief_complaint == "活动后气短 5 天"
    assert state["n"] == 2  # 确认确实重试了 1 次


def test_generate_case_gives_up_after_three_conflicts(tmp_path, monkeypatch):
    """连续 3 次冲突时抛友好提示，告诉用户已自动重试 3 次。"""
    monkeypatch.setattr("core.case_generator._CACHE_FILE",
                        tmp_path / "cache.json")
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = json.dumps(
        _VALID_PAYLOAD, ensure_ascii=False
    )
    conflicting = {
        "P77": _make_case(case_id="P77", chief_complaint="反复胸闷 1 周"),
    }
    call_count = {"n": 0}
    def _count_calls(*a, **k):
        call_count["n"] += 1
        return mock_resp

    with patch("core.case_generator.chat_completion",
               side_effect=_count_calls), \
         patch("core.case_generator.create_client",
               return_value=MagicMock()), \
         patch("core.case_generator.load_all_cases",
               return_value=conflicting):
        with pytest.raises(CaseLoadError, match="自动重试 3 次"):
            generate_case("心血管内科", "中", use_cache=False)
    assert call_count["n"] == 3


def _make_case(case_id: str, chief_complaint: str) -> object:
    from core.case_loader import Case
    return Case(case_id=case_id, chief_complaint=chief_complaint,
                scoring_category="问诊评分", department="心血管内科",
                difficulty="中", history={"现病史": "x"}, diagnosis="d",
                scoring_points=["p"])
