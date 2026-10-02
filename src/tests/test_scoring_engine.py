# -*- coding: utf-8 -*-
"""评分引擎单元测试。"""
import json
from types import SimpleNamespace

from core.case_loader import Case
from core.scoring_engine import (
    RUBRIC_COMPLETENESS,
    RUBRIC_PERFORMANCE,
    DimensionDetail,
    ScoringResult,
    build_scoring_prompt,
    llm_score,
    parse_llm_score,
    rule_score,
)


def _dim_score(dims: dict[str, object], region: str) -> float:
    """从 dimension 字典中安全取出该维度的得分（兼容 DimensionDetail / 旧格式）。"""
    v = dims[region]
    if isinstance(v, DimensionDetail):
        return v.score
    return float(v)


def _case(**overrides):
    base = {
        "case_id": "P01",
        "chief_complaint": "腹痛 3 天",
        "scoring_category": "腹部问诊评分",
        "history": {
            "现病史": "3 天前腹痛，餐后加重",
            "既往史": "慢性胃炎",
            "个人史": "吸烟",
            "婚育史": "已绝经",
            "家族史": "无特殊",
        },
    }
    base.update(overrides)
    return Case(**base)


def test_rule_score_range_and_method():
    result = rule_score(_case(), "3 天前腹痛，餐后加重，吸烟")
    assert result.method == "rule"
    assert 0 <= result.total <= 100
    assert set(RUBRIC_COMPLETENESS) <= set(result.dimensions)
    assert set(RUBRIC_PERFORMANCE) <= set(result.dimensions)


def test_rule_score_full_marks_on_complete_coverage():
    case = _case()
    # 覆盖所有病史信息点，应得较高分（各病史维度满分）
    all_facts = " ".join(case.history.values())
    result = rule_score(case, all_facts)
    assert result.total > 90


def test_rule_score_empty_inquiry_low_score():
    result = rule_score(_case(), "")
    # 未问到任何病史信息，病史维度应近似 0 分
    assert _dim_score(result.dimensions, "现病史") == 0.0
    assert _dim_score(result.dimensions, "既往史") == 0.0
    # 空问诊不应白拿过程维度与综合表现分，总分应为 0
    assert _dim_score(result.dimensions, "主诉") == 0.0
    assert result.total == 0.0


def test_scoring_result_roundtrip_json():
    result = rule_score(_case(), "3 天前腹痛")
    data = json.loads(result.to_json())
    assert data["总分"] == round(result.total, 1)
    assert data["评分方式"] == "rule"
    assert "分维度得分" in data


def test_parse_llm_score_with_code_fence():
    text = '```json\n{"总分": 88, "分维度得分": {"主诉": 5}, "优点": ["好"], "不足": [], "改进建议": "继续努力"}\n```'
    result = parse_llm_score(text)
    assert result.total == 88
    assert result.dimensions["主诉"] == 5.0
    assert result.strengths == ["好"]
    assert result.method == "llm"


def test_parse_llm_score_plain_json():
    text = '{"总分": 70, "分维度得分": {"现病史": 12}, "优点": [], "不足": [], "改进建议": ""}'
    result = parse_llm_score(text)
    assert result.total == 70
    assert result.dimensions["现病史"] == 12.0


def test_parse_llm_score_malformed_json_recovers():
    result = parse_llm_score("抱歉，我无法生成 JSON。")
    assert result.total == 0.0
    assert result.method == "llm"
    assert result.weaknesses  # 应有降级提示


def test_parse_llm_score_none_input():
    result = parse_llm_score(None)
    assert result.total == 0.0
    assert result.method == "llm"


def test_parse_llm_score_clamps_total_and_skips_bad_dims():
    text = '{"总分": 150, "分维度得分": {"现病史": "abc", "既往史": 10}, "优点": "不是列表"}'
    result = parse_llm_score(text)
    assert result.total == 100.0          # 总分钳制到上限
    assert result.dimensions["既往史"] == 10.0
    assert "现病史" not in result.dimensions  # 非数值维度被跳过
    assert result.strengths == []         # 非列表被归一为空列表


def _fake_client(fail_on_response_format=True):
    calls = []

    class _Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            if fail_on_response_format and "response_format" in kwargs:
                raise RuntimeError("response_format not supported")
            content = '{"总分": 80, "分维度得分": {"现病史": 10}, "优点": [], "不足": [], "改进建议": ""}'
            message = SimpleNamespace(content=content)
            choice = SimpleNamespace(message=message)
            return SimpleNamespace(choices=[choice])

    class _Chat:
        completions = _Completions()

    return SimpleNamespace(chat=_Chat()), calls


def test_llm_score_falls_back_without_response_format():
    client, calls = _fake_client(fail_on_response_format=True)
    result = llm_score(client, _case(), model="test-model")
    assert result.total == 80.0
    assert len(calls) == 2                      # 第一次失败后降级重试
    assert "response_format" not in calls[1]    # 第二次不再携带 response_format


def test_build_scoring_prompt_contains_dialogue():
    case = _case()
    history = [{"role": "user", "content": "哪里不舒服？"},
               {"role": "assistant", "content": "肚子疼。"}]
    prompt = build_scoring_prompt(case, history)
    assert "学生：哪里不舒服？" in prompt
    assert "SP：肚子疼。" in prompt
    assert case.chief_complaint in prompt


def test_scoring_result_from_dict_roundtrip():
    result = ScoringResult(
        total=88.5,
        dimensions={"现病史": 12.0},
        strengths=["问得不错"],
        weaknesses=["遗漏家族史"],
        suggestions="继续努力",
        method="llm",
    )
    restored = ScoringResult.from_dict(result.to_dict())
    assert restored.total == 88.5
    assert restored.dimensions == {"现病史": 12.0}
    assert restored.strengths == ["问得不错"]
    assert restored.weaknesses == ["遗漏家族史"]
    assert restored.suggestions == "继续努力"
    assert restored.method == "llm"


def test_scoring_result_to_markdown():
    result = ScoringResult(
        total=88.5,
        dimensions={"现病史": 12.0},
        strengths=["问得不错"],
        weaknesses=["遗漏家族史"],
        suggestions="继续努力",
    )
    md = result.to_markdown()
    assert "# 问诊评分报告" in md
    assert "88.5" in md
    assert "| 现病史 | 12.0 |" in md
    assert "## 优点" in md
    assert "## 改进建议" in md


def test_rule_score_chief_complaint_scored_directly():
    # 主诉应独立于病史直接评分：只采到主诉核心症状时，主诉得分 > 0，
    # 而未被问及的现病史仍为 0（不再随病史等比白拿分）。
    case = _case(chief_complaint="腹痛 3 天", history={"现病史": "餐后加重"})
    result = rule_score(case, "腹痛")
    assert _dim_score(result.dimensions, "主诉") > 0.0
    assert _dim_score(result.dimensions, "现病史") == 0.0


def test_rule_score_partial_coverage_gets_partial_credit():
    # 2-gram 模糊匹配：部分提及（未照抄原文）也能拿到部分现病史分数
    case = _case(history={"现病史": "上腹部隐痛伴恶心"})
    result = rule_score(case, "上腹隐痛")
    score = _dim_score(result.dimensions, "现病史")
    assert 0.0 < score < 15.0


def test_rule_score_returns_dimension_details_with_hit_miss():
    # 评分结果应包含每个维度的"已覆盖 / 建议补问"术语明细，便于学生看到扣分原因
    case = _case(history={"现病史": "腹痛持续 2 小时"})
    # 把整段事实复述出来，能完全覆盖
    result = rule_score(case, case.history["现病史"])
    detail = result.dimensions["现病史"]
    assert isinstance(detail, DimensionDetail)
    assert detail.score > 0
    assert detail.max == 15
    # 完全复述：missed 应为空或极少
    assert isinstance(detail.covered, list)
    assert isinstance(detail.missed, list)


def test_rule_score_synonym_recognized():
    # 同义词扩展：学生写"血糖高"，应被识别为覆盖了"糖尿病"事实点
    case = _case(history={"现病史": "糖尿病史5年，血糖控制不佳"})
    result = rule_score(case, "病人血糖高")
    score = _dim_score(result.dimensions, "现病史")
    assert score > 0


def test_parse_llm_score_clamps_dimension_scores():
    text = '{"总分": 80, "分维度得分": {"现病史": 50, "既往史": -3, "主诉": 4}}'
    result = parse_llm_score(text)
    assert result.dimensions["现病史"] == 15.0   # 钳制到现病史满分
    assert result.dimensions["既往史"] == 0.0    # 负分钳制到 0
    assert result.dimensions["主诉"] == 4.0       # 正常值不受影响
