"""SOAP 笔记评分模块单元测试。"""
from __future__ import annotations

from core.case_loader import Case
from core.soap_scorer import score_soap


def _case():
    return Case(
        case_id="P01",
        chief_complaint="腹痛 3 天",
        scoring_category="腹部问诊评分",
        history={
            "现病史": "3 天前开始上腹部隐痛，餐后加重，伴恶心，无呕吐",
            "既往史": "慢性胃炎 2 年",
            "个人史": "不吸烟不饮酒",
            "婚育史": "已婚已育",
            "家族史": "无特殊",
        },
        diagnosis="慢性胃炎急性发作",
        scoring_points=[],
    )


def test_empty_soap_low_total():
    result = score_soap(_case(), "", "", "", "")
    # 全空总分应较低
    assert result.total < 20.0
    assert result.dimensions["S - 主观资料"].score == 0.0


def test_subjective_matching_full_text_high_score():
    case = _case()
    # 学生 S 字段里把 case 全部主观资料都复述出来
    text = "\n".join([
        case.chief_complaint,
        case.history.get("现病史", ""),
        case.history.get("既往史", ""),
        case.history.get("个人史", ""),
        case.history.get("婚育史", ""),
        case.history.get("家族史", ""),
    ])
    result = score_soap(case, subjective=text, objective="", assessment="", plan="")
    s_detail = result.dimensions["S - 主观资料"]
    assert s_detail.score >= 20.0  # 接近满分
    assert isinstance(s_detail.covered, list)


def test_objective_dimension_baseline():
    case = _case()
    # 学生未填 O 维度
    result = score_soap(case, "腹痛", "", "")
    assert result.dimensions["O - 客观资料"].score == 0.0


def test_objective_long_text_full_score():
    case = _case()
    long_obj = "体格检查：腹软，剑下压痛（+），无反跳痛，肠鸣音 4 次/分。辅助检查：血常规 WBC 9.5×10^9/L，中性 70%。"
    result = score_soap(case, "", objective=long_obj)
    assert result.dimensions["O - 客观资料"].score == 25.0


def test_assessment_matching_diagnosis():
    case = _case()
    result = score_soap(case, "腹痛", assessment=case.diagnosis)
    assert result.dimensions["A - 评估诊断"].score > 0


def test_plan_scales_with_length():
    case = _case()
    short = score_soap(case, "腹痛", plan="做检查")
    long = score_soap(case, "腹痛", plan="完善胃镜检查；奥美拉唑抑酸；建议戒烟限酒；2 周后门诊复诊。")
    assert long.dimensions["P - 处理计划"].score > short.dimensions["P - 处理计划"].score


def test_total_in_range():
    case = _case()
    result = score_soap(
        case,
        subjective=case.chief_complaint + "。" + case.history["现病史"],
        objective="腹软无压痛",
        assessment=case.diagnosis,
        plan="做胃镜+奥美拉唑+两周后复诊",
    )
    assert 0 <= result.total <= 100


def test_markdown_renders():
    case = _case()
    result = score_soap(case, "腹痛", "", "", "")
    md = result.to_markdown()
    assert "SOAP 笔记评分" in md
    assert "S - 主观资料" in md
