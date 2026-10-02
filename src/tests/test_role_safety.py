"""角色一致性后置校验单测（A3）。"""
from __future__ import annotations

from core.role_engine import _looks_like_doctor, _safety_post_check


def test_clean_text_passes():
    assert _looks_like_doctor("我最近肚子有点疼，已经三天了") is False
    assert _looks_like_doctor("我晚上睡不太好") is False


def test_suggestion_phrases_trigger():
    assert _looks_like_doctor("建议您去做个胃镜") is True
    assert _looks_like_doctor("建议进一步检查") is True


def test_diagnosis_phrases_trigger():
    assert _looks_like_doctor("初步判断是胃溃疡") is True
    assert _looks_like_doctor("我考虑是胃炎") is True
    assert _looks_like_doctor("不排除阑尾炎的可能") is True


def test_empty_text_safe():
    assert _looks_like_doctor("") is False


def test_safety_post_check_inverts():
    assert _safety_post_check("我肚子疼") is True
    assert _safety_post_check("建议您做胃镜") is False
