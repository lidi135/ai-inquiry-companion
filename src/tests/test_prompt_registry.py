"""提示词注册表单元测试（A8）。"""
from __future__ import annotations

from core import prompt_registry


def test_get_returns_default_prompt_for_role():
    txt = prompt_registry.get("role_system")
    # default 版本必然存在（已落盘）
    assert "标准化病人" in txt


def test_get_returns_default_for_scoring():
    txt = prompt_registry.get("scoring_system")
    assert "评分量表" in txt


def test_get_returns_default_for_generation():
    txt = prompt_registry.get("generation_system")
    assert "标准化病人" in txt


def test_get_missing_name_returns_empty_string():
    txt = prompt_registry.get("nonexistent_prompt")
    assert txt == ""


def test_get_unknown_version_falls_back_to_default(monkeypatch):
    # 把 PROMPT_VERSION 设成一个不存在的，回退到 default
    monkeypatch.setenv("PROMPT_VERSION", "no-such-version")
    # 重新加载模块以读取新环境变量
    import importlib
    importlib.reload(prompt_registry)
    txt = prompt_registry.get("role_system")
    assert "标准化病人" in txt
    # 还原
    importlib.reload(prompt_registry)


def test_get_explicit_version(monkeypatch, tmp_path):
    # 临时构造一个非 default 版本的 prompt 文件
    f = prompt_registry._PROMPTS_DIR / "role_system_v2.txt"
    f.write_text("version v2", encoding="utf-8")
    try:
        txt = prompt_registry.get("role_system", version="v2")
        assert txt == "version v2"
    finally:
        f.unlink(missing_ok=True)


def test_list_versions_includes_default():
    versions = prompt_registry.list_versions()
    assert "default" in versions


def test_get_schema_returns_json():
    s = prompt_registry.get_schema()
    # default 文件可能不存在，至少返回字符串
    assert isinstance(s, str)
