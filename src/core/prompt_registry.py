# -*- coding: utf-8 -*-
"""提示词注册表 —— 把分散在各 .py 里的 system prompt 外提到 ``src/prompts/`` 目录。

设计目标：
    * 不破坏现有行为：缺文件时回退到代码内的默认 prompt，不抛错；
    * 支持版本化：通过环境变量 ``PROMPT_VERSION`` 或函数参数切换版本；
    * 可观测：每次调用记录日志（INFO/WARN），便于 A/B 实验后追溯。
"""
from __future__ import annotations

import os
from pathlib import Path

from .config import get_logger

logger = get_logger("core.prompt_registry")

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
# 路径打印，方便排查 fallback 与启动问题
logger.info("提示词目录：%s（存在=%s）", _PROMPTS_DIR, _PROMPTS_DIR.exists())

# 默认版本：可由环境变量 PROMPT_VERSION 覆盖
_DEFAULT_VERSION = os.getenv("PROMPT_VERSION", "default")


def _resolve(name: str, version: str) -> tuple[Path | None, str]:
    """根据 name/version 解析文件名。约定：
        * ``{name}.txt`` 为 default 版本；
        * ``{name}_{version}.txt`` 为自定义版本。
    """
    base = _PROMPTS_DIR / f"{name}"
    custom = _PROMPTS_DIR / f"{name}_{version}"
    # 优先版本化文件，再 default，再回退
    if version != "default" and custom.with_suffix(".txt").exists():
        return custom.with_suffix(".txt"), version
    if base.with_suffix(".txt").exists():
        return base.with_suffix(".txt"), "default"
    if version != "default" and custom.with_suffix(".txt").exists():
        return custom.with_suffix(".txt"), version
    return None, ""


def _read(name: str, version: str) -> str | None:
    """读取 prompt 文件内容；缺文件返回 None。"""
    path, _ = _resolve(name, version)
    if path is None:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning("读取提示词文件失败：%s（%s）", path, e)
        return None


def _read_schema(version: str) -> str | None:
    """读取 schema 提示（JSON 文件）。"""
    path, _ = _resolve("generation_schema", version)
    if path is None:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning("读取提示词 schema 失败：%s（%s）", path, e)
        return None


def get(name: str, version: str | None = None) -> str:
    """读取指定提示词。

    Args:
        name:    提示词名（不带后缀）：``role_system`` / ``scoring_system`` / ``generation_system``。
        version: 版本标识（默认 ``PROMPT_VERSION`` 环境变量或 ``default``）。
                 不存在的版本会自动回退到 ``default``。

    Returns:
        提示词字符串。
    """
    version = version or _DEFAULT_VERSION
    txt = _read(name, version)
    if txt is not None:
        logger.info("提示词 %s 命中版本 %s", name, version)
        return txt
    # 回退到 default 版本
    if version != "default":
        logger.info("提示词 %s 版本 %s 不存在，回退到 default", name, version)
        txt = _read(name, "default")
        if txt is not None:
            return txt
    logger.warning("提示词 %s 缺失，使用空字符串", name)
    return ""


def get_schema(version: str | None = None) -> str:
    """读取 JSON schema 提示（仅用于 generation 场景）。"""
    version = version or _DEFAULT_VERSION
    txt = _read_schema(version)
    if txt is not None:
        return txt
    if version != "default":
        txt = _read_schema("default")
        if txt is not None:
            return txt
    return ""


def list_versions() -> list[str]:
    """列出当前已注册的提示词版本（按文件名前缀扫描）。

    约定：``{name}.txt`` 视为 default；``{name}_{version}.txt`` 视为自定义版本。
    """
    versions: set[str] = set()
    if not _PROMPTS_DIR.exists():
        return []
    for p in _PROMPTS_DIR.glob("*.txt"):
        stem = p.stem
        # 形如 ``role_system`` 或 ``role_system_v2``
        parts = stem.split("_", 1)
        if len(parts) == 2:
            # 检查是否含非 default 版本字段（_ 之后含 . 或额外下划线视为自定义）
            suffix = parts[1]
            if suffix and suffix != "system":  # 'system' 是 generation_schema_xxx 等共用关键词
                versions.add(suffix)
        versions.add("default")
    return sorted(versions)
