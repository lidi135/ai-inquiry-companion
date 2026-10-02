# -*- coding: utf-8 -*-
"""中文医学口语同义词词典 —— 给规则评分使用。

加载 ``src/core/medical_synonyms.json`` 一次到内存，对外暴露：
    - ``load_synonyms()`` : 重新加载词典（缺文件时返回空 dict）
    - ``expand_term(term)`` : 返回某标准术语的所有同义形态（含原词）
    - ``contains_any(needle_set, text)`` : 判断文本中是否出现任一同义词

设计要点：
    * 文件读取失败不抛错 —— 评分必须能在没有词典的情况下继续运行；
    * 大小写不敏感（统一转小写后比较）；
    * 词典只读，无副作用（不缓存为全局可变状态，调用方按需复用）。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .config import get_logger

logger = get_logger("core.synonyms")

_DICT_FILE = Path(__file__).parent / "medical_synonyms.json"


def load_synonyms() -> dict[str, list[str]]:
    """读取词典文件。返回 ``{标准词: [同义词1, ...]}``；缺文件返回 ``{}``。"""
    if not _DICT_FILE.exists():
        logger.warning("同义词词典缺失：%s", _DICT_FILE)
        return {}
    try:
        raw = json.loads(_DICT_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("同义词词典解析失败：%s", e)
        return {}
    # 过滤掉键名以 _ 开头的元数据字段
    return {
        str(k): [str(x) for x in v]
        for k, v in raw.items()
        if not str(k).startswith("_") and isinstance(v, list)
    }


def expand_term(term: str, synonyms: dict[str, list[str]] | None = None) -> list[str]:
    """返回某术语的所有同义词（含原词），便于在文本里找任一即可。

    Args:
        term:     标准医学术语，如 "糖尿病"。
        synonyms: 词典；缺省时调用 ``load_synonyms`` 现读。

    Returns:
        同义词列表（含 ``term`` 自身）；词典缺失或 ``term`` 不在词典中时仅返回 ``[term]``。
    """
    synonyms = synonyms if synonyms is not None else load_synonyms()
    extras = synonyms.get(term, [])
    return [term] + [s for s in extras if s != term]


# 标点 + 空白，用于把文本切成"连续中英文字"短语再做包含判断
_TOKEN_SPLIT_RE = re.compile(r"[\s，。；、：,.!?？;:'\"“”‘’()（）\[\]【】/\\]+")


def contains_any(norm_text: str, candidates: list[str]) -> bool:
    """判断归一化文本 ``norm_text`` 中是否出现 ``candidates`` 中任一短语。

    实现：把文本切成"词"，再判断任一 candidate 是否为词边界完全等于词或词子串。
    适合口语化问诊文本（短句、含标点）场景，复杂度 O(N * M)。
    """
    if not norm_text or not candidates:
        return False
    lower = norm_text.lower()
    for cand in candidates:
        if not cand:
            continue
        c = cand.lower()
        if c and c in lower:
            return True
    return False
