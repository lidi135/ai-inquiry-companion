# -*- coding: utf-8 -*-
"""AI 生成病例（DeepSeek 半自动）+ 缓存与去重。

供侧边栏「生成新病例」控件调用：
    1. 按 (科室, 难度) 维度向 DeepSeek 申请 1 例结构化病例；
    2. 校验生成结果必填字段，通过即挂到内存中的病例库；
    3. 主诉归一化哈希写入 records/.cache/generated_cases.json，二次刷新
       同一组合时直接复用，避免重复调 API；
    4. 重复 case_id / 与手写病例主诉冲突时抛错。

注意：生成内容仅用于训练陪练，不构成诊疗建议。
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import Any

from .case_loader import (
    VALID_DIFFICULTIES,
    Case,
    CaseLoadError,
    load_all_cases,
    validate_case,
)
from .config import (
    CASES_DIR,
    RECORDS_DIR,
    SCORING_TEMPERATURE,
    get_logger,
    get_model_config,
)
from .prompt_registry import get as get_prompt
from .prompt_registry import get_schema
from .role_engine import chat_completion, create_client

logger = get_logger("core.case_generator")

# 缓存目录：放在 records/ 下，避免污染手写病例库 cases/
_CACHE_DIR = RECORDS_DIR / ".cache"
_CACHE_FILE = _CACHE_DIR / "generated_cases.json"

# 病例必填字段：缺失或空字符串即视为生成失败
_REQUIRED_KEYS = (
    "case_id",
    "主诉",
    "诊断",
    "科室",
    "难度",
    "病史",
    "评分维度",
)


# ---------------------------------------------------------------------------
# Prompt：约束模型输出严格 JSON，便于程序解析
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = (
    "你是一名资深医学教育专家，擅长编写医学生问诊训练用的虚拟标准化病例。"
    "请严格按用户指定的科室与难度生成 1 例中文病例，输出必须是合法 JSON，"
    "不得包含任何解释、Markdown 围栏或前后缀文字。"
)

_JSON_SCHEMA_HINT = """{
  "case_id": "Pxx（4 位数字编号，不能与已有病例重复）",
  "主诉": "一句话症状+时长，例如：腹痛 3 天",
  "诊断": "最终诊断（≤12 字）",
  "科室": "科室名，例如：消化内科",
  "难度": "易 / 中 / 难（三选一）",
  "病史": {
    "现病史": "症状发展、伴随、诊治经过（150-300 字）",
    "既往史": "既往疾病/手术/过敏史（可为空字符串）",
    "个人史": "吸烟/饮酒/职业等（可为空字符串）",
    "婚育史": "婚育/月经史（可为空字符串）",
    "家族史": "家族遗传病史（可为空字符串）"
  },
  "评分维度": ["询问主诉", "询问现病史", "询问既往史", "询问个人史",
               "询问家族史", "询问体格检查", "询问辅助检查",
               "沟通技巧", "逻辑条理"]
}"""


def build_generation_prompt(department: str, difficulty: str) -> str:
    """构造生成 prompt。

    优先读取 ``src/prompts/generation_system_{version}.txt`` 与 ``generation_schema_{version}.json``；
    任一缺失时回退到代码内默认值。

    Args:
        department: 目标科室。
        difficulty: "易" / "中" / "难"。

    Returns:
        用户侧提示词（含 JSON 模板说明）。
    """
    schema = get_schema().strip() or _JSON_SCHEMA_HINT
    return (
        f"请生成 1 例 {department}（难度：{difficulty}）的医学生问诊训练用例。"
        "要求：内容真实可信、避免冷僻诊断、对话时学生通过开放式问诊应能逐步明确诊断。"
        "评分维度覆盖问诊完整性与沟通技巧。"
        "请严格按以下 JSON 结构输出（不要 Markdown 围栏、不要注释）：\n"
        f"{schema}"
    )


# ---------------------------------------------------------------------------
# 解析 + 校验：剥离 Markdown 围栏 / 抽取 JSON / 字段映射到 Case
# ---------------------------------------------------------------------------
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_json(text: str) -> dict[str, Any]:
    """从模型输出中抽取 JSON 对象。

    兼容三种常见输出：
        1. 纯 JSON；
        2. 整体被 ```json ... ``` 包裹；
        3. JSON 前后夹杂解释文字（取首个 { 到最后一个 } 的子串）。
    """
    text = text.strip()
    fenced = _JSON_FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1)
    if not text.startswith("{"):
        first = text.find("{")
        last = text.rfind("}")
        if first == -1 or last == -1 or last <= first:
            raise ValueError("模型输出中找不到 JSON 对象")
        text = text[first : last + 1]
    return json.loads(text)


def _chief_complaint_hash(chief: str, department: str, difficulty: str) -> str:
    """主诉 + 科室 + 难度的归一化哈希，用于缓存去重。"""
    norm = re.sub(r"\s+", "", chief or "").lower()
    key = f"{department}|{difficulty}|{norm}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def parse_generated_case(payload: dict[str, Any]) -> Case:
    """把生成 JSON 映射到 Case，校验必填字段。

    Raises:
        CaseLoadError: 必填字段缺失或难度取值非法。
    """
    for key in _REQUIRED_KEYS:
        if key not in payload:
            raise CaseLoadError(f"生成病例缺少必填字段：{key}")
    difficulty = str(payload["难度"])
    if difficulty not in VALID_DIFFICULTIES:
        raise CaseLoadError(f"生成病例难度非法：{difficulty!r}")
    history = payload["病史"] if isinstance(payload["病史"], dict) else {}
    scoring_points = payload["评分维度"] if isinstance(payload["评分维度"], list) else []
    case = Case(
        case_id=str(payload["case_id"]),
        chief_complaint=str(payload["主诉"]),
        scoring_category="问诊评分",
        department=str(payload["科室"]),
        difficulty=difficulty,
        history={str(k): str(v) for k, v in history.items()},
        diagnosis=str(payload["诊断"]),
        scoring_points=[str(x) for x in scoring_points],
        patient_profile={"生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
    )
    problems = validate_case(case)
    if problems:
        raise CaseLoadError("生成病例校验未通过：" + "；".join(problems))
    return case


# ---------------------------------------------------------------------------
# 缓存层：去重 + 复用
# ---------------------------------------------------------------------------
def _load_cache() -> dict[str, dict[str, Any]]:
    """读取生成缓存；缺失或损坏返回空字典。"""
    if not _CACHE_FILE.exists():
        return {}
    try:
        return json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("生成病例缓存读取失败，将以空缓存继续")
        return {}


def _save_cache(cache: dict[str, dict[str, Any]]) -> None:
    """原子写：先写 .tmp 再替换，避免中途崩溃导致空文件。"""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _CACHE_FILE.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tmp.replace(_CACHE_FILE)


def _conflict_with_existing(new_case: Case, existing: dict[str, Case]) -> str | None:
    """检查新病例是否与已有病例冲突，返回冲突原因或 None。

    检查项：
        1. case_id 重复；
        2. 主诉归一化后完全相同（视为重复病例）。
    """
    if new_case.case_id in existing:
        return f"case_id {new_case.case_id} 已存在于病例库"
    norm = re.sub(r"\s+", "", new_case.chief_complaint).lower()
    for c in existing.values():
        if re.sub(r"\s+", "", c.chief_complaint).lower() == norm:
            return f"主诉与已有病例 {c.case_id} 重复"
    return None


def _retry_prompt_for_collision(
    original_prompt: str, reason: str, used_ids: set[str]
) -> str:
    """冲突重试的追加指令（提示模型换一个不重复的 case_id）。"""
    used_list = "、".join(sorted(used_ids)) or "（无）"
    return (
        f"{original_prompt}\n\n"
        "重要修正：上一轮你提交的 case_id 与已有病例冲突（"
        f"{reason}）。已使用的 case_id 有：{used_list}。"
        "请在保持其他字段不变的前提下，把 case_id 改为一个全新的、未在上述列表中出现过的编号，"
        "且主诉不要与已有病例完全一致。再次严格按 JSON 结构输出（不要 Markdown 围栏、不要注释）。"
    )


# ---------------------------------------------------------------------------
# 公开入口
# ---------------------------------------------------------------------------
def generate_case(
    department: str,
    difficulty: str,
    *,
    model_key: str = "deepseek",
    use_cache: bool = True,
    existing: dict[str, Case] | None = None,
) -> Case:
    """生成 1 例病例；优先复用缓存，缺失则调用 DeepSeek。

    Args:
        department: 目标科室。
        difficulty: "易" / "中" / "难"。
        model_key: 模型注册键（默认 deepseek = 最便宜的 deepseek-v4-flash）。
        use_cache: True 时命中缓存直接返回，不调 API。
        existing: 已加载的病例字典（用于去重）；缺省时从 CASES_DIR 全量加载。

    Returns:
        通过校验的 Case 对象。

    Raises:
        CaseLoadError: 必填字段缺失、难度非法、重复 case_id / 主诉。
        RuntimeError: 模型调用失败、JSON 解析失败等。
    """
    if difficulty not in VALID_DIFFICULTIES:
        raise CaseLoadError(f"非法难度：{difficulty!r}")

    existing = existing if existing is not None else load_all_cases(CASES_DIR)

    # 缓存 key 用 (科室, 难度) 维度组合，先看缓存里有没有匹配该科室难度的旧记录；
    # 若有则直接复用（即便主诉不完全相同，也视为同维度近似例，减少重复调 API）。
    cache = _load_cache() if use_cache else {}
    # 缓存 key 形如 "科室|难度|主诉哈希"，前缀（科室|难度）用于按维度命中
    cache_prefix = f"{department}|{difficulty}|"
    cached_payload: dict[str, Any] | None = None
    if use_cache:
        for k, v in cache.items():
            if k.startswith(cache_prefix):
                cached_payload = v
                logger.info("命中生成缓存：%s/%s", department, difficulty)
                break
        if cached_payload is not None:
            # 缓存的 value 应当是 dict；字符串形式的兼容解析
            if isinstance(cached_payload, str):
                try:
                    cached_payload = json.loads(cached_payload)
                except json.JSONDecodeError:
                    logger.warning("缓存记录已损坏，将重新生成")
                    cached_payload = None
            if cached_payload is not None:
                try:
                    return parse_generated_case(cached_payload)
                except CaseLoadError:
                    logger.warning("缓存记录已损坏，将重新生成")

    cfg = get_model_config(model_key)
    client = create_client(model_key)
    user_prompt = build_generation_prompt(department, difficulty)

    # 最多重试 3 次以避免 case_id / 主诉与已有病例冲突
    payload: dict[str, Any] | None = None
    case: Case | None = None
    last_conflict: str | None = None
    for attempt in range(1, 4):
        prompt_to_send = user_prompt
        if attempt > 1 and last_conflict is not None:
            used_ids = set(existing) | {c.case_id for c in (existing or {}).values()}
            prompt_to_send = _retry_prompt_for_collision(
                user_prompt, last_conflict, used_ids,
            )
        try:
            resp = chat_completion(
                client,
                model=cfg.model,
                temperature=SCORING_TEMPERATURE,
                max_tokens=1500,
                thinking=False,
                messages=[
                {"role": "system", "content": get_prompt("generation_system").strip() or _SYSTEM_PROMPT},
                {"role": "user", "content": prompt_to_send},
            ],
            )
        except Exception as e:
            raise RuntimeError(f"DeepSeek 生成病例失败：{e}") from e

        content = (resp.choices[0].message.content or "").strip()
        try:
            payload = _extract_json(content)
        except (json.JSONDecodeError, ValueError) as e:
            raise RuntimeError(f"解析生成结果失败：{e}\n原始输出：{content[:200]}") from e

        try:
            case = parse_generated_case(payload)
        except CaseLoadError as e:
            raise RuntimeError(f"生成结果字段不合法：{e}") from e

        conflict = _conflict_with_existing(case, existing)
        if conflict is None:
            break
        last_conflict = conflict
        logger.warning(
            "生成病例冲突（尝试 %d/3）：%s", attempt, conflict,
        )
        # 失败的尝试暂不写缓存，避免污染；下次还会重新生成
    else:
        # 三次都冲突 —— 通常是模型硬要复用已用 id，向用户友好提示
        assert case is not None and last_conflict is not None
        raise CaseLoadError(
            f"{last_conflict}。已自动重试 3 次仍冲突，请稍后重试或换一个科室/难度再试。"
        )

    assert case is not None and payload is not None

    if use_cache:
        # 用真实主诉做 key，写入缓存（key 形如 "科室|难度|主诉哈希"）
        real_key = _chief_complaint_hash(case.chief_complaint, department, difficulty)
        cache = _load_cache()
        cache[f"{department}|{difficulty}|{real_key}"] = payload
        _save_cache(cache)

    logger.info("生成新病例 %s：%s / %s", case.case_id, department, difficulty)
    return case


def reset_cache() -> None:
    """清空生成缓存（开发/调试用）。"""
    if _CACHE_FILE.exists():
        _CACHE_FILE.unlink()


__all__ = [
    "build_generation_prompt",
    "generate_case",
    "parse_generated_case",
    "reset_cache",
]
