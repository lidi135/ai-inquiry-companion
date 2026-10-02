# -*- coding: utf-8 -*-
"""角色引擎 —— AI 虚拟标准化病人（VSP）的核心。

角色引擎负责把"病例事实"和"角色人设"组织成送给大模型的 Prompt，
并通过低温度采样控制回复的稳定性，从而实现一个"不剧透诊断、口语化、
符合患者背景"的虚拟病人。

关键算法逻辑（三重约束）：
    1. 角色 System Prompt：约束模型扮演标准化病人，口语化、不主动诊断。
    2. 病例事实注入：把患者画像与病史作为事实，让回复与病例保持一致。
    3. 低温度采样：temperature=0.2~0.4，抑制随机发散，保证医学一致性。
"""
from __future__ import annotations

import re
import time
from typing import Any, Iterator

from .case_loader import Case, case_to_facts
from .config import DEFAULT_MODEL, SECRETS_FILE, TEMPERATURE, get_logger, get_model_config
from .prompt_registry import get as get_prompt

# 模块级 logger（统一命名空间见 config.get_logger）
logger = get_logger("core.role_engine")

# 角色 System Prompt：约束模型行为，防止剧透诊断
SYSTEM_PROMPT_TEMPLATE = """你是专业的标准化病人（SP）。请严格根据下面的患者信息模拟一位真实患者，与医学生进行问诊对话。

扮演要求：
1. 第一问用开放性问题开始 1—3 句话，不要一次性透露完整病情。
2. 对不清楚的医学细节保持模糊，或回答"我不清楚"，绝不主动给出诊断结论。
3. 当学生提出封闭式提问（如"是不是……"）时，用"我不懂，医生你看着办吧"之类的话回应，不要直接确认或否定诊断。
4. 回答要口语化，符合患者的文化水平与教育背景，不要使用医学术语。
5. 只描述自己的感受与经历，不评价医生，也不主动建议做哪些检查。
6. 你不是医生，不能给任何诊断、检查/用药建议、病情判断或解释（"I" = 患者）。
7. **再次强调**：任何时候都不能说"建议您……"、"应该做……"、"可能是……病"、"初步判断"、"您需要……"、"我考虑是……"这类医生话术。
8. 如果学生询问某个检查/检验的目的或必要性，仅以患者口吻表达感受（"我不懂"、"我没做这个"），不要给出医学理由。

患者信息如下：
{case_facts}
"""


def build_system_prompt(case: Case) -> str:
    """根据病例构建角色 System Prompt。

    优先读取 ``src/prompts/role_system_{version}.txt``；文件缺失或为空时
    回退到模块内 ``SYSTEM_PROMPT_TEMPLATE`` 默认值，保证向下兼容。

    Args:
        case: 病例对象。

    Returns:
        组装完成的 System Prompt 字符串。
    """
    template = get_prompt("role_system").strip() or SYSTEM_PROMPT_TEMPLATE
    return template.format(case_facts=case_to_facts(case))


def build_messages(case: Case, history: list[dict[str, str]] | None = None) -> list[dict[str, str]]:
    """组装完整消息列表（system + 历史对话）。

    Args:
        case:    病例对象。
        history: 历史对话，形如 [{"role": "user", "content": "..."}, ...]。

    Returns:
        可直接传给 OpenAI 兼容接口的 messages 列表。
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": build_system_prompt(case)}
    ]
    for turn in history or []:
        # 只保留 user / assistant 两种角色，防止非法角色导致 API 报错
        if turn.get("role") in ("user", "assistant"):
            messages.append({"role": turn["role"], "content": str(turn["content"])})
    return messages


def _read_api_key(env_var: str) -> str | None:
    """读取 API Key：优先环境变量，其次 secrets.toml。

    空值与占位值（以 "your-" 开头）视为未配置。
    """
    import os

    key = os.getenv(env_var)
    if key and not str(key).startswith("your-"):
        return str(key).strip()
    try:
        if SECRETS_FILE.exists():
            import tomllib  # Python 3.11+ 标准库

            data = tomllib.loads(SECRETS_FILE.read_text(encoding="utf-8"))
            val = data.get(env_var)
            if val and not str(val).startswith("your-"):
                return str(val).strip()
    except Exception:
        # secrets.toml 缺失/损坏或不支持 tomllib（<3.11）时静默降级为环境变量
        pass
    return None


def create_client(model_key: str = "deepseek", api_key: str | None = None, **kwargs: Any):
    """创建 OpenAI 兼容客户端（懒加载 openai 依赖）。

    Args:
        model_key: 模型注册名，见 config.MODEL_REGISTRY。
        api_key:   显式传入的 API Key；缺省时依次从环境变量、secrets.toml 读取。
        **kwargs:  透传给 openai.OpenAI 的额外参数。

    Returns:
        openai.OpenAI 客户端实例。
    """
    try:
        from openai import OpenAI  # 延迟导入，避免非推理环境强制依赖
    except ImportError as e:  # pragma: no cover
        raise ImportError("请先安装依赖：pip install openai") from e

    cfg = get_model_config(model_key)
    key = api_key or _read_api_key(cfg.api_key_env)
    if not key:
        raise ValueError(
            f"未找到 {cfg.name} 的 API Key，请设置环境变量 {cfg.api_key_env}"
            " 或在 .streamlit/secrets.toml 中配置同名变量"
        )
    return OpenAI(api_key=key, base_url=cfg.base_url, **kwargs)


# 可重试的 HTTP 状态码：限流与网关/服务端临时错误
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _is_retryable(exc: Exception) -> bool:
    """判断异常是否属于可恢复错误（限流/服务端抖动/网络超时）。"""
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(exc, "status", None)
    if status is not None:
        try:
            return int(status) in _RETRYABLE_STATUS
        except (TypeError, ValueError):
            pass
    # 无状态码：连接/超时/限流类异常按类型名判断
    name = type(exc).__name__.lower()
    return any(k in name for k in ("timeout", "connection", "ratelimit", "rate_limit"))


def chat_completion(
    client: Any,
    *,
    max_retries: int = 3,
    backoff_base: float = 1.0,
    thinking: bool = False,
    **kwargs: Any,
):
    """带指数退避重试的 chat.completions.create 封装。

    仅对可恢复错误（429 限流、5xx、连接/超时）重试；对业务性错误
    （如 402 余额不足、400 参数错误）不重试，直接抛出，避免无谓等待。

    默认关闭思考模式（thinking=False）：本项目只读取 content，关闭后可
    省 token、降延迟；需要思维链时传 thinking=True 开启。
    """
    if not thinking:
        # DeepSeek 通过 extra_body 的 thinking.type 控制思考模式开关
        extra = dict(kwargs.pop("extra_body", None) or {})
        extra["thinking"] = {"type": "disabled"}
        kwargs["extra_body"] = extra
    attempt = 0
    while True:
        attempt += 1
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            if not _is_retryable(e) or attempt > max_retries:
                raise
            delay = backoff_base * (2 ** (attempt - 1))
            logger.warning(
                "模型请求失败（%s），第 %d 次重试，%.1fs 后继续",
                type(e).__name__, attempt, delay,
            )
            time.sleep(delay)


def generate_patient_reply(
    client: Any,
    case: Case,
    history: list[dict[str, str]] | None = None,
    *,
    model: str | None = None,
    temperature: float = TEMPERATURE,
    max_tokens: int = 512,
) -> str:
    """调用大模型生成患者回复。

    Args:
        client:      OpenAI 兼容客户端。
        case:        病例对象。
        history:     历史对话。
        model:       模型标识；缺省使用 DEFAULT_MODEL。
        temperature: 采样温度，默认 0.3（低温度）。
        max_tokens:  最大生成长度。

    Returns:
        患者回复文本。
    """
    messages = build_messages(case, history)
    resp = chat_completion(
        client,
        model=model or DEFAULT_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    # 取第一个 choice 的 message.content
    return resp.choices[0].message.content or ""


def stream_patient_reply(
    client: Any,
    case: Case,
    history: list[dict[str, str]] | None = None,
    *,
    model: str | None = None,
    temperature: float = TEMPERATURE,
    max_tokens: int = 512,
) -> Iterator[str]:
    """流式生成患者回复（逐段 yield 文本，供打字机效果使用）。

    与 generate_patient_reply 使用相同的 Prompt 与参数，仅以 stream=True 调用，
    按增量 delta 逐段产出，降低首字延迟并改善交互体验。
    """
    messages = build_messages(case, history)
    stream = chat_completion(
        client,
        model=model or DEFAULT_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# 诊断关键词拆分：按括号、间隔号与常见标点切词（用于剧透检测/脱敏）
_DIAGNOSIS_SPLIT_RE = re.compile(r"[（(·、，。；;:：/]+")

# 诊断归一化：匹配前去掉空白与标点、英文转小写，提升对空格/大小写差异的鲁棒性
_DIAGNOSIS_NORMALIZE_RE = re.compile(r"[\s，。、；;:：,.·\-_/（）()]+")


def _normalize(text: str) -> str:
    """去除空白与常见标点并转小写，作为剧透匹配的归一化形式。"""
    return _DIAGNOSIS_NORMALIZE_RE.sub("", text or "").lower()


def diagnosis_terms(case: Case) -> list[str]:
    """提取病例真实诊断中的关键词。

    仅取括号（"（待排除…）"）之前的核心诊断，避免把"待排除"的内容也当作剧透词。
    """
    diag = (case.diagnosis or "").strip()
    if not diag:
        return []
    core = re.split(r"[（(]", diag)[0]
    return [t for t in _DIAGNOSIS_SPLIT_RE.split(core) if len(t) >= 2]


def detect_diagnosis_leak(
    case: Case, history: list[dict[str, str]] | None = None
) -> list[str]:
    """检测对话中虚拟病人是否泄露了真实诊断。

    Args:
        case:    病例对象。
        history: 问诊对话历史。

    Returns:
        泄露的诊断关键词列表（去重保序）；为空表示未检测到泄露。
    """
    terms = diagnosis_terms(case)
    leaked: list[str] = []
    for turn in history or []:
        if turn.get("role") != "assistant":
            continue
        content = _normalize(str(turn.get("content", "")))
        for t in terms:
            if t and _normalize(t) in content:
                leaked.append(t)
    return list(dict.fromkeys(leaked))


# ---------------------------------------------------------------------------
# 角色一致性后置校验（A3）
# ---------------------------------------------------------------------------
# "医生化剧本"特征词：模型若回复中含这些词，提示 SP 越界变成了医生。
_DOCTOR_PHRASES = (
    "建议您", "建议做", "建议查", "建议用", "建议进一步",
    "应该做", "应该查", "应该用", "需要进一步",
    "初步判断", "初步诊断", "初步考虑", "我考虑是", "考虑可能是",
    "可能是", "很可能", "考虑为", "不排除",
    "您可能患", "可能患", "疑似", "倾向诊断", "初步印象",
    "需要做", "应做", "须做", "需用药", "建议服",
    "根据您的描述", "从您的症状看",
)


def _looks_like_doctor(text: str) -> bool:
    """检测 SP 回复是否带"医生化剧本"（剧透/建议/诊断等越界表达）。

    命中任一关键词即返回 True。注意不做上下文分析，单纯关键词触发，
    适用于每轮生成后的轻量后置校验。
    """
    if not text:
        return False
    norm = text  # 不做标点去除，避免"建议："中的冒号被吞掉
    return any(p in norm for p in _DOCTOR_PHRASES)


def _safety_post_check(text: str) -> bool:
    """统一后置安全校验入口：医生化剧本 + 诊断泄露（诊断泄露需另传 case）。

    Args:
        text: SP 回复文本。

    Returns:
        True 表示"通过校验"，False 表示"未通过，需重新生成"。
    """
    return not _looks_like_doctor(text)


def _regenerate_without_doctor_leak(
    client: Any,
    case: Case,
    history: list[dict[str, str]] | None,
    *,
    model: str | None,
    temperature: float,
    max_tokens: int,
    max_retries: int = 2,
) -> str:
    """生成一条 SP 回复，若命中医生化剧本则附加强负面提示重试 ``max_retries`` 次。

    多次仍越界时返回最后一次结果（记录 warning 日志，不阻塞对话）。
    """
    base_history = list(history or [])
    for attempt in range(1, max_retries + 2):  # 总共尝试 max_retries+1 次
        try:
            text = generate_patient_reply(
                client, case, history=base_history,
                model=model, temperature=temperature, max_tokens=max_tokens,
            )
        except Exception:
            raise  # API 错误往外抛
        if _safety_post_check(text):
            return text
        # 命中医生化剧本：往历史里追加一条"自我纠正"提示，引导重生成
        base_history = list(base_history) + [
            {"role": "assistant", "content": text},
            {"role": "user", "content": "（系统提示：请用普通患者口吻重答，不要给任何诊断或建议。）"},
        ]
        logger.warning(
            "SP 回复触发医生化剧本（第 %d 次），自动重试",
            attempt,
        )
    logger.warning("SP 重试 %d 次仍含医生化剧本，使用最后一次结果", max_retries)
    return text


def generate_safe_patient_reply(
    client: Any,
    case: Case,
    history: list[dict[str, str]] | None = None,
    *,
    model: str | None = None,
    temperature: float = TEMPERATURE,
    max_tokens: int = 512,
) -> str:
    """调用大模型生成 SP 回复（带医生化剧本后置校验与自动重试）。

    教学场景下，"SP 主动给诊断/建议"会破坏训练真实性。本函数是对
    :func:`generate_patient_reply` 的加固版：生成后做后置校验，命中则
    重新生成，最多尝试 3 次；最终仍越界则返回最后一次结果（带 warning 日志）。

    流式场景使用 :func:`stream_patient_reply`，重试策略由调用方决定。
    """
    return _regenerate_without_doctor_leak(
        client, case, history,
        model=model, temperature=temperature, max_tokens=max_tokens,
        max_retries=2,
    )
