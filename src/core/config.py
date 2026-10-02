# -*- coding: utf-8 -*-
"""全局配置与多模型注册。

本模块集中管理：
    1. 采样超参数（温度、最大长度等）
    2. 支持的多家大模型 API 配置（DeepSeek / Qwen / GLM / GPT）
    3. 目录路径约定

通过统一的 OpenAI 兼容接口调用不同厂商模型，便于进行"多模型性能对比"实验。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
# 项目根目录（src 的上一级）
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
CASES_DIR: Path = ROOT_DIR / "cases"          # 病例库目录（JSON）
RECORDS_DIR: Path = ROOT_DIR / "records"      # 训练记录目录
SECRETS_FILE: Path = ROOT_DIR / ".streamlit" / "secrets.toml"  # 密钥配置

# ---------------------------------------------------------------------------
# 采样超参数
# ---------------------------------------------------------------------------
# 低温度采样：保证患者角色回复的稳定性与医学一致性，避免随机发散
TEMPERATURE: float = 0.3
# 评分引擎使用更低的温度，确保打分的可复现性
SCORING_TEMPERATURE: float = 0.1

# 默认模型（可通过环境变量或 secrets.toml 覆盖）
DEFAULT_MODEL: str = "deepseek-v4-flash"


@dataclass
class ModelConfig:
    """单个模型的 API 配置。

    Attributes:
        name:        模型显示名（如 "DeepSeek"）。
        model:       实际调用时的模型标识（如 "deepseek-v4-flash"）。
        base_url:    OpenAI 兼容接口地址。
        api_key_env: 存放 API Key 的环境变量名。
        provider:    厂商标识，用于结果聚合展示。
    """
    name: str
    model: str
    base_url: str
    api_key_env: str
    provider: str = ""


# ---------------------------------------------------------------------------
# 多模型注册表（用于 benchmark 实验中的模型对比）
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, ModelConfig] = {
    "deepseek": ModelConfig(
        name="DeepSeek",
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",
        provider="DeepSeek AI",
    ),
    "qwen": ModelConfig(
        name="Qwen（通义千问）",
        model="qwen-max",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_env="DASHSCOPE_API_KEY",
        provider="Alibaba",
    ),
    "glm": ModelConfig(
        name="GLM（智谱清言）",
        model="glm-4-plus",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        api_key_env="ZHIPU_API_KEY",
        provider="Zhipu AI",
    ),
    "gpt": ModelConfig(
        name="GPT",
        model="gpt-4o",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        provider="OpenAI",
    ),
}


def get_model_config(key: str) -> ModelConfig:
    """按注册名获取模型配置。

    Args:
        key: 注册名，如 "deepseek" / "qwen" / "glm" / "gpt"。

    Returns:
        对应的 ModelConfig。

    Raises:
        KeyError: 注册名不存在时抛出。
    """
    if key not in MODEL_REGISTRY:
        raise KeyError(f"未注册的模型：{key}，可选：{list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[key]


def get_logger(name: str | None = None) -> logging.Logger:
    """获取项目统一命名空间下的 logger。

    Args:
        name: 子模块名（如 "core.role_engine"）；缺省返回应用根 logger。

    Returns:
        以 "ai_ic" 为根的 logger 实例，便于统一过滤与日志等级配置。
    """
    return logging.getLogger("ai_ic" + (f".{name}" if name else ""))
