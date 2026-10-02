# -*- coding: utf-8 -*-
"""AI 问诊陪练助手 —— 核心算法源代码

本包实现"AI 虚拟标准化病人（VSP）+ 智能评分引擎"的核心算法：
    - config          全局配置与模型注册
    - case_loader     病例库加载与校验
    - role_engine     角色引擎（System Prompt + 病例事实注入 + 低温度采样）
    - scoring_engine  评分引擎（规则评分 + LLM 评分）

项目目录结构：
    src/
      app.py                  Streamlit 主应用（串联各模块）
      core/
        __init__.py
        config.py             配置与多模型注册
        case_loader.py        病例加载
        role_engine.py        角色引擎
        scoring_engine.py     评分引擎
      eval/
        __init__.py
        metrics.py            评估指标
        benchmark.py          多模型评测与零样本迁移
        imbalance.py          不平衡样本处理实验
"""
