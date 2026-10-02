# -*- coding: utf-8 -*-
"""pytest 公共配置：将 src 目录加入 sys.path，使测试能 import core / eval 包。"""
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
