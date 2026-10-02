# -*- coding: utf-8 -*-
"""app.py 结构级冒烟测试（系统测试）。

本机/CI 若未安装 streamlit / matplotlib，无法直接 `streamlit run app.py`。
此测试通过注入一个最小化的 streamlit 桩模块，在真实病例库上端到端执行
app.py 的模块级代码，验证其控制流（侧边栏、病例加载与筛选、会话状态、
主界面、成长档案区）不会因 API 误用或逻辑错误而崩溃。

注意：该测试不发起真实模型调用（chat_input 桩返回 None、button 桩返回 False），
也不依赖 matplotlib（成长档案为空时不会绘制雷达图）。
"""
from __future__ import annotations

import sys
import types

import pytest


class _SessionState(dict):
    """支持属性访问的会话状态容器。"""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key, value):
        self[key] = value


class _Ctx:
    """可空操作并支持 with 语句的伪控件。"""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __call__(self, *args, **kwargs):
        return None

    def __getattr__(self, _name):
        return self


class _Sidebar:
    """最小化的 st.sidebar 桩：各控件返回使主流程可行的默认值。"""

    def title(self, *a, **k):
        return None

    def subheader(self, *a, **k):
        return None

    def text(self, *a, **k):
        return None

    def caption(self, *a, **k):
        return None

    def warning(self, *a, **k):
        return None

    def markdown(self, *a, **k):
        return None

    def write(self, *a, **k):
        return None

    def divider(self, *a, **k):
        return None

    def success(self, *a, **k):
        return None

    def error(self, *a, **k):
        return None

    def info(self, *a, **k):
        return None

    def selectbox(self, label, options, index=0, **k):
        return options[0] if isinstance(options, (list, tuple)) else options

    def radio(self, label, options, horizontal=False, **k):
        return options[0]

    def checkbox(self, label, value=False, **k):
        return False

    def button(self, label, **k):
        return False

    def expander(self, label, expanded=False, **k):
        return _Ctx()


def _make_streamlit_stub():
    st = types.ModuleType("streamlit")
    st.session_state = _SessionState()
    st.sidebar = _Sidebar()

    def noop(*a, **k):
        return None

    def cache_data(*a, **k):
        def deco(f):
            return f
        return deco

    st.set_page_config = noop
    st.title = noop
    st.caption = noop
    st.subheader = noop
    st.write = noop
    st.markdown = noop
    st.error = noop
    st.warning = noop
    st.info = noop
    st.success = noop
    st.dataframe = noop
    st.metric = noop
    st.line_chart = noop
    st.pyplot = noop
    st.bar_chart = noop
    st.download_button = noop
    st.cache_data = cache_data

    st.chat_input = lambda *a, **k: None
    st.button = lambda *a, **k: False
    st.checkbox = lambda *a, **k: False
    def _selectbox_top(label, options, index=0, **k):
        return options[0] if isinstance(options, (list, tuple)) else options

    st.selectbox = _selectbox_top
    st.chat_message = lambda *a, **k: _Ctx()
    st.expander = lambda *a, **k: _Ctx()
    st.spinner = lambda *a, **k: _Ctx()
    st.toast = noop
    st.progress = noop
    st.text_area = noop
    st.write_stream = lambda gen, *a, **k: "".join(gen)

    def columns(spec):
        if isinstance(spec, int):
            return [_Ctx() for _ in range(spec)]
        return [_Ctx() for _ in spec]

    st.columns = columns

    def tabs(spec, *a, **k):
        if isinstance(spec, (list, tuple)):
            return [_Ctx() for _ in spec]
        return [_Ctx() for _ in range(int(spec))]

    st.tabs = tabs

    def stop():
        raise RuntimeError("st.stop() called")

    def rerun():
        raise RuntimeError("st.rerun() called")

    st.stop = stop
    st.rerun = rerun
    return st


@pytest.fixture()
def streamlit_stub(monkeypatch):
    stub = _make_streamlit_stub()
    monkeypatch.setitem(sys.modules, "streamlit", stub)
    # 隔离持久化副作用：冒烟测试不得写入真实 records 目录
    import core.records as records
    monkeypatch.setattr(records, "save_session", lambda state: None)
    monkeypatch.setattr(records, "load_session", lambda: None)
    monkeypatch.setattr(records, "clear_session", lambda: None)
    return stub


def test_app_runs_end_to_end(streamlit_stub):
    import app  # 模块级代码在此执行

    # 会话状态应被初始化
    assert "messages" in app.st.session_state
    assert app.st.session_state.messages == []
    # 主界面应正确读取真实病例库并完成加载（无异常即通过）
    assert app._load_cases() is not None
