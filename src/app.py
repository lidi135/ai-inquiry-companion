# -*- coding: utf-8 -*-
"""AI 问诊陪练助手 —— Streamlit 主应用。

该脚本串联核心算法模块，提供一个可交互的问诊陪练 Web 界面：
    1. 侧边栏按科室/难度筛选病例，并选择模型；
    2. 支持「技能训练」与「考试模拟」两种模式；
    3. 主界面与 AI 虚拟标准化病人（VSP）进行自然语言问诊对话；
    4. 结束问诊后调用评分引擎生成结构化评分报告，并持久化到成长档案；
    5. 规则评分（离线）为默认基线，LLM 评分需配置 API Key。
    6. 主导航：4 个 Tab（问诊训练 / SOAP 笔记 / MCQ 练习 / 教师视图）。
    7. 实时评分反馈：每 3-5 轮给学生提示「已覆盖 / 建议补问」维度。

运行方式：
    streamlit run app.py
"""
from __future__ import annotations

import html
import time
from collections.abc import Mapping

import streamlit as st

from core.case_generator import generate_case as generate_new_case
from core.case_loader import Case, load_all_cases
from core.config import CASES_DIR, MODEL_REGISTRY, TEMPERATURE
from core.records import (
    clear_session,
    load_history,
    load_session,
    save_record,
    save_session,
)
from core.role_engine import (
    create_client,
    detect_diagnosis_leak,
    generate_patient_reply,
    stream_patient_reply,
)
from core.scoring_engine import (
    RUBRIC_COMPLETENESS,
    RUBRIC_PERFORMANCE,
    DimensionDetail,
    ScoringResult,
    llm_score,
    rule_score,
)
from core.soap_scorer import score_soap
from core.teacher import dim_loss_ranking, load_class_overview

# 考试模拟：最多提问轮数（问满自动交卷）
EXAM_MAX_QUESTIONS = 10
# 单条提问最大长度（防超长输入导致 token 成本失控）
MAX_QUESTION_CHARS = 1000


def _esc(value: object) -> str:
    """对注入到 unsafe_allow_html 的动态内容做 HTML 转义，防注入。"""
    return html.escape(str(value))


def _dim_score(dimensions: Mapping[str, object], name: str) -> float:
    """从 dimension 字典中安全取出该维度的得分（兼容 DimensionDetail / 旧格式）。"""
    v: object = dimensions.get(name, 0.0)
    if isinstance(v, DimensionDetail):
        return v.score
    if v is None:
        return 0.0
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _dim_max(dimensions: Mapping[str, object], name: str, fallback: float = 100.0) -> float:
    """从 dimension 字典中安全取出该维度的满分（兼容 DimensionDetail）。"""
    v = dimensions.get(name)
    if isinstance(v, DimensionDetail):
        return v.max or fallback
    return fallback


# 页面基础配置
st.set_page_config(page_title="AI 问诊陪练助手", page_icon="🩺", layout="wide")

# ---------------------------------------------------------------------------
# 全局样式（自定义 CSS，配合 .streamlit/config.toml 主题色）
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --teal: #0E7490;
        --teal-dark: #0A5567;
        --navy: #12324A;
        --cyan: #14B8A6;
        --amber: #F59E0B;
        --bg: #F3F6F9;
        --card: #FFFFFF;
        --line: #E4EBF0;
        --text: #22313A;
        --muted: #5C7280;
    }

    /* 全局背景：柔和渐变 */
    .stApp {
        background: linear-gradient(180deg, #EDF4F7 0%, #F6F9FB 45%, #F3F6F9 100%);
    }
    .block-container { padding-top: 1.4rem; max-width: 1180px; }

    /* 侧边栏 */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #EAF3F5 0%, #E3EEF1 100%);
        border-right: 1px solid #D6E4E8;
    }
    section[data-testid="stSidebar"] h1 {
        color: #0A5567 !important;
        font-weight: 800;
        letter-spacing: .5px;
    }
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #0A5567 !important;
    }

    /* 主导航 Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #FFFFFF;
        border-radius: 12px;
        padding: 6px;
        border: 1px solid #E4EBF0;
        box-shadow: 0 2px 6px rgba(18, 50, 74, .04);
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        padding: 0 18px;
        font-weight: 600;
        color: #5C7280;
        border-radius: 8px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0E7490, #14B8A6) !important;
        color: #FFFFFF !important;
    }

    /* Hero 头图 */
    .hero {
        background: linear-gradient(135deg, #0F7B8E 0%, #16759A 55%, #12324A 100%);
        color: #FFFFFF;
        border-radius: 18px;
        padding: 28px 34px;
        margin-bottom: 16px;
        box-shadow: 0 10px 30px rgba(15, 123, 142, .22);
        position: relative;
        overflow: hidden;
    }
    .hero::after {
        content: "";
        position: absolute;
        right: -60px;
        top: -60px;
        width: 220px;
        height: 220px;
        background: radial-gradient(circle, rgba(255,255,255,.18) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-grid {
        display: grid;
        grid-template-columns: 1.35fr 1fr;
        gap: 26px;
        align-items: stretch;
        position: relative;
    }
    .hero-intro h1 { color: #FFFFFF !important; margin: 0 0 8px 0; font-size: 2.05rem; line-height: 1.15; font-weight: 800; }
    .hero-intro .sub { color: #D6EEF3; font-size: 1rem; margin-bottom: 18px; line-height: 1.65; }
    .hero-mode {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(255,255,255,0.14);
        border: 1px solid rgba(255,255,255,0.28);
        border-radius: 999px;
        padding: 6px 14px;
        font-size: 0.88rem;
        color: #FFFFFF;
    }
    .hero-stats {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
    }
    .hero-stat {
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.22);
        border-radius: 14px;
        padding: 14px 16px;
        backdrop-filter: blur(6px);
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .hero-stat .label { font-size: 0.78rem; color: #D6EEF3; letter-spacing: 1px; text-transform: uppercase; }
    .hero-stat .num { font-size: 1.85rem; font-weight: 800; line-height: 1.1; margin-top: 2px; }
    .hero-stat .unit { font-size: 0.82rem; color: #CFE9EE; margin-top: 2px; }
    .hero-stat .icon { font-size: 1.1rem; margin-bottom: 4px; }
    @media (max-width: 880px) {
        .hero-grid { grid-template-columns: 1fr; }
        .hero-stats { grid-template-columns: repeat(3, 1fr); }
    }

    /* 状态条 */
    .statusbar {
        background: #FFFFFF;
        border: 1px solid #E4EBF0;
        border-left: 4px solid #0E7490;
        border-radius: 12px;
        padding: 12px 18px;
        margin-bottom: 14px;
        font-size: 0.95rem;
        color: #22313A;
        box-shadow: 0 2px 8px rgba(18, 50, 74, .05);
    }
    .statusbar b { color: #0A5567; }
    .statusbar .stb-item { margin-right: 22px; }

    /* 欢迎卡片 */
    .welcome {
        background: #FFFFFF;
        border: 1px solid #E4EBF0;
        border-radius: 16px;
        padding: 24px 28px;
        margin: 10px 0 22px;
        box-shadow: 0 4px 14px rgba(18, 50, 74, .06);
    }
    .welcome h3 { color: #0A5567; margin: 0 0 14px 0; font-size: 1.2rem; }
    .welcome ol { list-style: none; counter-reset: w; margin: 0; padding: 0; }
    .welcome ol li {
        counter-increment: w;
        position: relative;
        padding: 8px 0 8px 40px;
        line-height: 1.7;
        color: #2A3B45;
    }
    .welcome ol li::before {
        content: counter(w);
        position: absolute;
        left: 0;
        top: 8px;
        width: 26px;
        height: 26px;
        background: linear-gradient(135deg, #14B8A6, #0E7490);
        color: #FFFFFF;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.85rem;
        font-weight: 700;
    }

    /* 卡片化指标 */
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E4EBF0;
        border-left: 4px solid #14B8A6;
        border-radius: 12px;
        padding: 14px 18px;
        box-shadow: 0 2px 8px rgba(18, 50, 74, .05);
    }
    [data-testid="stMetricLabel"] { color: #5C7280 !important; font-weight: 600; }
    [data-testid="stMetricValue"] { color: #22313A !important; }

    /* 按钮与标题 */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        border: 1px solid transparent;
        box-shadow: 0 2px 6px rgba(15, 123, 142, .18);
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0E7490, #14B8A6);
        color: #FFFFFF;
    }
    h2, h3, h4 { color: #16323F; }

    /* 聊天气泡 */
    [data-testid="stChatMessage"] {
        background: #FFFFFF;
        border: 1px solid #E4EBF0;
        border-radius: 14px;
        padding: 10px 16px;
        margin-bottom: 8px;
        box-shadow: 0 1px 4px rgba(18, 50, 74, .05);
    }

    /* 小节标题（带强调条） */
    .sec-title {
        font-size: 1.35rem;
        font-weight: 800;
        color: #16323F;
        margin: 22px 0 12px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .sec-title::before {
        content: "";
        width: 5px;
        height: 22px;
        background: linear-gradient(#14B8A6, #0E7490);
        border-radius: 3px;
    }
    .sec-sub { font-weight: 700; color: #0E7490; margin: 16px 0 8px; font-size: 1rem; }

    /* 评分卡片 */
    .scorecard {
        display: flex;
        align-items: center;
        gap: 26px;
        background: #FFFFFF;
        border: 1px solid #E4EBF0;
        border-radius: 16px;
        padding: 22px 28px;
        margin: 12px 0 18px;
        box-shadow: 0 6px 18px rgba(18, 50, 74, .07);
    }
    .score-ring {
        --val: 0;
        --ring: #0E7490;
        width: 118px;
        height: 118px;
        border-radius: 50%;
        position: relative;
        flex: none;
        background: conic-gradient(var(--ring) calc(var(--val) * 1%), #E9F0F3 0);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .score-ring::before {
        content: "";
        position: absolute;
        inset: 11px;
        background: #FFFFFF;
        border-radius: 50%;
    }
    .score-num { font-size: 2.1rem; font-weight: 800; color: #22313A; position: relative; line-height: 1; }
    .score-cap { font-size: .72rem; color: #5C7280; position: relative; margin-top: 2px; }
    .score-meta { flex: 1; }
    .score-title { font-size: 1.15rem; font-weight: 700; color: #0A5567; margin-bottom: 6px; }
    .score-method { color: #5C7280; font-size: .9rem; }

    /* 维度进度条 */
    .dimgrid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px 26px; margin: 10px 0 16px; }
    .dimrow { font-size: .9rem; }
    .dimrow .dim-head { display: flex; justify-content: space-between; margin-bottom: 4px; color: #2A3B45; }
    .dimrow .dim-head .dim-val { font-weight: 700; color: #0A5567; }
    .dim-bar { height: 8px; background: #E9F0F3; border-radius: 999px; overflow: hidden; }
    .dim-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #14B8A6, #0E7490); }

    /* 报告分区小节 */
    .rep-sec { margin-top: 14px; }
    .rep-h { font-weight: 700; color: #0A5567; margin-bottom: 6px; }
    .rep-h.tag-pos { color: #15803D; }
    .rep-h.tag-neg { color: #B45309; }
    .rep-list { color: #2A3B45; margin: 4px 0 0; padding-left: 18px; line-height: 1.7; }

    /* 已覆盖 / 建议补问 标签 */
    .tag-row { display: flex; flex-wrap: wrap; gap: 6px; margin: 4px 0 8px; }
    .tag {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: .8rem;
        border: 1px solid transparent;
    }
    .tag.ok { background: #ECFDF5; color: #15803D; border-color: #A7F3D0; }
    .tag.miss { background: #FFF7ED; color: #B45309; border-color: #FED7AA; }

    /* 实时提示卡 */
    .rt-tip {
        background: linear-gradient(135deg, #ECFDF5 0%, #F0FDFA 100%);
        border: 1px solid #A7F3D0;
        border-left: 4px solid #15803D;
        border-radius: 12px;
        padding: 12px 18px;
        margin: 10px 0 14px;
        font-size: .92rem;
        color: #14532D;
    }
    .rt-tip b { color: #15803D; }
    .rt-tip .rt-tip-miss b { color: #B45309; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 病例加载（缓存，避免每次 rerun 重复读盘）
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_cases() -> dict[str, Case]:
    """加载病例库，返回 {case_id: Case}。"""
    return load_all_cases(CASES_DIR)


# 内存中的临时病例：用户点「生成新病例」后追加，重启 Streamlit 后丢失。
if "_extra_cases" not in st.session_state:
    st.session_state._extra_cases = {}

# 上一次生成结果（用于在侧边栏展示成功/失败信息）
if "last_generated" not in st.session_state:
    st.session_state.last_generated = None


def _build_case_options(cases: dict[str, Case]) -> dict[str, str]:
    """构造侧边栏下拉选项：显示名 -> case_id。"""
    return {c.display_name: cid for cid, c in cases.items()}


def _mode_label(mode: str) -> str:
    """把模式名格式化为一致标签。"""
    return {"技能训练": "技能训练", "考试模拟": "考试模拟"}.get(mode, mode)


# ---------------------------------------------------------------------------
# 初始化会话状态
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "report" not in st.session_state:
    st.session_state.report = None

if "exam_start" not in st.session_state:
    st.session_state.exam_start = None

# 已弹过的实时评分提示（避免同一维度重复刷提示）
if "real_time_tips_shown" not in st.session_state:
    st.session_state.real_time_tips_shown = set()


# ---------------------------------------------------------------------------
# 侧边栏
# ---------------------------------------------------------------------------
st.sidebar.title("🩺 AI 问诊陪练助手")

st.sidebar.markdown('<div style="font-size:.85rem;color:#5C7280;margin-bottom:10px;">面向医学生的虚拟标准化病人（VSP）训练系统</div>', unsafe_allow_html=True)

cases = dict(_load_cases())
# 合并手写病例 + 用户在本次会话中临时生成的病例（复制副本，避免污染 cache_data 缓存）
cases.update(st.session_state._extra_cases)
if not cases:
    st.sidebar.warning("未在 cases/ 目录下找到病例 JSON，请先准备病例库。")
    st.stop()

# --- 训练模式分组 ---
st.sidebar.markdown("### 🎯 训练模式")
mode = st.sidebar.radio("选择模式", ["技能训练", "考试模拟"], horizontal=True, label_visibility="collapsed")

# --- 病例选择分组 ---
st.sidebar.markdown("### 📋 选择病例")
departments = sorted({c.department for c in cases.values() if c.department})
difficulties = ["易", "中", "难"]
dept_filter = st.sidebar.selectbox("科室筛选", ["全部"] + departments, key="_dept_filter")
diff_filter = st.sidebar.selectbox("难度筛选", ["全部"] + difficulties, key="_diff_filter")

filtered = {
    cid: c
    for cid, c in cases.items()
    if (dept_filter == "全部" or c.department == dept_filter)
    and (diff_filter == "全部" or c.difficulty == diff_filter)
}
if not filtered:
    st.sidebar.warning("没有符合筛选条件的病例，请调整筛选条件。")
    st.stop()

options = _build_case_options(filtered)
display_name = st.sidebar.selectbox("具体病例", list(options), key="_case_select")
case = filtered[options[display_name]]

# 恢复上次会话（仅当本轮尚无对话、且会话属于当前病例时，支持刷新/重进续答）
if not st.session_state.messages:
    saved = load_session()
    if saved and saved.get("case_id") == case.case_id:
        st.session_state.messages = list(saved.get("messages") or [])
        if saved.get("report"):
            st.session_state.report = ScoringResult.from_dict(saved["report"])
        if saved.get("exam_start") is not None:
            st.session_state.exam_start = saved["exam_start"]

# --- 患者基本信息 ---
st.sidebar.markdown("### 👤 患者信息")
profile = case.patient_profile
if profile:
    for k, v in profile.items():
        st.sidebar.text(f"{k}：{v}")
st.sidebar.caption(f"主诉：{case.chief_complaint}")
if case.department:
    st.sidebar.caption(f"科室：{case.department}　难度：{case.difficulty or '-'}")

# --- 模型与评分方式 ---
st.sidebar.markdown("### 🤖 模型与评分")
model_key = st.sidebar.selectbox("选择模型", list(MODEL_REGISTRY), index=0, key="_model_key")
use_llm_score = st.sidebar.checkbox("使用 LLM 智能评分（需配置 API Key）", value=False, key="_use_llm_score")

# 清空对话按钮
if st.sidebar.button("🔄 重新开始", use_container_width=True):
    st.session_state.messages = []
    st.session_state.report = None
    st.session_state.exam_start = None
    st.session_state.real_time_tips_shown = set()
    clear_session()
    st.rerun()

# ---------------------------------------------------------------------------
# AI 生成新病例（调用 DeepSeek，挂到本次会话内存中的病例库）
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### ✨ AI 助手")
with st.sidebar.expander("AI 生成新病例（DeepSeek）", expanded=False):
    gen_dept = st.selectbox(
        "目标科室", departments, key="_gen_dept",
    )
    gen_diff = st.selectbox(
        "难度", difficulties, index=1, key="_gen_diff",
    )
    if st.button("🚀 生成 1 例", key="_gen_btn"):
        try:
            with st.spinner("DeepSeek 正在生成病例……"):
                new_case = generate_new_case(
                    department=gen_dept,
                    difficulty=gen_diff,
                    model_key=model_key,
                    use_cache=True,
                    existing=cases,
                )
            st.session_state._extra_cases[new_case.case_id] = new_case
            st.session_state.last_generated = (
                f"已生成 {new_case.case_id} · {new_case.chief_complaint}"
            )
            st.success(st.session_state.last_generated)
            st.rerun()
        except Exception as e:
            st.error(f"生成失败：{e}")

if st.session_state.last_generated:
    st.sidebar.caption(f"📌 {st.session_state.last_generated}")


# ---------------------------------------------------------------------------
# 评分与记录
# ---------------------------------------------------------------------------
def _generate_report(
    case: Case,
    messages: list[dict[str, str]],
    use_llm_score: bool,
    model_key: str,
    mode: str,
) -> ScoringResult:
    """生成评分报告并持久化到成长档案。"""
    inquiry_text = "\n".join(
        m["content"] for m in messages if m["role"] == "user"
    )
    if use_llm_score:
        try:
            client = create_client(model_key)
            with st.spinner("正在生成智能评分……"):
                report = llm_score(
                    client, case, messages, model=MODEL_REGISTRY[model_key].model
                )
        except Exception as e:  # API 未配置等异常，回退到规则评分
            st.error(f"LLM 评分失败：{e}，已回退到规则评分。")
            report = rule_score(case, inquiry_text)
    else:
        report = rule_score(case, inquiry_text)

    # 诊断泄露检测：若虚拟病人越界剧透，在报告中提示（不透露具体诊断词）
    if detect_diagnosis_leak(case, messages):
        report.weaknesses.append("系统检测到虚拟病人在对话中可能存在越界回答，已记录。")

    try:
        save_record(report, case, mode=_mode_label(mode))
    except Exception:
        # 记录持久化失败不应阻断评分展示
        pass
    return report


def _persist() -> None:
    """把当前会话（对话、报告、考试计时）落盘，支持刷新/重进续答。"""
    save_session({
        "case_id": case.case_id,
        "messages": st.session_state.messages,
        "report": st.session_state.report.to_dict() if st.session_state.report else None,
        "exam_start": st.session_state.exam_start,
    })


def _conversation_markdown(case: Case, messages: list[dict[str, str]]) -> str:
    """把整段问诊对话渲染为 Markdown，用于成稿导出。"""
    user_turns = sum(1 for m in messages if m["role"] == "user")
    lines = [
        "# 问诊对话记录",
        "",
        f"- 病例：{case.display_name}",
        f"- 科室：{case.department or '未知'}　难度：{case.difficulty or '未知'}",
        f"- 提问轮数：{user_turns}",
        "",
        "---",
        "",
    ]
    for m in messages:
        who = "**🩺 医学生**" if m["role"] == "user" else "**🙍 患者**"
        lines.append(f"{who}：")
        lines.append(str(m["content"]))
        lines.append("")
    return "\n".join(lines)


def _real_time_tip_html(report: ScoringResult) -> str:
    """生成实时评分提示卡（已覆盖 / 建议补问）。"""
    covered = []
    missed = []
    for name, detail in report.dimensions.items():
        if isinstance(detail, DimensionDetail):
            covered.extend(detail.covered)
            missed.extend(detail.missed)
    if not covered and not missed:
        return ""
    cov_chips = "".join(f'<span class="tag ok">✓ {_esc(c)}</span>' for c in covered[:8])
    miss_chips = "".join(f'<span class="tag miss">✗ {_esc(m)}</span>' for m in missed[:8])
    return (
        '<div class="rt-tip">'
        f'💡 <b>实时评估（基于已采集问诊）</b><br>'
        f'已覆盖：{cov_chips or "—"}<br>'
        f'<span class="rt-tip-miss">建议补问：{miss_chips or "—"}</span>'
        '</div>'
    )


# ---------------------------------------------------------------------------
# 主导航：4 个 Tab
# ---------------------------------------------------------------------------
tab_train, tab_test, tab_teacher = st.tabs([
    "🩺 问诊训练",
    "📝 SOAP 笔记",
    "👨‍🏫 教师视图",
])

# ---------------------------------------------------------------------------
# Tab 1: 问诊训练
# ---------------------------------------------------------------------------
with tab_train:
    is_exam = mode == "考试模拟"
    user_turns = sum(1 for m in st.session_state.messages if m["role"] == "user")
    exam_finished = is_exam and user_turns >= EXAM_MAX_QUESTIONS

    # Hero 头图 + 关键统计
    _dept_count = len(departments)
    _case_count = len(cases)
    _model_count = len(MODEL_REGISTRY)
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-grid">
            <div class="hero-intro">
              <h1>🩺 AI 问诊陪练助手</h1>
              <div class="sub">面向医学生的虚拟标准化病人（VSP）问诊训练<br>训练 — 评分 — 反馈 — 成长</div>
              <span class="hero-mode">🎯 支持：技能训练 · 考试模拟</span>
            </div>
            <div class="hero-stats">
              <div class="hero-stat">
                <div class="icon">📚</div>
                <div class="label">真实病例</div>
                <div class="num">{_case_count}</div>
                <div class="unit">例 · 多难度</div>
              </div>
              <div class="hero-stat">
                <div class="icon">🏥</div>
                <div class="label">覆盖科室</div>
                <div class="num">{_dept_count}</div>
                <div class="unit">个临床专科</div>
              </div>
              <div class="hero-stat">
                <div class="icon">🤖</div>
                <div class="label">支持模型</div>
                <div class="num">{_model_count}</div>
                <div class="unit">种大模型热切换</div>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="statusbar">'
        f'<span class="stb-item">📁 当前病例　<b>{_esc(case.display_name)}</b></span>'
        f'<span class="stb-item">🤖 模型　<b>{_esc(MODEL_REGISTRY[model_key].name)}</b></span>'
        f'<span class="stb-item">🎯 模式　<b>{_esc(mode)}</b></span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # 首次进入（无对话）时展示使用引导
    if not st.session_state.messages:
        st.markdown(
            """
            <div class="welcome">
              <h3>👋 开始你的问诊训练</h3>
              <ol>
                <li>在左侧选择<span style="font-weight:600">病例</span>与<span style="font-weight:600">训练模式</span>（技能训练 / 考试模拟）；</li>
                <li>从「主诉」出发，像接诊医生一样逐条向患者提问；</li>
                <li>系统通过角色引擎扮演虚拟病人，逐句<span style="font-weight:600">流式</span>回复；</li>
                <li>每 3-5 轮会有<span style="font-weight:600">实时提示</span>告诉你哪些已覆盖、哪些还没问；</li>
                <li>问诊结束后点击<span style="font-weight:600">「生成评分报告」</span>，查看分维度得分与改进建议。</li>
              </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 考试进度条
    if is_exam:
        if st.session_state.exam_start is None:
            st.session_state.exam_start = time.time()
        elapsed = int(time.time() - st.session_state.exam_start)
        col1, col2 = st.columns(2)
        col1.metric("已提问", f"{user_turns} / {EXAM_MAX_QUESTIONS}")
        col2.metric("已用时间", f"{elapsed} 秒")
        # Streamlit 原生进度条：考试剩余轮数
        st.progress(
            min(1.0, user_turns / EXAM_MAX_QUESTIONS),
            text=f"考试进度 {user_turns}/{EXAM_MAX_QUESTIONS}",
        )
        st.caption("考试模拟：回合用尽后自动生成评分，也可随时\"交卷\"。")

    # 实时评分反馈：每累计 3 轮做一次中间评分（规则评分，无 LLM 成本）
    if st.session_state.messages and user_turns >= 3 and user_turns % 3 == 0 \
            and not exam_finished:
        inquiry_text = "\n".join(
            m["content"] for m in st.session_state.messages if m["role"] == "user"
        )
        try:
            interim = rule_score(case, inquiry_text)
            tip_html = _real_time_tip_html(interim)
            if tip_html:
                st.markdown(tip_html, unsafe_allow_html=True)
                # toast 形式再提醒一次（仅在新增维度时弹）
                covered_set = {
                    d.covered[0] if isinstance(d, DimensionDetail) and d.covered else ""
                    for d in interim.dimensions.values()
                }
                missed_set = {
                    d.missed[0] if isinstance(d, DimensionDetail) and d.missed else ""
                    for d in interim.dimensions.values()
                }
                new_missed = missed_set - st.session_state.real_time_tips_shown
                if new_missed:
                    miss_txt = "、".join(m for m in new_missed if m)
                    if miss_txt:
                        try:
                            st.toast(f"💡 建议补问：{miss_txt}", icon="💡")
                        except Exception:
                            pass
                    st.session_state.real_time_tips_shown |= new_missed
        except Exception:
            pass

    # 展示历史对话
    for msg in st.session_state.messages:
        role = "assistant" if msg["role"] == "assistant" else "user"
        with st.chat_message(role):
            st.markdown(msg["content"])

    # 问诊输入（考试结束后不再接受提问）
    question = None if exam_finished else st.chat_input("请向患者提问……")

    if question:
        # 0) 输入长度校验，防止超长输入导致 token 成本失控
        if len(question) > MAX_QUESTION_CHARS:
            question = question[:MAX_QUESTION_CHARS]
            st.warning(f"单条提问过长，已截取前 {MAX_QUESTION_CHARS} 字。")

        # 1) 记录学生提问
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # 2) 调用角色引擎流式生成患者回复
        try:
            client = create_client(model_key)
        except Exception as e:  # API 未配置，回滚本轮提问并给出提示
            st.session_state.messages.pop()
            st.error(f"模型调用失败：{e}。请检查 API Key 配置。")
            st.stop()

        try:
            with st.chat_message("assistant"):
                reply = st.write_stream(
                    stream_patient_reply(
                        client,
                        case,
                        history=st.session_state.messages,
                        model=MODEL_REGISTRY[model_key].model,
                        temperature=TEMPERATURE,
                    )
                )
        except Exception:  # 流式失败时降级为一次性生成
            with st.chat_message("assistant"):
                reply = generate_patient_reply(
                    client,
                    case,
                    history=st.session_state.messages,
                    model=MODEL_REGISTRY[model_key].model,
                    temperature=TEMPERATURE,
                )

        st.session_state.messages.append({"role": "assistant", "content": reply})

        # 3) 考试模式：问满即自动交卷
        if is_exam and sum(1 for m in st.session_state.messages if m["role"] == "user") >= EXAM_MAX_QUESTIONS:
            st.session_state.report = _generate_report(
                case, st.session_state.messages, use_llm_score, model_key, mode
            )
            st.rerun()

    if exam_finished:
        st.info("考试已结束，请查看下方评分报告与个人成长档案。")

    # 对话成稿导出（存在对话时可用）
    if st.session_state.messages:
        st.download_button(
            "下载对话记录（Markdown）",
            data=_conversation_markdown(case, st.session_state.messages),
            file_name=f"{case.case_id}_对话记录.md",
            mime="text/markdown",
        )

    # ---- 评分报告 ----
    col1, col2 = st.columns([1, 1])
    with col1:
        if not is_exam:
            if st.button("生成评分报告", disabled=len(st.session_state.messages) == 0):
                st.session_state.report = _generate_report(
                    case, st.session_state.messages, use_llm_score, model_key, mode
                )
                st.rerun()
        else:
            if st.button("交卷并生成评分", disabled=(exam_finished or len(st.session_state.messages) == 0)):
                st.session_state.report = _generate_report(
                    case, st.session_state.messages, use_llm_score, model_key, mode
                )
                st.rerun()

    with col2:
        if st.button("清空评分"):
            st.session_state.report = None

    # 分维度满分为量表配置，用于进度条归一化
    _SCORE_RUBRIC_MAX: dict[str, int] = {**RUBRIC_COMPLETENESS, **RUBRIC_PERFORMANCE}

    def _method_name(method: str) -> str:
        """把评分方式标识映射为友好中文。"""
        return {"rule": "规则评分（离线基线）", "llm": "LLM 智能评分"}.get(method, method)

    def _score_color(total: float) -> str:
        """根据总分返回进度环主题色（绿→青→橙→红）。"""
        if total >= 85:
            return "#15803D"
        if total >= 70:
            return "#0E7490"
        if total >= 60:
            return "#D97706"
        return "#DC2626"

    def _score_card_html(report: ScoringResult) -> str:
        """渲染总分环形进度卡片。"""
        total = float(report.total)
        return (
            '<div class="scorecard">'
            f'<div class="score-ring" style="--val:{total:.0f};--ring:{_score_color(total)};">'
            f'<span class="score-num">{total:.0f}</span><span class="score-cap">/ 100</span>'
            '</div>'
            '<div class="score-meta">'
            '<div class="score-title">问诊综合评分</div>'
            f'<div class="score-method">评分方式：{_esc(_method_name(report.method))}</div>'
            f'<div class="score-method">共 {len(report.dimensions)} 个评分维度</div>'
            '</div></div>'
        )

    def _dim_bars_html(dimensions: Mapping[str, object]) -> str:
        """渲染分维度得分进度条 + 已覆盖 / 建议补问 标签。"""
        rows = []
        for name, val in dimensions.items():
            max_score = _dim_max(dimensions, name, fallback=0) if isinstance(val, DimensionDetail) else (
                _SCORE_RUBRIC_MAX.get(name) or 100
            )
            score = _dim_score(dimensions, name)
            pct = max(0.0, min(100.0, score / max_score * 100 if max_score else 0))
            label = f"{score:g} / {max_score}"
            if isinstance(val, DimensionDetail):
                cov_chips = "".join(f'<span class="tag ok">✓ {_esc(c)}</span>' for c in val.covered[:6])
                miss_chips = "".join(f'<span class="tag miss">✗ {_esc(m)}</span>' for m in val.missed[:6])
            else:
                cov_chips = miss_chips = ""
            rows.append(
                '<div class="dimrow">'
                f'<div class="dim-head"><span>{_esc(name)}</span><span class="dim-val">{label}</span></div>'
                f'<div class="dim-bar"><div class="dim-fill" style="width:{pct:.0f}%"></div></div>'
                + (f'<div class="tag-row">{cov_chips}{miss_chips}</div>' if (cov_chips or miss_chips) else '')
                + '</div>'
            )
        return '<div class="dimgrid">' + "".join(rows) + '</div>'

    # 展示评分报告
    report = st.session_state.report
    if report is not None:
        st.markdown('<div class="sec-title">📋 评分报告</div>', unsafe_allow_html=True)
        st.markdown(_score_card_html(report), unsafe_allow_html=True)

        if report.dimensions:
            st.markdown('<div class="sec-sub">分维度得分</div>', unsafe_allow_html=True)
            st.markdown(_dim_bars_html(report.dimensions), unsafe_allow_html=True)

        if report.strengths:
            items = "".join(f"<li>{_esc(s)}</li>" for s in report.strengths)
            st.markdown(
                f'<div class="rep-sec"><div class="rep-h tag-pos">✓ 优点</div><ul class="rep-list">{items}</ul></div>',
                unsafe_allow_html=True,
            )
        if report.weaknesses:
            items = "".join(f"<li>{_esc(w)}</li>" for w in report.weaknesses)
            st.markdown(
                f'<div class="rep-sec"><div class="rep-h tag-neg">⚠ 不足</div><ul class="rep-list">{items}</ul></div>',
                unsafe_allow_html=True,
            )
        if report.suggestions:
            st.markdown(
                '<div class="rep-sec"><div class="rep-h">改进建议</div></div>',
                unsafe_allow_html=True,
            )
            st.info(report.suggestions)

        # 导出 JSON
        st.download_button(
            "下载评分报告（JSON）",
            data=report.to_json(),
            file_name=f"{case.case_id}_report.json",
            mime="application/json",
        )
        # 导出 Markdown
        st.download_button(
            "下载评分报告（Markdown）",
            data=report.to_markdown(),
            file_name=f"{case.case_id}_report.md",
            mime="text/markdown",
        )


# ---------------------------------------------------------------------------
# Tab 2: SOAP 笔记训练
# ---------------------------------------------------------------------------
with tab_test:
    st.markdown('<div class="sec-title">📝 SOAP 笔记训练</div>', unsafe_allow_html=True)
    st.caption(
        f"针对【{case.display_name} · {case.diagnosis or '无诊断'}】撰写 4 个 SOAP 笔记，"
        "系统按主观(S)/客观(O)/评估(A)/计划(P) 4 维各 25 分进行启发式评分。"
    )

    s = st.text_area("S - 主观资料（主诉+现病史+既往史等）", key="_soap_s", height=120,
                     placeholder="请记录患者的主观信息，如：腹痛 3 天...")
    o = st.text_area("O - 客观资料（体格检查+辅助检查）", key="_soap_o", height=120,
                     placeholder="如：腹软、剑下压痛，肠鸣音正常...")
    a = st.text_area("A - 评估诊断（初步诊断/鉴别诊断）", key="_soap_a", height=100,
                     placeholder="如：考虑慢性胃炎急性发作...")
    p = st.text_area("P - 处理计划（检查/治疗/健康教育）", key="_soap_p", height=120,
                     placeholder="如：完善胃镜、奥美拉唑抑酸、2 周复诊...")

    if st.button("🎯 评分", key="_soap_score_btn"):
        result = score_soap(case, s, o, a, p)
        st.markdown('<div class="sec-sub">评分结果</div>', unsafe_allow_html=True)
        # 总分卡片
        st.markdown(_score_card_html(
            ScoringResult(total=result.total, dimensions={}, method=result.method)
        ), unsafe_allow_html=True)
        # 4 维进度条
        st.markdown(_dim_bars_html(result.dimensions), unsafe_allow_html=True)
        st.download_button(
            "下载 SOAP 评分报告（Markdown）",
            data=result.to_markdown(),
            file_name=f"{case.case_id}_soap.md",
            mime="text/markdown",
        )


# ---------------------------------------------------------------------------
# Tab 3: 教师视图（班级聚合数据）
# ---------------------------------------------------------------------------
with tab_teacher:
    st.markdown('<div class="sec-title">👨‍🏫 教师视图</div>', unsafe_allow_html=True)
    st.caption("基于全部本地训练记录聚合统计；接入班级后可按 class_id 切分。")

    overview = load_class_overview("default")
    if overview["n_records"] == 0:
        st.info("尚无训练记录。请学生先完成问诊评分，再来查看班级统计。")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("记录数", overview["n_records"])
        c2.metric("学生数", len(overview["students"]))
        c3.metric("平均总分", f"{overview['avg_total']:.1f}")
        c4.metric("维度覆盖", len(overview["dim_avg"]))

        # 趋势图
        if overview["total_trend"]:
            st.markdown('<div class="sec-sub">最近 10 次总分趋势</div>', unsafe_allow_html=True)
            st.line_chart({"总分": [t["总分"] for t in overview["total_trend"]]})

        # 维度失分热力
        ranking = dim_loss_ranking("default", top_n=6)
        if ranking:
            st.markdown('<div class="sec-sub">班级弱项 TOP 5（按失分降序）</div>', unsafe_allow_html=True)
            rows = [
                {"维度": d, "平均失分": avg}
                for d, avg in ranking
            ]
            st.dataframe(rows, hide_index=True, use_container_width=True)

        # 历史记录
        st.markdown('<div class="sec-sub">训练历史</div>', unsafe_allow_html=True)
        history_rows = [
            {
                "时间": r.get("时间", ""),
                "学生": r.get("student_id", ""),
                "病例": r.get("病例", ""),
                "总分": r.get("总分"),
                "模式": r.get("模式", ""),
            }
            for r in overview["records"]
        ]
        st.dataframe(history_rows, hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# 个人成长档案（保留在 Tab 1 末尾，避免破坏老路径）
# ---------------------------------------------------------------------------
def _plot_radar(dimensions: Mapping[str, object]):
    """基于分维度得分绘制能力雷达图（matplotlib 极坐标）。

    兼容两种「分维度得分」格式：
        * 纯数字：``{"询问主诉": 80.0}``（旧记录）；
        * 嵌套 dict：``{"询问主诉": {"得分": 80.0, ...}}``（DimensionDetail 序列化）。
    """
    import math

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    available = {f.name for f in fm.fontManager.ttflist}
    font = next(
        (f for f in ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"] if f in available),
        "SimHei",
    )
    plt.rcParams["font.sans-serif"] = [font]
    plt.rcParams["axes.unicode_minus"] = False

    labels = list(dimensions.keys())
    values: list[float] = []
    for k in labels:
        v = dimensions[k]
        if isinstance(v, dict):
            raw = v.get("得分", v.get("score", 0.0))
        else:
            raw = v
        try:
            values.append(float(raw))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            values.append(0.0)
    n = len(labels)
    angles = [i / n * 2 * math.pi for i in range(n)]
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw={"polar": True})
    ax.plot(angles, values, "o-", linewidth=1.6, color="#2e5485")
    ax.fill(angles, values, color="#2e5485", alpha=0.2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, max(values) or 100)
    ax.set_title("问诊能力雷达图", fontsize=12, weight="bold")
    return fig


with tab_train:
    st.markdown("---")
    with st.expander("个人成长档案", expanded=False):
        history = load_history()
        if not history:
            st.caption("暂无训练记录。完成一次评分后，系统会自动写入成长档案。")
        else:
            scores = [float(r.get("总分", 0)) for r in history]
            st.write("**训练概览**")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("训练次数", len(history))
            m2.metric("平均总分", f"{sum(scores) / len(scores):.1f}" if scores else "—")
            m3.metric("最高分", f"{max(scores):.1f}" if scores else "—")
            m4.metric("最低分", f"{min(scores):.1f}" if scores else "—")

            st.write("**历史训练记录**")
            rows = [
                {
                    "时间": r.get("时间", ""),
                    "病例": r.get("病例", ""),
                    "模式": r.get("模式", ""),
                    "总分": r.get("总分"),
                }
                for r in history
            ]
            st.dataframe(rows, hide_index=True, use_container_width=True)

            st.write("**总分趋势**")
            st.line_chart({"总分": [r.get("总分", 0) for r in history]})

            latest = history[-1]
            latest_dims = latest.get("分维度得分") or {}
            if latest_dims:
                st.write("**能力雷达图（最近一次）**")
                fig = _plot_radar(latest_dims)
                st.pyplot(fig)


# 每次交互结束落盘会话状态，支持刷新/重进续答
_persist()
