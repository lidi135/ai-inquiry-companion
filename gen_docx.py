# -*- coding: utf-8 -*-
"""生成《大理大学医学生AI问诊陪练助手》应用设计方案（含案例与图表）"""
import os

import matplotlib
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

# ---------- 中文字体 ----------
_available = {f.name for f in fm.fontManager.ttflist}
ZH_FONT = next((f for f in ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"] if f in _available), "SimHei")
plt.rcParams["font.sans-serif"] = [ZH_FONT]
plt.rcParams["axes.unicode_minus"] = False

CWD = os.path.dirname(os.path.abspath(__file__))

HEI = "黑体"
SONG = "宋体"
ASCII = "Times New Roman"
MONO = "Consolas"

LQ = "\u201c"
RQ = "\u201d"


def Q(s):
    return LQ + s + RQ


# ================= 图表生成 =================
def fig_arch():
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.axis("off")
    layers = [
        ("应用层（前端）", "Streamlit Web 界面 · 会话管理 · 结果可视化", "#dbe9f6"),
        ("业务逻辑层", "病例加载 · Prompt 组装 · 角色引擎 · 对话管理 · 评分引擎 · 报告生成", "#cfe2f0"),
        ("模型层", "DeepSeek 大语言模型（OpenAI 兼容接口）· 可扩展本地/云端模型", "#c3d8ea"),
        ("数据 / 知识层", "病例库(JSON) · 评分量表(Rubric) · 医学知识库 · 训练记录库", "#b8cfe4"),
    ]
    y = 3.6
    for name, desc, color in layers:
        ax.add_patch(FancyBboxPatch((0.15, y - 0.68), 8.3, 0.66,
                                    boxstyle="round,pad=0.02", fc=color, ec="#2e5485", lw=1.2))
        ax.text(0.5, y - 0.22, name, fontsize=12, weight="bold", va="center", color="#1f3b73")
        ax.text(0.5, y - 0.52, desc, fontsize=8.8, va="center", color="#333333")
        if y < 3.6:
            ax.annotate("", xy=(4.3, y + 0.70), xytext=(4.3, y + 0.02),
                        arrowprops=dict(arrowstyle="->", lw=1.6, color="#1f3b73"))
        y -= 0.98
    ax.set_xlim(0, 8.6)
    ax.set_ylim(-0.2, 4.2)
    ax.set_title("图 1　系统总体架构图", fontsize=13, weight="bold", color="#1f3b73", pad=8)
    return fig


def fig_student_flow():
    steps = [
        ("登录并选择训练模式", "技能训练 / 多学科陪练 / 考试模拟"),
        ("选择病例", "查看病种 · 难度 · 患者基本信息"),
        ("自然语言问诊对话", "AI 患者口语化作答，保守病情细节"),
        ("结束问诊并生成报告", "系统基于 Rubric 自动评分"),
        ("查看反馈与成长档案", "总分 · 分维度得分 · 改进建议 · 历史趋势"),
    ]
    return _flow(steps, "图 2　学生端问诊训练操作流程")


def fig_data_flow():
    steps = [
        ("用户输入问诊问题", None),
        ("加载病例与对话历史", "病例 JSON + 上下文"),
        ("组装角色 Prompt", "System Prompt + 病例事实 + 对话上下文"),
        ("大模型生成患者回复", "口语化 · 低温度采样"),
        ("评分引擎结构化评分", "Rubric + JSON 输出"),
        ("生成报告并持久化", "训练记录库 · 个人成长档案"),
    ]
    return _flow(steps, "图 3　问诊训练数据流程")


def _flow(steps, title):
    n = len(steps)
    fig, ax = plt.subplots(figsize=(6.2, n * 0.95 + 1.0))
    ax.axis("off")
    h = 0.62
    gap = 0.42
    y = 0
    for i, (label, sub) in enumerate(steps):
        ax.add_patch(FancyBboxPatch((1.1, y), 4.6, h, boxstyle="round,pad=0.03",
                                    fc="#dbe9f6", ec="#2e5485", lw=1.2))
        if sub:
            ax.text(3.4, y + h - 0.20, label, ha="center", va="center", fontsize=10.5, weight="bold")
            ax.text(3.4, y + 0.16, sub, ha="center", va="center", fontsize=8.2, color="#444444")
        else:
            ax.text(3.4, y + h / 2, label, ha="center", va="center", fontsize=10.5, weight="bold")
        if i < n - 1:
            ax.annotate("", xy=(3.4, y + h + gap - 0.02), xytext=(3.4, y + h + 0.02),
                        arrowprops=dict(arrowstyle="->", lw=1.6, color="#1f3b73"))
        y += h + gap
    ax.set_xlim(0, 6.8)
    ax.set_ylim(-0.4, y + 0.2)
    ax.set_title(title, fontsize=13, weight="bold", color="#1f3b73", pad=6)
    return fig


def fig_radar():
    labels = ["主诉采集", "现病史", "既往史", "体格检查", "沟通人文", "问诊效率"]
    before = [62, 55, 58, 50, 60, 65]
    after = [85, 80, 82, 75, 88, 86]
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    before_c = before + before[:1]
    after_c = after + after[:1]
    angles_c = angles + angles[:1]
    fig, ax = plt.subplots(figsize=(5.4, 5.2), subplot_kw=dict(polar=True))
    ax.plot(angles_c, before_c, "o-", color="#9aa5b1", lw=1.6, label="训练前")
    ax.fill(angles_c, before_c, color="#9aa5b1", alpha=0.15)
    ax.plot(angles_c, after_c, "o-", color="#2e5485", lw=2.0, label="训练后")
    ax.fill(angles_c, after_c, color="#2e5485", alpha=0.20)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=7.5, color="gray")
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12), fontsize=9, frameon=False)
    ax.set_title("图 4　个人成长档案·问诊能力雷达图", fontsize=12.5, weight="bold", color="#1f3b73", pad=18)
    return fig


def fig_gantt():
    tasks = [
        ("需求调研", 0.0, 1.0),
        ("交互与逻辑开发", 0.5, 2.5),
        ("MVP 开发", 1.5, 2.5),
        ("评分引擎", 3.5, 2.5),
        ("测试与优化", 4.5, 2.5),
        ("试用与迭代", 5.5, 2.5),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    for i, (name, start, dur) in enumerate(tasks):
        ax.broken_barh([(start, dur)], (len(tasks) - 1 - i - 0.35, 0.7),
                       facecolors="#4a7bb5", edgecolors="#2e5485", alpha=0.9)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([t[0] for t in tasks][::-1], fontsize=10)
    ax.set_xlim(0, 8)
    ax.set_xticks(range(0, 9))
    ax.set_xlabel("周次", fontsize=10)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.set_title("图 5　8 周里程碑计划（甘特图）", fontsize=13, weight="bold", color="#1f3b73", pad=8)
    return fig


def save_all():
    paths = {}
    figs = {
        "fig1_arch.png": fig_arch(),
        "fig2_student_flow.png": fig_student_flow(),
        "fig3_data_flow.png": fig_data_flow(),
        "fig4_radar.png": fig_radar(),
        "fig5_gantt.png": fig_gantt(),
    }
    for name, f in figs.items():
        p = os.path.join(CWD, name)
        f.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(f)
        paths[name] = p
    return paths


# ================= docx 工具函数 =================
doc = Document()
style = doc.styles["Normal"]
style.font.name = ASCII
style.font.size = Pt(12)
style.element.rPr.rFonts.set(qn("w:eastAsia"), SONG)


def set_run(run, east=SONG, ascii_=ASCII, size=12, bold=False, color=None):
    run.font.name = ascii_
    run._element.rPr.rFonts.set(qn("w:eastAsia"), east)
    run.font.size = Pt(size)
    run.font.bold = bold
    if color:
        run.font.color.rgb = RGBColor(*color)
    return run


def title(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_run(p.add_run(text), HEI, HEI, 22, bold=True)


def subtitle(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_run(p.add_run(text), HEI, HEI, 14, bold=True, color=(0x40, 0x40, 0x40))


def h1(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    set_run(p.add_run(text), HEI, HEI, 16, bold=True, color=(0x1F, 0x3B, 0x73))


def h2(text):
    p = doc.add_paragraph()
    set_run(p.add_run(text), HEI, HEI, 13.5, bold=True, color=(0x2E, 0x54, 0x95))


def h3(text):
    p = doc.add_paragraph()
    set_run(p.add_run(text), HEI, HEI, 12, bold=True)


def para(text, indent=True):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.4
    if indent:
        p.paragraph_format.first_line_indent = Pt(24)
    set_run(p.add_run(text), SONG, ASCII, 12)


def bullet(text):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.line_spacing = 1.3
    set_run(p.add_run(text), SONG, ASCII, 12)


def numbered(text):
    p = doc.add_paragraph(style="List Number")
    p.paragraph_format.line_spacing = 1.3
    set_run(p.add_run(text), SONG, ASCII, 12)


def code(text):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.left_indent = Pt(18)
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    set_run(p.add_run(text), MONO, MONO, 10.5)


def make_table(headers, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, htext in enumerate(headers):
        t.rows[0].cells[i].text = ""
        set_run(t.rows[0].cells[i].paragraphs[0].add_run(htext), HEI, HEI, 11, bold=True)
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            set_run(cells[i].paragraphs[0].add_run(str(val)), SONG, ASCII, 10.5)
    if widths:
        for i, w in enumerate(widths):
            for r in t.rows:
                r.cells[i].width = Cm(w)
    return t


def add_image(path, width_cm=13.5, caption=None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(path, width=Cm(width_cm))
    if caption:
        cp = doc.add_paragraph()
        cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_run(cp.add_run(caption), SONG, SONG, 9.5, color=(0x66, 0x66, 0x66))


# ================= 生成图表 =================
img = save_all()

# ================= 封面 =================
for _ in range(4):
    doc.add_paragraph()
title("大理大学医学生AI问诊陪练助手")
subtitle("——" + Q("AI＋场景设计") + "赛项应用设计方案")
doc.add_paragraph()
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
set_run(p.add_run("参赛赛道：高校赛道 · AI＋场景设计"), SONG, SONG, 12)
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
set_run(p.add_run("应用领域：医学教育 / 临床技能训练"), SONG, SONG, 12)

# ================= 引言 =================
h1("引言")
para("本方案面向大理大学医学教育中的问诊（病史采集）技能训练场景，提出并设计一款基于大语言模型的" + Q("AI问诊陪练助手") + "。该应用通过构建高拟真的" + Q("AI虚拟标准化病人（VSP）") + "，让医学生能够在任意时间、任意地点进行反复、安全、可量化的问诊技能训练，并由智能评分引擎提供标准化、个性化的即时反馈，有效缓解临床教学中标准化病人（SP）资源稀缺、带教老师精力有限、训练机会不足等突出问题，助力医学生问诊能力与临床思维的系统化培养。")
para("方案以" + Q("Python + Streamlit + 大语言模型") + "为技术底座，采用分层架构与" + Q("角色引擎 + 病例事实约束 + 低温度采样") + "的核心机制，融合开源医学数据集与临床问诊评分量表，形成" + Q("训练—评分—反馈—成长") + "的完整教学闭环。")

# ================= 一、需求分析 =================
h1("一、需求分析")

h2("1.1 项目背景")
para("问诊（病史采集）是临床诊疗的起点，也是医学生从理论走向临床实践的关键环节。标准化的问诊训练通常依赖" + Q("标准化病人（SP）") + "与临床带教，然而这一模式在大理大学等院校的实际教学中面临多重现实约束：标准化病人招募与培训成本高、数量有限；带教老师临床任务繁重，难以对学生进行充分的一对一指导；学生进入临床实习的机会受限于医院资源。上述约束使得医学生普遍存在" + Q("练得少、练不真、练后无反馈") + "的困境，亟需一种低成本、高拟真、可反复训练的智能化解决方案。")

h2("1.2 医学生问诊训练的三大痛点")
h3("痛点一（P1）：训练机会稀缺、时间受限")
para("医学生问诊技能训练每周仅约 2 次，面对标准化病人或真实患者的机会十分有限。有限的练习次数难以让学生形成稳定、规范的问诊能力，也难以覆盖不同病种与复杂场景。")
h3("痛点二（P2）：陪练资源不足、真实性欠缺")
para("多学科陪练高度依赖标准化病人（SP）与带教老师，而 SP 数量有限、招募培训成本高；同学之间互练缺乏真实病情与医学细节，训练效果打折扣；带教老师临床任务重，每周仅能提供约 1—2 小时的一对一指导，无法满足每一位学生的训练需求。")
h3("痛点三（P3）：考核压力大、反馈不透明")
para("临床技能考核（如 OSCE）常为" + Q("一次定终身") + "，学生因紧张而不敢问、不会问；评分标准不透明，缺乏过程性、可量化的反馈，导致" + Q("练而无评、评而无据") + "，学生难以明确自身的不足与改进方向。")

h2("1.3 目标用户特征")
bullet("大理大学医学院大三及以上医学生，涵盖临床医学、护理学、药学等专业；")
bullet("已修完基础医学与诊断学等课程，具备一定理论储备，但缺乏临床问诊实操经验；")
bullet("学习动机强，但面对真实患者或标准化病人存在紧张与畏难情绪；")
bullet("时间呈碎片化，需要随时随地、可重复、低心理压力的训练环境；")
bullet("熟悉智能手机与 Web 应用，具备基本的信息化素养。")

h2("1.4 功能需求")
make_table(
    ["编号", "功能模块", "功能说明"],
    [
        ["FR1", "智能问诊陪练（SP1）", "提供 AI 虚拟标准化病人，支持自然语言对话式问诊，模拟真实问诊场景，开展问诊技能专项训练。"],
        ["FR2", "多学科病例库（SP2）", "覆盖内科、外科、妇产科、儿科、急诊等多学科病例，支持按病种、难度、科室筛选，实现个性化陪练。"],
        ["FR3", "考试模拟（SP3）", "模拟 OSCE 等临床技能考核，限时作答、限制追问暗示，考后自动生成成绩与评分报告。"],
        ["FR4", "智能评分反馈", "基于问诊评分量表（Rubric）自动评分，输出总分、分维度得分、优点、不足与改进建议。"],
        ["FR5", "病例与知识库管理", "内置丰富病例与医学知识（教材、诊疗指南、开源医学数据集），支持教师自定义病例。"],
        ["FR6", "个人成长档案", "记录历史训练与考试成绩，生成能力雷达图与趋势分析，追踪问诊能力成长。"],
    ],
    widths=[1.6, 4.0, 9.4],
)

h2("1.5 非功能需求")
bullet("可靠性：模型回复与评分需具备医学专业性与准确性，采用低温度采样并约束于知识库/病例事实，避免产生幻觉或误导性信息；")
bullet("安全性：涉及医患对话等敏感信息，支持本地化部署，数据不出本地；提供" + Q("仅用于教学、不构成医疗建议") + "的合规提示；")
bullet("易用性：界面简洁，Web 端即开即用，无需安装，学习成本低；")
bullet("性能：单次回复响应控制在数秒级，支持一定规模并发训练；")
bullet("可扩展性：病例、评分量表、模型均可插拔、可配置，便于后期扩展多学科与多院校；")
bullet("可维护性：模块化架构，前后端解耦，代码开源、文档齐全。")

# ================= 二、整体方案 =================
h1("二、整体方案")

h2("2.1 设计理念")
para("本应用以" + Q("AI虚拟标准化病人（VSP）+ 智能评分引擎") + "为核心，打造一个" + Q("随时随地、反复可练、即练即评") + "的问诊训练闭环。核心设计理念包括：")
bullet("拟真：用大模型模拟具备完整社会背景、心理状态与疾病表现的" + Q("真实患者") + "，而非简单问答机器人；")
bullet("标准化：以临床问诊评分量表为基准，实现评分可量化、可复现、可比较；")
bullet("个性化：根据学生水平与薄弱环节，提供差异化病例与针对性反馈；")
bullet("低门槛：轻量 Web 应用，本地即可部署，兼顾隐私保护与使用成本。")

h2("2.2 核心功能模块划分")
make_table(
    ["模块", "核心能力", "对应训练场景"],
    [
        ["M1 智能问诊陪练", "AI 虚拟标准化病人角色扮演、多轮自然语言问诊", "SP1：问诊技能专项训练"],
        ["M2 多学科病例库", "多病种、多难度、多科室病例的加载与筛选", "SP2：多学科陪练"],
        ["M3 考试模拟", "限时、限追问的标准化考核与自动判分", "SP3：临床技能考核模拟"],
        ["M4 智能评分反馈", "基于 Rubric 的分维度评分与改进建议", "训练与考核的即时反馈"],
        ["M5 病例与知识库", "病例管理、教师自定义、医学知识检索", "教学内容支撑"],
        ["M6 个人成长档案", "历史记录、能力雷达图、趋势分析", "过程性评价与学习追踪"],
    ],
    widths=[3.4, 6.6, 5.0],
)

h2("2.3 用户操作流程")
para("学生端操作流程如图 2 所示。用户登录后选择训练模式与病例，进入自然语言问诊对话；AI 患者以口语化方式作答并保守病情细节，问诊结束后系统基于 Rubric 自动评分，用户可查看分维度得分与改进建议，并在个人成长档案中追踪历史趋势。")
add_image(img["fig2_student_flow.png"], width_cm=10.5)
para("教师端可导入或自定义病例与评分量表、布置训练/考核任务，并查看学生训练记录与成绩统计，用于过程性评价与教学改进。")

h2("2.4 典型案例设计")
para("案例采用结构化 JSON 存储，覆盖多学科、多难度。以下列出四个典型病例，用于支撑 SP1/SP2/SP3 三类训练场景（详细病例字段见附录 B）。")
make_table(
    ["编号", "主诉", "科室 / 系统", "难度", "重点考查维度"],
    [
        ["P01", "上腹隐痛伴恶心 3 个月，加重 2 天", "消化内科", "中", "部位、性质、诱因、放射痛、伴随症状"],
        ["P02", "反复咳嗽、咳痰 2 周", "呼吸内科", "易", "咳嗽性质、痰液性状、时间规律、吸烟史"],
        ["P03", "腹泻伴腹痛 2 天", "消化内科", "易", "大便性状、饮食诱因、脱水、伴随症状"],
        ["P04", "活动后胸闷胸痛 1 周", "心血管内科", "难", "胸痛部位/性质/缓解方式、危险因素"],
    ],
    widths=[1.4, 5.6, 2.6, 1.4, 4.0],
)

# ================= 三、技术架构 =================
h1("三、技术架构")

h2("3.1 系统整体架构设计")
para("系统采用" + Q("应用层—业务逻辑层—模型层—数据/知识层") + "的分层架构，各层职责清晰、松耦合，便于迭代与扩展，如图 1 所示。")
add_image(img["fig1_arch.png"], width_cm=14.0)
bullet("应用层（前端）：基于 Streamlit 的 Web 应用，负责交互界面、会话管理、结果可视化；")
bullet("业务逻辑层：Python 后端，负责病例加载、Prompt 组装、角色引擎、对话管理、评分引擎、报告生成；")
bullet("模型层：DeepSeek 等大语言模型（兼容 OpenAI 接口），负责患者角色扮演与回复生成，可配置接入本地或云端模型；")
bullet("数据/知识层：病例库（JSON）、评分量表（Rubric）、医学知识库（教材/指南/开源数据集）、训练记录库。")

h2("3.2 关键技术选型")
make_table(
    ["类别", "选型", "说明"],
    [
        ["前端/应用框架", "Streamlit", "Python 生态，快速构建交互式 Web 应用，适合 MVP 与教学演示。"],
        ["后端语言", "Python 3.10+", "生态成熟，便于集成大模型 SDK 与数据处理。"],
        ["大语言模型", "DeepSeek（OpenAI 兼容接口）", "通过 base_url=https://api.deepseek.com 调用，采用低温度（0.2—0.4）保证稳定性与一致性。"],
        ["提示词工程", "System Prompt + 病例 JSON + 对话历史", "确保患者角色真实、不泄露诊断、不主动剧透病情。"],
        ["评分引擎", "Rubric + 结构化输出（JSON）", "输出总分、分维度得分、优点、不足、改进建议。"],
        ["知识库/数据集", "CMExam / CMB / MedDialog-CN 等", "CMExam（6 万余题，Apache-2.0）、CMB（含 CMB-Clin 74 个临床任务）、MedDialog-CN（110 万条医患对话）。"],
    ],
    widths=[3.0, 4.6, 7.4],
)
para("模型能力保障方面，据公开测评，主流国产大模型在中文医疗问答任务上的综合准确率可达 97.9%—99%，为问诊陪练的回复质量与评分可靠性提供了基础支撑。")

h2("3.3 数据流程")
para("系统数据流程如图 3 所示：用户输入经病例加载、Prompt 组装、模型推理、评分引擎，最终生成评分报告并持久化到训练记录库。")
add_image(img["fig3_data_flow.png"], width_cm=10.0)

h2("3.4 开发环境要求")
make_table(
    ["项目", "要求"],
    [
        ["操作系统", "Windows / macOS / Linux"],
        ["运行环境", "Python 3.10+，pip 包管理器"],
        ["核心依赖", "streamlit、openai"],
        ["密钥配置", ".streamlit/secrets.toml 中配置 DEEPSEEK_API_KEY 等密钥"],
        ["硬件", "本地开发/运行仅需普通 PC；大规模并发或私有化部署可迁移至云服务器"],
        ["目录结构", "app.py、cases/、prompts/、records/、docs/、.streamlit/secrets.toml"],
    ],
    widths=[4.0, 11.0],
)

# ================= 四、创新点 =================
h1("四、创新点")

h2("4.1 技术应用创新")
bullet("AI 虚拟标准化病人（VSP）：用大模型替代传统标准化病人，具备完整患者画像（性别、年龄、职业、文化、心理状态）与疾病脚本，实现高拟真、低成本、可规模化的陪练。")
bullet("动态角色引擎：通过" + Q("角色 System Prompt + 病例事实注入 + 低温度采样") + "三重约束，实现患者回答的口语化与背景一致性，并严格防止主动泄露诊断（如" + Q("我不懂，医生你看着办吧") + "）。")
bullet("可量化智能评分：将临床 Rubric 与大模型结合，实现从" + Q("主观印象") + "到" + Q("分维度量化评分") + "的转变，评分可复现、可追踪。")
bullet("本地化与隐私保护：支持本地部署，医患训练数据不出本地，兼顾医学伦理与数据安全。")

h2("4.2 场景结合创新")
bullet("精准对接医学教育痛点：直击 SP 资源稀缺、训练机会少、考核压力大三大痛点，实现" + Q("随时练、反复练、敢练") + "。")
bullet("多学科、多层次病例体系：覆盖临床/护理/药学多专业、多病种、多难度，并支持教师自定义病例，贴合院校实际教学。")
bullet("教学闭环：训练—评分—反馈—成长档案一体化，形成可量化的过程性评价，服务教学管理与改革。")

h2("4.3 交互设计创新")
bullet("多模态交互（可扩展）：在文本对话基础上，预留语音输入（ASR）与语音合成（TTS）接口，贴近真实问诊的语音交流体验。")
bullet("情感与沟通引导：在评分中纳入人文关怀、沟通技巧等维度，引导医学生不仅" + Q("问对") + "，更要" + Q("问好") + "。")
bullet("个性化成长档案：能力雷达图 + 历史趋势，帮助学生直观定位薄弱环节，实现个性化学习路径（见图 4）。")

# ================= 五、应用前景 =================
h1("五、应用前景")

h2("5.1 在医学生培养中的推广价值")
bullet("缓解教学资源压力：以低成本、可规模化的 AI 陪练，扩大问诊训练覆盖面，弥补 SP 资源不足；")
bullet("提升训练频次与质量：学生可随时随地反复练习，弥补临床实习机会不足，巩固问诊规范；")
bullet("支撑教学改革与考核：过程性、量化评价为教学管理与技能考核提供客观依据。")

h2("5.2 医学教育领域的应用价值")
bullet("可作为标准化病人培训的辅助工具，降低 SP 招募与培训成本；")
bullet("可推广至执业医师（含 OSCE）考前训练、住院医师规范化培训、继续医学教育；")
bullet("可与医学课程结合，形成" + Q("线上陪练 + 线下实操") + "的混合式教学模式。")

h2("5.3 未来发展方向")
bullet("从问诊向查体、辅助检查判读、诊疗决策等全流程扩展，构建" + Q("AI 临床思维训练") + "平台；")
bullet("接入多模态（语音、图像、影像），实现更真实的临床场景模拟；")
bullet("引入情感识别与自适应难度，实现真正的个性化学习；")
bullet("面向多院校、多机构开放，构建标准化病例与评分体系，逐步形成医学教育领域的行业标准与生态。")

# ================= 附录 =================
h1("附录")

h2("附录A　问诊评分量表（Rubric）")
para("评分采用百分制，其中" + Q("问诊完整性") + "占 80 分，" + Q("综合表现") + "占 20 分，评分量表可配置，教师可自定义维度与分值。", indent=False)
h3("问诊完整性（80 分）")
make_table(
    ["评分维度", "分值"],
    [
        ["开放式提问", "5"],
        ["主诉", "5"],
        ["现病史（起病、诱因、部位、性质、程度、时间）", "15"],
        ["既往史", "10"],
        ["系统回顾 / 伴随症状", "5"],
        ["体格检查", "8"],
        ["辅助检查", "5"],
        ["病情演变", "4"],
        ["一般情况（饮食、睡眠、二便）", "3"],
        ["个人史", "8"],
        ["家族史", "5"],
        ["婚育史", "7"],
    ],
    widths=[10.0, 3.0],
)
h3("综合表现（20 分）")
make_table(
    ["评分维度", "分值"],
    [
        ["问诊准确性", "5"],
        ["问诊全面性", "5"],
        ["医患沟通 / 人文关怀", "5"],
        ["问诊效率 / 条理性", "5"],
    ],
    widths=[10.0, 3.0],
)

h2("附录B　典型案例详情")
para("以下四个病例以结构化 JSON 组织，字段含病例编号、主诉、患者画像、现病史、既往史、个人史、婚育史、诊断与评分维度。", indent=False)

h3("P01　腹痛（消化内科）")
code('{ "case_id": "P01", "主诉": "上腹部隐痛伴恶心 3 个月，加重 2 天",')
code('  "患者画像": {"性别":"女","年龄":55,"职业":"农民","文化":"初中文化","就诊状态":"住院","心理状态":"担心花钱、依从性差"},')
code('  "现病史": "3 个月前无明显诱因出现上腹隐痛，餐后加重，伴反酸、嗳气，近 2 天加重",')
code('  "既往史": "慢性胃炎、高血压", "个人史": "长期务农、饮食不规律、吸烟",')
code('  "婚育史": "已绝经，孕 2 产 1", "诊断": "慢性胃炎（待排除消化性溃疡）",')
code('  "评分维度": ["主诉采集","部位性质","诱因","放射痛","伴随症状","一般情况","既往史","个人史","家族史","婚育史"] }')

h3("P02　咳嗽（呼吸内科）")
code('{ "case_id": "P02", "主诉": "反复咳嗽、咳痰 2 周",')
code('  "患者画像": {"性别":"男","年龄":42,"职业":"公司职员","文化":"本科","就诊状态":"门诊","心理状态":"工作压力大"},')
code('  "现病史": "2 周前受凉后咳嗽，夜间加重，咳少量白痰，无发热",')
code('  "既往史": "过敏性鼻炎", "个人史": "吸烟 15 年，每日约 1 包，偶饮酒",')
code('  "婚育史": "已婚，1 子", "诊断": "急性支气管炎（待排除哮喘/慢阻肺）",')
code('  "评分维度": ["咳嗽性质","痰液性状","时间规律","诱因","伴随症状","吸烟史","过敏史"] }')

h3("P03　腹泻（消化内科）")
code('{ "case_id": "P03", "主诉": "腹泻伴腹痛 2 天",')
code('  "患者画像": {"性别":"女","年龄":28,"职业":"学生","文化":"本科","就诊状态":"门诊","心理状态":"焦虑"},')
code('  "现病史": "2 天前进食不洁食物后腹泻，每日 5—6 次，水样便，伴脐周阵痛",')
code('  "既往史": "无特殊", "个人史": "无特殊", "婚育史": "未婚",')
code('  "诊断": "急性胃肠炎（待排除感染性肠炎）",')
code('  "评分维度": ["大便性状与次数","饮食诱因","伴随症状","脱水表现","发热","用药史"] }')

h3("P04　胸闷胸痛（心血管内科）")
code('{ "case_id": "P04", "主诉": "活动后胸闷胸痛 1 周",')
code('  "患者画像": {"性别":"男","年龄":60,"职业":"退休","文化":"初中文化","就诊状态":"门诊","心理状态":"紧张"},')
code('  "现病史": "1 周前爬楼后出现胸骨后压榨样疼痛，休息 3—5 分钟缓解",')
code('  "既往史": "高血压 10 年、糖尿病 5 年、高血脂", "个人史": "吸烟 40 年、久坐少动",')
code('  "婚育史": "已婚，1 子 1 女", "诊断": "冠心病·心绞痛（待排除心肌梗死）",')
code('  "评分维度": ["胸痛部位","性质","诱因","缓解方式","心血管危险因素","既往慢病史"] }')

h2("附录C　个人成长档案示例")
para("系统基于历史训练记录生成问诊能力雷达图，直观展示学生训练前后的能力变化，帮助定位薄弱环节（图 4）。")
add_image(img["fig4_radar.png"], width_cm=11.0)

h2("附录D　8 周里程碑计划")
add_image(img["fig5_gantt.png"], width_cm=13.5)
make_table(
    ["阶段", "时间", "主要任务与交付物"],
    [
        ["需求调研", "第 1 周", "调研目标用户与教师评分标准；下载 CMExam、CMB、MedDialog 等数据集；设计 3—5 个典型病例。"],
        ["交互与逻辑开发", "第 1—3 周", "完成界面原型与对话逻辑；搭建腹痛、腹泻、咳嗽等首批病例模板。"],
        ["MVP 开发", "第 2—4 周", "打通病例加载—对话—回复完整链路，发布可运行的最小可用版本。"],
        ["评分引擎", "第 4—6 周", "实现基于 Rubric 的自动评分，输出结构化评分报告。"],
        ["测试与优化", "第 5—7 周", "开展功能测试与医学准确性校验，优化回复与评分质量。"],
        ["试用与迭代", "第 6—8 周", "面向 30 名学生试用，收集反馈并迭代，完成成果展示材料。"],
    ],
    widths=[3.2, 2.6, 9.2],
)

h2("附录E　关键技术实现要点")
para("系统采用 OpenAI 兼容接口调用 DeepSeek 模型，核心初始化代码如下：", indent=False)
code("import json, glob")
code("import streamlit as st")
code("from openai import OpenAI")
code("")
code("client = OpenAI(")
code("    api_key=st.secrets[\"DEEPSEEK_API_KEY\"],")
code("    base_url=\"https://api.deepseek.com\"")
code(")")
code("MODEL = \"deepseek-v4-flash\"  # 可按需替换为其他模型")
para("患者角色 System Prompt 示例：", indent=False)
code("你是专业的标准化病人（SP），根据给定患者画像模拟真实患者。第一问用开放性问题开始 1—3 句话，不主动透露完整病情；对不清楚的医学细节保持模糊或回答\u201c我不清楚\u201d，不主动给出诊断；遇到封闭式提问（如\u201c是不是…\u201d）用\u201c我不懂，医生你看着办吧\u201d回应；回答口语化、符合患者文化水平与教育背景。")

# ================= 保存 =================
out = os.path.join(CWD, "大理大学医学生AI问诊陪练助手_设计方案_v2.docx")
doc.save(out)
print("已生成：", out)
print("使用字体：", ZH_FONT)
