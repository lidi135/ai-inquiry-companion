# -*- coding: utf-8 -*-
"""一次性脚本：扫描病例库做一致性检查。"""
import json
from collections import Counter
from pathlib import Path

CASES_DIR = Path("src/cases")
files = sorted(CASES_DIR.glob("*.json"))
print(f"== 总病例数: {len(files)} ==")
print(f"{'ID':<6} {'科室':<10} {'难度':<4} {'主诉':<22} {'诊断':<14} {'现病史长':<8} {'评分维度':<6}")
print("-" * 90)

VALID_DIFF = {"易", "中", "难"}
seen_ids = set()
seen_chiefs = {}
issues = []
dept_diff = {}
field_cov = {
    "主诉": 0, "诊断": 0, "科室": 0, "难度": 0,
    "现病史": 0, "评分维度": 0, "患者画像": 0,
    "既往史": 0, "个人史": 0, "婚育史": 0, "家族史": 0,
}
diff_counter = Counter()
total_score_points = 0
zero_score_ids = []
short_history_ids = []
ids_with_diag_leak = []

for fp in files:
    data = json.loads(fp.read_text(encoding="utf-8"))
    cid = data.get("case_id") or fp.stem
    chief = data.get("主诉", "") or ""
    diag = data.get("诊断", "") or ""
    dept = data.get("科室", "") or ""
    diff = data.get("难度", "") or ""
    history = data.get("病史", {}) or {}
    # 病例实际写法：病史字段（现病史/既往史/...）在顶层，不在病史子对象里
    history_keys = ["现病史", "既往史", "个人史", "婚育史", "家族史"]
    flat_history = {k: data.get(k, "") for k in history_keys}
    sp = data.get("评分维度", []) or []
    profile = data.get("患者画像", {}) or {}

    dept_diff.setdefault(dept, {"易": 0, "中": 0, "难": 0})
    if diff in VALID_DIFF:
        dept_diff[dept][diff] += 1
        diff_counter[diff] += 1

    print(
        f"{cid:<6} {dept[:8]:<10} {diff:<4} {chief[:20]:<22} "
        f"{diag[:12]:<14} {len(str(flat_history.get('现病史', ''))):<8} {len(sp):<6}"
    )

    if cid in seen_ids:
        issues.append(f"重复 case_id: {cid}")
    seen_ids.add(cid)

    norm_chief = chief.replace(" ", "").lower()
    if norm_chief and norm_chief in seen_chiefs:
        issues.append(
            f"主诉重复: {cid} vs {seen_chiefs[norm_chief]} -> {chief}"
        )
    elif norm_chief:
        seen_chiefs[norm_chief] = cid

    if diff and diff not in VALID_DIFF:
        issues.append(f"{cid}: 难度非法 {diff!r}")

    if diag and norm_chief and diag.replace(" ", "").lower() in norm_chief:
        ids_with_diag_leak.append(cid)

    if not sp:
        zero_score_ids.append(cid)
    if len(str(flat_history.get("现病史", ""))) < 30:
        short_history_ids.append(cid)
    total_score_points += len(sp)

    for k in field_cov:
        if k == "主诉":
            if chief: field_cov[k] += 1
        elif k == "诊断":
            if diag: field_cov[k] += 1
        elif k == "科室":
            if dept: field_cov[k] += 1
        elif k == "难度":
            if diff: field_cov[k] += 1
        elif k == "评分维度":
            if sp: field_cov[k] += 1
        elif k == "患者画像":
            if profile: field_cov[k] += 1
        else:
            if str(flat_history.get(k, "")).strip(): field_cov[k] += 1

print("\n== 字段覆盖统计 ==")
for k, v in field_cov.items():
    pct = v / len(files) * 100
    flag = "" if v == len(files) else "  ⚠"
    print(f"  {k}: {v}/{len(files)} ({pct:.0f}%){flag}")

print(f"\n== 难度分布: 易 {diff_counter['易']} / 中 {diff_counter['中']} / 难 {diff_counter['难']} ==")
print(f"== 评分维度平均: {total_score_points / len(files):.1f} 项/例 ==")
print(f"== 科室数: {len(dept_diff)} ==")
print("\n== 各科室难度分布 ==")
for d, c in sorted(dept_diff.items()):
    print(f"  {d}: 易{c.get('易', 0)} 中{c.get('中', 0)} 难{c.get('难', 0)}")

if zero_score_ids:
    print(f"\n== ⚠ {len(zero_score_ids)} 例缺失评分维度: {zero_score_ids}")
if short_history_ids:
    print(f"== ⚠ {len(short_history_ids)} 例现病史过短(<30字): {short_history_ids}")
if ids_with_diag_leak:
    print(f"== ⚠ {len(ids_with_diag_leak)} 例主诉中含诊断关键词: {ids_with_diag_leak}")

print()
if issues:
    print(f"== ❌ 发现 {len(issues)} 个问题 ==")
    for i in issues:
        print("  -", i)
else:
    print("== ✅ 一致性检查全部通过 ==")
