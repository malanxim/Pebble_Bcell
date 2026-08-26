"""report.py —— 汇总产出到 REPORT.md(结构按提示词第十四节)。"""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import config as C

SHORT = {"B naive": "Bnaive", "B memory": "Bmemory",
         "B intermediate": "Bintermediate", "Plasmablast": "Plasmablast"}


def _top_deg(subtype, n=15):
    f = C.DEG_DIR / f"{subtype.replace(' ', '')}_deg_continuous.csv"
    if not f.exists():
        return pd.DataFrame()
    d = pd.read_csv(f)
    d = d.dropna(subset=["padj", "log2FC_per10yr"])
    d = d[~d["is_IG"]] if "is_IG" in d.columns else d
    up = d.sort_values("log2FC_per10yr", ascending=False).head(n)
    dn = d.sort_values("log2FC_per10yr").head(n)
    return d, up, dn


def run(deg_results, enrich_results, comp_res):
    L = []
    L.append("# 年龄相关 B 细胞变化 —— Phase 1 分析报告\n")
    L.append("> 统计原则:**donor 是生物学重复**,DEG 用每亚型 donor 级 pseudobulk(原始 counts 求和);"
             "年龄作连续变量(每 10 岁)为主,<40 vs ≥65 与 Q1 vs Q4 为辅助/敏感性。"
             "组成变化与亚型内表达变化分开报告。免疫球蛋白(IG)基因保留完整结果,富集另给去IG视图。\n")

    # 1 数据与设计
    don = pd.read_csv(C.TAB_DIR / "donor_summary.csv")
    L.append("## 1. 数据与研究设计\n")
    L.append(f"- OneK1K PBMC,981 名健康供体,年龄 {int(don['age'].min())}–{int(don['age'].max())} "
             f"(均值 {don['age'].mean():.1f}),队列偏老(≥65 岁 {(don['age']>=65).sum()} 人)。\n")
    cov = pd.read_csv(C.TAB_DIR / "subtype_donor_coverage.csv", index_col=0)
    L.append("- 各亚型 donor 覆盖(≥20 细胞,连续年龄主分析):\n")
    L.append(cov.to_markdown() + "\n")
    L.append("- 协变量:sex(F/M)、pool(75 批次)。disease 全 normal、ethnicity 全 European(无变异,不入模型)。\n")
    L.append("- DEG 引擎:PyDESeq2(无 R 环境),design `~ sex + pool + age_scaled`(age_scaled=(age-mean)/10)。\n")

    # 4 组成
    L.append("## 4. 年龄与 B 细胞组成(donor 级比例,binomial GLM)\n")
    L.append(comp_res.to_markdown(index=False) + "\n")
    L.append("- 解读:每 10 岁亚型比例的对数几率变化;正=随龄增加,负=减少。\n"
             "- 这是**组成变化**,与亚型内表达变化(DEG,见下)是两类不同效应。\n")

    # 5 DEG
    L.append("## 5. 亚型内年龄相关 DEG(连续年龄,每 10 岁)\n")
    for s in C.MAIN_SUBTYPES + C.EXPLORATORY:
        got = _top_deg(s)
        if not got:
            continue
        d, up, dn = got
        nsig = int((d["padj"] < C.DEG_PADJ).sum())   # FDR<5%(年龄每10岁效应通常小,不设LFC量级门槛)
        nd = int(d["n_donors"].iloc[0]) if "n_donors" in d.columns else "?"
        tag = "【探索性,donor 少】" if s in C.EXPLORATORY else ""
        L.append(f"### {s} {tag}(n={nd} donor,显著 DEG={nsig})\n")
        cols = ["symbol", "log2FC_per10yr", "padj", "pct_donors"]
        L.append("**上调 top(去IG):**\n"); L.append(up[cols].round(3).to_markdown(index=False) + "\n")
        L.append("**下调 top(去IG):**\n"); L.append(dn[cols].round(3).to_markdown(index=False) + "\n")

    # 6/7 富集
    L.append("## 6-7. GO / 通路富集(ORA)与 GSEA\n")
    L.append("- 背景 = 该亚型受检基因;up/down DEG 分开;GSEA 用全基因按 Wald z 排序。"
             "详见 tables/enrich 与 tables/gsea 子目录 CSV。\n")
    for s in C.MAIN_SUBTYPES:
        f = C.ENR_DIR / f"{s.replace(' ','')}_ora.csv"
        if f.exists():
            ora = pd.read_csv(f)
            ora = ora[ora["set"] == "noIG"] if "set" in ora.columns else ora
            if "P-value" in ora.columns and len(ora):
                top = ora.sort_values("P-value").drop_duplicates("Term").head(6)
                L.append(f"### {s} 富集 top 项(去IG,按 P):\n")
                show = top[["Term", "direction", "P-value"]].copy()
                if "Overlap" in top.columns: show["Overlap"] = top["Overlap"].values
                L.append(show.round(4).to_markdown(index=False) + "\n")

    # 敏感性
    L.append("## 10. 敏感性与可信度\n")
    rows = []
    for s in C.MAIN_SUBTYPES:
        f = C.DEG_DIR / f"{s.replace(' ','')}_sensitivity_concordance.csv"
        if f.exists():
            sc = pd.read_csv(f)
            sc.insert(0, "subtype", s); rows.append(sc)
    if rows:
        sc_all = pd.concat(rows, ignore_index=True)
        L.append("- 不同最低细胞门槛(≥10/≥30)与主分析(≥20)的连续年龄效应一致性:\n")
        L.append(sc_all.to_markdown(index=False) + "\n")
    cf = C.DEG_DIR / "age_scheme_consistency.csv"
    if cf.exists():
        L.append("- 连续年龄 vs <40/≥65 vs Q1/Q4 一致性:\n")
        L.append(pd.read_csv(cf).to_markdown(index=False) + "\n")

    # 局限
    L.append("## 11. 主要结论与 12. 局限性\n")
    L.append("- 见上文每亚型的 DEG/组成数字(均标注 n donor、方向、每10岁效应、FDR、敏感性)。\n"
             "- **局限**:Plasmablast 仅 ~15 donor 达 ≥20 细胞,结果不稳定(已标探索性);"
             "ABC/增殖 B 太稀少,未作独立 DEG;无 R,DEG 用 PyDESeq2(与 DESeq2 等价但未经 R 版交叉);"
             "横断面设计无法区分 cohort 效应;TF 活性推断与全部 22 类图留 Phase 2。\n"
             "- **下一步**:Phase 2 转录因子 regulon/活性二轮;ATAC/CUT&Tag/蛋白验证因果。\n")

    out = C.OUT_DIR / "REPORT.md"
    out.write_text("".join(L), encoding="utf-8")
    print(f"[report] -> {out}")
