"""
config.py —— 年龄相关 B 细胞 DEG 分析(Phase 1)全部参数。

统计原则:donor 是生物学重复;DEG 用每亚型内 donor 级 pseudobulk(原始 counts 求和);
年龄作连续变量(每增 10 岁)为主,<40 vs ≥65 与 Q1 vs Q4 为辅助/敏感性。
运行环境:conda env pytorch_env(pydeseq2 + gseapy + statsmodels)。
"""
from __future__ import annotations
import os
from pathlib import Path

# ---------------------------------------------------------------- paths
PROJECT_ROOT = Path(os.environ.get(
    "BCELL_PROJECT_ROOT", Path(__file__).resolve().parent.parent
)).expanduser().resolve()
H5AD = Path(os.environ.get(
    "BCELL_H5AD",
    PROJECT_ROOT / "data" / "2a930352-0802-4af7-bd26-61f75c83e1b0"
    / "eQTLAutoimmune.h5ad",
)).expanduser().resolve()
OUT_DIR = Path(os.environ.get(
    "BCELL_OUTPUT_DIR", PROJECT_ROOT / "outputs" / "bcell_aging"
)).expanduser().resolve()
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"
PB_DIR = TAB_DIR / "pseudobulk"
DEG_DIR = TAB_DIR / "deg"
ENR_DIR = TAB_DIR / "enrich"
GSEA_DIR = TAB_DIR / "gsea"
for _d in (OUT_DIR, FIG_DIR, TAB_DIR, PB_DIR, DEG_DIR, ENR_DIR, GSEA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 输入列名
COL_DONOR = "donor_id"
COL_AGE = "age"
COL_SEX = "sex"
COL_POOL = "pool_number"
COL_CELLTYPE = "predicted.celltype.l2"     # Azimuth B 亚型(原始 obs 里就有)

# B 亚型(用原始 Azimuth 标签做 pseudobulk 分组;与我重聚类标签高度一致)
SUBTYPES = ["B naive", "B memory", "B intermediate", "Plasmablast"]
MAIN_SUBTYPES = ["B naive", "B memory", "B intermediate"]   # 主 DEG 对象
EXPLORATORY = ["Plasmablast"]                               # donor 太少,仅探索

# ---------------------------------------------------------------- donor 纳入门槛
MIN_CELLS = 20                  # 主分析:每 donor 每 亚型 ≥20 细胞
MIN_CELLS_SENS = [10, 20, 30]   # 敏感性扫描

# ---------------------------------------------------------------- 年龄设置
AGE_SCALE = 10                  # age_scaled = (age - mean)/10 -> 系数 = 每 10 岁
YOUNG_MAX = 40                  # 年轻组 <40
OLD_MIN = 65                    # 老年组 ≥65  (40-64 不进二分组比较)
AGE_BIN_EDGES = [17, 29, 39, 49, 59, 69, 79, 98]
AGE_BIN_LABELS = ["18-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
AGE_GROUP_COL = "age_group"     # young(<40) / middle(40-64) / old(≥65)
QUANTILE_COMPARE = [0.25, 0.75] # Q1 vs Q4 敏感性

# ---------------------------------------------------------------- 协变量
COVARIATES = ["sex", "pool"]    # disease/ethnicity 在本数据为常量,不入模型

# ---------------------------------------------------------------- DEG
DEG_PADJ = 0.05
DEG_LFC = 0.1                   # |log2FC_per10yr| 阈值(年龄连续,每10岁)
# 免疫球蛋白基因前缀(极高表达/方差,易主导结果;保留原始,富集另给"去IG"视图)
IG_PREFIXES = ("IGH", "IGL", "IGK", "IGJ", "IGLL")

# pseudobulk 基因过滤(保留所有过滤断基因,不限 HVG)
GENE_MIN_COUNT = 10             # 该 亚型 pseudobulk 总 counts 下限
GENE_MIN_CPM = 1
GENE_MIN_SAMP_PROP = 0.05       # 至少 5% 样本 CPM≥1
PB_CHUNK = 512                  # 基因块大小(内存安全)

# ---------------------------------------------------------------- 富集/GSEA
ORA_LIBRARIES = ["GO_Biological_Process_2023", "GO_Molecular_Function_2023",
                 "GO_Cellular_Component_2023", "Reactome_2022", "KEGG_2021_Human",
                 "MSigDB_Hallmark_2020"]
GSEA_LIBRARIES = ["MSigDB_Hallmark_2020", "GO_Biological_Process_2023",
                  "Reactome_2022", "KEGG_2021_Human"]
HALLMARK_LIB = "MSigDB_Hallmark_2020"
GSEA_RANK_BY = "stat"           # 按 Wald z 排序(保留方向+证据)
GSEA_SEED = 7
GSEA_MIN_SIZE, GSEA_MAX_SIZE = 15, 500

# ---------------------------------------------------------------- 火山图重点标注基因
HIGHLIGHT_GENES = [
    "MS4A1", "TCL1A", "IL4R", "FCER2", "IGHM", "IGHD", "CD27", "TNFRSF13B",
    "TBX21", "ITGAX", "FCRL5", "STAT1", "ISG15", "CD74", "HLA-DRA",
    "IRF4", "PRDM1", "XBP1", "MZB1", "JCHAIN",
    "SELL", "VPREB3", "CD72", "CXCR3", "FCRL2", "ZBTB32", "IRF7", "IFIT1",
    "BANK1", "CD79A", "PAX5", "BACH2", "AICDA", "S100A4",
]

# ---------------------------------------------------------------- B 细胞 marker 字典(Part 2 验证用,来自提示词)
MARKER_GENES = {
    "pan_B": ["MS4A1", "CD19", "CD79A", "CD79B", "CD37", "CD22", "CD74",
              "HLA-DRA", "BANK1", "BLNK", "EBF1", "PAX5", "SPIB", "BACH2", "CD40"],
    "naive": ["TCL1A", "IGHM", "IGHD", "IL4R", "FCER2", "VPREB3", "CD72",
              "TNFRSF13C", "SELL"],
    "memory": ["CD27", "TNFRSF13B", "AIM2", "S100A4", "ITGB1", "IGHG1", "IGHA1"],
    "plasmablast": ["CD38", "MZB1", "XBP1", "PRDM1", "IRF4", "SLAMF7", "DERL3",
                    "SDC1", "JCHAIN", "TNFRSF17"],
    "abc": ["ITGAX", "TBX21", "FCRL5", "FCRL2", "CXCR3", "ZBTB32"],
    "proliferating": ["MKI67", "TOP2A", "PCNA", "MCM2", "CDK1", "CCNB1"],
}

# ---------------------------------------------------------------- 画图
AGE_CMAP = "viridis"
RANDOM_SEED = 2026
