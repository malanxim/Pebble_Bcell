"""run.py —— 年龄相关 B 细胞 DEG 分析(Phase 1)主编排。

用法:
  python -u -m bcell_aging.run
顺序:data_check -> pseudobulk -> composition -> 每亚型 deg+enrich+图 -> 跨亚型图 -> report。
每亚型后 gc,内存安全(8GB)。
"""
from __future__ import annotations
import gc, time

from . import config as C
from . import data_check, pseudobulk, composition, deg, enrich, plots, report


def main():
    t0 = time.time()
    print("\n" + "=" * 70 + "\n Phase 1: 年龄相关 B 细胞 DEG 分析\n" + "=" * 70)

    obs, donors = data_check.run()
    gc.collect()

    pseudobulk.run(C.MIN_CELLS)
    gc.collect()

    try:
        comp_res = composition.run(donors)
    except Exception as e:
        print(f"[run] composition 失败(跳过): {str(e)[:80]}"); comp_res = None

    deg_results, enrich_results = {}, {}
    for s in C.MAIN_SUBTYPES + C.EXPLORATORY:
        print("\n" + "-" * 60)
        dr = deg.run_deg(s, C.MIN_CELLS)
        deg_results[s] = dr
        try:
            er = enrich.run_enrich(s, dr["continuous"])
        except Exception as e:
            print(f"[run] enrich {s} 失败(跳过): {str(e)[:80]}"); er = {"ora": None, "gsea": None}
        enrich_results[s] = er
        # 出图(需要 pseudobulk);任一图失败不影响整体
        try:
            pb = deg.get_pseudobulk(s, C.MIN_CELLS)
            plots.volcano(dr["continuous"], s)
            plots.ma_plot(dr["continuous"], s)
            plots.deg_heatmap(dr["continuous"], pb, s)
            plots.keygene_age_trend(dr["continuous"], pb, s)
            plots.ora_dotplot(er.get("ora"), s)
            del pb; gc.collect()
        except Exception as e:
            print(f"[run] {s} 出图部分失败(跳过): {str(e)[:80]}")

    # 跨亚型图
    try:
        plots.cross_subtype({s: deg_results[s]["continuous"] for s in C.MAIN_SUBTYPES
                             if "continuous" in deg_results.get(s, {})})
    except Exception as e:
        print(f"[run] cross_subtype 失败(跳过): {str(e)[:80]}")
    try:
        plots.consistency({s: (deg_results[s]["continuous"],
                               deg_results[s].get("young_old"), deg_results[s].get("Q4vsQ1"))
                           for s in C.MAIN_SUBTYPES if "continuous" in deg_results.get(s, {})})
    except Exception as e:
        print(f"[run] consistency 失败(跳过): {str(e)[:80]}")
    try:
        plots.gsea_nes_heatmap({s: enrich_results[s]["gsea"] for s in C.MAIN_SUBTYPES})
    except Exception as e:
        print(f"[run] gsea_nes_heatmap 失败(跳过): {str(e)[:80]}")

    try:
        report.run(deg_results, enrich_results, comp_res)
    except Exception as e:
        print(f"[run] report 失败(跳过): {str(e)[:80]}")
    print(f"\n[run] 全部完成,耗时 {((time.time()-t0)/60):.1f} min。")


if __name__ == "__main__":
    main()
