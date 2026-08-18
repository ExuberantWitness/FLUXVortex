"""Close figure hashes and LaTeX captions for the all-condition figure set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .fluxv_v5_all_conditions import INPUTS, OUTPUT_ROOT, REPO_ROOT, sha256


PLOT_SCRIPTS = (
    "plot_fluxv_v5_all_yang.py",
    "plot_fluxv_v5_all_fig14.py",
    "plot_fluxv_v5_all_baik_filtered.py",
    "plot_fluxv_v5_all_baik_raw.py",
    "plot_fluxv_v5_all_error_ratio.py",
    "plot_fluxv_v5_all_v5b_gate.py",
    "fluxv_v5_all_conditions_plot_helpers.py",
    "fluxv_v5_all_conditions_style.py",
    "finalize_fluxv_v5_all_conditions.py",
)

CAPTIONS = {
    "fig01_yang_all_conditions": (
        "Yang et al.全部六个安装角的周期平均升力与有符号阻力。实验误差棒仅表示"
        "PDF数字化不确定度（±0.4 gf），不是实验置信区间。论文没有公开相位载荷，故本图"
        "不作相位精度声明。v5a是被否决的development proxy，其六点均值与v1/v2重合；"
        "v5b未进入跨论文评分。"
    ),
    "fig02_izraelevitz_fig14_all_conditions": (
        "Izraelevitz/Scherer Figure 14全部公开工况。黑点保留14个原始实验观测，包括两个"
        "重复测量；FluxV指标另同时报告12个唯一工况口径。作者在theta=25 deg、psi=15/30 deg"
        "的参考点没有实验支持，不参与评分。误差棒为数字化的原图报告条，其统计含义未公开。"
    ),
    "fig03_baik_w1_w4_filtered": (
        "Baik W1--W4全部相位升阻力，采用与原文1 Hz处理相匹配的模型滤波：W1/W4保留至"
        "第7谐波，W2/W3保留至第3谐波；每工况评分400个唯一相位点，不做相位、幅值、均值"
        "或偏置拟合。负CD表示推力。Theodorsen仅有升力参考。"
    ),
    "figS01_baik_w1_w4_raw_numeric": (
        "Baik未滤波数值预测的补充诊断。公开实验曲线本身已经过1 Hz处理，因此该图只用于"
        "暴露数值模型的高频内容，不是raw-experiment验证，也不作为主精度排序口径。"
    ),
    "fig04_all_condition_error_ratio": (
        "全部工况逐项v5a/v4b误差比。Yang与Figure 14采用逐工况绝对误差，Baik采用每个"
        "W工况的相位RMSE；颜色是log2比值，格内标实际比值。蓝色小于1为改善，红色大于1"
        "为退化，不跨物理单位汇总。"
    ),
    "fig05_v5b_no_lev_gate": (
        "v5b进入三论文评分前的no-LEV精确退化门。共同Yang 15 deg运动下，全周期没有活跃"
        "LEV，但standalone v5b仍与current FluxV相差max abs dCL=0.556、dCD=0.529；因此"
        "v5b被阻断且没有三论文全工况曲线。本图是实现门控诊断，不是实验验证。"
    ),
}


def _manifest_key(path: Path, output_root: Path) -> str:
    resolved = path.resolve()
    try:
        return f"output/{resolved.relative_to(output_root.resolve()).as_posix()}"
    except ValueError:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()


def finalize(output_root: Path = OUTPUT_ROOT) -> dict[str, object]:
    figure_dir = output_root / "figures"
    expected = [
        figure_dir / f"{stem}.{suffix}"
        for stem in CAPTIONS
        for suffix in ("png", "pdf")
    ]
    missing = [path for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing figure outputs: {missing}")
    script_dir = REPO_ROOT / "platform/forward_flight_benchmarks"
    scripts = [script_dir / name for name in PLOT_SCRIPTS]
    figure_inputs = (
        output_root / "data/all_conditions_curves.csv",
        output_root / "data/all_conditions_metrics.csv",
        output_root / "data/build_manifest.json",
        INPUTS["v5b_gate"],
    )
    latex = []
    for stem, caption in CAPTIONS.items():
        latex.extend(
            (
                r"\begin{figure}[t]",
                r"\centering",
                rf"\includegraphics[width=\linewidth]{{figures/{stem}.pdf}}",
                rf"\caption{{{caption}}}",
                rf"\label{{fig:{stem}}}",
                r"\end{figure}",
                "",
            )
        )
    latex_path = figure_dir / "latex_includes.tex"
    latex_path.write_text("\n".join(latex), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "figure_hashes": {
            _manifest_key(path, output_root): sha256(path) for path in expected
        },
        "plot_source_hashes": {
            _manifest_key(path, output_root): sha256(path) for path in scripts
        },
        "figure_input_hashes": {
            _manifest_key(path, output_root): sha256(path) for path in figure_inputs
        },
        "auxiliary_hashes": {
            _manifest_key(latex_path, output_root): sha256(latex_path)
        },
        "captions": CAPTIONS,
        "v5b_crosspaper_status": "blocked_not_scored",
        "pdf_metadata": "deterministic CreationDate/ModDate omitted",
    }
    manifest_path = figure_dir / "figure_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    manifest = finalize(args.output_dir)
    print(f"closed {len(manifest['figure_hashes'])} figure hashes")


if __name__ == "__main__":
    main()
