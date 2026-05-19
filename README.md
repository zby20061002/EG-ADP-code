# EG-ADP Source Code

This package contains the source code and experimental scripts for:

EG-ADP: Semantic Region-Adaptive Privacy Perturbation for Facial Expression Recognition

## Directory structure

- scripts/: main scripts for reproducing the paper experiments.
- scripts/archive/: early exploratory scripts and parameter search experiments.
- tools/: dataset exploration and utility scripts.
- results/tables/: CSV files used to report paper tables.
- results/figures/: figures and visual outputs.
- results/analysis/: final CK+ analysis outputs.
- results/analysis_oulu/: Oulu-CASIA supplementary outputs.
- checkpoints/: selected emotion recognition model checkpoints.
- external/face-parsing/: BiSeNet face parsing dependency.

##主脚本映射

| Script |
|---|---|
|CK+ 预处理|脚本/00_preprocess_ckplus.py|
|奥卢-CASIA 预处理|脚本/01_preprocess_oulu.py|
|CK+ 情感模型|脚本/02_train_ckplus_emotion.py|
|奥卢-CASIA情绪模型|脚本/03_train_oulu_emotion.py|
|表5-1 CK+公平基线|脚本/04_run_ckplus_fair_baseline.py|
|表5-4 消融研究|脚本/05_run_ckplus_ablation.py|
|表5-5 ArcFace分层攻击|scripts/06_run_arcface_layered_attack.py|
|表5-6 / 表5-7 扩展基线|scripts/07_run_extended_baselines.py|
|表5-8 FER2013诊断|脚本/08_run_fer2013_diagnostic.py|
|图1 / 图2|脚本/09_plot_tradeoff_and_visuals.py|
|表5-2 科恩的d值|脚本/10_analysis_effect_size.py|

##笔记

许多原始脚本使用了硬编码的本地路径。在新机器上运行之前，请更新以下路径：

-数据根目录
-BiSeNet权重路径
-InsightFace模型根目录
-输出目录

建议未来改进：将所有路径移至配置文件中。
