import pandas as pd
import numpy as np

df = pd.read_csv('5fold_final_results_with_moderate_mosaic.csv')

# 解析 mean±std
def parse_mean_std(cell):
    parts = str(cell).split('±')
    return float(parts[0]), float(parts[1])

# 计算 EG-ADP (Ours_Mixed) 与各基线的效应量
ours_row = df[df['Method'] == 'Ours_Mixed'].iloc[0]
baselines = df[df['Method'] != 'Ours_Mixed']

results = []
for _, base_row in baselines.iterrows():
    for metric in ['Acc', 'Macro-F1', 'Top-1', 'AUC', 'EER']:
        m1, s1 = parse_mean_std(ours_row[metric])
        m2, s2 = parse_mean_std(base_row[metric])
        n_folds = 5
        pooled_sd = np.sqrt(((n_folds-1)*s1**2 + (n_folds-1)*s2**2) / (2*n_folds-2))
        d = (m1 - m2) / pooled_sd if pooled_sd != 0 else 0
        results.append((base_row['Method'], metric, d))

res_df = pd.DataFrame(results, columns=['Baseline', 'Metric', "Cohen's d"])
print(res_df)