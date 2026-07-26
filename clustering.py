# Wire Saw APC TF : Clustering code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 7


df = pd.read_csv(r'D:\chaewon\APC\02.TF\260719\big_data\data\data.csv', encoding='utf-8')


df[df['process_time']=='13.3Hr']['meas_eqp'].unique()

# <StringArray>
# ['BSAFS07', 'BSAFS06', 'BSAFS02', 'BSAFS08']
# Length: 4, dtype: str


df_r = df[df['process_time'] == '13.3Hr']
df_r2 = df_r[df_r['meas_eqp'] == 'BSAFS08']


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


feature_cols = ['avg_bow_bf_tail', 'range_wire_guide_10_99']

result_df_r2_r = df_r2.dropna(subset='avg_bow_bf_tail')
result_df_r2_r = result_df_r2_r.dropna(subset='range_wire_guide_10_99')
result_df_r2_r = df_r2.dropna(subset=feature_cols).copy()

X = result_df_r2_r[feature_cols].values

kmeans = KMeans(n_clusters=4, random_state=0, n_init='auto')
y_kmeans = kmeans.fit_predict(X)

df_r2['clustering_group'] = np.nan  # 일단 모든 행을 NaN 으로 초기화

df_r2.loc[result_df_r2_r.index, 'clustering_group'] = y_kmeans

df_r2['clustering_group'] = df_r2['clustering_group'].astype('Int64')

# --- 시각화 코드 (원본 유지) ---
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis', alpha=0.6, label='Wafer Data')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.9, marker='X', edgecolors='black', label='Centroids')

plt.title('avg_bow_bf_tail vs range_wire_guide_10_99')
plt.xlabel('avg_bow_bf_tail')
plt.ylabel('range_wire_guide_10_99')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# 군집 중심 좌표 출력
print("군집 중심 좌표:\n", centers)

# 결과
# 군집 중심 좌표:
# [[7.34174146e-02 1.03901006e+01]
#  [1.24105669e-02 1.66014862e+01]
#  [4.41677585e-02 7.48156208e+00]
#  [2.14596344e-01 1.32188041e+01]]


df_r2['clustering_group'].value_counts()

# 결과
# clustering_group    count
# 0                   3194
# 2                   2842
# 3                   2791
# 1                   1642


cluster0_df_r2 = df_r2[df_r2['clustering_group']==0]
cluster1_df_r2 = df_r2[df_r2['clustering_group']==1]
cluster2_df_r2 = df_r2[df_r2['clustering_group']==2]
cluster3_df_r2 = df_r2[df_r2['clustering_group']==3]
print(cluster0_df_r2.shape, cluster1_df_r2.shape, cluster2_df_r2.shape, cluster3_df_r2.shape)

print('min ~ max')
print(cluster0_df_r2['range_wire_guide_10_99'].min(), cluster0_df_r2['range_wire_guide_10_99'].max())
print(cluster1_df_r2['range_wire_guide_10_99'].min(), cluster1_df_r2['range_wire_guide_10_99'].max())
print(cluster2_df_r2['range_wire_guide_10_99'].min(), cluster2_df_r2['range_wire_guide_10_99'].max())
print(cluster3_df_r2['range_wire_guide_10_99'].min(), cluster3_df_r2['range_wire_guide_10_99'].max())

# 결과
# (3194, 324) (1642, 324) (2842, 324) (2791, 324)
# min ~ max
# 8.936965812 11.89166667
# 14.83956044 19.9
# 5.052777778 8.929166667
# 11.73839009 14.96612587


cluster0_df_r2.to_csv(r'./133/cluster0_df.csv', index=False, encoding='utf-8')
cluster1_df_r2.to_csv(r'./133/cluster1_df.csv', index=False, encoding='utf-8')
cluster2_df_r2.to_csv(r'./133/cluster2_df.csv', index=False, encoding='utf-8')
cluster3_df_r2.to_csv(r'./133/cluster3_df.csv', index=False, encoding='utf-8')
