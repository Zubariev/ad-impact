import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm

# # === Step 1: Load dataset ===
# df = pd.read_excel("/Users/upqrade/econ/data/input_pca.xlsx")

# # Keep only numeric columns
# data = df.select_dtypes(include=[np.number]).dropna()

# # === Step 2: Correlation analysis ===
# corr_matrix = data.corr().abs()  # absolute correlations
# threshold = 0.85  # correlation threshold

# # Upper triangle of correlation matrix
# upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# # Drop columns with high correlation
# to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
# selected_features = [col for col in data.columns if col not in to_drop]

# print("Highly correlated columns removed:", to_drop)
# print("Selected features for Expanding PCA:", selected_features)

# # === Step 3: Standardize selected data ===
# data_selected = data[selected_features]
# scaler = StandardScaler()
# scaled_data = scaler.fit_transform(data_selected)

# # === Step 4: Define Expanding PCA function ===
# def expanding_pca(data, min_window=10, n_components=2):
#     """
#     Apply Expanding PCA on time-series data.
    
#     Args:
#         data: ndarray (T x N) where T = time, N = features
#         min_window: minimum number of rows before starting PCA
#         n_components: number of PCA components
        
#     Returns:
#         pcs: array of principal component scores over time
#         explained_var: array of explained variance ratio for PC1
#     """
#     T, N = data.shape
#     pcs = []
#     explained_var = []

#     for t in range(1, T+1):
#         if t < min_window:
#             pcs.append([np.nan] * n_components)
#             explained_var.append(np.nan)
#         else:
#             window = data[:t, :]  # expanding window
#             pca = PCA(n_components=n_components)
#             pca.fit(window)
#             pcs.append(pca.transform(window)[-1])  # last observation’s PCs
#             explained_var.append(pca.explained_variance_ratio_[0])  # PC1 variance ratio
    
#     return np.array(pcs), np.array(explained_var)

# # === Step 5: Apply Expanding PCA ===
# pcs, ev = expanding_pca(scaled_data, min_window=10, n_components=2)

# # === Step 6: Save results ===
# result_df = pd.DataFrame({
#     "PC1": pcs[:, 0],
#     "PC2": pcs[:, 1],
#     "ExplainedVar_PC1": ev
# })

# print("✅ Expanding PCA finished. Results saved to mdpca_results.xlsx")

df = pd.read_excel("/Users/upqrade/econ/data/main.xlsx")

df.columns = df.columns.str.lower()

df['display_ins_scaled'] = (df['display_ins'] - df['display_ins'].min()) / (df['display_ins'].max() - df['display_ins'].min())
df['max_click_scaled'] = (df['max_click'] - df['max_click'].min()) / (df['max_click'].max() - df['max_click'].min())
df['max_ins_scaled'] = (df['max_ins'] - df['max_ins'].min()) / (df['max_ins'].max() - df['max_ins'].min())
df['all_regions_alarm_scaled'] = (df['all_regions_alarm'] - df['all_regions_alarm'].min()) / (df['all_regions_alarm'].max() - df['all_regions_alarm'].min())
df['visits_dynamics_scaled'] = (df['visits_dynamics'] - df['visits_dynamics'].min()) / (df['visits_dynamics'].max() - df['visits_dynamics'].min())
df['tv_sov_scaled'] = (df['tv_sov'] - df['tv_sov'].min()) / (df['tv_sov'].max() - df['tv_sov'].min())
df['tv_sov_d_scaled'] = (df['tv_sov_d'] - df['tv_sov_d'].min()) / (df['tv_sov_d'].max() - df['tv_sov_d'].min())
df['radio_wgrp_scaled'] = (df['radio_wgrp'] - df['radio_wgrp'].min()) / (df['radio_wgrp'].max() - df['radio_wgrp'].min())
# df['social_ins_scaled'] = (df['social_ins'] - df['social_ins'].min()) / (df['social_ins'].max() - df['social_ins'].min())
df['video_ins_scaled'] = (df['video_ins'] - df['video_ins'].min()) / (df['video_ins'].max() - df['video_ins'].min())
df['ooh_grp_scaled'] = (df['ooh_grp'] - df['ooh_grp'].min()) / (df['ooh_grp'].max() - df['ooh_grp'].min())

df.to_excel("/Users/upqrade/econ/data/mdpca_results.xlsx", index=False)