from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
df = pd.read_csv('./pca_dataset.csv')
df.head()
print(df.head())


X_scaled = StandardScaler().fit_transform(df)
X_scaled[:5]
print(X_scaled)

features = X_scaled.T
cov_matrix = np.cov(features)
cov_matrix[:5]


values, vectors = np.linalg.eig(cov_matrix)
values[:5]

# Just from this, we can calculate the percentage of explained variance per principal component:

explained_variances = []
for i in range(len(values)):
    explained_variances.append(values[i] / np.sum(values))

print(np.sum(explained_variances), explained_variances)

projected_1 = X_scaled.dot(vectors.T[0])
projected_2 = X_scaled.dot(vectors.T[1])
res = pd.DataFrame(projected_1, columns=["PC1"])
res["PC2"] = projected_2
res["Y"] = projected_1
res.head()
