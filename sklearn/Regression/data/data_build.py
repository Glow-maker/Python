from sklearn.datasets import make_regression

import pandas as pd
import numpy as np

X, y = make_regression(n_samples=1000, n_features=50, noise=10, random_state=1)
weights = np.random.rand(X.shape[0])
setIDs = np.random.randint(0, 101, size=X.shape[0])  # Generates random integers between 0 and 100


# 将数据转换为 pandas DataFrame
features_df = pd.DataFrame(X, columns=[f'Feature_{i}' for i in range(X.shape[1])])
target_df = pd.DataFrame(y, columns=['target'])
weights_df = pd.DataFrame(weights, columns=['weight'])
setID_df = pd.DataFrame(setIDs, columns=['setID'])
df = pd.concat([features_df, target_df, weights_df, setID_df], axis=1)

# 导出为 Parquet 文件
df.to_parquet(r'C:\Users\jintao\OneDrive\work\Vscode\sklearn\Regression\data\test1.parquet')