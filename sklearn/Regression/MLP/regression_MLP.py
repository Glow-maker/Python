from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
import numpy as np
import pandas as pd

# 创建一个简单的回归数据集
X, y = make_regression(n_samples=200, n_features=1, noise=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
mlp = MLPRegressor(random_state=42, max_iter=500)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)

# 计算评估指标
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

metrics = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}
df = pd.DataFrame([metrics])
print(df)