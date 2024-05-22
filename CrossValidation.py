## 導入Python數據處理套件
import numpy as np
import pandas as pd
## 導入繪圖套件
import matplotlib.pyplot as plt
## 導入回歸模型套件
from sklearn.linear_model import LinearRegression, Lasso, Ridge
## 導入多項式套件，建構多項式迴歸模型所需的套件
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
## 導入區分訓練集與測試集套件
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
##交叉驗證
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score


#---------------------------------資料整理-------------------------------------#
# 匯入excel檔案
df = pd.read_excel(r'/Users/zhen/Desktop/VS studio/upgrade version/tes3_standard.xlsx')

# 選取指定的資料行
#selected_rows = [0, 1, 3, 4, 6, 8, 9, 10, 11 ,13 ,14]  work_1[1, 2, 4, 5, 7, 9, 10, 11, 12 ,14 ,15] # Python中的索引是從0開始
#selected_rows = [0, 5, 9, 11 ,14]  work_2[1, 6, 10, 12 ,15]
selected_rows = [0, 6, 7, 8 ,12]  #work_3[1, 7, 8, 9 ,13]
#selected_rows = [2 ,4]  work_4[3, 5]

df_selected = df.iloc[selected_rows]

# 重新構建 DataFrame 中只包含需要的列
E = df_selected[['E']]
A = df_selected[['A']]
C = df_selected[['C']]
N = df_selected[['N']]
O = df_selected[['O']]
SATI = df_selected[['SATI']]
data_dict = {'E': E.values.flatten(), 'A': A.values.flatten(), 'C': C.values.flatten(), 'N': N.values.flatten(), 'O': O.values.flatten(), 'SATI': SATI.values.flatten()}
df = pd.DataFrame(data_dict)


#---------------------------------預測模型-------------------------------------#
## 將數據集分為訓練集與測試集
X = df[['E', 'A', 'C', 'N', 'O']]
y = df['SATI']

## 分離訓練和測試數據
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)






# 创建 Lasso 回归模型实例
lasso_regressor = Lasso(alpha=0.00030000000009381756)  # alpha 是正则化强度
lasso_regressor.fit(X_train, y_train)  # 训练模型

# 使用测试集评估模型
lasso_score = lasso_regressor.score(X_test, y_test)
print('Lasso Score: ', lasso_score)



# 使用交叉验证评估模型性能

scores = cross_val_score(lasso_regressor, X, y, cv=5, scoring='neg_mean_squared_error')
print("Mean Squared Error scores:", -scores)
print("Average MSE:", -scores.mean())




from sklearn.model_selection import GridSearchCV

# 參數網格
param_grid = {'alpha': [0.004,0.006,0.007,0.008,0.01,0.0071]}

# GridSearchCV
grid_search = GridSearchCV(Lasso(), param_grid, cv=5, scoring='neg_mean_squared_error')

# 尋找參數
grid_search.fit(X, y)

# 输出最佳参数和对应的性能评分
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score (MSE):", -grid_search.best_score_)
