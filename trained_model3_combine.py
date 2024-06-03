#---------------------------------套件們-------------------------------------#
# 導入必要的套件
import numpy as np
import pandas as pd
# 導入回歸模型套件
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import train_test_split
# 導入模型保存套件
import joblib

#---------------------------------資料整理和模型訓練的功能-------------------------------------#
def process_and_train(work_number, model_type='lasso'):
    # 匯入excel檔案
    df = pd.read_excel(r'C:/Users/YUN/Desktop/VS studio/upgrade version/tes3_standard.xlsx')

    # 選取指定的資料行
    rows_dict = {
        'work_1': [0, 1, 3, 4, 6, 8, 9, 10, 11, 13, 14],
        'work_2': [0, 5, 9, 11, 14],
        'work_3': [0, 6, 7, 8, 12],
        'work_4': [3, 5]
    }
    selected_rows = rows_dict[work_number]
    df_selected = df.iloc[selected_rows]

    # 重新構建 DataFrame
    columns = ['E', 'A', 'C', 'N', 'O', 'SATI']
    df = pd.DataFrame({col: df_selected[[col]].values.flatten() for col in columns})

    # 分離數據為訓練集和測試集
    X = df[['E', 'A', 'C', 'N', 'O']]
    y = df['SATI']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 訓練模型
    if model_type == 'lasso':
        alphas = {
            'work_1': 0.007,
            'work_2': 0.01,
            'work_3': 0.005
        }
        regressor = Lasso(alpha=alphas[work_number])
    elif model_type == 'linear':
        regressor = LinearRegression()

    regressor.fit(X_train, y_train)
    accuracy = regressor.score(X_test, y_test)
    print(f'{work_number} {model_type.capitalize()} Score: ', accuracy)

    # 模型保存
    joblib.dump(regressor, f'{work_number}_{model_type}_model.joblib')

# 主程式
def main():
    works = {
        'work_1': 'lasso',
        'work_2': 'lasso',
        'work_3': 'lasso',
        'work_4': 'linear'  # work_4 使用線性迴歸
    }
    for work, model in works.items():
        process_and_train(work, model)

if __name__ == "__main__":
    main()
