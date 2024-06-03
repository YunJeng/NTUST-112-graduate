#-----------------------------Need upgrade-------------------------------------#
#1. 模型參數調整 check
#2. 結果 & 推薦的頁面優化及完成 check
#3. 回上一頁功能(或是能從推薦去跳轉到不同工作內容的推薦)
#4. start pages
#5. 照片 & 背景 check
    #要換成自己拍的台科大照片
#6. 顏色 & 字體 check
    #交疊的雷達圖有點太大，要縮小

#---------------------------------套件們-------------------------------------#
from flask import Flask, request, render_template, session
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # 設定matplotlib使用'Agg'後端來儲存圖表，無需顯示伺服器
from sklearn.linear_model import Lasso, LinearRegression
#from sklearn.model_selection import train_test_split

df = pd.read_excel(r'/home/12yun06/mysite/tes3_standard.xlsx')

# 選取指定的資料行
rows_dict = {
    'work_1': [0, 1, 3, 4, 6, 8, 9, 10, 11, 13, 14],
    'work_2': [0, 5, 9, 11, 14],
    'work_3': [0, 6, 7, 8, 12],
    'work_4': [3, 5]
}
selected_rows1 = [0, 1, 3, 4, 6, 8, 9, 10, 11, 13, 14]
selected_rows2 = [0, 5, 9, 11, 14]
selected_rows3 = [0, 6, 7, 8, 12]
selected_rows4 = [3, 5]
df_selected1 = df.iloc[selected_rows1]
df_selected2 = df.iloc[selected_rows2]
df_selected3 = df.iloc[selected_rows3]
df_selected4 = df.iloc[selected_rows4]

# 重新構建 DataFrame
columns = ['E', 'A', 'C', 'N', 'O', 'SATI']
df1 = pd.DataFrame({col: df_selected1[[col]].values.flatten() for col in columns})
df1*=100
df2 = pd.DataFrame({col: df_selected2[[col]].values.flatten() for col in columns})
df2*=100
df3 = pd.DataFrame({col: df_selected3[[col]].values.flatten() for col in columns})
df3*=100
df4 = pd.DataFrame({col: df_selected4[[col]].values.flatten() for col in columns})
df4*=100
# 分離數據為訓練集和測試集
X1 = df1[['E', 'A', 'C', 'N', 'O']]
y1 = df1['SATI']
X2 = df2[['E', 'A', 'C', 'N', 'O']]
y2 = df2['SATI']
X3 = df3[['E', 'A', 'C', 'N', 'O']]
y3 = df3['SATI']
X4 = df4[['E', 'A', 'C', 'N', 'O']]
y4 = df4['SATI']

regressor1 = Lasso(alpha=0.007)
regressor2 = Lasso(alpha=0.01)
regressor3 = Lasso(alpha=0.8)
regressor4 = LinearRegression()

regressor1.fit(X1, y1)
regressor2.fit(X2, y2)
regressor3.fit(X3, y3)
regressor4.fit(X4, y4)

app = Flask(__name__)
app.secret_key = 'your_secret_key'   #設定一個安全鑰匙以安全簽署會話(session)

#---------------------------------預測模型-------------------------------------#
# 載入預訓練的機器學習模型
models = {
    'work_1': regressor1,
    'work_2': regressor2,
    'work_3': regressor3,
    'work_4': regressor4
}

#---------------------------------Big5計算-------------------------------------#
def process_input(input_list):
    answers = list(map(int, input_list))

    # Reverse scoring for certain questions
    for index in [1, 3, 4, 5, 6, 7, 8, 11, 15, 19, 20, 22, 23, 24, 27]:
        answers[index] = 6 - answers[index] # 反向計分
    big5 = {
        'E': sum(answers[i] for i in range(len(answers)) if (i % 5) == 0),
        'A': sum(answers[i] for i in range(len(answers)) if (i % 5) == 1),
        'C': sum(answers[i] for i in range(len(answers)) if (i % 5) == 2),
        'N': sum(answers[i] for i in range(len(answers)) if (i % 5) == 3),
        'O': sum(answers[i] for i in range(len(answers)) if (i % 5) == 4)
    }
    for dimension, value in big5.items():
        big5[dimension] = ((value - 6) / 24) * 100
    return big5

#---------------------------------輸出圖形（雷達圖）-------------------------------------#
def plot_radar(ax, labels, stats, color, alpha):
    stats = np.array(stats)
    stats = np.concatenate((stats, [stats[0]]))
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    ax.fill(angles, stats, color=color, alpha=alpha)
    for angle, stat in zip(angles, stats):
        ax.text(angle, stat, f'{stat:.1f}%', horizontalalignment='center', verticalalignment='center', fontsize=12, color='black')

def generate_radar_chart(big5, filename, overlay_data=None,figsize=(5, 5)):
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    # Plot main radar chart
    big5_labels = np.array(list(big5.keys()))
    big5_stats = np.array(list(big5.values()))
    plot_radar(ax, big5_labels, big5_stats, color='#115799b2', alpha=0.5)

    if overlay_data:
        overlay_stats = np.array(list(overlay_data.values()))   # 將值提取為一維數組
        plot_radar(ax, big5_labels, overlay_stats, color='#408080', alpha=0.5)

    ax.set_yticklabels([])
    ax.set_xticks(np.linspace(0, 2*np.pi, len(big5_labels), endpoint=False))
    ax.set_xticklabels(big5_labels, fontsize=15)

    plt.savefig(filename, transparent=True)
    plt.close()

#---------------------------------API-------------------------------------#
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        page = int(request.form.get('page', 1))
        input_list = session.get('answers', [])

        for i in range(1, 11):  # Each page has 10 questions
            question_index = (page - 1) * 10 + i
            response = request.form.get(f'question{question_index}')
            if response is not None:
                response = int(response)
                if len(input_list) >= question_index:
                    input_list[question_index - 1] = response
                else:
                    input_list.append(response)

        session['answers'] = input_list
        next_page = page + 1
        if next_page > 3:
            big5_scores = process_input(input_list)
            # 原本big5變數的雷達圖
            generate_radar_chart(big5_scores, '/home/12yun06/mysite/static/big5.png')
            # 原本big5變數與產能規劃的交疊雷達圖
            overlay_data1 = {
                'E': 64.4,
                'A': 74.6,
                'C': 81.1,
                'N': 26.9,
                'O': 66.7
            }
            generate_radar_chart(big5_scores, '/home/12yun06/mysite/static/big5_capacity.png', overlay_data1, figsize=(3.5, 3.5))
            # 原本big5變數與設施規劃的交疊雷達圖
            overlay_data2 = {
                'E': 66.7,
                'A': 75.0,
                'C': 66.7,
                'N': 33.3,
                'O': 70.0
            }
            generate_radar_chart(big5_scores, '/home/12yun06/mysite/static/big5_facility.png', overlay_data2, figsize=(3.5, 3.5))
            # 原本big5變數與排程最佳化的交疊雷達圖
            overlay_data3 = {
                'E': 47.5,
                'A': 63.3,
                'C': 75.8,
                'N': 42.5,
                'O': 64.2
            }
            generate_radar_chart(big5_scores, '/home/12yun06/mysite/static/big5_scheduling.png', overlay_data3, figsize=(3.5, 3.5))
            # 原本big5變數與自動化控制的交疊雷達圖
            overlay_data4 = {
                'E': 47.9,
                'A': 81.3,
                'C': 68.8,
                'N': 25.0,
                'O': 66.7
            }
            generate_radar_chart(big5_scores, '/home/12yun06/mysite/static/big5_automation.png', overlay_data4, figsize=(3.5, 3.5))
            #generate_radar_chart(big5_scores)
            input_data = pd.DataFrame([big5_scores])
            predictions = {name: np.clip(model.predict(input_data)[0], 0, 100) for name, model in models.items()}
            return render_template('result.html', predictions=predictions, big5_scores=big5_scores, radar_charts=['big5.png', 'big5_capacity.png', 'big5_facility.png', 'big5_scheduling.png', 'big5_automation.png'])
        else:
            return render_template(f'indexP{next_page}.html', page=next_page)
    else:
        session['answers'] = []
        return render_template('indexP1.html', page=1)

@app.route('/Capacity_Planning')
def detail_Capacity_Planning():
    return render_template('Capacity_Planning.html')

@app.route('/Scheduling_Optimization')
def detail_Scheduling_Optimization():
    return render_template('Scheduling_Optimization.html')

@app.route('/Facility_Planning')
def detail_Facility_Planning():
    return render_template('Facility_Planning.html')

@app.route('/Automation_Control')
def detail_Automation_Control():
    return render_template('Automation_Control.html')

#---------------------------------啟動應用程式-------------------------------------#
if __name__ == '__main__':
    app.run(debug=True) # 啟動應用程式，開啟調試模式