import pandas as pd 
import matplotlib.pyplot as plt 

df_ARIMA = pd.read_csv('./data/result/final-result-ARIMA.txt')
df = pd.read_csv('./data/result/final-result.txt')
df_BOA = pd.read_csv('./data/result/final-result-BOA.txt')

df_ARIMA.columns = ['code', 'score']
df.columns = ['code', 'score']
df_BOA.columns = ['code', 'score']

names = ['ARIMA(benchmark)', 'without BOA', 'with BOA']

score = [df_ARIMA["score"].mean(), df["score"].mean(), df_BOA["score"].mean()]

fig, ax = plt.subplots()

b = ax.bar(range(len(names)), score)

# for rect in b:
#     w = rect.get_height()
#     ax.text(w, rect.get_x()+rect.get_width()/2, '%d' %
#             int(w), ha='left', va='center')
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names)


 
plt.title('Performance Comparison', loc='center')
 
plt.show()