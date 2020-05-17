import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./data/result/final-result.txt')
df_BOA = pd.read_csv('./data/result/final-result-BOA.txt')
df_ARIMA = pd.read_csv('./data/result/final-result-ARIMA.txt')
df.columns = ['code', 'score']
df_BOA.columns = ['code', 'score']
df_ARIMA.columns = ['code', 'score']
df_score = df['score']
df_BOA_score = df_BOA['score']
df_ARIMA_score = df_ARIMA['score']
df_final = pd.DataFrame({
    'ARIMA': df_ARIMA_score,
    'with BOA': df_BOA_score,
    'without BOA': df_score,
})
x = []
for i in range(1000):
    x.append(i/10000.0)
df_final.plot.kde(ind=x)
plt.title("Density function of score with/out BOA module")
plt.xlabel("score")
plt.axvline(df_score.mean(), linestyle='--', label='without BOA mean', color = 'green')
plt.axvline(df_BOA_score.mean(), linestyle='--', label='with BOA mean', color='orange')
plt.axvline(df_ARIMA_score.mean(), linestyle='--', label='ARIMA mean', color='blue')
plt.show()
