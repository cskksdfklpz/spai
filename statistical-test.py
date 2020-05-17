import pandas as pd

codes = open('./data/stock-codes.txt')

code = codes.readline()
code = code.strip(" ")
code = code.strip('\n')
code = code.strip('\t')
print(code)
df = pd.read_csv('./data/features'+code+'.csv')
df = df.drop('Date', axis=1)
df = df.drop('Close', axis=1)
df = df.drop('Close_y', axis=1)
df = df.drop('Unnamed: 0', axis=1)
features = df.columns
count = {feature: 0 for feature in features}

while code:
    code = code.strip('\s')
    code = code.strip('\n')
    code = code.strip('\t')
    print(code)
    df = pd.read_csv('./data/features'+code+'.csv')
    df = df.drop('Date', axis=1)
    df = df.drop('Close', axis=1)
    df = df.drop('Close_y', axis=1)
    df = df.drop('Unnamed: 0', axis=1)
    df_corr = df.corr(method='pearson')
    features = df.columns
    
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            if df_corr.values[i,j] > 0.8:
                count[features[i]] += 1
    code = codes.readline()

for key in count.items():
    count[key[0]] /= 147.0
list = sorted(count.items(), key=lambda x:float(x[1]), reverse=True)
n = 0
names = []
scores = []
for item in list:
    key = item[0]
    val = item[1]
    names.append(key)
    scores.append(val)
    c = " & "
    if n%3 == 2:
        c = " \\\\ "
    print(key+" & " + str(val)[0:5] + c)
    n += 1


import matplotlib.pyplot as plt

fig, ax = plt.subplots()
b = ax.barh(range(len(names)), scores)
 
#为横向水平的柱图右侧添加数据标签。
for rect in b:
    w = rect.get_width()
    ax.text(w, rect.get_y()+rect.get_height()/2, '%d' %
            int(w), ha='left', va='center')
 
#设置Y轴纵坐标上的刻度线标签。
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names)
 
#不要X横坐标上的label标签。
plt.xticks(())
 
plt.title('Duplication Index', loc='center')
 
plt.show()
