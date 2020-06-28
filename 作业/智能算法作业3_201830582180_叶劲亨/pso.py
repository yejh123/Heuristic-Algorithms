import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('data2.csv', delimiter=",")

w_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
c_list = [0.5,1.0,1.5,2.0,2.5,3.0,3.5]

average = pd.DataFrame(columns=['w','c','g_best'])

i  = 0
for w in w_list:
    for c in c_list:
        temp = data[(data['w']==w) & (data['c1=c2']==c)]
        average.loc[i] = [w, c, np.average(temp['g_best'])]
        i += 1

# 绘图
res = np.array(average['g_best']).reshape((len(w_list), len(c_list))).T
plt.figure(figsize=(20,10))

sns.heatmap(res, annot=True, fmt='.3f', xticklabels=w_list, yticklabels=c_list)