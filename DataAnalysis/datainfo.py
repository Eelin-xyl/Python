import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv(r'Python\\DataAnalysis\\SpotifyFeatures(cleaned).csv', encoding='gbk')

dataset['genre'].hist()

plt.show()

dic = {}
for i in dataset.genre:
    if i not in dic:
        dic[i] = 1
    else:
        dic[i] += 1
for j in dic:
    print(j, dic[j])