import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Python\\CW3\\playerdata.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# from boto import sns
# sns.boxplot(x = dataset.ejection_fraction)
# plt.show()

dataset.drop('TARGET', axis=1).corrwith(dataset.TARGET).plot(kind = 'bar', grid = True,
                                                   figsize = (12, 8),
                                                   title = "Correlation with Target")
plt.show()

df_1 = dataset[['GS','MIN','2FGM','2FGA','3FGM','3FGA','FTM','FTA','OREB',
                'DREB','AST','STL','BCK','TO','PF','PTS']]
# print(df_1.describe())
for item in df_1.columns:
    plt.subplot(4,4,list(df_1.columns).index(item) + 1)
    plt.boxplot(df_1[item],patch_artist = True, labels=[item])
    plt.ylabel('value')
plt.tight_layout()
plt.show()