import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv(r'Python\\DataAnalysis\\SpotifyFeatures2.csv')

corr = dataset.corrwith(dataset.genre)
# visualizing correlaiton with heatmap
plt.figure(figsize=(20,8))
sns.heatmap(corr, vmax=1, vmin=-1, center=0,linewidth=.5,square=True, annot = True, annot_kws = {'size':8},fmt='.1f', cmap='BrBG_r')
plt.title('Correlation')
plt.show()

# dataset=dataset.drop('track_name',axis=1)
# dataset=dataset.drop('track_id',axis=1)
# dataset=dataset.drop('artist_name',axis=1)
# dataset=dataset.drop('key',axis=1)

# dataset.drop('genre', axis=1).corrwith(dataset.genre).plot(kind = 'bar', grid = True,
#                                                    figsize = (12, 8),
#                                                    title = "Correlation with Target")
# plt.show()

sns.boxplot(x = dataset.popularity)
plt.show()
dataset.drop(dataset[dataset.popularity > 80].index, inplace=True)
# dataset.drop(dataset[(dataset.duration_ms > 0.3*10**6) | (dataset.duration_ms < 90000)].index, inplace=True)
# dataset.drop(dataset[dataset.instrumentalness > 0.1].index, inplace=True)
# dataset.drop(dataset[dataset.liveness > 0.51].index, inplace=True)
# dataset.drop(dataset[dataset.loudness < -21.5].index, inplace=True)
# dataset.drop(dataset[dataset.speechiness > 0.21].index, inplace=True)

print(dataset.info())

# dataset = dataset.iloc[:,1:]
# dataset = dataset.drop(columns=0, axis=1)

# dataset.to_csv('Python\DataAnalysis\SpotifyFeatures3.csv')
# print('OK')
# df_1 = dataset[['popularity','acousticness','danceability',
#                 'duration_ms','energy','instrumentalness','liveness','loudness','mode','speechiness']]
# print(df_1.describe())
# for item in df_1.columns:
#     plt.subplot(4,3,list(df_1.columns).index(item) + 1)
#     plt.boxplot(df_1[item],patch_artist = True, labels=[item])
#     plt.ylabel('value')
# plt.tight_layout()
# plt.show()