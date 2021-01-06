import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Python\\MachineLearning\\project\\SpotifyAudioFeaturesApril2019.csv')

dataset=dataset.drop('artist_name',axis=1)
dataset=dataset.drop('track_name',axis=1)
dataset=dataset.drop('track_id',axis=1)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


dataset.drop('popularity', axis=1).corrwith(dataset.popularity).plot(kind = 'bar', grid = True,
                                                   figsize = (12, 8),
                                                   title = "Correlation with Target")
plt.show()

df_1 = dataset[['acousticness','danceability','duration_ms','energy','instrumentalness','key',
                'liveness','loudness','mode','speechiness','tempo','time_signature','valence','popularity']]
# print(df_1.describe())
for item in df_1.columns:
    plt.subplot(4,4,list(df_1.columns).index(item) + 1)
    plt.boxplot(df_1[item],patch_artist = True, labels=[item])
    plt.ylabel('value')
plt.tight_layout()
plt.show()